"""Check the native writer against saved expectations and LibRaw readback."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
from pathlib import Path

import numpy as np
import rawpy

import ci_fixture
import ticoraw_fixture
from tiff_fixture import read, serialize, tree_digest

# One case per model covers every capture count and both HE codes.
HIGH_EFFICIENCY = (
    ("NIKON Z 8", 4, 13),
    ("NIKON Z f", 8, 14),
    ("NIKON Z6_3", 16, 13),
    ("NIKON Z5_2", 32, 14),
)


def pixels(index: int, shape: tuple[int, int] = (48, 64)) -> np.ndarray:
    y, x = np.indices(shape)
    return ((53 * x + 97 * y + index * 311 + (x * y + index * 7) % 127) % 16384).astype(
        np.uint16
    )


REASONS = {
    "he_payload": "Expected a Nikon HE codestream.",
    "he_code": "Unsupported Nikon RAW compression code.",
    "he_signature": "Unsupported Nikon HE codestream profile.",
    "he_profile": "Unsupported Nikon HE codestream profile.",
    "he_length": "Nikon HE codestream length disagrees with its RAW strip.",
    "he_truncated": "Corrupt Nikon HE codestream.",
}


def rejections(executable: Path, directory: Path) -> list[str]:
    passed = []
    for change in (
        "model",
        "he_payload",
        "he_code",
        "he_signature",
        "he_profile",
        "he_length",
        "he_truncated",
        "selector",
        "cfa",
        "crop",
        "timestamp",
        "timestamp_sign",
        "serial",
        "iso",
        "shutter",
        "reversed",
        "duplicate",
        "hardlink",
        "existing",
    ):
        folder = directory / f"reject-{change}"
        folder.mkdir()
        if change.startswith("he_") and change != "he_payload":
            sources = ticoraw_fixture.sequence(
                folder, 4, model="NIKON Z 8", code=13, seed=0
            )[0]
        else:
            sources = ci_fixture.sequence(folder)
        target = folder / "output.nefx"
        tree = read(sources[-1].read_bytes())
        exif = tree.tags[34665].children[0]
        maker = exif.tags[37500].children[0]
        raw = tree.tags[330].children[1]
        if change == "model":
            tree.set(272, 2, b"NIKON D850\0")
        elif change in ("he_payload", "he_code"):
            data = bytearray(maker.tags[0x51].payload)
            struct.pack_into("<H", data, 10, 13 if change == "he_payload" else 15)
            maker.set(0x51, 7, bytes(data))
        elif change in ("he_signature", "he_profile"):
            # The CAP vendor signature, or the packet order of the picture header.
            stream = bytearray(raw.images[273][0])
            stream[6 if change == "he_signature" else 63] ^= 0x20
            raw.images[273] = [bytes(stream)]
        elif change in ("he_length", "he_truncated"):
            stream = raw.images[273][0]
            if change == "he_length":
                stream = stream[:-1]
            else:
                # Cut the entropy data short and declare the shorter length.
                stream = bytearray(stream[: len(stream) // 2] + b"\xff\x11")
                stream[42:46] = len(stream).to_bytes(4, "big")
            raw.images[273] = [bytes(stream)]
            raw.numbers(279, 4, len(stream))
        elif change == "selector":
            maker.set(0x5E, 7, b"9999" + bytes(24))
        elif change == "cfa":
            raw.set(33422, 1, bytes((2, 1, 1, 0)))
        elif change == "crop":
            maker.numbers(0x45, 3, 0, 0, 64, 48)
        elif change == "timestamp":
            exif.set(36867, 2, b"2026:02:30 00:00:00\0")
        elif change == "timestamp_sign":
            exif.set(36867, 2, b"2026:02:28 -0:00:00\0")
        elif change == "serial":
            exif.set(42033, 2, b"different-camera\0")
        elif change == "iso":
            exif.numbers(34855, 3, 200)
        elif change == "shutter":
            maker.numbers(0xA7, 4, 999)
        sources[-1].write_bytes(serialize(tree))
        if change == "reversed":
            sources.reverse()
        elif change == "duplicate":
            sources[-1] = sources[0]
        elif change == "hardlink":
            sources[-1].unlink()
            os.link(sources[0], sources[-1])
        elif change == "existing":
            target.write_bytes(b"preserve")
        result = subprocess.run(
            [str(executable), "--merge", str(target), *map(str, sources)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 1, (change, result)
        assert result.stderr, change
        assert REASONS.get(change, "") in result.stderr, (change, result.stderr)
        if change == "existing":
            assert target.read_bytes() == b"preserve"
        else:
            assert not target.exists(), change
        assert not list(folder.glob(".cnefx-*")), change
        passed.append(change)
    return passed


def high_efficiency(executable: Path, directory: Path) -> list[dict]:
    golden = ci_fixture.GOLDEN["high_efficiency"]["cases"]
    cases = []
    for seed, (model, count, code) in enumerate(HIGH_EFFICIENCY):
        name = f"{model.replace(' ', '_')}-{count}-{code}"
        expected = golden[name]
        folder = directory / f"he-{name}"
        (folder / "he").mkdir(parents=True)
        (folder / "twin").mkdir()
        sources, streams, mosaics = ticoraw_fixture.sequence(
            folder / "he", count, model=model, code=code, seed=1000 * seed
        )
        hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in sources]
        assert [hashlib.sha256(s).hexdigest() for s in streams] == expected[
            "codestream_sha256"
        ]
        for source, mosaic, pinned in zip(
            sources, mosaics, expected["mosaic_sha256"], strict=True
        ):
            assert hashlib.sha256(mosaic.astype("<u2").tobytes()).hexdigest() == pinned
            plane = source.with_suffix(".samples")
            subprocess.run(
                [str(executable), "--decode", str(plane), str(source)],
                check=True,
                timeout=30,
            )
            assert hashlib.sha256(plane.read_bytes()).hexdigest() == pinned
        # The same samples stored packed must merge to the same RGB samples.
        original = ci_fixture.pixels
        ci_fixture.pixels = lambda index, shape: mosaics[index]
        try:
            twins = ci_fixture.sequence(
                folder / "twin", count, model=model, shape=mosaics[0].shape
            )
        finally:
            ci_fixture.pixels = original
        outputs = []
        for inputs in (sources, twins):
            target = inputs[0].parent / "output.nefx"
            subprocess.run(
                [str(executable), "--merge", str(target), *map(str, inputs)],
                check=True,
                timeout=60,
            )
            with rawpy.imread(str(target)) as image:
                outputs.append(image.raw_image[..., :3].copy())
        assert outputs[0].shape == outputs[1].shape
        assert np.array_equal(outputs[0], outputs[1])
        assert hashes == [hashlib.sha256(p.read_bytes()).hexdigest() for p in sources]
        cases.append(
            {
                "model": model,
                "count": count,
                "compression": code,
                "decoded_samples": int(sum(m.size for m in mosaics)),
                "merged_samples": int(outputs[0].size),
                "decode_matches_pinned_oracle": True,
                "merge_matches_packed_twin": True,
                "input_sha256": hashes,
            }
        )
    return cases


def run(executable: Path, directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    cases = []
    ci_fixture.pixels = pixels
    for model in ("NIKON Z 8", "NIKON Z f", "NIKON Z6_3", "NIKON Z5_2"):
        for count in (4, 8, 16, 32):
            for corrected in (False, True):
                folder = (
                    directory / f"{model.replace(' ', '_')}-{count}-{int(corrected)}"
                )
                folder.mkdir()
                sources = ci_fixture.sequence(folder, count, model=model)
                hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in sources]
                target = folder / "output.nefx"
                command = [
                    str(executable),
                    "--merge-corrected" if corrected else "--merge",
                    str(target),
                    *map(str, sources),
                ]
                result = subprocess.run(
                    command, capture_output=True, text=True, timeout=30
                )
                if result.returncode:
                    raise RuntimeError(f"{model}/{count}: {result.stderr}")
                golden = ci_fixture.merge_expectations(model, count, 1)
                assert hashes == golden["input_sha256"]
                expected = golden["outputs"][0]
                with rawpy.imread(str(target)) as image:
                    ci_fixture.check_rgb(image.raw_image[..., :3], expected)
                assert tree_digest(target.read_bytes()) == expected["tree_sha256"]
                assert hashes == [
                    hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
                ]
                cases.append(
                    {
                        "model": model,
                        "count": count,
                        "corrected": corrected,
                        "samples": expected["samples"],
                        "samples_match": True,
                        "metadata_and_payloads_equal": True,
                        "input_sha256": hashes,
                        "output_sha256": hashlib.sha256(
                            target.read_bytes()
                        ).hexdigest(),
                    }
                )
    report = {
        "evidence_kind": "authored_native_saved_format_regression_real_libraw",
        "expectations_sha256": ci_fixture.GOLDEN_SHA256,
        "passed": True,
        "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "rawpy": rawpy.__version__,
        "rawpy_libraw": rawpy.libraw_version,
        "cases": cases,
        "high_efficiency": high_efficiency(executable, directory),
        "rejections": rejections(executable, directory),
    }
    (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--work-directory", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.executable.resolve(), args.work_directory.resolve())
    print(json.dumps({"cases": len(report["cases"]), "passed": report["passed"]}))


if __name__ == "__main__":
    main()
