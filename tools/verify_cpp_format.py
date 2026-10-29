"""Compare the C++ decode/writer with independent Python and LibRaw readback."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
from pathlib import Path

import ci_fixture
import numpy as np
import rawpy
from nx_ctudio._tiff import read, serialize
from nx_ctudio.nefx import encode
from nx_ctudio.pixel_shift import reconstruct


def pixels(index: int, shape: tuple[int, int] = (48, 64)) -> np.ndarray:
    y, x = np.indices(shape)
    return ((53 * x + 97 * y + index * 311 + (x * y + index * 7) % 127) % 16384).astype(
        np.uint16
    )


def compare_tree(actual, expected) -> None:
    assert set(actual.tags) == set(expected.tags)
    pointer_tags = {273, 279, 513, 514, 324, 325}
    for code, tag in expected.tags.items():
        other = actual.tags[code]
        assert tag.kind == other.kind, code
        assert tag.nikon_header == other.nikon_header, code
        assert len(tag.children) == len(other.children), code
        if tag.children:
            for left, right in zip(other.children, tag.children, strict=True):
                compare_tree(left, right)
        elif code not in pointer_tags | {305, 306}:
            assert tag.payload == other.payload, code
    assert actual.images == expected.images


def rejections(executable: Path, directory: Path) -> list[str]:
    passed = []
    for change in (
        "model",
        "he",
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
        sources = ci_fixture.sequence(folder)
        target = folder / "output.nefx"
        tree = read(sources[-1].read_bytes())
        exif = tree.tags[34665].children[0]
        maker = exif.tags[37500].children[0]
        raw = tree.tags[330].children[1]
        if change == "model":
            tree.set(272, 2, b"NIKON D850\0")
        elif change == "he":
            data = bytearray(maker.tags[0x51].payload)
            struct.pack_into("<H", data, 10, 13)
            maker.set(0x51, 7, bytes(data))
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
        if change == "existing":
            assert target.read_bytes() == b"preserve"
        else:
            assert not target.exists(), change
        assert not list(folder.glob(".cnefx-*")), change
        passed.append(change)
    return passed


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
                expected = reconstruct(
                    [pixels(i) for i in range(count)],
                    (4, 4, 56, 40),
                    b"0100" + bytes([(4, 8, 16, 32).index(count) + 1]) + bytes(23),
                    100,
                )
                with rawpy.imread(str(target)) as image:
                    np.testing.assert_array_equal(image.raw_image[..., :3], expected)
                compare_tree(
                    read(target.read_bytes()),
                    read(encode(sources[0].read_bytes(), expected, count)),
                )
                assert hashes == [
                    hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
                ]
                cases.append(
                    {
                        "model": model,
                        "count": count,
                        "corrected": corrected,
                        "samples": int(expected.size),
                        "different_samples": 0,
                        "metadata_and_payloads_equal": True,
                        "input_sha256": hashes,
                        "output_sha256": hashlib.sha256(
                            target.read_bytes()
                        ).hexdigest(),
                    }
                )
    report = {
        "evidence_kind": "authored_cpp_python_format_comparison_real_libraw",
        "passed": True,
        "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "rawpy": rawpy.__version__,
        "rawpy_libraw": rawpy.libraw_version,
        "cases": cases,
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
