# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.4.6", "rawpy==0.27.1", "tifffile==2026.9.15"]
# ///
"""Verify nonzero sparse metadata through the full synthetic NEF merge pipeline."""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import struct
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
import rawpy
import tifffile
from sparse_reference import Selection, repair
from verify_radial import correct, parameters
from verify_vendor_matrix import DLL_SHA256, reconstruct, sha256, windows


def patch_fixture(source: Path, target: Path, selection: Selection, iso: int) -> dict:
    content = bytearray(source.read_bytes())
    with tifffile.TiffFile(io.BytesIO(content)) as image:
        exif = image.pages[0].tags[34665].valueoffset
        count = struct.unpack_from("<H", content, exif)[0]
        tags = {
            tag.code: tag
            for index in range(count)
            for tag in [tifffile.TiffTag.fromfile(image, offset=exif + 2 + 12 * index)]
        }
        iso_tag = tags[0x8827]
        if iso_tag.count != 1 or iso_tag.dtype != 3:
            raise ValueError("Expected one SHORT ISO value.")
        maker = tags[0x927C]
        base = maker.valueoffset + 10
        with tifffile.TiffFile(
            io.BytesIO(content[base : base + maker.count - 10])
        ) as notes:
            tag = notes.pages[0].tags[0x5E]
            if tag.count != 28:
                raise ValueError("Unexpected sparse selector size.")
            offset = base + tag.valueoffset
        if content[offset : offset + 4] != b"0100":
            raise ValueError("Unexpected sparse selector version.")
        selector = bytearray(selection.pack())
        selector[0] = content[offset + 4]
        replacements = (
            (offset + 4, bytes(selector)),
            (iso_tag.valueoffset, struct.pack("<H", iso)),
        )
    patches = []
    for offset, data in replacements:
        patches.append(
            {
                "offset": offset,
                "before_hex": content[offset : offset + len(data)].hex(),
                "after_hex": data.hex(),
            }
        )
        content[offset : offset + len(data)] = data
    with target.open("xb") as stream:
        stream.write(content)
    return {
        "source": str(source),
        "source_sha256": sha256(source),
        "path": str(target),
        "sha256": sha256(target),
        "patches": patches,
    }


def preprocess(
    frames: list[np.ndarray], selection: Selection, iso: int
) -> list[np.ndarray]:
    height, width = frames[0].shape
    if len(frames) in (4, 16):
        if iso > 3200:
            return frames
        output = []
        for start in range(0, len(frames), 4):
            group = [p.ravel().tolist() for p in frames[start : start + 4]]
            output.extend(
                np.asarray(p, dtype=np.uint16).reshape(height, width)
                for p in repair(group, width, height, selection)
            )
        return output
    half = len(frames) // 2
    output = [p.copy() for p in frames[:half]]
    for index in range(half):
        a, b = frames[index], frames[index + half]
        output[index][2:] = ((a[2:].astype(np.uint32) + b[:-2]) >> 1).astype(np.uint16)
        if iso <= 3200:
            for y, x in selection.coordinates(width, height):
                output[index][y, x] = b[y - 2, x]
                output[index][y + 2, x] = a[y + 2, x]
    return output


def expected_output(
    frames: list[np.ndarray],
    selection: Selection,
    iso: int,
    crop: tuple[int, ...],
    radial: bool,
    correction_parameters: tuple[list[int], list[list[int]]],
) -> np.ndarray:
    height, width = frames[0].shape
    reduced = preprocess(frames, selection, iso)
    scale = 2 if len(reduced) == 16 else 1
    output = reconstruct(reduced, (1, 1, width - 1, height - 1))
    if radial:
        output = correct(output, frames[0].shape, scale, *correction_parameters)
    left, top, crop_width, crop_height = crop
    return output[
        scale * (top - 1) : scale * (top - 1 + crop_height),
        scale * (left - 1) : scale * (left - 1 + crop_width),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dll", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--wine-prefix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--counts", type=int, nargs="+", choices=(4, 8, 16, 32), default=[4, 8, 16, 32]
    )
    parser.add_argument("--wine", default="/usr/lib/wine/wine64")
    args = parser.parse_args()
    logging.getLogger("tifffile").setLevel(logging.ERROR)
    if sha256(args.dll) != DLL_SHA256:
        parser.error("Unexpected DLL checksum.")
    manifest = json.loads(args.manifest.read_text())
    if manifest["evidence_kind"] != "synthetic_packed14_input_fixture":
        parser.error("Expected synthetic packed-14 inputs.")
    args.output.mkdir(parents=True, exist_ok=False)
    width, height = manifest["width"], manifest["height"]
    selection = Selection(height, width, first_row=12, inner_step=1, inner_after=5)
    crop = tuple(manifest["crop_left_top_width_height"])
    report: dict = {
        "schema_version": 1,
        "evidence_kind": "live_full_file_vendor_synthetic_sparse_pipeline",
        "dll_sha256": DLL_SHA256,
        "probe_sha256": sha256(args.probe),
        "manifest_sha256": sha256(args.manifest),
        "selection": asdict(selection),
        "rawpy_version": rawpy.__version__,
        "libraw_version": list(rawpy.libraw_version),
        "reference_sha256": sha256(Path(__file__).with_name("sparse_reference.py")),
        "radial_reference_sha256": sha256(Path(__file__).with_name("verify_radial.py")),
        "sampling_reference_sha256": sha256(
            Path(__file__).with_name("verify_vendor_matrix.py")
        ),
        "runner_sha256": sha256(Path(__file__)),
        "cases": [],
        "passed": False,
    }
    environment = dict(
        os.environ, WINEPREFIX=str(args.wine_prefix.resolve()), WINEDEBUG="-all"
    )
    for count in args.counts:
        for iso in (3200, 3201):
            case: dict = {"input_count": count, "iso": iso, "passed": False}
            try:
                records = sorted(
                    (r for r in manifest["files"] if r["input_count"] == count),
                    key=lambda r: r["frame_index"],
                )
                if len(records) != count:
                    raise ValueError("Incomplete input group.")
                directory = args.output / f"mode-{count}-iso-{iso}"
                directory.mkdir()
                files, frames = [], []
                case["fixtures"] = []
                for record in records:
                    source = Path(record["path"])
                    if sha256(source) != record["sha256"]:
                        raise ValueError("Input checksum mismatch.")
                    target = directory / source.name
                    case["fixtures"].append(
                        patch_fixture(source, target, selection, iso)
                    )
                    files.append(target)
                    with rawpy.imread(str(target)) as raw:
                        raw.unpack()
                        frames.append(raw.raw_image.copy())
                correction_parameters = parameters(files[0])
                case["outputs"] = []
                for radial in (False, True):
                    mode = "correct" if radial else "normal"
                    output = directory / f"{mode}.nefx"
                    observed: dict = {"radial": radial, "path": str(output)}
                    case["outputs"].append(observed)
                    result = subprocess.run(
                        [
                            args.wine,
                            str(args.probe),
                            windows(args.dll),
                            mode,
                            "1",
                            windows(output),
                            *map(windows, files),
                        ],
                        env=environment,
                        capture_output=True,
                        text=True,
                        timeout=60,
                        check=False,
                    )
                    observed.update(
                        return_code=result.returncode,
                        stderr=result.stderr,
                        events=[
                            json.loads(line)
                            for line in result.stdout.splitlines()
                            if line.startswith("{")
                        ],
                    )
                    if result.returncode:
                        raise ValueError("Vendor merge failed.")
                    expected = expected_output(
                        frames, selection, iso, crop, radial, correction_parameters
                    )
                    with rawpy.imread(str(output)) as raw:
                        raw.unpack()
                        actual = raw.raw_image[..., :3].copy()
                    if expected.shape != actual.shape:
                        raise ValueError("Output geometry mismatch.")
                    diff = expected.astype(np.int32) - actual.astype(np.int32)
                    opposite = expected_output(
                        frames,
                        selection,
                        3201 if iso == 3200 else 3200,
                        crop,
                        radial,
                        correction_parameters,
                    )
                    observed.update(
                        sha256=sha256(output),
                        samples=int(actual.size),
                        differences=int(np.count_nonzero(diff)),
                        max_abs_error=int(np.max(np.abs(diff))),
                        opposite_iso_differences=int(
                            np.count_nonzero(opposite != actual)
                        ),
                    )
                case["passed"] = all(
                    o["differences"] == 0 and o["opposite_iso_differences"] > 0
                    for o in case["outputs"]
                )
            except subprocess.TimeoutExpired as error:
                diagnostics = error.stderr or b""
                case.update(
                    error_kind="timeout",
                    error=str(error),
                    stderr=(
                        diagnostics.decode("utf-8", errors="replace")
                        if isinstance(diagnostics, bytes)
                        else diagnostics
                    ),
                )
            except (
                OSError,
                ValueError,
                RuntimeError,
                KeyError,
                rawpy.LibRawError,
            ) as error:
                case["error"] = str(error)
            report["cases"].append(case)
            (args.output / "report.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            print(
                json.dumps({k: v for k, v in case.items() if k != "fixtures"}),
                flush=True,
            )
    report["passed"] = all(case["passed"] for case in report["cases"])
    report["samples"] = sum(
        o.get("samples", 0) for c in report["cases"] for o in c.get("outputs", [])
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
