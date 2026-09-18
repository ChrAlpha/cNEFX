# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.4.6", "rawpy==0.27.1"]
# ///
"""Verify recovered sampling against a local DLL using distinct synthetic frames.

Requires the external packed14-fixtures manifest and a compiled probe executable.
This does not test authentic captures, sparse repair, or chromatic correction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import rawpy

DLL_SHA256 = "c82782c51be65d9528f6751bfca9d2bdc7b728baa3ee8eca2e84ba713ed9031c"
PAIRS = ((4, 1), (8, 1), (8, 2), (16, 1), (16, 4), (32, 1), (32, 2), (32, 4), (32, 8))
PHASE_X = (0, 1, 1, 0)
PHASE_Y = (0, 0, 1, 1)
CHANNELS = ((2, 1, 0, 1), (1, 2, 1, 0), (0, 1, 2, 1), (1, 0, 1, 2))


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def windows(path: Path) -> str:
    return "Z:" + str(path.resolve()).replace("/", "\\")


def groups(count: int, outputs: int) -> list[list[int]]:
    if (count, outputs) == (32, 4):
        return [
            list(range(i * 4, i * 4 + 4)) + list(range(16 + i * 4, 20 + i * 4))
            for i in range(4)
        ]
    size = count // outputs
    return [list(range(i * size, (i + 1) * size)) for i in range(outputs)]


def reconstruct(inputs: list[np.ndarray], crop: tuple[int, ...]) -> np.ndarray:
    frames = inputs
    count = len(frames)
    if count in (8, 32):
        half = count // 2
        frames = [frame.copy() for frame in inputs[:half]]
        for index in range(half):
            frames[index][2:] = (
                (inputs[index][2:].astype(np.uint32) + inputs[index + half][:-2]) >> 1
            ).astype(np.uint16)
    scale = 2 if len(frames) == 16 else 1
    height, width = frames[0].shape
    output = np.zeros((scale * (height - 1), scale * (width - 1), 3), dtype=np.uint16)
    for group in range(len(frames) // 4):
        for frame in range(4):
            raw = frames[group * 4 + frame]
            for corner, (py, px) in enumerate(((0, 0), (0, 1), (1, 1), (1, 0))):
                ys = np.arange(py, height - 1, 2)
                xs = np.arange(px, width - 1, 2)
                samples = raw[np.ix_(ys + 1 - PHASE_Y[frame], xs + 1 - PHASE_X[frame])]
                channel = CHANNELS[frame][corner]
                target = np.ix_(
                    scale * ys + PHASE_Y[group], scale * xs + PHASE_X[group]
                )
                plane = output[:, :, channel]
                if channel == 1 and frame >= 2:
                    samples = ((samples.astype(np.uint32) + plane[target]) >> 1).astype(
                        np.uint16
                    )
                plane[target] = samples
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wine-prefix", type=Path, required=True)
    parser.add_argument("--wine", default="/usr/lib/wine/wine64")
    args = parser.parse_args()
    if sha256(args.dll) != DLL_SHA256:
        parser.error("DLL checksum differs from the analyzed binary.")
    manifest = json.loads(args.manifest.read_text())
    if manifest["evidence_kind"] != "synthetic_packed14_input_fixture":
        parser.error("Expected a synthetic packed-14 manifest.")
    args.output.mkdir(parents=True, exist_ok=False)
    environment = dict(
        os.environ, WINEPREFIX=str(args.wine_prefix.resolve()), WINEDEBUG="-all"
    )
    report: dict[str, object] = {
        "schema_version": 1,
        "evidence_kind": "live_vendor_dll_synthetic_distinct_frame_matrix",
        "dll_sha256": DLL_SHA256,
        "probe_sha256": sha256(args.probe),
        "manifest_sha256": sha256(args.manifest),
        "rawpy": rawpy.__version__,
        "libraw": list(rawpy.libraw_version),
        "cases": [],
    }
    for count, output_count in PAIRS:
        started = time.monotonic()
        case: dict[str, object] = {"inputs": count, "outputs": output_count}
        try:
            records = sorted(
                (r for r in manifest["files"] if r["input_count"] == count),
                key=lambda r: r["frame_index"],
            )
            if len(records) != count:
                raise ValueError(f"Expected {count} input records.")
            files = [Path(record["path"]) for record in records]
            arrays = []
            for record, path in zip(records, files, strict=True):
                if sha256(path) != record["sha256"]:
                    raise ValueError("Fixture checksum mismatch.")
                with rawpy.imread(str(path)) as raw:
                    raw.unpack()
                    arrays.append(raw.raw_image.copy())
            prefix = args.output / f"merge-{count}-{output_count}.nefx"
            command = [
                args.wine,
                str(args.probe),
                windows(args.dll),
                "normal",
                str(output_count),
                windows(prefix),
                *map(windows, files),
            ]
            result = subprocess.run(
                command,
                env=environment,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            events = [
                json.loads(line)
                for line in result.stdout.splitlines()
                if line.startswith("{")
            ]
            case.update(
                return_code=result.returncode,
                events=events,
                stderr=result.stderr,
                comparisons=[],
            )
            if result.returncode == 0:
                for index, members in enumerate(groups(count, output_count)):
                    path = (
                        prefix
                        if output_count == 1
                        else Path(f"{prefix}.part-{index + 1}.nefx")
                    )
                    expected = reconstruct(
                        [arrays[i] for i in members],
                        tuple(manifest["crop_left_top_width_height"]),
                    )
                    with rawpy.imread(str(path)) as raw:
                        raw.unpack()
                        actual = raw.raw_image[..., :3]
                        if expected.shape != actual.shape:
                            raise ValueError("Unexpected output geometry.")
                        diff = expected.astype(np.int32) - actual.astype(np.int32)
                        selected = [arrays[i] for i in members]
                        crop = tuple(manifest["crop_left_top_width_height"])
                        reversed_result = reconstruct(list(reversed(selected)), crop)
                        controls = {
                            "reversed_frame_order": int(
                                np.count_nonzero(reversed_result != actual)
                            )
                        }
                        if len(selected) in (8, 32):
                            half = len(selected) // 2
                            swapped = reconstruct(
                                selected[half:] + selected[:half], crop
                            )
                            controls["swapped_pair_halves"] = int(
                                np.count_nonzero(swapped != actual)
                            )
                        case["comparisons"].append(
                            {
                                "input_indices": members,
                                "shape": list(actual.shape),
                                "sample_count": int(actual.size),
                                "different_samples": int(np.count_nonzero(diff)),
                                "max_abs_error": int(np.abs(diff).max()),
                                "output_sha256": sha256(path),
                                "negative_control_different_samples": controls,
                            }
                        )
            case["ok"] = (
                result.returncode == 0
                and len(case["comparisons"]) == output_count
                and all(
                    item["different_samples"] == 0
                    and all(
                        value > 0
                        for value in item["negative_control_different_samples"].values()
                    )
                    for item in case["comparisons"]
                )
            )
        except subprocess.TimeoutExpired as exc:
            case.update(
                ok=False,
                error="Vendor process exceeded 60 seconds.",
                stdout=(exc.stdout or b"").decode("utf-8", errors="replace")
                if isinstance(exc.stdout, bytes)
                else (exc.stdout or ""),
                stderr=(exc.stderr or b"").decode("utf-8", errors="replace")
                if isinstance(exc.stderr, bytes)
                else (exc.stderr or ""),
            )
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            rawpy.LibRawError,
        ) as exc:
            case.update(ok=False, error=f"{type(exc).__name__}: {exc}")
        case["elapsed_seconds"] = round(time.monotonic() - started, 3)
        report["cases"].append(case)
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps({"inputs": count, "outputs": output_count, "ok": case["ok"]}),
            flush=True,
        )
    return int(not all(case["ok"] for case in report["cases"]))


if __name__ == "__main__":
    raise SystemExit(main())
