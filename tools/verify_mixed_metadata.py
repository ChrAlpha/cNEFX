# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.4.6", "rawpy==0.27.1", "tifffile==2026.9.15"]
# ///
"""Probe which input owns ISO and sparse selection in synthetic merge groups."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import rawpy
from sparse_reference import Selection
from verify_sparse_pipeline import expected_output, patch_fixture
from verify_vendor_matrix import DLL_SHA256, sha256, windows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "dll", "probe", "wine-prefix", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument(
        "--counts", nargs="+", type=int, choices=(4, 8, 16, 32), default=[4, 8, 16, 32]
    )
    args = parser.parse_args()
    if sha256(args.dll) != DLL_SHA256:
        parser.error("Unexpected DLL checksum.")
    manifest = json.loads(args.manifest.read_text())
    if manifest["evidence_kind"] != "synthetic_packed14_input_fixture":
        parser.error("Expected synthetic packed-14 inputs.")
    args.output.mkdir(exist_ok=False, parents=True)
    base = Selection(
        manifest["height"], manifest["width"], first_row=12, inner_step=1, inner_after=5
    )
    variants = (
        ("first-low", 3200, 3201, base, base),
        ("first-high", 3201, 3200, base, base),
        ("different-selection", 3200, 3200, base, replace(base, first_row=40)),
        ("first-no-selection", 3200, 3200, replace(base, outer_count=0), base),
    )
    report = {
        "schema_version": 1,
        "evidence_kind": "live_vendor_synthetic_mixed_metadata",
        "dll_sha256": DLL_SHA256,
        "probe_sha256": sha256(args.probe),
        "manifest_sha256": sha256(args.manifest),
        "code_sha256": {
            name: sha256(Path(__file__).with_name(name))
            for name in (
                "verify_mixed_metadata.py",
                "verify_sparse_pipeline.py",
                "sparse_reference.py",
                "verify_vendor_matrix.py",
            )
        },
        "cases": [],
        "passed": False,
    }
    for count in args.counts:
        for name, first_iso, rest_iso, first_selection, rest_selection in variants:
            case = {
                "name": name,
                "input_count": count,
                "first_iso": first_iso,
                "rest_iso": rest_iso,
                "first_selection": asdict(first_selection),
                "rest_selection": asdict(rest_selection),
                "passed": False,
            }
            try:
                directory = args.output / f"{count}-{name}"
                directory.mkdir()
                records = sorted(
                    (r for r in manifest["files"] if r["input_count"] == count),
                    key=lambda r: r["frame_index"],
                )
                if len(records) != count:
                    raise ValueError("Incomplete input group.")
                files, frames = [], []
                case["fixtures"] = []
                for index, record in enumerate(records):
                    source = Path(record["path"])
                    if sha256(source) != record["sha256"]:
                        raise ValueError("Source hash mismatch.")
                    path = directory / source.name
                    case["fixtures"].append(
                        patch_fixture(
                            source,
                            path,
                            first_selection if index == 0 else rest_selection,
                            first_iso if index == 0 else rest_iso,
                        )
                    )
                    files.append(path)
                    with rawpy.imread(str(path)) as raw:
                        raw.unpack()
                        frames.append(raw.raw_image.copy())
                output = directory / "result.nefx"
                result = subprocess.run(
                    [
                        "/usr/lib/wine/wine64",
                        str(args.probe),
                        windows(args.dll),
                        "normal",
                        "1",
                        windows(output),
                        *map(windows, files),
                    ],
                    env=dict(
                        os.environ,
                        WINEPREFIX=str(args.wine_prefix.resolve()),
                        WINEDEBUG="-all",
                    ),
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=False,
                )
                case.update(
                    return_code=result.returncode,
                    stderr=result.stderr,
                    events=[
                        json.loads(line)
                        for line in result.stdout.splitlines()
                        if line.startswith("{")
                    ],
                )
                if result.returncode != 0:
                    raise ValueError("Vendor merge failed.")
                with rawpy.imread(str(output)) as raw:
                    raw.unpack()
                    actual = raw.raw_image[..., :3].copy()
                crop = tuple(manifest["crop_left_top_width_height"])
                expected = expected_output(
                    frames,
                    first_selection,
                    first_iso,
                    crop,
                    False,
                    ([0, 0], [[0, 0, 0], [0, 0, 0]]),
                )
                counterfactual = expected_output(
                    frames,
                    rest_selection,
                    rest_iso,
                    crop,
                    False,
                    ([0, 0], [[0, 0, 0], [0, 0, 0]]),
                )
                if actual.shape != expected.shape:
                    raise ValueError("Unexpected geometry.")
                differences = int(np.count_nonzero(actual != expected))
                other = int(np.count_nonzero(actual != counterfactual))
                case.update(
                    output_sha256=sha256(output),
                    samples=int(actual.size),
                    differences=differences,
                    other_metadata_differences=other,
                    passed=differences == 0 and other > 0,
                )
            except subprocess.TimeoutExpired as error:
                diagnostics = error.stderr or b""
                case.update(
                    error_kind="timeout",
                    error=str(error),
                    stderr=diagnostics.decode(errors="replace")
                    if isinstance(diagnostics, bytes)
                    else diagnostics,
                )
            except (OSError, ValueError, KeyError, rawpy.LibRawError) as error:
                case["error"] = str(error)
            report["cases"].append(case)
            (args.output / "report.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            print(
                json.dumps(
                    {
                        k: v
                        for k, v in case.items()
                        if k
                        not in (
                            "fixtures",
                            "events",
                            "first_selection",
                            "rest_selection",
                        )
                    }
                ),
                flush=True,
            )
    report["passed"] = all(case["passed"] for case in report["cases"])
    report["samples"] = sum(case.get("samples", 0) for case in report["cases"])
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
