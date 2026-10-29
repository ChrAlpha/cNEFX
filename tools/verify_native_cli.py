"""Run the native CLI against saved synthetic Nikon pixel-shift references."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import rawpy
import tifffile


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def shutter_field(data: bytes) -> tuple[int, int]:
    with tifffile.TiffFile(io.BytesIO(data)) as image:
        offset = image.pages[0].tags[34665].valueoffset
        count = struct.unpack_from("<H", data, offset)[0]
        entries = [
            tifffile.TiffTag.fromfile(image, offset=offset + 2 + 12 * i)
            for i in range(count)
        ]
        tag = next(tag for tag in entries if tag.code == 37500)
        base = tag.valueoffset + 10
        with tifffile.TiffFile(io.BytesIO(data[base:])) as maker:
            shutter = maker.pages[0].tags[0xA7]
            if shutter.dtype != 4 or shutter.count != 1:
                raise ValueError("Unsupported synthetic shutter-count field.")
            return base + shutter.valueoffset, shutter.value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pipeline_reports", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--executable", type=Path, help="Native cnefx executable.")
    parser.add_argument("--chromatic-aberration-correction", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.output.resolve().is_relative_to(root):
        parser.error("Research output must be outside the repository.")
    args.output.mkdir(exist_ok=False)
    prefix = (
        [str(args.executable.resolve())]
        if args.executable
        else [sys.executable, "-m", "nx_ctudio.cli"]
    )
    report = {
        "evidence_kind": "native_cli_saved_synthetic_vendor_comparison",
        "passed": False,
        "cases": [],
        "runtime": "cnefx" if args.executable else "python_reference",
        "chromatic_aberration_correction": args.chromatic_aberration_correction,
        "pipeline_reports": [
            {"path": str(p), "sha256": digest(p)} for p in args.pipeline_reports
        ],
        "code_sha256": {
            str(p.relative_to(root)): digest(p)
            for p in [
                *sorted((root / "src/nx_ctudio").glob("*.py")),
                *sorted((root / "src").glob("*.cpp")),
                *sorted((root / "src").glob("*.hpp")),
                *sorted((root / "include/cnefx").glob("*.hpp")),
                root / "CMakeLists.txt",
            ]
        },
        "runner_sha256": digest(Path(__file__)),
        "limitations": (
            "Synthetic capture metadata; no live Nikon or Lightroom execution."
        ),
    }
    expected_cases = 0
    try:
        if args.executable:
            report["executable_sha256"] = digest(args.executable)
            report["executable_version"] = subprocess.run(
                [*prefix, "--version"],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            ).stdout.strip()
        for corpus, source_report in enumerate(args.pipeline_reports):
            source = json.loads(source_report.read_text())
            if (
                not source["passed"]
                or source["evidence_kind"]
                != "live_full_file_vendor_synthetic_sparse_pipeline"
            ):
                raise ValueError("Expected a passing saved vendor pipeline report.")
            expected_cases += len(source["cases"])
            for case in source["cases"]:
                files, patches = [], []
                name = f"{corpus}-{case['input_count']}-{case['iso']}"
                directory = args.output / f"inputs-{name}"
                directory.mkdir()
                first_shutter = None
                for index, item in enumerate(case["fixtures"]):
                    path = Path(item["path"])
                    if digest(path) != item["sha256"]:
                        raise ValueError("Input hash mismatch.")
                    data = bytearray(path.read_bytes())
                    offset, original_shutter = shutter_field(data)
                    if first_shutter is None:
                        first_shutter = original_shutter
                    struct.pack_into("<I", data, offset, first_shutter + index)
                    copied = directory / path.name
                    with copied.open("xb") as stream:
                        stream.write(data)
                    files.append(str(copied))
                    patches.append(
                        {
                            "source": str(path),
                            "source_sha256": item["sha256"],
                            "copy": str(copied),
                            "sha256": digest(copied),
                            "patch_offset": offset,
                            "purpose": "Synthetic consecutive shutter counts only",
                        }
                    )
                output = args.output / f"merge-{name}.nefx"
                command = [
                    *prefix,
                    "merge",
                    *files,
                    "--output",
                    str(output),
                    "--json",
                ]
                if args.chromatic_aberration_correction:
                    command.append("--chromatic-aberration-correction")
                result = subprocess.run(
                    command, capture_output=True, text=True, timeout=180, check=False
                )
                observed = {
                    "synthetic_identity_copies": patches,
                    "inputs": case["input_count"],
                    "iso": case["iso"],
                    "returncode": result.returncode,
                    "stderr": result.stderr,
                    "passed": False,
                }
                report["cases"].append(observed)
                if result.returncode:
                    observed["stdout"] = result.stdout
                    raise ValueError("Native CLI failed.")
                observed["cli"] = json.loads(result.stdout)
                if (
                    observed["cli"]["chromatic_aberration_correction"]
                    != args.chromatic_aberration_correction
                ):
                    raise ValueError("Native CLI correction mode differs from request.")
                vendor = next(
                    item
                    for item in case["outputs"]
                    if item["radial"] == args.chromatic_aberration_correction
                )
                reference = Path(vendor["path"])
                if digest(reference) != vendor["sha256"]:
                    raise ValueError("Vendor output hash mismatch.")
                with rawpy.imread(str(reference)) as image:
                    expected = image.raw_image[..., :3].copy()
                with rawpy.imread(str(output)) as image:
                    actual = image.raw_image[..., :3]
                    if actual.shape != expected.shape:
                        raise ValueError("Native and vendor geometry differ.")
                    differences = int(np.count_nonzero(expected != actual))
                observed.update(
                    samples=int(expected.size),
                    different_samples=differences,
                    passed=differences == 0,
                )
                rerun = subprocess.run(
                    command, capture_output=True, text=True, timeout=30, check=False
                )
                observed["overwrite_rejected"] = (
                    rerun.returncode == 1
                    and digest(output) == observed["cli"]["sha256"]
                )
                observed["passed"] &= observed["overwrite_rejected"]
        report["passed"] = (
            bool(expected_cases)
            and len(report["cases"]) == expected_cases
            and all(case["passed"] for case in report["cases"])
        )
    except (Exception, KeyboardInterrupt) as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    with (args.output / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "passed": report["passed"]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
