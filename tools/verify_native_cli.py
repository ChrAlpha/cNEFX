"""Run the native CLI against saved synthetic Nikon pixel-shift references."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import rawpy


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pipeline_reports", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.output.resolve().is_relative_to(root):
        parser.error("Research output must be outside the repository.")
    args.output.mkdir(exist_ok=False)
    report = {
        "evidence_kind": "native_cli_saved_synthetic_vendor_comparison",
        "passed": False,
        "cases": [],
        "code_sha256": {
            str(p.relative_to(root)): digest(p)
            for p in sorted((root / "src/nx_ctudio").glob("*.py"))
        },
        "runner_sha256": digest(Path(__file__)),
        "limitations": (
            "Synthetic capture metadata; no live Nikon or Lightroom execution."
        ),
    }
    expected_cases = 0
    try:
        for source_report in args.pipeline_reports:
            source = json.loads(source_report.read_text())
            if not source["passed"]:
                raise ValueError("Expected a passing saved vendor pipeline report.")
            expected_cases += len(source["cases"])
            for case in source["cases"]:
                files = []
                for item in case["fixtures"]:
                    path = Path(item["path"])
                    if digest(path) != item["sha256"]:
                        raise ValueError("Input hash mismatch.")
                    files.append(str(path))
                output = args.output / f"merge-{case['input_count']}-{case['iso']}.nefx"
                command = [
                    sys.executable,
                    "-m",
                    "nx_ctudio.cli",
                    "merge",
                    *files,
                    "--output",
                    str(output),
                    "--json",
                ]
                result = subprocess.run(
                    command, capture_output=True, text=True, timeout=180, check=False
                )
                observed = {
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
                vendor = next(item for item in case["outputs"] if not item["radial"])
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
