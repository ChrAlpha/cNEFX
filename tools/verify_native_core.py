"""Compare the native merge core with saved synthetic Nikon output evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import rawpy

from nx_ctudio.pixel_shift import reconstruct
from nx_ctudio.radial import RadialCorrection
from nx_ctudio.raw import read_frame


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(pipeline_report: Path, radial: bool = False) -> list[dict]:
    report = json.loads(pipeline_report.read_text())
    if not report["passed"]:
        raise ValueError("Expected a completed passing vendor pipeline report.")
    cases = []
    for case in report["cases"]:
        frames = []
        for fixture in case["fixtures"]:
            path = Path(fixture["path"])
            if digest(path) != fixture["sha256"]:
                raise ValueError("Fixture hash mismatch.")
            samples, metadata = read_frame(path)
            frames.append(samples)
            if len(frames) == 1:
                first = metadata
        correction = None
        if radial:
            parameters = first["radial"]
            if not parameters["available"]:
                raise ValueError("Fixture lacks supported radial correction metadata.")
            correction = RadialCorrection(
                tuple(parameters["center_q8"]),
                tuple(tuple(row) for row in parameters["coefficients_q20"]),
            )
        native = reconstruct(
            frames,
            first["crop"],
            b"0100" + first["selector"],
            case["iso"],
            radial=correction,
        )
        vendor = next(
            output for output in case["outputs"] if output["radial"] == radial
        )
        path = Path(vendor["path"])
        if digest(path) != vendor["sha256"]:
            raise ValueError("Saved vendor output hash mismatch.")
        with rawpy.imread(str(path)) as image:
            actual = image.raw_image[..., :3]
            if actual.shape != native.shape:
                raise ValueError("Output geometry differs.")
            differences = int(np.count_nonzero(native != actual))
        cases.append(
            {
                "input_count": case["input_count"],
                "iso": case["iso"],
                "output": str(path),
                "output_sha256": digest(path),
                "samples": int(native.size),
                "different_samples": differences,
                "passed": differences == 0,
            }
        )
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pipeline_reports", nargs="+", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--radial-correction", action="store_true")
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Report already exists.")
    report = {
        "evidence_kind": "native_core_saved_synthetic_vendor_comparison",
        "passed": False,
        "radial_correction": args.radial_correction,
        "cases": [],
        "pipeline_reports": [
            {"path": str(p), "sha256": digest(p)} for p in args.pipeline_reports
        ],
        "code_sha256": {
            str(p): digest(p)
            for p in (
                Path(__file__),
                Path("reference/nx_ctudio/pixel_shift.py"),
                Path("reference/nx_ctudio/_sparse.py"),
                Path("reference/nx_ctudio/radial.py"),
                Path("reference/nx_ctudio/raw.py"),
            )
        },
    }
    try:
        for path in args.pipeline_reports:
            report["cases"].extend(verify(path, args.radial_correction))
        report["passed"] = bool(report["cases"]) and all(
            case["passed"] for case in report["cases"]
        )
    except (Exception, KeyboardInterrupt) as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    with args.report.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"report": str(args.report), "passed": report["passed"]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
