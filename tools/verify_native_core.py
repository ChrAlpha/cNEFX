"""Compare the native merge core with saved synthetic Nikon output evidence."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import rawpy
import tifffile

from nx_ctudio.pixel_shift import reconstruct


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(pipeline_report: Path) -> list[dict]:
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
            with rawpy.imread(str(path)) as image:
                frames.append(image.raw_image.copy())
        source = Path(case["fixtures"][0]["path"])
        with tifffile.TiffFile(source) as image:
            maker = image.pages[0].tags[34665].value["MakerNote"]
        with tifffile.TiffFile(io.BytesIO(maker[10:])) as image:
            crop = image.pages[0].tags[0x45].value
            selector = image.pages[0].tags[0x5E].value
        native = reconstruct(frames, crop, selector, case["iso"])
        vendor = next(output for output in case["outputs"] if not output["radial"])
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
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Report already exists.")
    report = {
        "evidence_kind": "native_core_saved_synthetic_vendor_comparison",
        "passed": False,
        "cases": [],
        "pipeline_reports": [
            {"path": str(p), "sha256": digest(p)} for p in args.pipeline_reports
        ],
        "code_sha256": {
            str(p): digest(p)
            for p in (
                Path(__file__),
                Path("src/nx_ctudio/pixel_shift.py"),
                Path("src/nx_ctudio/_sparse.py"),
            )
        },
    }
    try:
        for path in args.pipeline_reports:
            report["cases"].extend(verify(path))
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
