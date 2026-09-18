"""Replay the pinned scalar lattice against native selector-0 Lab conversion."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from kaiji_color import quantized_rgb_lab_bounds, rgb_to_lab

REPORT_HASH = "be1f9c18a696eca3368f0034ff16030484457291d85faa4b9eaf673f01e34c35"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.output.exists() or args.output.resolve().is_relative_to(root):
        parser.error("Output must be a new path outside the repository.")
    snapshot = args.oracle.read_bytes()
    if hashlib.sha256(snapshot).hexdigest() != REPORT_HASH:
        parser.error("Unexpected scalar oracle report hash.")
    oracle = json.loads(snapshot)
    if oracle.get("passed") is not True or len(oracle["cases"]) != 8:
        parser.error("Incomplete scalar oracle report.")
    planes = {}
    for case in oracle["cases"]:
        if case["processing"] != 0 or case["matrix"] != 1:
            continue
        data = (args.oracle.parent / f"p0-f{case['flags']}-m1.bin").read_bytes()
        if (
            len(data) != 3 * 33**3 * 2
            or hashlib.sha256(data).hexdigest() != case["sha256"]
        ):
            parser.error("Unexpected pinned scalar plane bytes.")
        planes[case["flags"]] = np.frombuffer(data, dtype="<u2").reshape(3, 33**3)
    if set(planes) != {0, 1}:
        parser.error("Missing initialized RGB/Lab plane pair.")
    outside = 0
    max_error = [0, 0, 0]
    max_width = [0, 0, 0]
    differing_words = 0
    for rgb, lab in zip(planes[1].T.tolist(), planes[0].T.tolist(), strict=True):
        low, high = quantized_rgb_lab_bounds(rgb)
        center = rgb_to_lab([value / 65280.0 for value in rgb])
        for channel in range(3):
            outside += not low[channel] <= lab[channel] <= high[channel]
            differing_words += center[channel] != lab[channel]
            max_error[channel] = max(
                max_error[channel], abs(center[channel] - lab[channel])
            )
            max_width[channel] = max(max_width[channel], high[channel] - low[channel])
    report = {
        "evidence_kind": "native_selector0_lab_quantization_interval_replay",
        "oracle_sha256": REPORT_HASH,
        "implementation_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), root / "tools/kaiji_color.py")
        },
        "points": 33**3,
        "channel_words": 3 * 33**3,
        "out_of_interval_words": outside,
        "center_different_words": differing_words,
        "center_max_absolute_error": max_error,
        "max_interval_width": max_width,
        "passed": outside == 0,
        "limits": [
            "RGB has half-code quantization uncertainty; exact input is unknown.",
            "Intervals include 1e-9 Lab-unit numerical margin for binary64 arithmetic.",
            "Only selector-0 output conversion is tested, not upstream processing.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
