"""Check C++ reconstruction against pinned independent reference expectations."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
from pathlib import Path

import numpy as np

from ci_fixture import GOLDEN, GOLDEN_SHA256


def run(executable: Path, directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    cases = []
    expectations = {
        (case["count"], case["seed"], case["iso"], case["radial"]): case
        for case in GOLDEN["core"]
    }
    for count in (4, 8, 16, 32):
        for seed in range(8):
            height, width = ((18, 24), (68, 74))[seed % 2]
            rng = np.random.default_rng(510 + seed)
            frames = [
                rng.integers(0, 16384, (height, width), dtype=np.uint16)
                for _ in range(count)
            ]
            if seed in (2, 3):
                for frame in frames:
                    frame.fill(0 if seed == 2 else 16383)
            selector = b"0100" + struct.pack(
                "<BB6H3BxBxB3x", 0, 0, height, width, 2, 6, 0, 0, 1, 1, 1, 0, 5
            )
            crop = (1 + seed % 3, 1 + seed % 3, width - 4, height - 4)
            for iso in (3200, 3201):
                for corrected in (False, True):
                    centers = (192, -128)
                    coefficients = ((-120, 350, -90), (280, -180, 70))
                    if seed == 7:
                        centers = (2**31 - 1, -(2**31))
                        coefficients = ((2**31 - 1,) * 3,) * 2
                    payload = (
                        struct.pack(
                            "<9I", count, width, height, *crop, iso, int(corrected)
                        )
                        + selector
                        + struct.pack("<8i", *centers, *sum(coefficients, ()))
                        + b"".join(frame.astype("<u2").tobytes() for frame in frames)
                    )
                    source, target = directory / "input.bin", directory / "output.bin"
                    source.write_bytes(payload)
                    expected = expectations[count, seed, iso, corrected]
                    assert (
                        hashlib.sha256(payload).hexdigest() == expected["input_sha256"]
                    )
                    result = subprocess.run(
                        [str(executable), "--replay", str(source), str(target)],
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    if result.returncode:
                        raise RuntimeError(result.stderr)
                    actual = target.read_bytes()
                    matches = (
                        len(actual) == expected["samples"] * 2
                        and hashlib.sha256(actual).hexdigest()
                        == expected["expected_sha256"]
                    )
                    case = {
                        "count": count,
                        "seed": seed,
                        "iso": iso,
                        "radial": corrected,
                        "samples": expected["samples"],
                        "samples_match": matches,
                        "input_sha256": hashlib.sha256(payload).hexdigest(),
                        "expected_sha256": expected["expected_sha256"],
                    }
                    cases.append(case)
                    if not matches:
                        raise AssertionError(json.dumps(case))
    report = {
        "evidence_kind": "synthetic_native_saved_reference_regression",
        "expectations_sha256": GOLDEN_SHA256,
        "passed": True,
        "cases": cases,
        "samples": sum(case["samples"] for case in cases),
        "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "numpy": np.__version__,
    }
    (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--work-directory", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.executable.resolve(), args.work_directory.resolve())
    print(json.dumps({"cases": len(report["cases"]), "samples": report["samples"]}))


if __name__ == "__main__":
    main()
