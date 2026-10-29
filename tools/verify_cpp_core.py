"""Differential C++ reconstruction checks against the retained Python reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nx_ctudio._sparse import Selection  # noqa: E402
from nx_ctudio.pixel_shift import reconstruct  # noqa: E402
from nx_ctudio.radial import RadialCorrection  # noqa: E402


def run(executable: Path, directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    cases = []
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
            selector = (
                b"0100"
                + Selection(
                    height, width, first_row=6, inner_step=1, inner_after=5
                ).pack()
            )
            crop = (1 + seed % 3, 1 + seed % 3, width - 4, height - 4)
            for iso in (3200, 3201):
                for corrected in (False, True):
                    centers = (192, -128)
                    coefficients = ((-120, 350, -90), (280, -180, 70))
                    if seed == 7:
                        centers = (2**31 - 1, -(2**31))
                        coefficients = ((2**31 - 1,) * 3,) * 2
                    radial = RadialCorrection(centers, coefficients)
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
                    expected = reconstruct(
                        frames,
                        crop,
                        selector,
                        iso,
                        radial=radial if corrected else None,
                    )
                    result = subprocess.run(
                        [str(executable), "--replay", str(source), str(target)],
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    if result.returncode:
                        raise RuntimeError(result.stderr)
                    actual = np.frombuffer(target.read_bytes(), dtype="<u2").reshape(
                        expected.shape
                    )
                    differences = int(np.count_nonzero(actual != expected))
                    case = {
                        "count": count,
                        "seed": seed,
                        "iso": iso,
                        "radial": corrected,
                        "samples": int(expected.size),
                        "different_samples": differences,
                        "input_sha256": hashlib.sha256(payload).hexdigest(),
                        "expected_sha256": hashlib.sha256(
                            expected.astype("<u2").tobytes()
                        ).hexdigest(),
                    }
                    cases.append(case)
                    if differences:
                        raise AssertionError(json.dumps(case))
    report = {
        "evidence_kind": "synthetic_cpp_python_differential",
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
