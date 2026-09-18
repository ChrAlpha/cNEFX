# /// script
# requires-python = ">=3.12"
# dependencies = ["gmpy2==2.2.1", "numpy==2.4.6"]
# ///
"""Replay pinned scalar artifacts against native Lab quantization enclosures."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import gmpy2
import numpy as np
from kaiji_output import (
    ENCODE_OFFSET,
    ENCODE_SCALE,
    LAB_MAX,
    LAB_MIN,
    LAB_OFFSET,
    LAB_POWER,
    LAB_SLOPE,
    LAB_THRESHOLD,
    MATRIX,
    RGB_SCALE,
    TRANSFER_OFFSET,
    TRANSFER_POWER,
    TRANSFER_SCALE,
    TRANSFER_SLOPE,
    TRANSFER_THRESHOLD,
    XYZ_MAX,
    XYZ_SCALE,
    FloatArray,
    rgb_to_icc_lab,
    validate_triplets,
)

REPORT_HASH = "be1f9c18a696eca3368f0034ff16030484457291d85faa4b9eaf673f01e34c35"
ROME_HASH = "45c8b91ddd61a103711d755a57c45db4ce8431b9ec9ce7a1f4fac28fe25582e2"
GRID = 33
POINTS = GRID**3


def outward(values: FloatArray, upper: bool) -> FloatArray:
    return np.nextafter(values, np.inf if upper else -np.inf)


def power_bound(values: FloatArray, exponent: float, upper: bool) -> FloatArray:
    """MPFR encloses real powers in adjacent binary64 values without a ULP budget."""
    rounding = gmpy2.RoundUp if upper else gmpy2.RoundDown
    with gmpy2.context(precision=53, round=rounding):
        power = gmpy2.mpfr(exponent)
        return np.fromiter(
            (float(gmpy2.mpfr(float(value)) ** power) for value in values.flat),
            dtype=np.float64,
            count=values.size,
        ).reshape(values.shape)


def rgb_interval(codes: object) -> tuple[FloatArray, FloatArray]:
    """Invert trunc(round64(round64(rgb * 65280) + 0.5)) conservatively.

    Each nextafter also encloses the rounding cell of the preceding binary64
    operation. Endpoints are closed overapproximations, including clipped ends.
    """
    values = validate_triplets(codes, encoded=True)
    lo_product = outward(outward(values, False) - 0.5, False)
    hi_product = outward(outward(values + 1.0, True) - 0.5, True)
    lo = outward(outward(lo_product, False) / RGB_SCALE, False)
    hi = outward(outward(hi_product, True) / RGB_SCALE, True)
    return np.clip(lo, 0.0, 1.0), np.clip(hi, 0.0, 1.0)


def _transfer_bound(values: FloatArray, upper: bool) -> FloatArray:
    result = np.empty_like(values)
    small = values <= TRANSFER_THRESHOLD
    result[small] = outward(values[small] * TRANSFER_SLOPE, upper)
    base = outward(values[~small] * TRANSFER_SCALE, upper)
    base = outward(base + TRANSFER_OFFSET, upper)
    result[~small] = power_bound(base, TRANSFER_POWER, upper)
    return result


def _xyz_bound(linear: FloatArray, upper: bool) -> FloatArray:
    r, g, b = (linear[..., index] for index in range(3))
    channels = []
    for row in MATRIX:
        value = outward(g * row[1], upper) + outward(r * row[0], upper)
        value = outward(value, upper) + outward(b * row[2], upper)
        channels.append(outward(value, upper))
    xyz = np.clip(np.stack(channels, axis=-1), 0.0, XYZ_MAX)
    return outward(xyz * XYZ_SCALE, upper)


def _lab_f_bound(xyz: FloatArray, upper: bool) -> FloatArray:
    result = np.empty_like(xyz)
    small = xyz <= LAB_THRESHOLD
    result[small] = outward(outward(xyz[small] * LAB_SLOPE, upper) + LAB_OFFSET, upper)
    result[~small] = power_bound(xyz[~small], LAB_POWER, upper)
    return result


def encoded_bounds(codes: object) -> tuple[FloatArray, FloatArray]:
    """Enclose the tail under IEEE binary64 operations and faithful vendor pow.

    Both piecewise functions are increasing, including their small positive
    jumps. Selector-0 matrix entries are positive. Chroma subtraction needs
    opposite endpoints; evaluating only eight RGB corners would not suffice.
    """
    rgb_lo, rgb_hi = rgb_interval(codes)
    f_lo = _lab_f_bound(_xyz_bound(_transfer_bound(rgb_lo, False), False), False)
    f_hi = _lab_f_bound(_xyz_bound(_transfer_bound(rgb_hi, True), True), True)
    bounds = []
    for upper, same, opposite in ((False, f_lo, f_hi), (True, f_hi, f_lo)):
        lightness = outward(outward(same[..., 1] * 116.0, upper) - 16.0, upper)
        a = outward(same[..., 0] - opposite[..., 1], upper)
        b = outward(same[..., 1] - opposite[..., 2], upper)
        lab = np.stack(
            [lightness, outward(a * 500.0, upper), outward(b * 200.0, upper)],
            axis=-1,
        )
        lab = np.clip(lab, LAB_MIN, LAB_MAX)
        encoded = outward(outward(lab * ENCODE_SCALE, upper) + ENCODE_OFFSET, upper)
        bounds.append(np.clip(np.trunc(encoded), 0, int(RGB_SCALE)))
    return bounds[0], bounds[1]


def _load_planes(path: Path) -> tuple[dict, dict]:
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != REPORT_HASH:
        raise ValueError("Unexpected scalar report hash.")
    source = json.loads(content)
    if (
        source.get("passed") is not True
        or source.get("grid") != GRID
        or source.get("selector") != 0
        or ROME_HASH not in source.get("input_hashes", {}).values()
    ):
        raise ValueError("Unexpected scalar report provenance.")
    planes = {}
    for case in source["cases"]:
        key = case["processing"], case["flags"], case["matrix"]
        processing, flags, matrix = key
        if key in planes or case.get("passed") is not True:
            raise ValueError("Invalid or repeated scalar case.")
        filename = f"p{processing}-f{flags}-m{matrix}.bin"
        data = path.with_name(filename).read_bytes()
        if (
            len(data) != 3 * POINTS * 2
            or hashlib.sha256(data).hexdigest() != case["sha256"]
        ):
            raise ValueError(f"Unexpected plane bytes: {filename}.")
        planes[key] = np.frombuffer(data, dtype="<u2").reshape(3, POINTS).T
    if set(planes) != set(itertools.product((0, 2), (0, 1), (0, 1))):
        raise ValueError("Incomplete scalar case matrix.")
    return source, planes


def verify(path: Path) -> dict:
    source, planes = _load_planes(path)
    comparisons = []
    checks = {}
    for processing in (0, 2):
        rgb = planes[processing, 1, 1]
        reference = planes[processing, 0, 1]
        validate_triplets(reference, encoded=True)
        midpoint = rgb.astype(np.float64) / RGB_SCALE
        reconstructed = rgb_to_icc_lab(midpoint)
        lower, upper = encoded_bounds(rgb)
        difference = reconstructed.astype(np.int32) - reference.astype(np.int32)
        violations = (reference < lower) | (reference > upper)
        rgb_lo, rgb_hi = rgb_interval(rgb)
        ideal_lo = np.clip((rgb.astype(np.float64) - 0.5) / RGB_SCALE, 0.0, 1.0)
        ideal_hi = np.clip((rgb.astype(np.float64) + 0.5) / RGB_SCALE, 0.0, 1.0)
        comparisons.append(
            {
                "processing": processing,
                "points": POINTS,
                "channel_values": 3 * POINTS,
                "midpoint_max_absolute_code_difference": np.abs(difference)
                .max(axis=0)
                .tolist(),
                "midpoint_differing_values": (difference != 0).sum(axis=0).tolist(),
                "interval_violations": violations.sum(axis=0).tolist(),
                "maximum_encoded_interval_width": (upper - lower).max(axis=0).tolist(),
                "rgb_rounding_inflation": float(
                    max((ideal_lo - rgb_lo).max(), (rgb_hi - ideal_hi).max())
                ),
            }
        )
        checks[f"p{processing}_interval_inclusion"] = not bool(np.any(violations))
        checks[f"p{processing}_rgb_matrix_invariant"] = np.array_equal(
            rgb, planes[processing, 1, 0]
        )
        checks[f"p{processing}_zero_matrix_lab"] = bool(
            np.all(planes[processing, 0, 0] == [0, 32768, 32768])
        )
    checks["processing_modes_equal"] = all(
        np.array_equal(planes[0, flags, matrix], planes[2, flags, matrix])
        for flags, matrix in itertools.product((0, 1), repeat=2)
    )
    return {
        "evidence_kind": "native_scalar_tail_quantization_interval_replay",
        "source_report_sha256": REPORT_HASH,
        "source_report": str(path.resolve()),
        "source_plane_hashes": {
            f"p{c['processing']}-f{c['flags']}-m{c['matrix']}.bin": c["sha256"]
            for c in source["cases"]
        },
        "implementation_sha256": hashlib.sha256(
            Path(__file__).with_name("kaiji_output.py").read_bytes()
        ).hexdigest(),
        "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "numerical_model": {
            "basic_operations": "Binary64 with outward nextafter at each operation.",
            "powers": "MPFR 53-bit directed bounds enclose both faithful results.",
            "gmpy2": gmpy2.version(),
            "mpfr": gmpy2.mpfr_version(),
            "quantization_half_step": 0.5 / RGB_SCALE,
            "code_tolerance": 0,
            "vendor_pow_assumption": "Faithful rounding, not independently proven.",
        },
        "comparisons": comparisons,
        "checks": checks,
        "passed": all(checks.values()),
        "limits": [
            "Inclusion is conditional on the stated floating-point model.",
            "Interval inclusion is not exact unquantized-RGB or rendering parity.",
            "The report replays a synthetic state and bypasses the engine factory.",
            "Vendor runtime pow accuracy and full image rendering remain unverified.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.output.exists() or args.output.resolve().is_relative_to(root):
        parser.error("Output must be a new file outside the repository.")
    try:
        report = verify(args.input_report)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"verify_kaiji_output: {exc}\n")
    print(json.dumps({"report": str(args.output), "passed": report["passed"]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
