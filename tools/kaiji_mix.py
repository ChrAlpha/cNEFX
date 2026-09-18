"""Native Z8 family-12 blend selection with external calibration tables."""

from __future__ import annotations

import hashlib
import struct
from bisect import bisect_left
from collections.abc import Sequence

CALIBRATION_BYTES = 3162
CALIBRATION_HASH = "4f255aa63f356fd4a2a985a818ce933dd23f7666d186a0ad1bfff08b8c8a0e6d"


def _signed(value: int, bits: int) -> int:
    return (value + (1 << (bits - 1))) % (1 << bits) - (1 << (bits - 1))


def _trunc(numerator: int, denominator: int) -> int:
    quotient = abs(numerator) // abs(denominator)
    return -quotient if (numerator < 0) != (denominator < 0) else quotient


def _rounded_short(numerator: int, denominator: int) -> int:
    # This leaf uses numerator-sign rounding even with a negative denominator.
    numerator = _signed(numerator, 32)
    if -(1 << 30) <= numerator < (1 << 30):
        quotient = _trunc(2 * numerator, denominator)
        result = _trunc(quotient + (-1 if numerator < 0 else 1), 2)
    else:
        result = _trunc(numerator, denominator)
    return _signed(result, 16)


def _matrix_product(camera: Sequence[int], matrix: Sequence[int]) -> list[int]:
    return [
        _rounded_short(
            sum(camera[i * 3 + k] * matrix[k * 3 + j] for k in range(3)), 16384
        )
        for i in range(3)
        for j in range(3)
    ]


def blend_mix(
    calibration: bytes,
    matrix: Sequence[int],
    red: int,
    blue: int,
    wb_mode: int,
    wb_value: int,
) -> int:
    """Evaluate Rome0x2ace00 for camera1071 and Flexible family12.

    Matrix is nine signed Q14 words from ColorBalance; red/blue are signed
    interpretations of the TOP 16-bit gains. WB mode/value are explicit inputs,
    not a claim of general metadata or application-override recovery.
    """
    if (
        type(calibration) is not bytes
        or len(calibration) != CALIBRATION_BYTES
        or hashlib.sha256(calibration).hexdigest() != CALIBRATION_HASH
    ):
        raise ValueError("Only the pinned Z8 blend calibration is supported.")
    if len(matrix) != 9 or any(
        type(v) is not int or not -32768 <= v <= 32767 for v in matrix
    ):
        raise ValueError("Expected nine signed16 matrix words.")
    if any(type(v) is not int or not -32768 <= v <= 32767 for v in (red, blue)):
        raise ValueError("Expected signed16 white-balance levels.")
    if type(wb_mode) is not int or not 0 <= wb_mode <= 21:
        raise ValueError("White-balance mode must be in [0, 21].")
    if type(wb_value) is not int or not -(1 << 31) <= wb_value < (1 << 31):
        raise ValueError("White-balance value must be signed32.")
    if wb_mode == 21:
        return 0
    camera = struct.unpack_from("<9h", calibration)
    reference = struct.unpack_from("<2h", calibration, 3158)
    combined = _matrix_product(camera, matrix)
    gains = [
        _rounded_short(value * 2048, level * 4) if level else 0
        for value, level in zip(reference, (red, blue), strict=True)
    ]
    rgb = [
        _rounded_short(
            combined[i] * gains[0]
            + combined[i + 1] * 2048
            + combined[i + 2] * gains[1],
            16384,
        )
        for i in (0, 3, 6)
    ]
    if rgb[0] == 0:
        return 0
    a, b = (_rounded_short(v * 6848, rgb[0]) for v in rgb[1:])
    x = min(43, max(0, _trunc(_signed(a - b, 16) + 6144, 256)))
    y = min(31, max(0, _trunc(_signed(-a - b, 16) + 1024, 128)))
    grid = struct.unpack_from("<h", calibration, 342 + (y * 44 + x) * 2)[0]
    if 1 <= wb_mode <= 16:
        return grid
    thresholds = list(struct.iter_unpack("<2h", calibration[18:342]))
    position = max(0, bisect_left([v[0] for v in thresholds], _trunc(wb_value, 4)) - 1)
    return max(grid, thresholds[position][1])
