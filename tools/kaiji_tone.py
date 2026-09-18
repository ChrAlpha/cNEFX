"""Native Kaiji highlights/shadows/white/black curves in engine parameter units."""

from __future__ import annotations

import hashlib
import struct

from flexible_math import interpolate, round_divide32
from kaiji_grading import COEFFICIENT_HASH


def _truncate(numerator: int, denominator: int) -> int:
    value = abs(numerator) // denominator
    return -value if numerator < 0 else value


def tone_tables(
    coefficients: bytes, highlights: int, shadows: int, white: int, black: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Generate the family-12 no-custom-curve path at table offset 0x2e044.

    Highlights/shadows are post-application engine units, not NP3 input values.
    Contrast, custom-curve replacement and saturation are separate stages.
    """
    if (
        len(coefficients) != 422416
        or hashlib.sha256(coefficients).hexdigest() != COEFFICIENT_HASH
    ):
        raise ValueError("Only the pinned Z8 coefficient resource is supported.")
    for value, limit in zip(
        (highlights, shadows, white, black), (120, 120, 100, 100), strict=True
    ):
        if type(value) is not int or not -limit <= value <= limit:
            raise ValueError("Tone control is outside the supported integer range.")

    def word(offset: int, signed: bool = False) -> int:
        return struct.unpack_from("<h" if signed else "<H", coefficients, offset)[0]

    white_delta = round_divide32(white * word(0x62364 if white < 0 else 0x62366), 100)
    black_delta = round_divide32(black * word(0x62368 if black < 0 else 0x6236A), 100)
    slope = round_divide32((white_delta - black_delta + 4096) * 64, 256)
    offset = black_delta * 64
    high_limit, high_quad, high_linear, high_constant = 256, 0, 0, 0
    if white_delta < 0:
        high_limit = word(0x6236C)
        distance = 256 - high_limit
        high_quad = round_divide32(
            262144 - slope * high_limit - offset, distance**2
        ) - round_divide32(slope, distance)
        high_linear = slope - 2 * high_quad * high_limit
        high_constant = int((262144.0 - high_quad * 65536.0) - high_linear * 256)
    low_limit, low_quad, low_linear = 0, 0, 0
    if black_delta > 0:
        low_limit = word(0x6236E)
        low_quad = round_divide32(slope, low_limit) + round_divide32(
            -(slope * low_limit + offset), low_limit**2
        )
        low_linear = slope - 2 * low_quad * low_limit
    nodes = []
    for i in range(257):
        base = word(0x5CAE8 + i * 2)
        h = word(0x5CCEA + i * 8 + (2 if highlights < 0 else 3) * 2, True)
        s = word(0x5D4F2 + i * 8 + (2 if shadows < 0 else 3) * 2, True)
        adjusted = (
            base
            + round_divide32(h * abs(highlights), 120)
            + round_divide32(s * abs(shadows), 120)
        )
        # The vendor stores this intermediate as a signed 16-bit word.
        adjusted = (adjusted + 32768) % 65536 - 32768
        if i < low_limit:
            curve = _truncate(low_quad * i * i + low_linear * i, 64)
        elif i <= high_limit:
            curve = _truncate(slope * i + offset, 64)
        else:
            curve = _truncate(high_quad * i * i + high_linear * i + high_constant, 64)
        curve = (curve + 32768) % 65536 - 32768
        nodes.append(max(0, min(4095, adjusted + curve - base)))
    source = tuple(nodes)
    expanded = tuple(interpolate(source, i, interval_shift=4) for i in range(4096))
    return source, expanded
