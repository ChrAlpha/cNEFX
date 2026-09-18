"""Native 32-sector Blender tables for the pinned Z8 synthetic model-0 state."""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Sequence

from flexible_math import round_divide32
from kaiji_grading import COEFFICIENT_HASH


def _signed(value: int, bits: int) -> int:
    return (value + (1 << (bits - 1))) % (1 << bits) - (1 << (bits - 1))


def blender_tables(
    coefficients: bytes, regions: Sequence[Sequence[int]]
) -> tuple[tuple[int, ...], ...]:
    """Return six 32-word tables, before later engine-wide parameter blending."""
    if (
        len(coefficients) != 422416
        or hashlib.sha256(coefficients).hexdigest() != COEFFICIENT_HASH
    ):
        raise ValueError("Only the pinned Z8 coefficient resource is supported.")
    if len(regions) != 8 or any(len(region) != 3 for region in regions):
        raise ValueError("Expected eight hue/chroma/brightness triples.")
    if any(type(v) is not int or not -100 <= v <= 100 for row in regions for v in row):
        raise ValueError("Blender settings must be integers in [-100, 100].")

    def word(offset: int) -> int:
        return struct.unpack_from("<h", coefficients, offset)[0]

    hue = [0] * 32
    chroma = [0] * 32
    brightness = [[0, 0] for _ in range(32)]
    for region, settings in enumerate(regions):
        h, c, b = settings
        h_index, c_index, b_index = [region + (8 if v >= 0 else 0) for v in settings]
        for sector in range(32):
            hue[sector] = _signed(
                hue[sector]
                + round_divide32(
                    word(0x605E8 + sector * 32 + h_index * 2) * abs(h), 100
                ),
                16,
            )
            delta = round_divide32(
                word(0x6166A + sector * 32 + c_index * 2) * abs(c), 100
            )
            chroma[sector] = _signed(chroma[sector] + delta, 16)
            for axis in range(2):
                if c > 0:
                    brightness[sector][axis] += round_divide32(
                        word(0x615EA + sector * 4 + axis * 2) * delta, 1024
                    )
                brightness[sector][axis] += round_divide32(
                    word(0x609E8 + sector * 64 + axis * 32 + b_index * 2) * abs(b), 100
                )
    quarter = struct.unpack_from("<H", coefficients, 0x2777A)[0]
    step = (quarter + 256) >> 9
    result = [[] for _ in range(6)]
    for sector in range(32):
        angle = hue[sector]
        if angle < 0:
            angle += quarter * 4
        if not 0 <= angle < quarter * 4:
            raise ValueError(
                "Hue accumulation exceeds the recovered single-wrap domain."
            )
        if angle < quarter:
            sin_pos, cos_pos, sin_sign, cos_sign = quarter - angle, angle, 1, 1
        elif angle < 2 * quarter:
            sin_pos, cos_pos, sin_sign, cos_sign = (
                angle - quarter,
                2 * quarter - angle,
                1,
                -1,
            )
        elif angle < 3 * quarter:
            sin_pos, cos_pos, sin_sign, cos_sign = (
                3 * quarter - angle,
                angle - 2 * quarter,
                -1,
                -1,
            )
        else:
            sin_pos, cos_pos, sin_sign, cos_sign = (
                angle - 3 * quarter,
                4 * quarter - angle,
                -1,
                1,
            )
        sine = word(0x611E8 + sin_pos // step * 2) * sin_sign
        cosine = word(0x611E8 + cos_pos // step * 2) * cos_sign
        scale = max(0, _signed(chroma[sector] + 1024, 16))
        base = [word(0x1C528 + channel * 64 + sector * 2) for channel in range(6)]
        transformed = (
            base[0] * cosine - base[2] * sine,
            base[1] * cosine - base[3] * sine,
            base[2] * cosine + base[0] * sine,
            base[1] * sine + base[3] * cosine,
        )
        for channel, value in enumerate(transformed):
            rounded = round_divide32(_signed(value * scale, 32), 1 << 20)
            result[channel].append(max(-512, min(511, rounded)))
        for axis in range(2):
            delta = round_divide32(_signed(brightness[sector][axis], 32), 128)
            result[4 + axis].append(_signed(delta + base[4 + axis], 16))
    return tuple(tuple(channel) for channel in result)
