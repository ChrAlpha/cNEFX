"""Native reconstruction of Kaiji's 65-node Grading curves and expanded LUTs."""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Sequence

from flexible_math import interpolate, round_divide32

COEFFICIENT_HASH = "d44561f16bf877a264ecfbb947f5b6635d6b0e9151fa70ca601d9b131804822f"


def grading_tables(
    coefficients: bytes,
    groups: Sequence[Sequence[int]],
    blending: int,
    balance: int,
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    """Generate three source curves and three expanded tables for pinned Z8 data.

    Groups are parameter-state order, each containing hue/chroma/brightness.
    UI labels and upstream NP3 import are deliberately outside this function.
    """
    if (
        len(coefficients) != 422416
        or hashlib.sha256(coefficients).hexdigest() != COEFFICIENT_HASH
    ):
        raise ValueError("Only the pinned Z8 coefficient resource is supported.")
    if len(groups) != 3 or any(len(group) != 3 for group in groups):
        raise ValueError("Expected three hue/chroma/brightness triples.")
    for hue, chroma, brightness in groups:
        if any(type(value) is not int for value in (hue, chroma, brightness)):
            raise ValueError("Grading settings must be integers.")
        if not (0 <= hue <= 360 and 0 <= chroma <= 100 and -100 <= brightness <= 100):
            raise ValueError("Grading setting is outside the supported range.")
    if type(blending) is not int or not 0 <= blending <= 100:
        raise ValueError("Blending must be an integer in [0, 100].")
    if type(balance) is not int or not -100 <= balance <= 100:
        raise ValueError("Balance must be an integer in [-100, 100].")

    def signed(offset: int) -> int:
        return struct.unpack_from("<h", coefficients, offset)[0]

    def unsigned(offset: int) -> int:
        return struct.unpack_from("<H", coefficients, offset)[0]

    def node(value: int) -> int:
        quotient = abs(value) // 16
        return max(0, min(65, (quotient if value >= 0 else -quotient) + 1))

    shift = round_divide32(balance * unsigned(0x1BCE4), 100)
    width = round_divide32(blending * unsigned(0x1BCE8), 100) + signed(0x1BCE6)
    boundaries = [signed(0x61A6A + 2 * i) - shift for i in range(4)]
    weights = [[0] * 65 for _ in range(3)]

    def ramp(group: int, start: int, end: int, rising: bool) -> None:
        for i in range(node(start), node(end)):
            value = round_divide32((i * 16 - start) * 256, width)
            weights[group][i] = value if rising else 256 - value

    for i in range(node(boundaries[0])):
        weights[0][i] = 256
    ramp(0, boundaries[0], boundaries[0] + width, False)
    ramp(1, boundaries[1] - width, boundaries[1], True)
    for i in range(node(boundaries[1]), node(boundaries[2])):
        weights[1][i] = 256
    ramp(1, boundaries[2], boundaries[2] + width, False)
    ramp(2, boundaries[3] - width, boundaries[3], True)
    for i in range(node(boundaries[3]), 65):
        weights[2][i] = 256

    brightness_nodes = [0] * 65
    color_nodes = [[0] * 65 for _ in range(3)]
    for group_index, (hue, chroma, brightness) in enumerate(groups):
        level = round_divide32(unsigned(0x1BCEA) * brightness, 100)
        colors = [
            round_divide32(signed(0x61AF4 + hue * 6 + channel * 2) * chroma, 100)
            for channel in range(3)
        ]
        for i, weight in enumerate(weights[group_index]):
            brightness_nodes[i] += round_divide32(level * weight, 256)
            for channel in range(3):
                color_nodes[channel][i] += round_divide32(colors[channel] * weight, 256)
    sources = tuple(
        tuple(
            max(
                0,
                min(
                    4095,
                    i * 64
                    + round_divide32(brightness_nodes[i], 16)
                    + round_divide32(
                        unsigned(0x61A72 + i * 2) * color_nodes[channel][i], 1024
                    ),
                ),
            )
            for i in range(65)
        )
        for channel in range(3)
    )
    expanded = tuple(
        tuple(interpolate(source, i, interval_shift=6) for i in range(4096))
        for source in sources
    )
    return sources, expanded
