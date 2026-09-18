"""Recovered camera lookup sections for the pinned Z8 family-12 Kaiji path."""

from __future__ import annotations

import hashlib
import struct

from flexible_math import interpolate, round_divide32
from kaiji_grading import COEFFICIENT_HASH


def camera_ranges(
    coefficients: bytes, *, model_index: int, selector: int, saturation_bias: int
) -> list[tuple[int, bytes]]:
    """Generate independent camera sections, with state+152a/+152c both zero.

    This covers input scales, secondary saturation, correction/rotation tables,
    and final gamut compression. Other sections remain outside this function.
    """
    if (
        len(coefficients) != 422416
        or hashlib.sha256(coefficients).hexdigest() != COEFFICIENT_HASH
    ):
        raise ValueError("Only the pinned Z8 coefficient resource is supported.")
    if type(model_index) is not int or model_index != 14:
        raise ValueError("Only the validated model index 14 is supported.")
    if type(selector) is not int or selector not in (0, 1, 2):
        raise ValueError("Only matrix selectors 0, 1 and 2 are supported.")
    if type(saturation_bias) is not int or not 0 <= saturation_bias <= 24:
        raise ValueError("Saturation bias must be an integer in [0, 24].")
    result = [(0x48, coefficients[0x1BCDE:0x1BCE4])]

    def word(offset: int) -> int:
        return struct.unpack_from("<H", coefficients, offset)[0]

    def words(offset: int, data, code: str = "H") -> None:
        result.append((offset, struct.pack(f"<{len(data)}{code}", *data)))

    def curve(source_offset: int, target: int, source, shift: int, signed=False):
        words(source_offset, source, "h" if signed else "H")
        expanded = tuple(
            interpolate(source, i, interval_shift=shift, signed=signed)
            for i in range((len(source) - 1) << shift)
        )
        words(target, expanded, "h" if signed else "H")

    shifts = [coefficients[0x1C504 + i * 3 + selector] for i in range(12)]
    result.append((0x61EA, bytes(shifts)))
    words(0x61F8, [1 << min(3, shift) for shift in shifts], "i")

    for source_offset, target, resource in (
        (0x69FA, 0x18BC6, 0x5F99A),
        (0x6A3C, 0x1ABC6, 0x5FA60),
    ):
        source = tuple(word(resource + i * 6 + selector * 2) for i in range(33))
        curve(source_offset, target, source, 7)

    for source_offset, target, resource in (
        (0x6A7E, 0x20BC6, 0x276F4),
        (0x6AC0, 0x20FC6, 0x27736),
    ):
        curve(
            source_offset,
            target,
            struct.unpack_from("<33h", coefficients, resource),
            4,
            True,
        )

    amount = saturation_bias - 12
    side = 0x42 if amount > 0 else 0
    camera_index = coefficients[0x60042]
    secondary = []
    for i in range(33):
        node = word(0x25990 + model_index * 0x84 + side + i * 2)
        base = word(0x26D28 + model_index * 0x42 + i * 2)
        value = base + round_divide32((node - 256) * abs(amount), 12)
        scale = word(0x602D0 + camera_index * 2 + i * 24)
        secondary.append(round_divide32(scale * value, 256) & 65535)
    curve(0x6568, 0xABC6, tuple(secondary), 7)

    result.append((0x30246, coefficients[0x5DCFA:0x5DD12]))
    base = 0x5DE16 + model_index * 10
    words(
        0x3025E,
        (
            word(base),
            word(base + 2),
            coefficients[base + 4],
            word(base + 6),
            word(base + 8),
        ),
    )
    for source_offset, target, resource in (
        (0x30268, 0x3036C, 0x5DD12),
        (0x302EA, 0x3236C, 0x5DD94),
    ):
        curve(
            source_offset, target, struct.unpack_from("<65H", coefficients, resource), 6
        )
    return result
