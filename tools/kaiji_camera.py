"""Recovered camera lookup sections for the pinned Z8 family-12 Kaiji path."""

from __future__ import annotations

import hashlib
import struct

from flexible_math import interpolate, round_divide32
from kaiji_grading import COEFFICIENT_HASH


def camera_ranges(
    coefficients: bytes,
    *,
    model_index: int,
    selector: int,
    saturation_bias: int,
    hue: int = 128,
    strength: int = 256,
    blend_mix: int = 0,
    camera_index: int = 0,
) -> list[tuple[int, bytes]]:
    """Generate camera sections with explicit internal calibration controls.

    State+152c/+1533/+153e are zero. Hue supports the ordinary encoded
    range116..140, neutral sentinel255, and the historical invalid zero control.
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
    if type(hue) is not int or not (116 <= hue <= 140 or hue in (0, 255)):
        raise ValueError("Unsupported encoded hue control.")
    if type(strength) is not int or not 0 <= strength <= 256:
        raise ValueError("Strength must be an integer in [0, 256].")
    if type(blend_mix) is not int or not 0 <= blend_mix <= 10:
        raise ValueError("Blend mix must be an integer in [0, 10].")
    if type(camera_index) is not int or not 0 <= camera_index <= 108:
        raise ValueError("Camera index must be an integer in [0, 108].")
    result = [(0x48, coefficients[0x1BCDE:0x1BCE4])]

    def word(offset: int) -> int:
        return struct.unpack_from("<H", coefficients, offset)[0]

    def signed_word(offset: int) -> int:
        return struct.unpack_from("<h", coefficients, offset)[0]

    def trunc(value: int, denominator: int) -> int:
        quotient = abs(value) // denominator
        return -quotient if value < 0 else quotient

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
    words(0x61E6, (0, (10 - blend_mix) * 2048 // 10))
    result.append((0x61EA, bytes(shifts)))
    words(0x61F8, [1 << min(3, shift) for shift in shifts], "i")
    words(
        0x6528,
        tuple(
            trunc(
                word(0x5F8DA + selector * 2 + i * 6)
                * word(0x5FB26 + camera_index * 12),
                256,
            )
            for i in range(32)
        ),
    )

    result.append(
        (
            0x65AA,
            coefficients[
                0x1FE28 + model_index * 0x240 : 0x1FE28 + (model_index + 1) * 0x240
            ],
        )
    )
    for bank, (source_offset, target) in enumerate(
        (
            (0x67EA, 0x6BC6),
            (0x682C, 0x8BC6),
            (0x686E, 0xCBC6),
            (0x68B0, 0xEBC6),
            (0x68F2, 0x10BC6),
            (0x6934, 0x12BC6),
            (0x6976, 0x14BC6),
            (0x69B8, 0x16BC6),
            (0x6B02, 0x1CBC6),
        )
    ):
        resource = 0x226A8 + model_index * 0x252 + bank * 66
        curve(
            source_offset, target, struct.unpack_from("<33H", coefficients, resource), 7
        )
    resource = 0x2506C + model_index * 130
    curve(0x6B44, 0x1EBC6, struct.unpack_from("<65h", coefficients, resource), 6, True)

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
    saturation_index = coefficients[0x60042 + camera_index * 6]
    secondary = []
    for i in range(33):
        node = word(0x25990 + model_index * 0x84 + side + i * 2)
        base = word(0x26D28 + model_index * 0x42 + i * 2)
        value = base + round_divide32((node - 256) * abs(amount), 12)
        scale = word(0x602D0 + saturation_index * 2 + i * 24)
        secondary.append(round_divide32(scale * value, 256) & 65535)
    curve(0x6568, 0xABC6, tuple(secondary), 7)

    result.append((0x21546, coefficients[0x2A8C4:0x2A8D2]))
    u, v = (max(-4096, min(4095, signed_word(o))) for o in (0x2A8C4, 0x2A8C6))
    chroma = min(4095, abs(v - u) + abs(v + u) + abs(v) + abs(u))
    correction = word(0x2A8D0) - (
        (chroma * coefficients[0x2A8CD]) >> (coefficients[0x2A8CC] & 31)
    )
    words(
        0x23758, (max(0, min(1023, word(0x2A8CE), max(0, min(4095, correction)))),), "i"
    )
    mix = word(0x2A3A8 + camera_index * 12)
    source = tuple(
        trunc(word(0x29DA2 + i * 6) * mix + word(0x277A0 + i * 38) * (256 - mix), 256)
        & 65535
        for i in range(257)
    )
    curve(0x21554, 0x21758, source, 4)
    hue_delta = 0 if hue == 255 else trunc((hue - 128) * strength, 256)
    matrix_index = 4 * hue_delta + 48
    matrix = [signed_word(0x2A8D2 + i * 2) for i in range(4)]
    matrix.extend(
        (
            signed_word(0x2A8F6 + matrix_index * 2),
            signed_word(0x2A8F8 + matrix_index * 2),
            signed_word(0x2A8DE),
            signed_word(0x2A8FA + matrix_index * 2),
            signed_word(0x2A8FC + matrix_index * 2),
        )
    )
    words(0x2375C, tuple(matrix) + (0, 0, 0), "h")
    result.append(
        (
            0x23774,
            coefficients[0x2ABC0 + model_index * 64 : 0x2ABC0 + (model_index + 1) * 64],
        )
    )
    source = tuple(
        min(2047, word(0x2B040 + model_index * 130 + i * 2)) for i in range(65)
    )
    curve(0x237B4, 0x23836, source, 6)

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
