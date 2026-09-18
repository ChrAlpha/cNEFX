"""Native table composition for a bounded synthetic Z8 Flexible Color state."""

from __future__ import annotations

import struct
from collections.abc import Sequence

from flexible_math import interpolate, round_divide32
from kaiji_blender import blender_tables
from kaiji_camera import camera_ranges
from kaiji_contrast import contrast_tables
from kaiji_grading import grading_tables
from kaiji_input import input_matrix
from kaiji_tone import tone_tables

from nx_ctudio.picture_curve import generate_curve


def parameter_ranges(
    coefficients: bytes, values: Sequence[int], *, curve_payload: bytes | None = None
) -> list[tuple[int, bytes]]:
    """Generate model14/strength256/selector0/brightness6 parameter sections."""
    if len(values) != 41 or any(type(value) is not int for value in values):
        raise ValueError("Expected 41 normalized integer controls.")
    if any(not -100 <= v <= 100 for v in values[:5]):
        raise ValueError("Contrast and tone controls must be in [-100, 100].")
    result = []

    def words(offset: int, data, code: str = "H") -> None:
        result.append((offset, struct.pack(f"<{len(data)}{code}", *data)))

    groups = [values[30 + i * 3 : 33 + i * 3] for i in range(3)]
    sources, expanded = grading_tables(coefficients, groups, values[39], values[40])
    for offset, row in zip(
        (0x4E, 0xD0, 0x152, 0x1D4, 0x21D4, 0x41D4), sources + expanded, strict=True
    ):
        words(offset, row)
    words(0x61D4, input_matrix(coefficients, values[5], 256, 0), "h")
    regions = [values[6 + i * 3 : 9 + i * 3] for i in range(8)]
    for channel, row in enumerate(
        blender_tables(coefficients, regions, model_index=14)
    ):
        words(0x6228 + channel * 128, row, "h")
        words(0x6268 + channel * 128, row, "h")
        words(0x213C6 + channel * 64, [max(-512, min(511, v)) for v in row], "h")
    contrast = round_divide32(values[0] * 64, 25)
    sources, expanded = contrast_tables(coefficients, contrast, 6, model_index=14)
    words(0x25836, [256 - abs(contrast), abs(contrast)])
    for offset, row in zip((0x2583A, 0x25A3C, 0x25C3E, 0x25E40), sources, strict=True):
        words(offset, row)
    for offset, row in zip((0x26044, 0x2A044), expanded, strict=True):
        words(offset, row, "i")
    if curve_payload is None:
        source, expanded_tone = tone_tables(
            coefficients,
            round_divide32(values[1] * 6, 5),
            round_divide32(values[2] * 6, 5),
            values[3],
            values[4],
        )
    else:
        source = generate_curve(curve_payload, sample_max=4095)
        expanded_tone = tuple(
            interpolate(source, i, interval_shift=4) for i in range(4096)
        )
    words(0x2E044, source)
    words(0x2E246, expanded_tone)
    return result


def generate_table(
    coefficients: bytes,
    values: Sequence[int],
    *,
    blend_mix: int = 0,
    camera_index: int = 0,
    curve_payload: bytes | None = None,
) -> bytes:
    """Build all 213872 bytes without a vendor-generated baseline.

    Camera state+152c/+1533/+153e stays zero; hue is128, family12,
    model14, strength256, selector0, brightness6. An explicit custom curve replaces
    highlights/shadows/white/black; its activation must be resolved by the caller.
    The leading output-space matrix remains zero, suitable for scalar RGB only.
    """
    values = tuple(values)
    ranges = parameter_ranges(coefficients, values, curve_payload=curve_payload)
    ranges.extend(
        camera_ranges(
            coefficients,
            model_index=14,
            selector=0,
            saturation_bias=round_divide32(values[5] * 6, 25) + 12,
            blend_mix=blend_mix,
            camera_index=camera_index,
        )
    )
    output, mask = bytearray(0x34370), bytearray(0x34370)
    for offset, data in ranges:
        if (
            offset < 0
            or offset + len(data) > len(output)
            or any(mask[offset : offset + len(data)])
        ):
            raise ValueError("Native table sections overlap or exceed the output.")
        output[offset : offset + len(data)] = data
        mask[offset : offset + len(data)] = bytes([1]) * len(data)
    for offset, size in (
        (0, 72),
        (0x61F6, 2),
        (0x21756, 2),
        (0x26042, 2),
        (0x3436C, 4),
    ):
        if any(mask[offset : offset + size]):
            raise ValueError("Native section overwrites an initial matrix or padding.")
        mask[offset : offset + size] = bytes([1]) * size
    if not all(mask):
        raise ValueError("Native table layout is incomplete.")
    return bytes(output)
