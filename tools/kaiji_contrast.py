"""Native family-12 contrast/brightness tables before the separate tone stage."""

from __future__ import annotations

import hashlib
import struct

from flexible_math import round_divide32
from kaiji_grading import COEFFICIENT_HASH


def contrast_tables(
    coefficients: bytes,
    contrast: int,
    brightness: int,
    *,
    model_index: int = 0,
    output_selector: int = 0,
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    """Return four 257-node uint16 curves and two 4096-node int32 tables.

    Supports the regular family-12 path with +0x174f clear and +0xe3c in0..1.
    Contrast is in engine units; brightness is the engine's biased byte value.
    """
    if (
        len(coefficients) != 422416
        or hashlib.sha256(coefficients).hexdigest() != COEFFICIENT_HASH
    ):
        raise ValueError("Only the pinned Z8 coefficient resource is supported.")
    if type(model_index) is not int or model_index not in (0, 14):
        raise ValueError("Only model indices 0 and 14 are supported.")
    if type(contrast) is not int or not -256 <= contrast <= 256:
        raise ValueError("Contrast must be an integer in [-256, 256].")
    if type(brightness) is not int or not 0 <= brightness <= 12:
        raise ValueError("Brightness must be an integer in [0, 12].")
    if type(output_selector) is not int or output_selector not in (0, 1):
        raise ValueError("Only output selectors 0 and 1 are supported.")

    def word(offset: int) -> int:
        return struct.unpack_from("<H", coefficients, offset)[0]

    def trunc(value: int, denominator: int) -> int:
        quotient = abs(value) // denominator
        return -quotient if value < 0 else quotient

    side = 2 if contrast <= 0 else 4
    strength = abs(contrast)
    brightness_side = 0 if brightness <= 6 else 2
    brightness_amount = abs(brightness - 6)
    sources = []
    for bank, alternate in ((0, False), (0, True), (1, False), (1, True)):
        base = (0x35E7E if bank == 0 else 0x44362) + model_index * 0x606
        brightness_base = (0x52846 if bank == 0 else 0x5708E) + model_index * 0x404
        row = []
        for i in range(257):
            delta = trunc(
                (word(brightness_base + i * 4 + brightness_side) - 4096)
                * brightness_amount,
                6,
            )
            row.append(word(base + i * 6 + (side if alternate else 0)) + delta)
        if output_selector == 1:
            remapped = []
            for i in range(257):
                position, fraction = divmod(word(0x5B8D6 + i * 2), 16)
                remapped.append(
                    round_divide32(
                        row[position] * 16
                        + (row[position + 1] - row[position]) * fraction,
                        16,
                    )
                )
            row = remapped
        sources.append(tuple(value & 65535 for value in row))
    expanded = []
    for bank in range(2):
        base, alternate = sources[2 * bank : 2 * bank + 2]
        row = []
        for index in range(4096):
            position, fraction = divmod(index, 16)
            left = base[position] + round_divide32(
                fraction * (base[position + 1] - base[position]), 16
            )
            right = alternate[position] + round_divide32(
                fraction * (alternate[position + 1] - alternate[position]), 16
            )
            row.append(left * (256 - strength) + right * strength)
        expanded.append(tuple(row))
    return tuple(sources), tuple(expanded)
