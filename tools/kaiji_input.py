"""Native family-12 saturation matrix for pinned external Z8 coefficients."""

from __future__ import annotations

import hashlib
import struct

from flexible_math import round_divide32
from kaiji_grading import COEFFICIENT_HASH


def input_matrix(
    coefficients: bytes, saturation: int, strength: int, selector: int
) -> tuple[int, ...]:
    """Return the nine signed words at generated-table offset 0x61d4."""
    if (
        len(coefficients) != 422416
        or hashlib.sha256(coefficients).hexdigest() != COEFFICIENT_HASH
    ):
        raise ValueError("Only the pinned Z8 coefficient resource is supported.")
    for value, low, high in (
        (saturation, -50, 50),
        (strength, 0, 256),
        (selector, 0, 2),
    ):
        if type(value) is not int or not low <= value <= high:
            raise ValueError("Input-matrix setting is outside the supported range.")
    factor = struct.unpack_from("<H", coefficients, 0x62438 + saturation * 2)[0]
    scale = round_divide32(strength * (factor - 1024) + 262144, 256)
    result = []
    for i in range(9):
        value = struct.unpack_from("<h", coefficients, 0x5F424 + i * 6 + selector * 2)[
            0
        ]
        if i >= 3:
            value = round_divide32(value * scale, 1024)
        result.append((value + 32768) % 65536 - 32768)
    return tuple(result)
