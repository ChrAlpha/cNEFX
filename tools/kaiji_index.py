"""Native camera-table selector leaf with externally supplied Nikon resources."""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Sequence

from kaiji_grading import COEFFICIENT_HASH

MODE_TABLE_HASH = "61a0dff3e76374ff454b9d3adef021de4828975ddc27d1b7903453d8bd8a7888"
BOUNDS = (
    (0, 65535),
    (0, 65535),
    (0, 19),
    (0, 12),
    (0, 24),
    (-256, 256),
    (0, 40),
    (0, 108),
    (0, 2),
    (0, 4),
)


def camera_index(coefficients: bytes, mode_table: bytes, values: Sequence[int]) -> int:
    """Evaluate RVA0x27e230 for ten explicit raw scalar arguments.

    Order: two unsigned levels, row, brightness, saturation, signed contrast,
    auxiliary control, offset, processing selector, correction mode0..4.
    Processing 0 ignores the mode; processing 2 requires correction mode1..4.
    The resource pointer is the omitted ninth ABI argument. Input names remain
    internal; this function does not infer them from camera metadata.
    """
    if len(values) != 10:
        raise ValueError("Expected ten camera selector arguments.")
    for i, (value, (low, high)) in enumerate(zip(values, BOUNDS, strict=True)):
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"Camera selector argument {i} is outside its range.")
    if values[8] not in (0, 2):
        raise ValueError("Only processing selectors 0 and 2 are supported.")
    if values[8] == 2 and values[9] == 0:
        raise ValueError("Processing selector 2 requires correction mode 1 through 4.")
    if (
        type(coefficients) is not bytes
        or len(coefficients) != 422416
        or hashlib.sha256(coefficients).hexdigest() != COEFFICIENT_HASH
    ):
        raise ValueError("Only the pinned Z8 coefficient resource is supported.")
    if (
        type(mode_table) is not bytes
        or len(mode_table) != 76
        or hashlib.sha256(mode_table).hexdigest() != MODE_TABLE_HASH
    ):
        raise ValueError("Only the pinned external mode-correction table is supported.")
    (
        level_a,
        level_b,
        row,
        brightness,
        saturation,
        contrast,
        auxiliary,
        offset,
        processing,
        mode,
    ) = values
    thresholds = struct.unpack_from("<5H", coefficients, 0xA82)
    band = 0
    while band < 5 and max(level_a, level_b) >= thresholds[band]:
        band += 1
    column = coefficients[0xA9F + row * 13 + brightness]
    contrast_band = int(contrast * 0.046875 + 12.5)
    correction = 0
    if processing == 2:
        node, fraction = divmod(offset, 6)
        index = (mode - 1) * 19 + node
        if fraction:
            a = struct.unpack_from("b", mode_table, index)[0]
            b = struct.unpack_from("b", mode_table, index + 1)[0]
            numerator = (6 - fraction) * a + fraction * b + 3
            correction = abs(numerator) // 6 * (-1 if numerator < 0 else 1)
        else:
            correction = mode_table[index]
    total = (
        coefficients[0xA8C + band]
        + coefficients[0xBA3 + contrast_band * 8 + column]
        + coefficients[0xC6B + auxiliary]
        + coefficients[0xC94 + saturation]
        + offset
        + correction
    ) & 255
    return min(108, total)
