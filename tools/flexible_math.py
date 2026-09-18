"""Native integer leaves recovered from Rome2; no color rendering is implied."""

from __future__ import annotations

from collections.abc import Sequence

INT32_MIN = -(2**31)
INT32_MAX = 2**31 - 1


def _int32(value: int) -> None:
    if type(value) is not int or not INT32_MIN <= value <= INT32_MAX:
        raise ValueError("Expected a signed 32-bit integer.")


def round_divide32(numerator: int, denominator: int) -> int:
    """RVA 0x29d8e0: round signed division, retaining 32-bit addition wrap."""
    _int32(numerator)
    _int32(denominator)
    if denominator <= 0:
        raise ValueError("Only positive denominators are supported.")
    half = denominator // 2
    adjusted = numerator + (half if numerator >= 0 else -half)
    adjusted = (adjusted + 2**31) % 2**32 - 2**31
    quotient = abs(adjusted) // denominator
    return quotient if adjusted >= 0 else -quotient


def clamp32(value: int, minimum: int, maximum: int) -> int:
    """RVA 0x29d8d0 with ordered inclusive bounds."""
    for argument in (value, minimum, maximum):
        _int32(argument)
    if minimum > maximum:
        raise ValueError("Clamp bounds must be ordered.")
    return min(max(value, minimum), maximum)


def interpolate(
    samples: Sequence[int], index: int, *, interval_shift: int, signed: bool = False
) -> int:
    """Interpolate 16-bit nodes using the recovered delta-rounding convention.

    The valid index interval is [0, (len(samples) - 1) * 2**shift).
    The exact final node is excluded because the vendor reads the next node.
    Supported unsigned shifts are 4, 6 and 7; signed shifts are 4 and 6.
    """
    _int32(index)
    if type(signed) is not bool or type(interval_shift) is not int:
        raise ValueError("Unsupported interpolation mode.")
    if interval_shift not in ((4, 6) if signed else (4, 6, 7)):
        raise ValueError("Unsupported interpolation mode.")
    if len(samples) < 2 or not 0 <= index < (len(samples) - 1) << interval_shift:
        raise ValueError("Interpolation index requires two adjacent samples.")
    low, high = (-32768, 32767) if signed else (0, 65535)
    if any(type(value) is not int or not low <= value <= high for value in samples):
        raise ValueError("Samples must fit the selected 16-bit representation.")
    position, fraction = divmod(index, 1 << interval_shift)
    delta = (samples[position + 1] - samples[position]) * fraction
    correction = (delta + (1 << (interval_shift - 1)) - (delta < 0)) >> interval_shift
    bits = (samples[position] + correction) & 65535
    return bits - 65536 if signed and bits >= 32768 else bits
