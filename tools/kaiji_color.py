"""Recovered selector-0 output conversion; not a complete Picture Control renderer."""

from __future__ import annotations

import math
from collections.abc import Sequence


def _lab_components(rgb: Sequence[float]) -> tuple[float, float, float]:
    if len(rgb) != 3 or any(not math.isfinite(value) for value in rgb):
        raise ValueError("Expected three finite nonlinear RGB values.")
    linear = []
    for value in rgb:
        value = max(0.0, min(1.0, value))
        linear.append(
            value * 0.07739938080495357
            if value <= 0.04045
            else math.pow(value * 0.9478672985781991 + 0.05213270142180095, 2.4)
        )
    r, g, b = linear
    # Preserve the recovered multiplication/addition order and binary64 constants.
    x = (g * 0.385066415 + r * 0.436071339) + b * 0.143057225
    y = (g * 0.716880769 + r * 0.222502613) + b * 0.060614910
    z = (g * 0.097068147 + r * 0.013922978) + b * 0.713984047
    xyz = (
        max(0.0, min(0.9642, x)) * 1.0371292263015972,
        max(0.0, min(1.0, y)),
        max(0.0, min(0.8249, z)) * 1.212268153715602,
    )
    return tuple(
        value * 7.787 + 0.13793103448275862
        if value <= 0.008856
        else math.pow(value, 0.3333333333333333)
        for value in xyz
    )


def _encode(lightness: float, a: float, b: float) -> tuple[int, int, int]:
    return (
        int(max(0.0, min(100.0, lightness)) * 652.8000000000001 + 0.5),
        int(max(-128.0, min(127.0, a)) * 256.0 + 32768.5),
        int(max(-128.0, min(127.0, b)) * 256.0 + 32768.5),
    )


def rgb_to_lab(rgb: Sequence[float]) -> tuple[int, int, int]:
    """Convert pre-quantization nonlinear RGB into recovered ICC Lab uint16 words."""
    x, y, z = _lab_components(rgb)
    return _encode(y * 116.0 - 16.0, (x - y) * 500.0, (y - z) * 200.0)


def quantized_rgb_lab_bounds(
    rgb: Sequence[int],
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Conservatively propagate half-code RGB uncertainty through selector 0.

    The matrix/transfer operations are monotone for selector 0. Chroma bounds use
    opposing component endpoints. A 1e-9 Lab-unit margin covers binary64 rounding;
    this is a quantization-aware diagnostic, not bit-exact rendering evidence.
    """
    if len(rgb) != 3 or any(type(v) is not int or not 0 <= v <= 65280 for v in rgb):
        raise ValueError("Expected three RGB integer codes in [0, 65280].")
    low = _lab_components([(value - 0.5) / 65280.0 for value in rgb])
    high = _lab_components([(value + 0.5) / 65280.0 for value in rgb])
    margin = 1e-9
    return (
        _encode(
            low[1] * 116.0 - 16.0 - margin,
            (low[0] - high[1]) * 500.0 - margin,
            (low[1] - high[2]) * 200.0 - margin,
        ),
        _encode(
            high[1] * 116.0 - 16.0 + margin,
            (high[0] - low[1]) * 500.0 + margin,
            (high[1] - low[2]) * 200.0 + margin,
        ),
    )
