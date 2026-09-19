"""Native bounded Nlg forward/inverse tables with vendor float32 operation order."""

from __future__ import annotations

import numpy as np


def tables(
    scale: float, base: int, offset: int, half_range: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Generate the two 65536-word tables using the observed float32 order."""
    if (
        type(scale) is not float
        or not np.isfinite(scale)
        or not 1 / 256 <= scale <= 256
        or any(type(v) is not int or not 0 <= v <= 65535 for v in (base, offset))
        or type(half_range) is not bool
    ):
        raise ValueError(
            "Expected finite bounded scale and stored-width table controls."
        )
    scale32 = np.float32(scale)
    index = np.arange(65536, dtype=np.float32)
    value = index * (scale32 + scale32)
    value = value * np.float32(512)
    addend = np.float32(base) * np.float32(512)
    if half_range:
        value = value + value
        addend = addend + addend
    value = np.sqrt(value + addend)
    if half_range:
        value = value * np.float32(0.5)
    value = value - np.float32(offset) * np.float32(0.25)
    forward = np.clip(
        np.trunc(value * np.float32(4) + np.float32(0.5)), 0, 65535
    ).astype("<u2")
    reciprocal = np.float32(1) / scale32
    factor = np.float32(float(reciprocal) * 0.5)
    value = index + np.float32(offset)
    value = value * value
    value = value * np.float32(1 / (4096 if half_range else 8192))
    value = (value - np.float32(base)) * factor
    inverse = np.clip(np.trunc(value), 0, 65535).astype("<u2")
    return forward, inverse
