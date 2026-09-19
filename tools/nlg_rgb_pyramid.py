"""Bounded Nlg RGB pyramid reduction and residual reconstruction.

The spatial filter and initial Bayer-to-RGB stage are separate operations.
"""

from __future__ import annotations

import numpy as np


def _validate(pixels: np.ndarray) -> None:
    if (
        not isinstance(pixels, np.ndarray)
        or pixels.ndim != 3
        or pixels.shape[2] != 3
        or pixels.dtype != np.dtype("<u2")
        or any(not 1 <= n <= 2048 for n in pixels.shape[:2])
        or np.any(pixels > 16383)
    ):
        raise ValueError("Expected bounded uint16 RGB samples in [0,16383].")


def downsample(pixels: np.ndarray) -> np.ndarray:
    """Average 2x2 blocks with half-up rounding and reflected odd edges."""
    _validate(pixels)
    height, width = pixels.shape[:2]
    y, x = np.arange(0, height, 2), np.arange(0, width, 2)
    ny = np.where(y + 1 < height, y + 1, max(0, height - 2))
    nx = np.where(x + 1 < width, x + 1, max(0, width - 2))
    values = pixels.astype(np.int64)
    total = (
        values[y[:, None], x]
        + values[y[:, None], nx]
        + values[ny[:, None], x]
        + values[ny[:, None], nx]
    )
    return ((total + 2) >> 2).astype("<u2")


def reconstruct(fine: np.ndarray, coarse: np.ndarray) -> np.ndarray:
    """Add each coarse-minus-block-mean residual back to its original pixels.

    Fine and coarse RGB samples are unsigned 14-bit. Green clamps at 4095 while
    red/blue clamp at 16383. Coarse axes are the ceiling of half the fine axes.
    """
    _validate(fine)
    _validate(coarse)
    height, width = fine.shape[:2]
    if coarse.shape != ((height + 1) // 2, (width + 1) // 2, 3):
        raise ValueError("Expected the matching half-size coarse RGB image.")
    delta = coarse.astype(np.int64) - downsample(fine).astype(np.int64)
    corrected = (
        fine.astype(np.int64)
        + delta[np.arange(height)[:, None] // 2, np.arange(width) // 2]
    )
    return np.minimum(np.maximum(corrected, 0), [16383, 4095, 16383]).astype("<u2")
