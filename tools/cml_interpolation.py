"""Bounded native replay of the CML RGB16, grid32, mode1 consumer."""

from __future__ import annotations

import numpy as np


def transform(
    pixels: np.ndarray, grid: np.ndarray, input_lut: np.ndarray, output_lut: np.ndarray
) -> np.ndarray:
    """Use captured tables; this does not generate a CML transform from ICCs."""
    for array, shape in (
        (grid, (32, 32, 32, 4)),
        (input_lut, (3, 65536)),
        (output_lut, (3, 65536)),
    ):
        if array.shape != shape or array.dtype != np.dtype("<u2"):
            raise ValueError("Unsupported CML table shape or type.")
    if pixels.ndim != 2 or pixels.shape[1] != 3 or pixels.dtype != np.dtype("<u2"):
        raise ValueError("Expected RGB16 pixels.")
    values = input_lut[np.arange(3), pixels]
    # SIMD converts float32 coordinates to Q14 using round-to-nearest-even.
    scaled = np.rint(values.astype(np.float32) * np.float32(31 * 16384 / 65535))
    fixed = scaled.astype(np.int64)
    base, fraction = fixed >> 14, fixed & 16383
    order = np.argsort(-fraction, axis=1, kind="stable")
    weights = np.take_along_axis(fraction, order, axis=1)
    index = base.copy()
    result = grid[index[:, 0], index[:, 1], index[:, 2], :3].astype(np.int64)
    result *= 16384 - weights[:, 0, None]
    rows = np.arange(len(pixels))
    for step in range(3):
        axis = order[:, step]
        index[rows, axis] = np.minimum(31, index[rows, axis] + 1)
        value = grid[index[:, 0], index[:, 1], index[:, 2], :3].astype(np.int64)
        weight = weights[:, step]
        if step < 2:
            weight = weight - weights[:, step + 1]
        result += value * weight[:, None]
    return output_lut[np.arange(3), result >> 14]
