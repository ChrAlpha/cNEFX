"""Bounded native mode5 vignette with captured numeric lens parameters.

Only aspect1, unit radial normalization/cap, identity transfer tables, disabled
auxiliary correction and fixed strength0.4 are covered. Lens parameter recovery
from source metadata is outside this module.
"""

from __future__ import annotations

import numpy as np


def _numbers(values: tuple, length: int, bound: float) -> np.ndarray:
    if (
        type(values) is not tuple
        or len(values) != length
        or any(type(v) not in (int, float) for v in values)
        or any(not np.isfinite(v) or abs(v) > bound for v in values)
    ):
        raise ValueError("Expected bounded finite numeric parameters.")
    return np.array(values, dtype=np.float32)


def _parameters(size: tuple, offset: tuple, coefficients: tuple):
    if (
        type(size) is not tuple
        or len(size) != 2
        or any(type(v) is not int or not 2 <= v <= 16384 for v in size)
    ):
        raise ValueError("Expected integer image dimensions between 2 and 16384.")
    return _numbers(offset, 2, 64), _numbers(coefficients, 4, 16)


def radial_table(
    size: tuple[int, int],
    offset: tuple[float, float],
    coefficients: tuple[float, float, float, float],
) -> np.ndarray:
    """Generate the mode5 float32 polynomial table for the unit-exponent branch."""
    center, weights = _parameters(size, offset, coefficients)
    width, height = size
    diagonal = 1 - int(
        np.sqrt(np.float32(height * height) + np.float32(width * width))
        * np.float32(-2)
    )
    expanded_width = np.float32(width) + np.abs(center[0])
    expanded_height = np.float32(height) + np.abs(center[1])
    count = 1 - int(
        np.sqrt(expanded_height * expanded_height + expanded_width * expanded_width)
        * np.float32(-2)
    )
    radius = np.minimum(np.arange(count), diagonal).astype(np.float32) * (
        np.float32(1) / np.float32(diagonal)
    )
    square = radius * radius
    polynomial = (
        ((square * weights[0] + weights[1]) * square + weights[2]) * square + weights[3]
    ) * square
    polynomial = np.maximum(polynomial, np.float32(0))
    if np.any(polynomial >= 1):
        raise ValueError("Only the unit-exponent radial branch is verified.")
    return (polynomial * np.float32(4096)).astype("<u2")


def transform(
    pixels: np.ndarray,
    size: tuple[int, int],
    origin: tuple[int, int],
    offset: tuple[float, float],
    coefficients: tuple[float, float, float, float],
) -> np.ndarray:
    """Apply the observed RGB16 radial branch using native table generation."""
    center, _ = _parameters(size, offset, coefficients)
    if (
        not isinstance(pixels, np.ndarray)
        or pixels.ndim != 3
        or pixels.shape[2] != 3
        or pixels.dtype != np.dtype("<u2")
        or any(not 1 <= value <= 1024 for value in pixels.shape[:2])
    ):
        raise ValueError("Expected a bounded RGB16 tile.")
    if (
        type(origin) is not tuple
        or len(origin) != 2
        or any(type(v) is not int or v < 0 for v in origin)
        or origin[0] + pixels.shape[1] > size[0]
        or origin[1] + pixels.shape[0] > size[1]
    ):
        raise ValueError("Expected an integer tile origin inside the image.")
    table = radial_table(size, offset, coefficients)
    height, width = pixels.shape[:2]
    cx = (np.float32(size[0] * 0.5) + center[0]) - np.float32(0.5)
    cy = (np.float32(size[1] * 0.5) + center[1]) - np.float32(0.5)
    anchor = np.float32(origin[0] + width - 1) - cx
    descending = np.arange(width - 1, -1, -1, dtype=np.float32)
    dy = np.arange(origin[1], origin[1] + height, dtype=np.float32) - cy
    # Preserve the leaf's expanded polynomial and per-operation float32 rounding.
    distance_squared = (
        (descending - (anchor + anchor)) * descending + anchor * anchor
    )[None, :] + (dy * dy)[:, None]
    if np.any(distance_squared < 0):
        raise ValueError(
            "Negative rounded squared radius is outside the verified branch."
        )
    index = np.minimum(
        (np.sqrt(distance_squared) * np.float32(4)).astype(np.int64), len(table) - 1
    )
    values = pixels.astype(np.int64)
    correction = np.minimum((values * table[index, None]) >> 12, 65535)
    return np.minimum(values + ((correction * 3276) >> 13), 65535).astype("<u2")
