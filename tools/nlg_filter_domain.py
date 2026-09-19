"""Bounded scalar Nlg RGB/filter-domain conversion and zero-strength clipping.

The adaptive 5x5 filter is not implemented here. Domain arrays use int32 numeric
values in Y, Cb, Cr order to preserve unsigned luma and signed chroma explicitly.
"""

from __future__ import annotations

import numpy as np


def _geometry(samples: np.ndarray, dtype: str) -> None:
    if (
        not isinstance(samples, np.ndarray)
        or samples.ndim != 3
        or samples.shape[2] != 3
        or samples.dtype != np.dtype(dtype)
        or any(not 1 <= size <= 2048 for size in samples.shape[:2])
    ):
        raise ValueError("Expected a bounded image with three little-endian channels.")


def _domain(samples: np.ndarray) -> None:
    _geometry(samples, "<i4")
    if np.any(samples < [0, -32768, -32768]) or np.any(samples > [65535, 32767, 32767]):
        raise ValueError("Expected unsigned 16-bit luma and signed 16-bit chroma.")


def _half(values: np.ndarray) -> np.ndarray:
    return (values - (values < 0) + 1) >> 1


def to_domain(pixels: np.ndarray) -> np.ndarray:
    """Convert RGB14 to integer luma and signed blue/red color differences."""
    _geometry(pixels, "<u2")
    if np.any(pixels > 16383):
        raise ValueError("Expected RGB samples in [0,16383].")
    red, green, blue = pixels.astype(np.int32).transpose(2, 0, 1)
    luma = green + ((red + blue + 1) >> 1)
    return np.stack((luma, 2 * blue - luma, 2 * red - luma), axis=-1).astype("<i4")


def from_domain(samples: np.ndarray) -> np.ndarray:
    """Convert to RGB with signed rounding, then clamp R/B to 14 and G to 12 bits."""
    _domain(samples)
    luma, cb, cr = samples.astype(np.int32).transpose(2, 0, 1)
    red, blue = _half(luma + cr), _half(luma + cb)
    green = luma - _half(red + blue)
    return np.clip(
        np.stack((red, green, blue), axis=-1), 0, [16383, 4095, 16383]
    ).astype("<u2")


def zero_strength(samples: np.ndarray) -> np.ndarray:
    """Apply the filter's final bounds when both Y and C strengths are zero.

    Zero strength does not bypass the internal luma/chroma clamps.
    """
    _domain(samples)
    return np.clip(samples, [0, -6143, -6143], [8190, 6142, 6142]).astype("<i4")
