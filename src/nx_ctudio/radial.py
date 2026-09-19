"""Recovered pixel-shift red/blue correction before the final sensor crop."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class RadialCorrection:
    center_q8: tuple[int, int]
    coefficients_q20: tuple[tuple[int, int, int], tuple[int, int, int]]

    def __post_init__(self) -> None:
        if (
            len(self.center_q8) != 2
            or len(self.coefficients_q20) != 2
            or any(len(triple) != 3 for triple in self.coefficients_q20)
        ):
            raise ValueError("Expected two radial centers and two coefficient triples.")
        values = (*self.center_q8, *self.coefficients_q20[0], *self.coefficients_q20[1])
        if any(
            type(value) is not int or not -(2**31) <= value < 2**31 for value in values
        ):
            raise ValueError("Radial parameters must be signed 32-bit integers.")


def _weight(distance: NDArray[np.float64]) -> NDArray[np.float64]:
    distance = np.abs(distance)
    squared = distance * distance
    cubed = squared * distance
    return np.where(
        distance <= 1,
        1 - 2 * squared + cubed,
        np.where(distance <= 2, 4 - 8 * distance + 5 * squared - cubed, 0),
    )


def correct_crop(
    planes: NDArray[np.uint16],
    raw_shape: tuple[int, int],
    scale: int,
    crop: tuple[int, int, int, int],
    correction: RadialCorrection,
) -> NDArray[np.uint16]:
    """Sample the final crop from uncropped RGB, using bounded float64 row blocks.

    Out-of-domain taps are omitted without renormalizing weights. Green is
    copied unchanged. The input must remain unmodified throughout sampling.
    """
    raw_height, raw_width = raw_shape
    if scale not in (1, 2) or min(raw_shape) < 8 or any(n % 2 for n in raw_shape):
        raise ValueError("Expected even RAW dimensions and scale one or two.")
    height, width = scale * (raw_height - 1), scale * (raw_width - 1)
    if planes.shape != (height, width, 3) or planes.dtype != np.uint16:
        raise ValueError("Expected full reconstructed RGB uint16 geometry.")
    if int(planes.max()) > 16383:
        raise ValueError("RGB samples exceed the 14-bit code range.")
    left, top, crop_width, crop_height = crop
    if not (
        left >= 1
        and top >= 1
        and crop_width > 0
        and crop_height > 0
        and left + crop_width <= raw_width
        and top + crop_height <= raw_height
    ):
        raise ValueError("Crop is outside the recovered sampling domain.")
    x0, y0 = scale * (left - 1), scale * (top - 1)
    output = planes[y0 : y0 + scale * crop_height, x0 : x0 + scale * crop_width].copy()
    cx = scale * ((raw_width - 1) / 2 + correction.center_q8[0] / 256 - 1)
    cy = scale * ((raw_height - 1) / 2 + correction.center_q8[1] / 256 - 1)
    radius = scale * math.sqrt(raw_width * raw_width + raw_height * raw_height) / 2
    dx = np.arange(x0, x0 + output.shape[1], dtype=np.float64)[None, :] - cx
    for start in range(0, output.shape[0], 64):
        stop = min(start + 64, output.shape[0])
        dy = np.arange(y0 + start, y0 + stop, dtype=np.float64)[:, None] - cy
        normalized = np.sqrt(dx * dx + dy * dy) / radius
        squared = normalized * normalized
        for channel, triple in zip((0, 2), correction.coefficients_q20, strict=True):
            a, b, c = [value / 1048576 for value in triple]
            factor = squared * b + a + squared * normalized * c + 1
            sx, sy = factor * dx + cx, factor * dy + cy
            supported = (sx > -2) & (sx < width + 1) & (sy > -2) & (sy < height + 1)
            # Extreme signed metadata can map far beyond int64; those taps are zero.
            sx, sy = np.where(supported, sx, 0), np.where(supported, sy, 0)
            ix, iy = np.trunc(sx).astype(np.int64), np.trunc(sy).astype(np.int64)
            accumulated = np.zeros_like(sx)
            for row in (-1, 0, 1, 2):
                y = iy + row
                wy = _weight(y - sy)
                for column in (-1, 0, 1, 2):
                    x = ix + column
                    wx = _weight(x - sx)
                    valid = supported & (y >= 0) & (y < height) & (x >= 0) & (x < width)
                    samples = planes[
                        y.clip(0, height - 1), x.clip(0, width - 1), channel
                    ]
                    accumulated += np.where(
                        valid, samples.astype(np.float64) * wx * wy, 0
                    )
            output[start:stop, :, channel] = np.clip(accumulated, 0, 16383).astype(
                np.uint16
            )
    return output
