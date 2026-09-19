"""Bounded native Lab16-to-RGB16 stage from the observed image renderer.

Only the D50 Lab path with identity auxiliary curves and three sampled output
curves is covered. Matrix and curve data must be supplied explicitly.
"""

from __future__ import annotations

import numpy as np


def _xyz(pixels: np.ndarray, width: int) -> np.ndarray:
    lab = pixels.astype(np.float32)
    lightness = lab[:, 0] * np.float32(0.0015259021893143654)
    low_y = lightness * np.float32(0.0011070563923567533)
    f = np.where(
        lightness > np.float32(8),
        (lightness + np.float32(16)) * np.float32(0.008620689623057842),
        low_y * np.float32(7.787036895751953) + np.float32(0.13793103396892548),
    )
    x = (lab[:, 1] - np.float32(32768)) * np.float32(0.00390625) * np.float32(
        0.0020000000949949026
    ) + f
    z = f - (
        (lab[:, 2] - np.float32(32768))
        * np.float32(0.00390625)
        * np.float32(0.004999999888241291)
    )

    def inverse(values: np.ndarray) -> np.ndarray:
        return np.where(
            values > np.float32(0.2068965584039688),
            (values * values) * values,
            (values - np.float32(0.13793103396892548))
            * np.float32(0.12841854989528656),
        )

    result = np.column_stack(
        (
            inverse(x) * np.float32(0.9642000198364258),
            np.where(lightness > np.float32(8), (f * f) * f, low_y),
            inverse(z) * np.float32(0.8248999714851379),
        )
    )
    # Each row's remainder uses double precision before returning to float32.
    tail = np.arange(len(pixels)) % width >= width // 4 * 4
    lab = pixels[tail].astype(np.float64)
    lightness = lab[:, 0] * 0.0015259021896696422
    low_y = lightness * 0.0011070564598794539
    f = np.where(
        lightness <= 8.000000000000334,
        low_y * 7.787037037037037 + 0.13793103448275862,
        lightness * 0.008620689655172414 + 0.13793103448275862,
    )
    x = (lab[:, 1] - 32768) * 0.0000078125 + f
    z = f - (lab[:, 2] - 32768) * 0.00001953125

    def inverse64(values: np.ndarray) -> np.ndarray:
        return np.where(
            values <= 0.20689655172414081,
            (values - 0.13793103448275862) * 0.12841854934601665,
            (values * values) * values,
        )

    result[tail] = np.column_stack(
        (
            inverse64(x) * 0.9642,
            np.where(lightness <= 8.000000000000334, low_y, (f * f) * f),
            inverse64(z) * 0.8249,
        )
    )
    return result


def transform(
    pixels: np.ndarray, matrix: np.ndarray, curves: np.ndarray, width: int
) -> np.ndarray:
    """Convert packed Lab16 rows with a float32 matrix and sampled output curves.

    Pixels have shape (height * width, 3); width preserves four-pixel groups and
    each row's double-precision tail. Curves have shape (3, 65536), with finite,
    nondecreasing float32 values in [0, 65535]. Matrix entries are bounded by 16.
    """
    if (
        not isinstance(pixels, np.ndarray)
        or pixels.ndim != 2
        or pixels.shape[1] != 3
        or pixels.dtype != np.dtype("<u2")
    ):
        raise ValueError("Expected packed Lab16 uint16 pixels.")
    if type(width) is not int or width <= 0 or len(pixels) == 0 or len(pixels) % width:
        raise ValueError(
            "Expected a positive integer width and complete nonempty rows."
        )
    if (
        not isinstance(matrix, np.ndarray)
        or matrix.shape != (3, 3)
        or matrix.dtype != np.dtype("<f4")
        or not np.all(np.isfinite(matrix))
        or np.any(np.abs(matrix) > 16)
    ):
        raise ValueError(
            "Expected a finite float32 3x3 matrix with entries in [-16, 16]."
        )
    if (
        not isinstance(curves, np.ndarray)
        or curves.shape != (3, 65536)
        or curves.dtype != np.dtype("<f4")
        or not np.all(np.isfinite(curves))
        or np.any(curves < 0)
        or np.any(curves > 65535)
        or np.any(curves[:, 1:] < curves[:, :-1])
    ):
        raise ValueError(
            "Expected three monotonic float32 curves of 65536 values in [0, 65535]."
        )
    source = _xyz(pixels, width)
    output = np.empty_like(pixels)
    for channel, row in enumerate(matrix):
        # Preserve the renderer's separate float32 products and left-to-right sum.
        rgb = (source[:, 0] * row[0] + source[:, 1] * row[1]) + source[:, 2] * row[2]
        coordinate = np.clip(rgb * np.float32(65535), np.float32(0), np.float32(65535))
        index = coordinate.astype(np.int32)
        fraction = coordinate - index.astype(np.float32)
        curve = curves[channel]
        values = curve[index] * (np.float32(1) - fraction)
        values += curve[np.minimum(index + 1, 65535)] * fraction
        output[:, channel] = np.rint(values).astype("<u2")
    return output
