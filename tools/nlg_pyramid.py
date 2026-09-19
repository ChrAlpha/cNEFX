"""Native Nlg spatial planes from a bounded, already reconstructed RGB pyramid.

RGB pyramid construction and source parameter recovery remain outside this module.
"""

from __future__ import annotations

import numpy as np


def _mirror(indices: np.ndarray, extent: int) -> np.ndarray:
    if extent == 1:
        return np.zeros_like(indices)
    period = 2 * (extent - 1)
    values = indices % period
    return np.minimum(values, period - values)


def _expand(plane: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    y, x = np.arange(shape[0]), np.arange(shape[1])
    by, bx = y // 2, x // 2
    ny = _mirror(by + np.where(y & 1, 1, -1), plane.shape[0])
    nx = _mirror(bx + np.where(x & 1, 1, -1), plane.shape[1])
    source = plane.astype(np.int64)
    return (
        source[by[:, None], bx] * 9
        + source[by[:, None], nx] * 3
        + source[ny[:, None], bx] * 3
        + source[ny[:, None], nx]
    ) >> 4


def _smooth(plane: np.ndarray) -> np.ndarray:
    height, width = plane.shape
    y, x = np.arange(height), np.arange(width)
    horizontal = (
        plane[:, _mirror(x - 1, width)] + 2 * plane + plane[:, _mirror(x + 1, width)]
    ) >> 2
    return (
        horizontal[_mirror(y - 1, height)]
        + 2 * horizontal
        + horizontal[_mirror(y + 1, height)]
    ) >> 2


def spatial_planes(
    pixels: np.ndarray,
    amounts: tuple[int, int, int],
    previous: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the two smoothed spatial planes in vendor output14/output15 order.

    Amounts are (chroma, plane14, plane15), each an unsigned byte. Optional previous
    planes must have ceil(height/2) by ceil(width/2) samples, with both axes >=2.
    Input RGB and previous planes are little-endian uint16 values in [0,16383].
    """
    if (
        not isinstance(pixels, np.ndarray)
        or pixels.ndim != 3
        or pixels.shape[2] != 3
        or pixels.dtype != np.dtype("<u2")
        or any(not 1 <= n <= 2048 for n in pixels.shape[:2])
        or np.any(pixels > 16383)
    ):
        raise ValueError("Expected a bounded RGB14 image in a uint16 array.")
    if (
        type(amounts) is not tuple
        or len(amounts) != 3
        or any(type(v) is not int or not 0 <= v <= 255 for v in amounts)
    ):
        raise ValueError("Expected three integer byte-valued Nlg amounts.")
    if previous is not None:
        shape = tuple((n + 1) // 2 for n in pixels.shape[:2])
        if (
            type(previous) is not tuple
            or len(previous) != 2
            or min(shape) < 2
            or any(
                not isinstance(p, np.ndarray)
                or p.shape != shape
                or p.dtype != np.dtype("<u2")
                or np.any(p > 16383)
                for p in previous
            )
        ):
            raise ValueError("Expected two bounded, nonsingleton half-size planes.")
    red, green, blue = pixels.astype(np.int64).transpose(2, 0, 1)
    cb = (blue * 512 - green * 339 - red * 173) >> 10
    cr = (red * 512 - green * 429 - blue * 83) >> 10
    choose = cr * 339 < cb * 429
    chroma = (
        np.where(
            choose & (cr * 173 >= cb * -512),
            cb * 1815,
            np.where(~choose & (cr * -512 < cb * 83), cr * 1436, cb * -352 + cr * -731),
        )
        >> 10
    )
    chroma *= amounts[0]
    chroma = np.clip((chroma + (chroma >> 63) + 128) >> 8, 0, 16383)
    luma = np.minimum(chroma + ((green * 601 + blue * 117 + red * 306) >> 10), 16383)
    result = []
    for index, amount in enumerate(amounts[1:]):
        value = (luma * amount) >> 8
        if previous is not None:
            value = np.minimum(value + _expand(previous[index], value.shape), 16383)
        result.append(_smooth(value).astype("<u2"))
    return result[0], result[1]
