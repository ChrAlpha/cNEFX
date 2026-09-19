"""Bounded native RGB16-to-Lab16 matrix branch of the Nikon renderer."""

from __future__ import annotations

import numpy as np

F = np.float32


def _cube_root(value: np.ndarray) -> np.ndarray:
    """Mirror the renderer's SIMD log/exp approximation and rounding order."""
    safe = np.maximum(value, F(1.1754943508222875e-38))
    bits = safe.view(np.uint32)
    mantissa = ((bits & np.uint32(0x807FFFFF)) | np.uint32(0x3F000000)).view(F)
    lower = mantissa < F(0.7071067690849304)
    mantissa = mantissa + np.where(lower, mantissa, F(0))
    exponent = ((bits >> 23).astype(np.int32) - 126 - lower.astype(np.int32)).astype(F)
    delta = mantissa - F(1)
    ratio = delta / (mantissa + F(1))
    square = ratio * ratio
    polynomial = (
        (F(0.29884400963783264) * square + F(0.3997655212879181)) * square
        + F(0.6666679382324219)
    ) * square
    logarithm = ((polynomial - delta) * ratio + F(1.1730463711501216e-7) * exponent) + (
        F(0.6931470632553101) * exponent + delta
    )
    scaled = logarithm * F(0.3333333432674408)
    n = np.rint(F(1.4426950216293335) * scaled).astype(np.int32)
    nf = n.astype(F)
    r = (F(-0.6931470632553101) * nf + scaled) + F(-1.1730463711501216e-7) * nf
    q = F(0.0999748557806015) * (r * r)
    p = (F(0.008320826105773449) * (r * r) + F(0.5)) * r
    power = ((n + 127).astype(np.uint32) << 23).view(F)
    return ((p + q + F(1)) / ((q - p) + F(1))) * power


def transform(
    pixels: np.ndarray, matrix: np.ndarray, curves: np.ndarray, width: int
) -> np.ndarray:
    """Apply normalized input curves and a nonnegative RGB-to-XYZ matrix.

    Input pixels are packed uint16 rows. Matrix entries are float32 in [0, 16];
    each monotonic input curve has 65536 float32 values in [0, 1]. The four-pixel
    SIMD groups and each row's double-precision remainder remain distinct.
    """
    if (
        not isinstance(pixels, np.ndarray)
        or pixels.ndim != 2
        or pixels.shape[1] != 3
        or pixels.dtype != np.dtype("<u2")
    ):
        raise ValueError("Expected packed RGB16 uint16 pixels.")
    if type(width) is not int or width <= 0 or len(pixels) == 0 or len(pixels) % width:
        raise ValueError(
            "Expected a positive integer width and complete nonempty rows."
        )
    if (
        not isinstance(matrix, np.ndarray)
        or matrix.shape != (3, 3)
        or matrix.dtype != np.dtype("<f4")
        or not np.all(np.isfinite(matrix))
        or np.any(matrix < 0)
        or np.any(matrix > 16)
    ):
        raise ValueError(
            "Expected a finite float32 3x3 matrix with entries in [0, 16]."
        )
    if (
        not isinstance(curves, np.ndarray)
        or curves.shape != (3, 65536)
        or curves.dtype != np.dtype("<f4")
        or not np.all(np.isfinite(curves))
        or np.any(curves < 0)
        or np.any(curves > 1)
        or np.any(curves[:, 1:] < curves[:, :-1])
    ):
        raise ValueError("Expected three monotonic float32 curves in [0, 1].")
    rgb = np.column_stack([curve[pixels[:, i]] for i, curve in enumerate(curves)])
    xyz = np.column_stack(
        [
            (rgb[:, 0] * row[0] + rgb[:, 1] * row[1]) + rgb[:, 2] * row[2]
            for row in matrix
        ]
    )
    normalized = xyz * np.array([1.0371291637420654, 1, 1.2122682332992554], dtype=F)
    root = _cube_root(normalized)
    f = np.where(
        normalized > F(0.008856452070176601),
        root,
        F(7.787036895751953) * normalized + F(0.13793103396892548),
    )
    lightness = np.where(
        xyz[:, 1] > F(0.008856452070176601),
        F(116) * root[:, 1] - F(16),
        F(903.2963256835938) * xyz[:, 1],
    ) * F(655.3499755859375)
    a = ((f[:, 0] - f[:, 1]) * F(500)) * F(256) + F(32768)
    b = ((f[:, 1] - f[:, 2]) * F(200)) * F(256) + F(32768)
    result = np.rint(np.clip(np.column_stack((lightness, a, b)), 0, 65535)).astype(
        "<u2"
    )
    tail = np.arange(len(pixels)) % width >= width // 4 * 4
    values = xyz[tail].astype(np.float64) * [1.0371292263015972, 1, 1.212268153715602]
    f = np.where(
        values <= 0.008856,
        values * 7.787 + 0.13793103448275862,
        values ** (1 / 3),
    )
    lightness = np.where(
        values[:, 1] <= 0.008856,
        values[:, 1] * 591977.655,
        f[:, 1] * 76020.6 - 10485.6,
    )
    a = (f[:, 0] - f[:, 1]) * 128000
    b = (f[:, 1] - f[:, 2]) * 51200
    result[tail] = np.clip(
        np.floor(np.column_stack((lightness + 0.5, a + 32768.5, b + 32768.5))),
        0,
        65535,
    ).astype("<u2")
    return result
