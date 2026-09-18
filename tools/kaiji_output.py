"""Recovered selector-0/1 Kaiji output tails; not a complete color renderer."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
WordArray = NDArray[np.uint16]

# Binary64 constants from Rome2.dll preparation RVA 0x2ae7b0 and tail 0x2b08d2.
MATRIX = np.array(
    [
        [0.436071339, 0.385066415, 0.143057225],
        [0.222502613, 0.716880769, 0.060614910],
        [0.013922978, 0.097068147, 0.713984047],
    ]
)
MATRIX_SELECTOR1 = np.array(
    [
        [0.609775615, 0.205241544, 0.149182841],
        [0.311134109, 0.625655490, 0.063210401],
        [0.019469045, 0.060874584, 0.744556371],
    ]
)
RGB_SCALE = 65280.0
TRANSFER_THRESHOLD = 0.04045
TRANSFER_SLOPE = 0.07739938080495357
TRANSFER_SCALE = 0.9478672985781991
TRANSFER_OFFSET = 0.05213270142180095
TRANSFER_POWER = 2.4
XYZ_MAX = np.array([0.9642, 1.0, 0.8249])
XYZ_SCALE = np.array([1.0371292263015972, 1.0, 1.212268153715602])
LAB_THRESHOLD = 0.008856
LAB_SLOPE = 7.787
LAB_OFFSET = 0.13793103448275862
LAB_POWER = 0.3333333333333333
LAB_MIN = np.array([0.0, -128.0, -128.0])
LAB_MAX = np.array([100.0, 127.0, 127.0])
ENCODE_SCALE = np.array([652.8000000000001, 256.0, 256.0])
ENCODE_OFFSET = np.array([0.5, 32768.5, 32768.5])


def validate_triplets(values: object, *, encoded: bool = False) -> FloatArray:
    """Require nonempty finite (..., 3) RGB or unsigned ICC-code triplets."""
    array = np.asarray(values)
    if array.ndim < 1 or array.shape[-1] != 3 or not array.size:
        raise ValueError("Expected nonempty (..., 3) triplets.")
    if array.dtype.kind not in ("iu" if encoded else "iuf"):
        raise ValueError("Expected real numeric triplets.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Triplets must be finite.")
    maximum = RGB_SCALE if encoded else 1.0
    if np.any(array < 0) or np.any(array > maximum):
        raise ValueError(f"Triplets must be in the range [0, {maximum}].")
    return array.astype(np.float64)


def _power(values: FloatArray, exponent: float) -> FloatArray:
    return np.fromiter(
        (math.pow(float(value), exponent) for value in values.flat),
        dtype=np.float64,
        count=values.size,
    ).reshape(values.shape)


def rgb_to_icc_lab(rgb: object, *, selector: int = 0) -> WordArray:
    """Convert unquantized nonlinear RGB to the recovered uint16 Lab encoding.

    The input has already been clipped by the upstream Kaiji evaluator. Separate
    NumPy operations preserve the DLL's arithmetic order without fused products.
    Python's platform pow is not claimed bit-identical to the vendor runtime.
    """
    if type(selector) is not int or selector not in (0, 1):
        raise ValueError("Only output selectors 0 and 1 are supported.")
    rgb = validate_triplets(rgb)
    if selector == 1:
        linear = np.maximum(_power(rgb, 2.19921875), rgb * 0.03125)
        matrix = MATRIX_SELECTOR1
    else:
        linear = np.empty_like(rgb)
        small = rgb <= TRANSFER_THRESHOLD
        linear[small] = rgb[small] * TRANSFER_SLOPE
        linear[~small] = _power(
            rgb[~small] * TRANSFER_SCALE + TRANSFER_OFFSET, TRANSFER_POWER
        )
        matrix = MATRIX
    r, g, b = (linear[..., index] for index in range(3))
    xyz = np.stack([(g * row[1] + r * row[0]) + b * row[2] for row in matrix], axis=-1)
    xyz = np.clip(xyz, 0.0, XYZ_MAX) * XYZ_SCALE
    f = np.empty_like(xyz)
    small = xyz <= LAB_THRESHOLD
    f[small] = xyz[small] * LAB_SLOPE + LAB_OFFSET
    f[~small] = _power(xyz[~small], LAB_POWER)
    fx, fy, fz = (f[..., index] for index in range(3))
    lab = np.stack([fy * 116.0 - 16.0, (fx - fy) * 500.0, (fy - fz) * 200.0], axis=-1)
    lab = np.clip(lab, LAB_MIN, LAB_MAX)
    return np.trunc(lab * ENCODE_SCALE + ENCODE_OFFSET).astype(np.uint16)
