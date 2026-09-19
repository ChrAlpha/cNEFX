"""Recovered RGB16 standard transfer tables for the observed RAW LUT stages."""

from __future__ import annotations

import ctypes
import ctypes.util

import numpy as np


def srgb_lut(*, inverse: bool = False) -> np.ndarray:
    """Generate utilities GetStdLut(0, inverse, 16, 16) using native libm powf.

    Each arithmetic operation rounds to float32; final RGB16 conversion truncates.
    Exact vendor parity is validated on the recorded Linux libm runtime only.
    """
    if type(inverse) is not bool:
        raise ValueError("Transfer direction must be boolean.")
    library = ctypes.util.find_library("m")
    if library is None:
        raise ValueError("A native libm with powf is required.")
    libm = ctypes.CDLL(library)
    if libm.fegetround() != 0:
        raise ValueError("Native transfer tables require nearest-even rounding.")
    power = libm.powf
    power.argtypes = [ctypes.c_float, ctypes.c_float]
    power.restype = ctypes.c_float
    f = np.float32
    x = np.arange(65536, dtype=f) * (f(1) / f(65535))
    mask = x > f(0.04045 if inverse else 0.0031308)
    if inverse:
        y = x * f(0.07739938050508499)
        base, exponent = (x[mask] + f(0.055)) * f(0.9478673338890076), f(2.4)
    else:
        y = x * f(12.92)
        base, exponent = x[mask], f(0.4166666567325592)
    nonlinear = np.fromiter(
        (power(float(value), float(exponent)) for value in base),
        dtype=f,
        count=len(base),
    )
    y[mask] = nonlinear if inverse else nonlinear * f(1.055) - f(0.055)
    return (np.minimum(y, f(1)) * f(65535)).astype("<u2")
