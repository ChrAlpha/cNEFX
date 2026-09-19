"""Bounded native Nlg final-row arithmetic for the observed nonnegative branch.

Source decoding, state/table generation, and coarse spatial-grid construction
are outside this module. Captured pointers are never used by native processing.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np
from renderer_nlg_interpolation import interpolate


@dataclass(frozen=True)
class Parameters:
    matrix: np.ndarray
    curves: np.ndarray
    spatial_curve: np.ndarray
    difference_curves: np.ndarray
    gain_curve: np.ndarray


def decode_state(state: bytes) -> Parameters:
    """Extract numeric fields from the fixed, observed Nlg state branch."""
    if not isinstance(state, bytes) or len(state) != 0x10F9C0:
        raise ValueError("Expected the complete 0x10f9c0-byte Nlg state.")
    words = {
        0x48: 0,
        0x5E: 0,
        0x60: 0,
        0x62: 256,
        0x66: 38,
        0x68: 48,
        0x6A: 256,
        0x70: 1126,
        0x74: 1228,
        0x76: 4096,
        0x78: 4096,
        0x7C: 0,
    }
    flags = {
        0x5C: 1,
        0x64: 0,
        0x6C: 205,
        0x6D: 4,
        0x7A: 0,
        0x7E: 0,
        0x7F: 0,
        0x2E8D: 0,
        0x2EC3: 1,
        0x309A: 0,
    }
    if (
        any(struct.unpack_from("<H", state, k)[0] != v for k, v in words.items())
        or any(state[k] != v for k, v in flags.items())
        or struct.unpack_from("<i", state, 0x10F7D0)[0] != 0
    ):
        raise ValueError("Unsupported Nlg controls or optional correction branch.")

    def table(offset: int, count: int) -> np.ndarray:
        return np.frombuffer(state, dtype="<u2", offset=offset, count=count).copy()

    if any(np.any(table(offset, 4096) != 1024) for offset in (0xF7D0, 0x197D0)):
        raise ValueError("Only identity Nlg tone tables are verified.")
    difference = np.stack([table(0x117D0, 4096), table(0x137D0, 4096)])
    if any(
        not np.array_equal(table(offset, 4096), difference[i])
        for i, offset in enumerate((0x157D0, 0x177D0))
    ):
        raise ValueError("Only shared Nlg difference curves are verified.")
    gain = table(0x1B7D0, 4096)
    if not np.array_equal(gain, table(0x1D7D0, 4096)):
        raise ValueError("Only shared Nlg gain curves are verified.")
    result = Parameters(
        table(0x4A, 9).view("<i2").reshape(3, 3),
        np.stack([table(0x2F7D0 + i * 0x40000, 65536) for i in range(3)]),
        table(0xB7D0, 8192),
        difference,
        gain,
    )
    _validate_parameters(result)
    return result


def _array(value: np.ndarray, shape: tuple, maximum: int, signed: bool = False):
    if (
        not isinstance(value, np.ndarray)
        or value.shape != shape
        or value.dtype != np.dtype("<i2" if signed else "<u2")
        or np.any(value < 0)
        or np.any(value > maximum)
    ):
        raise ValueError("Expected bounded nonnegative Nlg numeric arrays.")


def _validate_parameters(parameters: Parameters) -> None:
    if not isinstance(parameters, Parameters):
        raise ValueError("Expected numeric Nlg parameters.")
    _array(parameters.matrix, (3, 3), 1024, True)
    if np.any(parameters.matrix.sum(axis=1) != 1024):
        raise ValueError("Only nonnegative unit-sum Q10 matrices are verified.")
    _array(parameters.curves, (3, 65536), 32767)
    _array(parameters.spatial_curve, (8192,), 4095)
    _array(parameters.difference_curves, (2, 4096), 2048)
    _array(parameters.gain_curve, (4096,), 1024)


def transform_row(
    pixels: np.ndarray,
    coarse0: np.ndarray,
    coarse1: np.ndarray,
    origin: tuple[int, int],
    parameters: Parameters,
) -> tuple[np.ndarray, np.ndarray]:
    """Return RGB16 and uint16 auxiliary gain for one bounded input row.

    Pixels are already halved, planar-source RGB represented as (width, 3).
    Coarse planes contain three logical rows, each ceil((left%8+width)/8)+2
    samples. Keep both interpolation truncations and all integer operation order.
    """
    if (
        not isinstance(pixels, np.ndarray)
        or pixels.ndim != 2
        or pixels.shape[1] != 3
        or not 1 <= len(pixels) <= 1024
    ):
        raise ValueError("Expected between 1 and 1024 RGB input pixels.")
    _array(pixels, pixels.shape, 32767)
    if (
        type(origin) is not tuple
        or len(origin) != 2
        or any(type(v) is not int or not 0 <= v < 16384 for v in origin)
        or origin[0] + len(pixels) > 16384
    ):
        raise ValueError("Expected a bounded nonnegative integer row origin.")
    _validate_parameters(parameters)
    width = len(pixels)
    coarse_width = ((origin[0] & 7) + width + 7) // 8
    for coarse in (coarse0, coarse1):
        _array(coarse, (3, coarse_width + 2), 4095)

    products = pixels.astype(np.int64)[:, None, :] * parameters.matrix[None, :, :]
    transformed = ((((products + 256) >> 9).sum(axis=2)) + 1) >> 1
    rgb = np.column_stack([parameters.curves[c, transformed[:, c]] for c in range(3)])
    rgb = rgb.astype(np.int64) >> 3
    x = np.arange(width) + (origin[0] & 7)
    spatial = []
    for coarse in (coarse0, coarse1):
        phases = interpolate(coarse[:, x[:, None] // 8 + np.arange(3)])
        horizontal = phases[:, np.arange(width), x % 8]
        values = interpolate(horizontal.T)[:, origin[1] & 7].astype(np.int64)
        square = np.minimum(((values + 48) ** 2) >> 11, 8191)
        indices = np.maximum(square - 38, 0)
        spatial.append(parameters.spatial_curve[indices].astype(np.int64))

    red, green, blue = rgb.T
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
    luma = np.minimum(
        ((205 * chroma + 128) >> 8) + ((green * 601 + red * 306 + blue * 117) >> 10),
        4095,
    )
    if np.any(luma < 0):
        raise ValueError("Negative Nlg luminance is outside the verified branch.")
    blend = []
    for coarse, amount in zip(spatial, (1126, 1228), strict=True):
        index = (luma >> 1) - (coarse >> 1) + 2048
        near, far = parameters.difference_curves[:, index].astype(np.int64)
        value = ((near * amount) >> 11) + ((far * (2048 - amount)) >> 11) - 1024
        value = (((value * parameters.gain_curve[spatial[0]]) >> 10) + 1024) & 8191
        blend.append(value)
    gain = np.clip((blend[1] * blend[0]) >> 10, 0, 8191)
    output = (np.minimum((rgb * gain[:, None]) >> 7, 32767) * 2).astype("<u2")
    auxiliary = np.minimum(gain >> 4, 255).astype("<u2")
    return output, auxiliary
