"""Native Nlg black/gain preparation, nonlinear limiting and RGB12 lookup."""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Parameters:
    black: int
    gains: tuple[int, int, int]
    offsets: tuple[int, int, int]
    nonlinear: int
    cutoff_base: int
    green_slope_q8: int
    blue_coupling_q8: int
    red_coupling_q8: int
    green_upper: int


def decode_parameters(record: bytes) -> Parameters:
    if type(record) is not bytes or len(record) != 0xB6:
        raise ValueError("Expected the complete 0xb6-byte Nlg level record.")
    return Parameters(
        struct.unpack_from("<H", record, 0x0C)[0],
        struct.unpack_from("<3H", record, 0x0E),
        struct.unpack_from("<3H", record, 0x14),
        record[0x1C],
        struct.unpack_from("<H", record, 0x1E)[0],
        struct.unpack_from("<H", record, 0x20)[0],
        struct.unpack_from("<H", record, 0x22)[0],
        struct.unpack_from("<H", record, 0x24)[0],
        struct.unpack_from("<h", record, 0x26)[0],
    )


def _signed(values: np.ndarray, bits: int) -> np.ndarray:
    sign = 1 << (bits - 1)
    return ((values + sign) & ((1 << bits) - 1)) - sign


def _round_gain(products: np.ndarray) -> np.ndarray:
    products = _signed(products, 32)
    # The sign correction and 512 addition are themselves 32-bit operations.
    return _signed(products - (products < 0) + 512, 32) >> 10


def _validate_image(samples: np.ndarray, dtype: str) -> None:
    if (
        not isinstance(samples, np.ndarray)
        or samples.ndim != 3
        or samples.shape[2] != 3
        or samples.dtype != np.dtype(dtype)
        or any(not 1 <= n <= 2048 for n in samples.shape[:2])
    ):
        raise ValueError("Expected a bounded three-channel little-endian image.")


def _validate_parameters(parameters: Parameters) -> None:
    if type(parameters) is not Parameters:
        raise ValueError("Expected decoded numeric gain parameters.")
    vectors = (parameters.gains, parameters.offsets)
    if any(type(v) is not tuple or len(v) != 3 for v in vectors):
        raise ValueError("Expected three gains and three offsets.")
    unsigned = (
        parameters.black,
        *parameters.gains,
        *parameters.offsets,
        parameters.cutoff_base,
        parameters.green_slope_q8,
        parameters.blue_coupling_q8,
        parameters.red_coupling_q8,
    )
    if (
        any(type(v) is not int or not 0 <= v <= 65535 for v in unsigned)
        or type(parameters.nonlinear) is not int
        or not 0 <= parameters.nonlinear <= 255
        or type(parameters.green_upper) is not int
        or not -32768 <= parameters.green_upper <= 32767
    ):
        raise ValueError("Expected numeric parameters within their stored widths.")


def prepare(pixels: np.ndarray, parameters: Parameters) -> np.ndarray:
    """Recover 0x1056e0 into signed16 RGB, including both gain branches."""
    _validate_image(pixels, "<u2")
    _validate_parameters(parameters)
    source = pixels.astype(np.int64)
    difference = source - parameters.black
    if not parameters.nonlinear:
        difference = _signed(difference, 16)
    scaled = _round_gain(difference * np.array(parameters.gains) * 2)
    if not parameters.nonlinear:
        return np.clip(scaled + parameters.offsets, -32768, 32767).astype("<i2")

    red = np.clip(scaled[..., 0], -16384, 16383)
    green = np.where(
        scaled[..., 1] <= parameters.green_upper,
        np.maximum(scaled[..., 1], -16384),
        parameters.green_upper,
    )
    green = _signed(green, 16)
    blue = np.clip(scaled[..., 2], -16384, 16383)
    cutoff = np.clip(
        parameters.cutoff_base
        - (
            _signed((green - parameters.cutoff_base) * parameters.green_slope_q8, 32)
            >> 8
        ),
        -16384,
        16383,
    )
    blue_below = blue < cutoff
    both_above = (red >= cutoff) & ~blue_below
    from_blue = np.clip(
        cutoff - (_signed((blue - cutoff) * parameters.blue_coupling_q8, 32) >> 8),
        -16384,
        16383,
    )
    from_red = np.clip(
        cutoff - (_signed((red - cutoff) * parameters.red_coupling_q8, 32) >> 8),
        -16384,
        16383,
    )
    coupled_red = np.where(
        both_above, cutoff, np.where(blue_below, np.minimum(red, from_blue), red)
    )
    coupled_blue = np.where(
        both_above, cutoff, np.where(blue_below, blue, np.minimum(blue, from_red))
    )
    coupled = np.stack((coupled_red, green, coupled_blue), axis=-1)
    return np.clip(coupled + parameters.offsets, -32768, 32767).astype("<i2")


def lookup(prepared: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Recover 0x6af00: shared uint16 LUT, signed input, RGB12 output."""
    _validate_image(prepared, "<i2")
    if (
        not isinstance(table, np.ndarray)
        or table.dtype != np.dtype("<u2")
        or table.shape != (65536,)
    ):
        raise ValueError("Expected the shared 65536-word little-endian lookup table.")
    indices = np.clip(prepared.astype(np.int64) * 2, 0, 65535)
    return np.minimum((table[indices].astype(np.int64) + 2) >> 2, 4095).astype("<u2")
