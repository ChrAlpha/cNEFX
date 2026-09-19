"""Bounded CML Lab16/Lch16 identity-profile conversions, SIMD forward path.

This is Nikon's integer encoding, not a generic ICC Lab/Lch conversion. The
forward scalar fallback and additional ICC grids or curves are not covered.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=1)
def _tables() -> tuple[np.ndarray, ...]:
    angles = np.arange(65536, dtype=np.float64) * 9.587379924285257e-05
    sine, cosine = np.sin(angles), np.cos(angles)
    result = (
        np.floor(
            np.arctan2(np.arange(8193, dtype=np.float64), 8192.0) * 10430.378350470453
            + 0.5
        ).astype(np.int64),
        np.trunc(np.minimum(sine * 32768, 32767)).astype(np.int64),
        np.trunc(np.minimum(cosine * 32768, 32767)).astype(np.int64),
        np.trunc(32768 / np.maximum(np.abs(sine), np.abs(cosine))).astype(np.int64),
    )
    for table in result:
        table.flags.writeable = False
    return result


def _validate(pixels: np.ndarray) -> None:
    if (
        not isinstance(pixels, np.ndarray)
        or pixels.ndim != 2
        or pixels.shape[1] != 3
        or pixels.dtype != np.dtype("<u2")
        or len(pixels) == 0
    ):
        raise ValueError("Expected nonempty packed uint16 three-channel pixels.")


def lab_to_lch(pixels: np.ndarray) -> np.ndarray:
    """CML0xf4b0/0x11180: float32 magnitude, nearest-even and atan lookup."""
    _validate(pixels)
    atan, _, _, _ = _tables()
    xy = pixels[:, 1:].astype(np.float32) - np.float32(32768)
    absolute = np.abs(xy)
    maximum = np.max(absolute, axis=1)
    ratio = np.divide(
        np.min(absolute, axis=1) * np.float32(8192),
        maximum,
        out=np.zeros(len(pixels), dtype=np.float32),
        where=maximum != 0,
    )
    hue = atan[np.rint(ratio).astype(np.int64)]
    hue = np.where(absolute[:, 0] < absolute[:, 1], 16384 - hue, hue)
    hue = np.where(xy[:, 0] < 0, 32768 - hue, hue)
    hue = np.where(xy[:, 1] < 0, -hue, hue) & 65535
    hue = np.where(maximum == 0, 0, hue)
    chroma = np.rint(np.sqrt(xy[:, 1] * xy[:, 1] + xy[:, 0] * xy[:, 0]))
    return np.column_stack((pixels[:, 0], chroma, hue)).astype("<u2")


def lch_to_lab(pixels: np.ndarray) -> np.ndarray:
    """CML0x9180: hue-dependent chroma clamp, signed Q15 toward-zero products."""
    _validate(pixels)
    _, sine, cosine, limit = _tables()
    hue = pixels[:, 2]
    chroma = np.minimum(pixels[:, 1].astype(np.int64), limit[hue])

    def component(table: np.ndarray) -> np.ndarray:
        product = chroma * table[hue]
        return 32768 + np.sign(product) * (np.abs(product) // 32768)

    return np.column_stack((pixels[:, 0], component(cosine), component(sine))).astype(
        "<u2"
    )
