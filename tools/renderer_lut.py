"""Bounded native grid65 RGB-to-Lab stage from the observed image renderer.

This identity-curve path does not cover general ICC transforms or the later
Lab/Lch stages of a complete NX Studio render.
"""

from __future__ import annotations

import struct

import numpy as np
from cml_luts import icc_tags


def normalized_grid(profile: bytes) -> np.ndarray:
    """Decode the observed v2.2 D50 RGB/Lab grid65 mft2 identity-curve subset."""
    tags = icc_tags(profile)
    if (
        profile[8:16] != b"\x02\x20\0\0scnr"
        or profile[20:24] != b"Lab "
        or profile[68:80] != struct.pack(">3i", 63190, 65536, 54061)
    ):
        raise ValueError("Expected a v2.2 D50 scanner-class RGB/Lab ICC profile.")
    table = tags.get(b"A2B0", b"")
    identity = struct.pack(">9i", 65536, 0, 0, 0, 65536, 0, 0, 0, 65536)
    if (
        len(table) != 1672396
        or table[:12] != b"mft2\0\0\0\0\x03\x03\x41\0"
        or table[12:48] != identity
        or struct.unpack_from(">HH", table, 48) != (4096, 3)
    ):
        raise ValueError("Expected the recovered grid65 RGB/Lab mft2 layout.")
    for offset, count in ((52, 4096), (1672378, 3)):
        curves = np.frombuffer(table, dtype=">u2", count=count * 3, offset=offset)
        curves = curves.reshape(3, count).astype(np.int64)
        identity_curve = np.arange(count, dtype=np.int64) * 65535 // (count - 1)
        if np.any(np.abs(curves - identity_curve) > 2):
            raise ValueError("Only the observed identity-curve bypass is supported.")
    grid = np.frombuffer(table, dtype=">u2", count=65**3 * 3, offset=24628)
    grid = grid.reshape(65, 65, 65, 3).astype("<u2")
    lightness = np.minimum(grid[..., 0].astype(np.int64), 65280)
    grid[..., 0] = (lightness * 65535 + 32640) // 65280
    return grid


def transform(pixels: np.ndarray, grid: np.ndarray, width: int) -> np.ndarray:
    """Transform packed RGB16 rows using an already normalized grid65 lattice.

    Pixels have shape (height * width, 3); width preserves the renderer's SIMD
    groups and scalar tails. No source white balance or later stages are applied.
    """
    if grid.shape != (65, 65, 65, 3) or grid.dtype != np.dtype("<u2"):
        raise ValueError("Expected a normalized RGB/Lab uint16 grid65 lattice.")
    if pixels.ndim != 2 or pixels.shape[1] != 3 or pixels.dtype != np.dtype("<u2"):
        raise ValueError("Expected packed RGB16 pixels.")
    if type(width) is not int or width <= 0 or len(pixels) == 0 or len(pixels) % width:
        raise ValueError(
            "Expected a positive integer width and complete nonempty rows."
        )
    scale = np.float32(64) * np.float32(16384 / 65535)
    fixed = np.rint(pixels.astype(np.float32) * scale).astype(np.int64)
    base, fraction = fixed >> 14, fixed & 16383
    order = np.argsort(-fraction, axis=1, kind="stable")
    weights = np.take_along_axis(fraction, order, axis=1)
    index = base.copy()
    result = grid[tuple(index.T)].astype(np.int64) * (16384 - weights[:, :1])
    rows = np.arange(len(pixels))
    for step in range(3):
        axis = order[:, step]
        index[rows, axis] = np.minimum(64, index[rows, axis] + 1)
        weight = weights[:, step]
        if step < 2:
            weight = weight - weights[:, step + 1]
        result += grid[tuple(index.T)].astype(np.int64) * weight[:, None]
    # Each row's four-pixel groups truncate; its remaining pixels round half up.
    rounding = np.where(rows % width >= width // 4 * 4, 8192, 0)
    return ((result + rounding[:, None]) >> 14).astype("<u2")


def convert(profile: bytes, pixels: np.ndarray, width: int) -> np.ndarray:
    """Decode a supported ICC and apply its bounded RGB16-to-Lab16 stage."""
    return transform(pixels, normalized_grid(profile), width)
