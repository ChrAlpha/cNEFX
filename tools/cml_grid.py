"""Compose the recovered Kaiji ICC to matrix/TRC RGB CML lattice natively."""

from __future__ import annotations

import hashlib
import struct

import numpy as np
from cml_luts import icc_tags, input_table

D50 = struct.pack(">3i", 63190, 65536, 54061)
SRGB_HASH = "49caea94c9d36322910350ee37f1fa09629bed70e01cf615a6863ad3f8d1475e"


def inverse_matrix(matrix: np.ndarray) -> np.ndarray:
    """CML0x1920 inverse with the vendor's operation order and singular gate."""
    if matrix.shape != (9,) or not np.all(np.isfinite(matrix)):
        raise ValueError("Expected nine finite matrix coefficients.")
    a, b, c, d, e, f, g, h, i = matrix.astype(np.float64)
    p, q, r = e * i - f * h, c * h, b * f - c * e
    determinant = (p * a - (b * i - q) * d) + r * g
    if not np.isfinite(determinant) or abs(determinant * 65536) < 1:
        raise ValueError("Unsupported singular output matrix.")
    return np.array(
        [
            p,
            q - b * i,
            r,
            g * f - d * i,
            a * i - g * c,
            d * c - a * f,
            d * h - g * e,
            g * b - a * h,
            a * e - d * b,
        ]
    ) * (1.0 / determinant)


def build_grid(source: bytes, target: bytes) -> np.ndarray:
    """Perceptual RGB/Lab mft2 input to v2.2 D50 matrix/TRC RGB output.

    No proofing, black-point compensation, extra LUTs or image preprocessing.
    """
    a2b = input_table(source)
    if hashlib.sha256(target).hexdigest() != SRGB_HASH:
        raise ValueError("Only the pinned Nikon sRGB output profile is validated.")
    tags = icc_tags(target)
    if (
        source[8:16] != bytes.fromhex("0220000073636e72")
        or target[8:16] != bytes.fromhex("022000006d6e7472")
        or source[68:80] != D50
        or target[68:80] != D50
        or target[20:24] != b"XYZ "
        or any(key.startswith((b"A2B", b"B2A")) for key in tags)
    ):
        raise ValueError("Unsupported ICC composition branch.")
    matrix = []
    for key in (b"rXYZ", b"gXYZ", b"bXYZ"):
        tag = tags.get(key, b"")
        if len(tag) != 20 or tag[:8] != b"XYZ \0\0\0\0":
            raise ValueError("Expected three XYZ output matrix columns.")
        matrix.extend(struct.unpack_from(">3i", tag, 8))
    inverse = inverse_matrix(np.asarray(matrix, dtype=np.float64) * (1.0 / 65536))
    nodes = np.frombuffer(a2b, dtype=">u2", count=33**3 * 3, offset=24628)
    nodes = nodes.reshape(33, 33, 33, 3)
    curves = np.frombuffer(a2b, dtype=">u2", count=9, offset=240250).reshape(3, 3)
    coordinate = np.indices((32, 32, 32)).reshape(3, -1).T.astype(np.float64)
    coordinate *= 65535.0 / 31
    position = coordinate * (32 * (1.0 / 65535))
    lower = position.astype(np.int64)
    fractions = position - lower
    lower = np.minimum(lower, 32)
    fractions[coordinate >= 65535] = 0
    order = np.argsort(-fractions, axis=1, kind="stable")
    fraction = np.take_along_axis(fractions, order, axis=1)
    weights = np.column_stack(
        (
            1 - fraction[:, 0],
            fraction[:, 0] - fraction[:, 1],
            fraction[:, 1] - fraction[:, 2],
            fraction[:, 2],
        )
    )
    vertices = [nodes[tuple(lower.T)].astype(np.float64)]
    index, rows = lower.copy(), np.arange(len(lower))
    for step in range(3):
        axis = order[:, step]
        index[rows, axis] = np.minimum(32, index[rows, axis] + 1)
        vertices.append(nodes[tuple(index.T)].astype(np.float64))
    # Double tetrahedral accumulation is ordered from the farthest vertex.
    lab = vertices[3] * weights[:, 3:4] + vertices[2] * weights[:, 2:3]
    lab = lab + vertices[1] * weights[:, 1:2]
    lab = lab + vertices[0] * weights[:, 0:1]
    for channel in range(3):
        position = np.clip(lab[:, channel], 0, 65535) * (2 * (1.0 / 65535))
        lower = position.astype(np.int64)
        delta = curves[channel, np.minimum(lower + 1, 2)].astype(np.float64)
        delta -= curves[channel, lower]
        lab[:, channel] = delta * (position - lower) + curves[channel, lower]
    lab[:, 0] *= 0.0015318627450980394
    lab[:, 1:] = lab[:, 1:] * 0.00390625 - 128.0
    lightness, a, b = lab.T
    low = lightness <= 7.9996247999999985
    y = np.where(low, lightness * 0.0011070519207350825, 0.0)
    fy = np.where(
        low,
        y * 7.787 + 0.13793103448275862,
        lightness * 0.008620689655172414 + 0.13793103448275862,
    )
    y = np.where(low, y, fy * fy * fy)
    fx, fz = a * 0.002 + fy, fy - b * 0.005

    def cube(value: np.ndarray) -> np.ndarray:
        return np.where(
            value <= 0.20689270648275859,
            (value - 0.13793103448275862) * 0.1284191601386927,
            value * value * value,
        )

    white = np.asarray(struct.unpack(">3i", D50)) * (1.0 / 65536)
    x, y, z = cube(fx) * white[0], y * white[1], cube(fz) * white[2]
    m = inverse
    rgb = (
        np.stack(
            (
                y * m[3] + x * m[0] + z * m[6],
                x * m[1] + y * m[4] + z * m[7],
                x * m[2] + y * m[5] + z * m[8],
            ),
            axis=1,
        )
        * 65535.0
    )
    biased = rgb * 0.74999618524452583 + 6755399441063936.0
    quantized = (biased.view(np.uint64) & 0xFFFFFFFF).astype(np.uint32).view(np.int32)
    result = np.zeros((32**3, 4), dtype="<u2")
    result[:, :3] = np.clip(quantized, 0, 65535)
    return result.reshape(32, 32, 32, 4)
