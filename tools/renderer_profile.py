"""Native sampled matrix/TRC ICC parameters for the bounded renderer kernels."""

from __future__ import annotations

import struct

import numpy as np
from cml_luts import icc_tags, profile_curves

D50 = struct.pack(">3i", 63190, 65536, 54061)


def inverse_matrix(matrix: np.ndarray) -> np.ndarray:
    """Invert with the renderer's double determinant and float32 cofactors."""
    if (
        not isinstance(matrix, np.ndarray)
        or matrix.shape != (3, 3)
        or matrix.dtype != np.dtype("<f4")
        or not np.all(np.isfinite(matrix))
        or np.any(np.abs(matrix) > 16)
    ):
        raise ValueError("Expected a finite bounded float32 3x3 matrix.")
    a, b, c, d, e, f, g, h, i = matrix.astype(np.float64).ravel()
    determinant = ((e * i - f * h) * a + (f * g - d * i) * b) + (d * h - e * g) * c
    if determinant <= 1 / 65536:
        raise ValueError("Matrix determinant must exceed 1/65536.")
    a, b, c, d, e, f, g, h, i = matrix.ravel()
    cofactors = np.array(
        [
            e * i - h * f,
            h * c - b * i,
            b * f - e * c,
            f * g - d * i,
            i * a - g * c,
            d * c - f * a,
            h * d - e * g,
            b * g - h * a,
            e * a - b * d,
        ],
        dtype="<f4",
    )
    result = (
        (cofactors.astype(np.float64) * (1 / determinant)).astype("<f4").reshape(3, 3)
    )
    if np.any(np.abs(result) > 16):
        raise ValueError("Inverse matrix exceeds the bounded renderer range.")
    return result


def sampled_curve(words: np.ndarray, *, inverse: bool = False) -> np.ndarray:
    """Expand normalized input or unquantized inverse output TRC samples."""
    if (
        type(inverse) is not bool
        or not isinstance(words, np.ndarray)
        or words.ndim != 1
        or not 2 <= len(words) <= 65536
        or words.dtype != np.dtype("<u2")
        or words[0] != 0
        or words[-1] != 65535
        or np.any(words[1:] <= words[:-1])
    ):
        raise ValueError("Expected a strictly increasing full-domain uint16 curve.")
    if inverse:
        values = words.astype(np.float64)
        queries = np.arange(65536, dtype=np.float64)
        lower = np.clip(
            np.searchsorted(values, queries, side="left") - 1, 0, len(values) - 2
        )
        reciprocal = 1 / (values[lower + 1] - values[lower])
        return (
            ((queries - values[lower]) * reciprocal + lower)
            * (65535 / (len(values) - 1))
        ).astype("<f4")
    values = words.astype(np.float32)
    scale = np.float32(1 / 65535)
    coordinate = np.arange(65536, dtype=np.float32) * (
        scale * np.float32(len(values) - 1)
    )
    lower = coordinate.astype(np.int32)
    fraction = coordinate - lower.astype(np.float32)
    expanded = (
        values[lower] * (np.float32(1) - fraction)
        + values[np.minimum(lower + 1, len(values) - 1)] * fraction
    ) * scale
    if np.any(expanded[1:] < expanded[:-1]):
        raise ValueError("Expanded curve is outside the monotonic kernel range.")
    return expanded


def parameters(
    profile: bytes, *, inverse: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Read a v2.2 D50 monitor RGB/XYZ profile with full-domain sampled TRCs.

    Return a row-major matrix and three float32 curves. The forward direction
    supplies RGB-to-Lab inputs; inverse supplies Lab-to-RGB outputs. Other ICC
    classes, LUT profiles, parametric TRCs and nonmonotonic samples or float32
    expansions are excluded.
    """
    if type(inverse) is not bool:
        raise ValueError("Expected an explicit boolean transform direction.")
    tags = icc_tags(profile)
    if (
        profile[8:16] != b"\x02\x20\0\0mntr"
        or profile[20:24] != b"XYZ "
        or profile[68:80] != D50
        or any(key.startswith((b"A2B", b"B2A")) for key in tags)
    ):
        raise ValueError("Expected a v2.2 D50 monitor matrix/TRC RGB/XYZ profile.")
    columns = []
    for signature in (b"rXYZ", b"gXYZ", b"bXYZ"):
        tag = tags.get(signature, b"")
        if len(tag) != 20 or tag[:8] != b"XYZ \0\0\0\0":
            raise ValueError("Expected three complete XYZ colorant tags.")
        columns.append(struct.unpack_from(">3i", tag, 8))
    matrix = np.asarray(columns, dtype="<f4").T.copy() * np.float32(1 / 65536)
    if np.any(matrix < 0) or np.any(matrix > 16):
        raise ValueError("Colorants exceed the bounded nonnegative matrix range.")
    curves = np.stack(
        [sampled_curve(w, inverse=inverse) for w in profile_curves(profile)]
    )
    return (inverse_matrix(matrix) if inverse else matrix), curves
