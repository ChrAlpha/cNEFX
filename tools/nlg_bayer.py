"""Native bounded Nlg Bayer reduction and directional auxiliary classification.

This is the initial 0x69f20 substage, before black/gain correction, lookup,
interpolation and chroma smoothing. Pattern values use the owning context's CFA.
"""

from __future__ import annotations

import numpy as np


def classification_table(low: int, high: int) -> np.ndarray:
    """Build the 1024-byte first-matching-threshold table from 0x1053a0."""
    if any(type(v) is not int or not 0 <= v <= 65535 for v in (low, high)):
        raise ValueError("Expected two unsigned 16-bit classification thresholds.")
    step = (high - low) >> 3
    values = np.arange(1024)
    result = np.full(1024, 8, dtype=np.uint8)
    for index in range(7, -1, -1):
        result[values <= ((low + index * step) & 65535)] = index
    return result


def _coordinates(extent: int, phase: int) -> np.ndarray:
    starts = 4 * np.arange((extent + 2) // 4)
    offsets = np.array([phase, 1 - phase, phase + 2, 3 - phase])
    indices = starts[:, None] + offsets
    return np.where(indices < extent, indices, extent - 2 + (indices & 1))


def _auxiliary(blocks: np.ndarray, table: np.ndarray, shift: int) -> np.ndarray:
    a, b, c, d, e, f, g, h = (
        blocks[:, :, y, x]
        for y, x in ((0, 1), (1, 0), (0, 3), (1, 2), (2, 1), (3, 0), (2, 3), (3, 2))
    )
    sequences = (
        ((f + 3 * b) >> 2, (a + 3 * e) >> 2, (h + 3 * d) >> 2, (c + 3 * g) >> 2),
        ((c + 3 * a) >> 2, (b + 3 * d) >> 2, (g + 3 * e) >> 2, (f + 3 * h) >> 2),
        (f, (b + 2 * e + h) >> 2, (a + 2 * d + g) >> 2, c),
        ((a + b) >> 1, (c + 3 * (d + e) + f) >> 3, (g + h) >> 1),
    )
    scores, monotonic = [], []
    for index, sequence in enumerate(sequences):
        differences = np.diff(np.stack(sequence, axis=-1), axis=-1)
        total = np.abs(differences).sum(axis=-1)
        scores.append((total >> 5) if index < 3 else ((total * 3) >> 6))
        monotonic.append(
            np.all(differences > 0, axis=-1) | np.all(differences < 0, axis=-1)
        )
    scores = np.stack(scores, axis=-1)
    selected = scores.argmax(axis=-1)[..., None]
    magnitude = np.take_along_axis(scores, selected, axis=-1)[..., 0]
    suppress = np.take_along_axis(np.stack(monotonic, axis=-1), selected, axis=-1)[
        ..., 0
    ]
    values = table[magnitude].astype(np.uint32)
    return np.where(suppress, values >> (shift & 31), values).astype(np.uint8)


def reduce_bayer(
    samples: np.ndarray,
    pattern: int,
    *,
    precision: int = 0,
    monochrome: bool = False,
    low: int = 10,
    high: int = 30,
    monotonic_shift: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce even Bayer axes by four, retaining CFA phase at partial blocks.

    Pattern 0/1/2/3 denotes RGGB/GRBG/GBRG/BGGR. Precision 0/1 shifts 14-bit
    samples right by one; precision 2/3 doubles 12-bit samples. RGB and auxiliary
    output axes are (input extent + 2) // 4. Only this initial substage is run.
    """
    if (
        not isinstance(samples, np.ndarray)
        or samples.ndim != 2
        or samples.dtype != np.dtype("<u2")
        or any(not 2 <= n <= 2048 or n % 2 for n in samples.shape)
    ):
        raise ValueError("Expected an even, bounded little-endian uint16 Bayer image.")
    if (
        type(pattern) is not int
        or not 0 <= pattern <= 3
        or type(precision) is not int
        or not 0 <= precision <= 3
        or type(monochrome) is not bool
        or type(monotonic_shift) is not int
        or not 0 <= monotonic_shift <= 255
    ):
        raise ValueError("Unsupported CFA, precision, monochrome or shift control.")
    table = classification_table(low, high)
    if np.any(samples > (16383 if precision < 2 else 4095)):
        raise ValueError("Bayer samples exceed the selected normalization range.")
    y = _coordinates(samples.shape[0], pattern // 2)
    x = _coordinates(samples.shape[1], pattern % 2)
    rgb = np.empty((len(y), len(x), 3), dtype="<u2")
    auxiliary = np.empty((len(y), len(x)), dtype=np.uint8)
    for start in range(0, len(y), 32):
        rows = y[start : start + 32]
        blocks = samples[rows[:, None, :, None], x[None, :, None, :]].astype(np.int64)
        blocks = (blocks >> 1) if precision < 2 else (blocks * 2)
        red = (blocks[:, :, ::2, ::2].sum(axis=(2, 3)) + 2) >> 2
        blue = (blocks[:, :, 1::2, 1::2].sum(axis=(2, 3)) + 2) >> 2
        green = (
            blocks[:, :, ::2, 1::2].sum(axis=(2, 3))
            + blocks[:, :, 1::2, ::2].sum(axis=(2, 3))
            + 4
        ) >> 3
        if monochrome:
            green = (blocks.sum(axis=(2, 3)) + 8) >> 4
            red = blue = np.zeros_like(green)
        rgb[start : start + 32] = np.minimum(
            np.stack((red, green, blue), axis=-1), 8191
        )
        auxiliary[start : start + 32] = _auxiliary(blocks, table, monotonic_shift)
    return rgb, auxiliary
