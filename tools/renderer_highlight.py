"""Native bounded green-reference highlight recovery with explicit six-pixel halo.

Only the observed Bayer 1/0, even-origin RGB16 branch is covered. Halo pixels
are supplied by the caller; source decoding and image-border extension are not
inferred here.
"""

from __future__ import annotations

import numpy as np


def transfer_tables() -> tuple[np.ndarray, np.ndarray]:
    """Return the square-root encoder and rounded-square decoder, in that order."""
    values = np.arange(65536, dtype=np.int64)
    encode = np.minimum(np.floor(np.sqrt(values) * 256 + 0.5), 65535).astype("<u2")
    decode = ((values * values + 32768) >> 16).astype("<u2")
    return encode, decode


def _validate(padded: np.ndarray) -> None:
    if (
        not isinstance(padded, np.ndarray)
        or padded.ndim != 3
        or padded.shape[2] != 3
        or padded.dtype != np.dtype("<u2")
        or any(not 14 <= size <= 1036 or size % 2 for size in padded.shape[:2])
    ):
        raise ValueError("Expected even RGB16 tile dimensions with a six-pixel halo.")


def _smooth_excess(encoded: np.ndarray, channel: int, phase: int, threshold: int):
    sparse = np.zeros(encoded.shape[:2], dtype=np.int64)
    sparse[phase::2, phase::2] = np.maximum(
        encoded[phase::2, phase::2, channel] - threshold, 0
    )
    horizontal = np.zeros_like(sparse)
    horizontal[:, 4:-4] = (
        sparse[:, :-8]
        + 4 * sparse[:, 2:-6]
        + 6 * sparse[:, 4:-4]
        + 4 * sparse[:, 6:-2]
        + sparse[:, 8:]
    ) >> 4
    vertical = np.zeros_like(sparse)
    vertical[4:-4] = (
        horizontal[:-8]
        + 4 * horizontal[2:-6]
        + 6 * horizontal[4:-4]
        + 4 * horizontal[6:-2]
        + horizontal[8:]
    ) >> 4
    center = vertical[6:-6, 6:-6]
    x = (vertical[6:-6, 5:-7] + vertical[6:-6, 7:-5]) >> 1
    y = (vertical[5:-7, 6:-6] + vertical[7:-5, 6:-6]) >> 1
    diagonal = (
        vertical[5:-7, 5:-7]
        + vertical[5:-7, 7:-5]
        + vertical[7:-5, 5:-7]
        + vertical[7:-5, 7:-5]
    ) >> 2
    rows, columns = np.indices(center.shape)
    return np.where(
        rows % 2 == phase,
        np.where(columns % 2 == phase, center, x),
        np.where(columns % 2 == phase, y, diagonal),
    )


def recover(padded: np.ndarray) -> np.ndarray:
    """Recover the interior before the caller's eligibility selection and gain.

    The two color planes use separable (1,4,6,4,1)/16 integer filtering on Bayer
    sites, then bilinear interpolation. Each filtering pass truncates separately.
    """
    _validate(padded)
    encode, decode = transfer_tables()
    encoded = encode[padded].astype(np.int64)
    threshold = int(encode[15974])
    center = encoded[6:-6, 6:-6]
    corrected = center.copy()
    for channel, phase in ((0, 0), (2, 1)):
        excess = _smooth_excess(encoded, channel, phase, threshold)
        changed = np.minimum(
            np.maximum(center[:, :, channel] - excess, center[:, :, 1]), threshold
        )
        corrected[:, :, channel] = np.where(
            center[:, :, channel] > center[:, :, 1], changed, center[:, :, channel]
        )
    return decode[corrected]


def transform(padded: np.ndarray) -> np.ndarray:
    """Apply observed recovery selection and fixed Q10 gains [4096,4096,4096]."""
    restored = recover(padded)
    source = padded[6:-6, 6:-6]
    eligible = np.all(source <= 16384, axis=2)
    selected = np.where(eligible[:, :, None], restored, source).astype(np.int64)
    return np.minimum(selected * 4, 65535).astype("<u2")
