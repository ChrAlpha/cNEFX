"""Bounded native three-point, eight-phase Nlg interpolation."""

from __future__ import annotations

import numpy as np

_WEIGHTS = np.array(
    [
        [225, 286, 1],
        [167, 338, 7],
        [119, 370, 23],
        [81, 382, 49],
        [49, 382, 81],
        [23, 370, 119],
        [7, 338, 167],
        [1, 286, 225],
    ],
    dtype=np.uint32,
)
_WEIGHTS.setflags(write=False)


def interpolate(samples: np.ndarray) -> np.ndarray:
    """Expand at most 65536 little-endian uint16 triples into eight phases."""
    if (
        not isinstance(samples, np.ndarray)
        or samples.ndim == 0
        or samples.shape[-1] != 3
        or samples.dtype != np.dtype("<u2")
        or samples.size // 3 > 65536
    ):
        raise ValueError("Expected at most 65536 little-endian uint16 triples.")
    totals = samples.astype(np.uint32) @ _WEIGHTS.T
    return (totals >> 9).astype("<u2")
