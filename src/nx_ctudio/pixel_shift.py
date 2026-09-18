"""Native reconstruction of ordered Z8 pixel-shift RAW buffers.

The numerical contract is backed by synthetic Nikon DLL oracle comparisons.
This module does not establish capture identity or decode compressed files.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ._sparse import Selection, repair

PHASE_X = (0, 1, 1, 0)
PHASE_Y = (0, 0, 1, 1)
CHANNELS = ((2, 1, 0, 1), (1, 2, 1, 0), (0, 1, 2, 1), (1, 0, 1, 2))


def _preprocess(
    frames: list[NDArray[np.uint16]], selection: Selection, iso: int
) -> list[NDArray[np.uint16]]:
    height, width = frames[0].shape
    coordinates = selection.coordinates(width, height) if iso <= 3200 else []
    if len(frames) in (4, 16):
        if not coordinates:
            return frames
        output = []
        for start in range(0, len(frames), 4):
            output.extend(
                plane.reshape(height, width)
                for plane in repair(
                    [p.reshape(-1) for p in frames[start : start + 4]],
                    width,
                    height,
                    selection,
                )
            )
        return output
    half = len(frames) // 2
    output = []
    for index in range(half):
        first, second = frames[index], frames[index + half]
        combined = first.copy()
        for row in range(2, height, 128):
            end = min(row + 128, height)
            combined[row:end] = (
                (first[row:end].astype(np.uint32) + second[row - 2 : end - 2]) >> 1
            ).astype(np.uint16)
        for y, x in coordinates:
            combined[y, x] = second[y - 2, x]
            combined[y + 2, x] = first[y + 2, x]
        output.append(combined)
    return output


def reconstruct(
    frames: list[NDArray[np.uint16]],
    crop: tuple[int, int, int, int],
    selector: bytes,
    iso: int,
) -> NDArray[np.uint16]:
    """Return cropped RGB without mutating any input; radial correction is off."""
    if len(frames) not in (4, 8, 16, 32):
        raise ValueError("Expected 4, 8, 16 or 32 ordered RAW frames.")
    shape = frames[0].shape
    if len(shape) != 2 or min(shape) < 8:
        raise ValueError("Expected two-dimensional RAW planes at least 8x8.")
    if any(p.shape != shape or p.dtype != np.uint16 for p in frames):
        raise ValueError("RAW frames must have identical uint16 geometry.")
    if any(int(p.max()) > 16383 for p in frames):
        raise ValueError("RAW samples exceed the 14-bit code range.")
    if iso <= 0:
        raise ValueError("Expected a positive ISO value.")
    left, top, crop_width, crop_height = crop
    height, width = shape
    if not (
        left >= 1
        and top >= 1
        and crop_width > 0
        and crop_height > 0
        and left + crop_width <= width
        and top + crop_height <= height
    ):
        raise ValueError("Crop is outside the recovered sampling domain.")
    selection = Selection.from_bytes(selector)
    reduced = _preprocess(frames, selection, iso)
    scale = 2 if len(reduced) == 16 else 1
    output = np.zeros((scale * crop_height, scale * crop_width, 3), dtype=np.uint16)
    # Sample directly into the final crop; avoid a full-resolution intermediate.
    for group in range(len(reduced) // 4):
        for frame in range(4):
            raw = reduced[group * 4 + frame]
            for corner, (py, px) in enumerate(((0, 0), (0, 1), (1, 1), (1, 0))):
                y0, x0 = (py - (top - 1)) % 2, (px - (left - 1)) % 2
                xs = np.arange(x0, crop_width, 2)
                channel = CHANNELS[frame][corner]
                for row in range(y0, crop_height, 128):
                    ys = np.arange(row, min(row + 128, crop_height), 2)
                    samples = raw[
                        np.ix_(ys + top - PHASE_Y[frame], xs + left - PHASE_X[frame])
                    ]
                    target = np.ix_(
                        scale * ys + PHASE_Y[group], scale * xs + PHASE_X[group]
                    )
                    plane = output[:, :, channel]
                    if channel == 1 and frame >= 2:
                        samples = (
                            (samples.astype(np.uint32) + plane[target]) >> 1
                        ).astype(np.uint16)
                    plane[target] = samples
    return output
