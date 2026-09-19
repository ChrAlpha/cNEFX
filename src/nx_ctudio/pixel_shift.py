"""Native reconstruction of ordered Z8 pixel-shift RAW buffers.

The numerical contract is backed by synthetic Nikon DLL oracle comparisons.
This module does not establish capture identity or decode compressed files.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from ._sparse import Selection, repair

PHASE_X = (0, 1, 1, 0)
PHASE_Y = (0, 0, 1, 1)
CHANNELS = ((2, 1, 0, 1), (1, 2, 1, 0), (0, 1, 2, 1), (1, 0, 1, 2))


def _average_pair(
    first: NDArray[np.uint16],
    second: NDArray[np.uint16],
    coordinates: list[tuple[int, int]],
) -> NDArray[np.uint16]:
    height = first.shape[0]
    combined = first.copy()
    for row in range(2, height, 128):
        end = min(row + 128, height)
        combined[row:end] = (
            (first[row:end].astype(np.uint32) + second[row - 2 : end - 2]) >> 1
        ).astype(np.uint16)
    for y, x in coordinates:
        combined[y, x] = second[y - 2, x]
        combined[y + 2, x] = first[y + 2, x]
    return combined


def reconstruct(
    frames: list[NDArray[np.uint16]],
    crop: tuple[int, int, int, int],
    selector: bytes,
    iso: int,
) -> NDArray[np.uint16]:
    """Return cropped RGB without mutating any input; radial correction is off."""
    return reconstruct_from_reader(len(frames), frames.__getitem__, crop, selector, iso)


def reconstruct_from_reader(
    count: int,
    read_frame: Callable[[int], NDArray[np.uint16]],
    crop: tuple[int, int, int, int],
    selector: bytes,
    iso: int,
) -> NDArray[np.uint16]:
    """Read each zero-based capture index once, retaining at most four source planes.

    Reads follow reconstruction groups, not capture order in paired modes.
    Sparse repair retains ordered dependencies within each four-frame group.
    The final RGB array is still held in memory. Inputs are never mutated.
    """
    if count not in (4, 8, 16, 32):
        raise ValueError("Expected 4, 8, 16 or 32 ordered RAW frames.")
    if iso <= 0:
        raise ValueError("Expected a positive ISO value.")
    selection = Selection.from_bytes(selector)
    initial = [read_frame(0)]
    shape = initial[0].shape
    if len(shape) != 2 or min(shape) < 8 or any(axis % 2 for axis in shape):
        raise ValueError("Expected even two-dimensional RAW planes at least 8x8.")

    def checked(index: int) -> NDArray[np.uint16]:
        samples = initial.pop() if index == 0 else read_frame(index)
        if samples.shape != shape or samples.dtype != np.uint16:
            raise ValueError("RAW frames must have identical uint16 geometry.")
        if int(samples.max()) > 16383:
            raise ValueError("RAW samples exceed the 14-bit code range.")
        return samples

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
    coordinates = selection.coordinates(width, height) if iso <= 3200 else []
    paired = count in (8, 32)
    reduced_count = count // 2 if paired else count
    scale = 2 if reduced_count == 16 else 1
    output = np.zeros((scale * crop_height, scale * crop_width, 3), dtype=np.uint16)
    for start in range(0, reduced_count, 4):
        if paired:
            reduced = [
                _average_pair(checked(i), checked(i + count // 2), coordinates)
                for i in range(start, start + 4)
            ]
        else:
            reduced = [checked(i) for i in range(start, start + 4)]
            if coordinates:
                reduced = [
                    plane.reshape(height, width)
                    for plane in repair(
                        [p.reshape(-1) for p in reduced], width, height, selection
                    )
                ]
        _sample_group(output, reduced, crop, scale, start // 4)
        del reduced
    return output


def _sample_group(
    output: NDArray[np.uint16],
    frames: list[NDArray[np.uint16]],
    crop: tuple[int, int, int, int],
    scale: int,
    group: int,
) -> None:
    left, top, crop_width, crop_height = crop
    # Sample directly into the final crop; avoid a full-resolution intermediate.
    for frame, raw in enumerate(frames):
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
                    samples = ((samples.astype(np.uint32) + plane[target]) >> 1).astype(
                        np.uint16
                    )
                plane[target] = samples
