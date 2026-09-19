"""Bounded native scalar Nlg adaptive spatial filtering in the integer domain."""

from __future__ import annotations

import struct

import numpy as np
from nlg_filter_domain import zero_strength


def _blend_table(params: bytes) -> np.ndarray:
    low, high = struct.unpack_from("<2H", params, 0x1E)
    step = ((high - low) >> 3) & 511
    result = np.full(4096, 8, dtype=np.uint8)
    index = np.arange(4096)
    for level in range(7, -1, -1):
        result[index <= ((low + level * step) & 65535)] = level
    return result


def _mirror_domain(samples: np.ndarray) -> np.ndarray:
    axes = []
    for extent in samples.shape[:2]:
        coords = np.arange(-2, extent + 2)
        period = max(1, 2 * (extent - 1))
        modulo = coords % period
        axes.append(np.minimum(modulo, period - modulo))
    return samples[axes[0][:, None], axes[1]]


def _rounded(value, bits):
    return (value - (value < 0) + (1 << (bits - 1))) >> bits


def _truncated(value, bits):
    return np.where(value < 0, -(np.abs(value) >> bits), value >> bits)


def _round_clip(values, bits, limits):
    limits = np.array(limits)
    return np.clip(_rounded(values, bits), -limits, limits - 1)


def _activities(window):
    shifted = window >> [2, 3, 3]
    deltas = (
        np.abs(np.diff(shifted, axis=2)),
        np.abs(np.diff(shifted, axis=3)),
        np.abs(shifted[:, :, :-1, :-1] - shifted[:, :, 1:, 1:]),
        np.abs(shifted[:, :, :-1, 1:] - shifted[:, :, 1:, :-1]),
    )
    ys, cs = [], []
    for direction, delta in enumerate(deltas):
        values = (delta.sum(axis=2) >> 2).sum(axis=2)
        y, c = values[..., 0], values[..., 1:].sum(axis=-1)
        if direction > 1:
            y, c = (y * 35) >> 5, (c * 35) >> 5
        ys.append(y)
        cs.append(c)
    ys, cs = np.stack(ys, axis=-1), np.stack(cs, axis=-1)
    scores = (4 * ys + cs) & 65535
    direction = scores.argmin(axis=-1)
    activity = []
    for index in (direction, direction ^ 1):
        y = np.take_along_axis(ys, index[..., None], axis=-1)[..., 0]
        c = np.take_along_axis(cs, index[..., None], axis=-1)[..., 0]
        activity.extend(
            (
                np.minimum(1023, ((y * 4) & 65535) >> 3),
                np.minimum(1023, ((y + 4 * c) & 65535) >> 3),
            )
        )
    blend_index = np.clip((scores.max(axis=-1) - scores.min(axis=-1)) >> 3, 0, 4095)
    return direction, activity, blend_index


def _impulse(window, words, params):
    center = window[:, :, 2, 2]
    neighbors = np.stack(
        [
            window[:, :, y, x]
            for y in range(1, 4)
            for x in range(1, 4)
            if (y, x) != (2, 2)
        ],
        axis=-2,
    )
    low, high = neighbors.min(axis=-2), neighbors.max(axis=-2)
    threshold = 2 * np.array([words[11], words[12], words[12]])
    mode = np.array([params[0x14], params[0x15], params[0x15]]) != 0
    delta = np.where(
        center < low, center - low, np.where(center >= high, center - high, 0)
    )
    selected = np.where(center < low, delta < -threshold, delta >= threshold)
    return np.where(selected == mode, delta, 0)


def _response(center, neighbor, thresholds, shifts, large=False):
    delta = center - neighbor
    magnitude = np.abs(delta)
    damping = magnitude[..., 0] - 2 * thresholds[0]
    damping = np.clip(damping + (damping >> shifts[0]), 0, 8190)
    y = np.clip(magnitude[..., 0] - damping, 0, 510 if large else 1022)
    damping = (
        (magnitude[..., 0] << shifts[2])
        + magnitude[..., 1:].sum(axis=-1)
        - 2 * thresholds[1]
    )
    damping = np.clip(damping + (damping >> shifts[1]), 0, 12285)
    chroma = np.clip(
        magnitude[..., 1:] - damping[..., None], 0, 4094 if large else 8190
    )
    values = np.concatenate((y[..., None], chroma), axis=-1)
    return np.where(delta < 0, -values, values)


def _square_response(window, center, thresholds, shifts, large=False):
    radius, bits = (2, 2) if large else (1, 1)
    limits = [1024, 8192, 8192] if large else [2048, 16384, 16384]
    sums = np.zeros_like(center)
    for x in range(2 - radius, 3 + radius):
        column = np.zeros_like(center)
        for y in range(2 - radius, 3 + radius):
            if (y, x) != (2, 2):
                column += _response(
                    center, window[:, :, y, x], thresholds, shifts, large
                )
        sums += _round_clip(column, bits, limits)
    sums = _round_clip(sums, bits, limits)
    return (
        _round_clip(sums * 5, 3, [512, 4096, 4096])
        if large
        else _round_clip(sums, 1, [1024, 8192, 8192])
    )


def _directional_response(window, center, thresholds, shifts, direction):
    dy, dx = np.array([1, 0, 1, 1])[direction], np.array([0, 1, 1, -1])[direction]
    flat = window.reshape(*window.shape[:2], 25, 3)
    sums = np.zeros_like(center)
    for offset in (-2, -1, 1, 2):
        position = (2 + dy * offset) * 5 + 2 + dx * offset
        neighbor = np.take_along_axis(flat, position[..., None, None], axis=2)[:, :, 0]
        sums += _response(center, neighbor, thresholds, shifts)
    sums = _round_clip(sums, 2, [1024, 8192, 8192])
    return _round_clip(sums * 7, 3, [1024, 8192, 8192])


def _filter_windows(window, params, table):
    words = struct.unpack_from("<19H", params)
    direction, activity, blend_index = _activities(window)
    small = [
        np.minimum(511, words[0] + np.maximum(0, words[2] - activity[0]) * params[8]),
        np.minimum(4095, words[1] + np.maximum(0, words[3] - activity[1]) * params[9]),
    ]
    large = [
        np.minimum(
            255, (words[5] + np.maximum(0, words[7] - activity[2]) * params[0x12]) >> 1
        ),
        np.minimum(
            2047, (words[6] + np.maximum(0, words[8] - activity[3]) * params[0x13]) >> 1
        ),
    ]
    correction = _impulse(window, words, params)
    original = window[:, :, 2, 2]
    center = original - correction
    weight = table[blend_index][..., None].astype(np.int64)
    shifts = [params[o] & 31 for o in (0x1A, 0x1B, 0x1C)]
    isotropic = _square_response(window, center, small, shifts)
    directional = _directional_response(window, center, small, shifts, direction)
    blended = _truncated(isotropic * (8 - weight) + directional * weight, 3)
    second_center = np.clip(center - blended, [0, -6143, -6143], [8190, 6142, 6142])
    large_shifts = [params[o] & 31 for o in (0x22, 0x23, 0x24)]
    second = _square_response(window, second_center, large, large_shifts, large=True)
    correction = correction + blended + second
    final = original - _truncated(
        correction * [params[0x25], params[0x26], params[0x26]], 5
    )
    return np.clip(final, [0, -6143, -6143], [8190, 6142, 6142])


def filter_domain(samples: np.ndarray, params: bytes) -> np.ndarray:
    """Apply the recovered 5x5 scalar branch with captured filter parameters.

    Nonzero strengths require Y in [0,8190], Cb/Cr in [-6143,6142], strengths
    <=31, impulse thresholds <=4095 and chroma weighting shifts <=8.
    Zero strengths retain the broader unsigned/signed 16-bit domain contract.
    The 39-byte parameter block starts at offset 0x84 in a level record.
    """
    if (
        not isinstance(samples, np.ndarray)
        or samples.ndim != 3
        or samples.shape[2] != 3
        or samples.dtype != np.dtype("<i4")
        or any(not 1 <= n <= 2048 for n in samples.shape[:2])
    ):
        raise ValueError("Expected bounded little-endian int32 domain samples.")
    if type(params) is not bytes or len(params) != 39:
        raise ValueError("Expected a 39-byte filter block.")
    thresholds = struct.unpack_from("<2H", params, 0x16)
    if (
        max(params[0x25:0x27]) > 31
        or max(thresholds) > 4095
        or any(params[o] > 8 for o in (0x1C, 0x24))
    ):
        raise ValueError(
            "Unsupported filter strengths, thresholds or weighting shifts."
        )
    if params[0x25:0x27] == b"\x00\x00":
        return zero_strength(samples)
    if np.any(samples < [0, -6143, -6143]) or np.any(samples > [8190, 6142, 6142]):
        raise ValueError("Nonzero strengths require the bounded Nlg domain.")
    samples = samples.astype(np.int64)
    padded, table = _mirror_domain(samples), _blend_table(params)
    windows = np.lib.stride_tricks.sliding_window_view(padded, (5, 5), axis=(0, 1))
    windows = windows.transpose(0, 1, 3, 4, 2)
    result = np.empty(samples.shape, dtype="<i4")
    for start in range(0, samples.shape[0], 16):
        result[start : start + 16] = _filter_windows(
            windows[start : start + 16], params, table
        )
    return result
