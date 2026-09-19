"""RGB scale transitions checked with an independent block reference."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from nlg_rgb_pyramid import downsample, reconstruct


def reference_down(pixels):
    height, width, _ = pixels.shape
    out = np.empty(((height + 1) // 2, (width + 1) // 2, 3), dtype="<u2")
    for y in range(len(out)):
        y0 = 2 * y
        y1 = y0 + 1 if y0 + 1 < height else max(0, height - 2)
        for x in range(out.shape[1]):
            x0 = 2 * x
            x1 = x0 + 1 if x0 + 1 < width else max(0, width - 2)
            for c in range(3):
                out[y, x, c] = (
                    int(pixels[y0, x0, c])
                    + int(pixels[y0, x1, c])
                    + int(pixels[y1, x0, c])
                    + int(pixels[y1, x1, c])
                    + 2
                ) // 4
    return out


def reference_up(fine, coarse):
    averages = reference_down(fine)
    out = np.empty_like(fine)
    for y in range(len(fine)):
        for x in range(fine.shape[1]):
            for c, limit in enumerate((16383, 4095, 16383)):
                delta = int(coarse[y // 2, x // 2, c]) - int(
                    averages[y // 2, x // 2, c]
                )
                out[y, x, c] = min(max(int(fine[y, x, c]) + delta, 0), limit)
    return out


@pytest.mark.parametrize(
    "shape", [(1, 1), (1, 7), (9, 1), (3, 3), (4, 4), (7, 8), (8, 7), (65, 43)]
)
def test_block_arithmetic_and_reflected_odd_edges(shape):
    rng = np.random.default_rng(sum(shape))
    fine = rng.integers(0, 16384, (*shape, 3), dtype="<u2")
    coarse = rng.integers(
        0, 16384, ((shape[0] + 1) // 2, (shape[1] + 1) // 2, 3), dtype="<u2"
    )
    np.testing.assert_array_equal(downsample(fine), reference_down(fine))
    np.testing.assert_array_equal(reconstruct(fine, coarse), reference_up(fine, coarse))


def test_half_up_rounding_and_reflection_instead_of_clamping():
    image = np.zeros((3, 3, 3), dtype="<u2")
    image[0, 0] = 2
    image[1, 1] = 10
    np.testing.assert_array_equal(downsample(image)[:, :, 0], [[3, 3], [3, 3]])
    image = np.array([[[0, 0, 0], [1, 2, 3]]], dtype="<u2")
    np.testing.assert_array_equal(downsample(image), [[[1, 1, 2]]])


def test_reconstruction_uses_fine_residuals_and_per_channel_limits():
    fine = np.array(
        [[[0, 0, 0], [16383, 16383, 16383]], [[16383, 16383, 16383], [0, 0, 0]]],
        dtype="<u2",
    )
    actual = reconstruct(fine, np.full((1, 1, 3), 16383, dtype="<u2"))
    np.testing.assert_array_equal(actual[0], [[8191, 4095, 8191], [16383, 4095, 16383]])
    actual = reconstruct(fine, np.zeros((1, 1, 3), dtype="<u2"))
    np.testing.assert_array_equal(actual[0], [[0, 0, 0], [8191, 4095, 8191]])


def test_noncontiguous_readonly_and_no_mutation():
    fine = np.arange(7 * 16 * 3, dtype="<u2").reshape(7, 16, 3)[:, ::2]
    fine.flags.writeable = False
    coarse = downsample(fine)[:, ::-1]
    coarse.flags.writeable = False
    before = fine.copy()
    np.testing.assert_array_equal(reconstruct(fine, coarse), reference_up(fine, coarse))
    np.testing.assert_array_equal(fine, before)


@pytest.mark.parametrize(
    "bad",
    [
        None,
        [],
        np.zeros((3, 3), dtype="<u2"),
        np.zeros((3, 3, 4), dtype="<u2"),
        np.zeros((0, 3, 3), dtype="<u2"),
        np.zeros((2049, 1, 3), dtype="<u2"),
        np.zeros((3, 3, 3), dtype=">u2"),
        np.zeros((3, 3, 3), dtype=np.int16),
        np.full((3, 3, 3), 16384, dtype="<u2"),
    ],
)
def test_rejects_invalid_rgb_inputs(bad):
    with pytest.raises(ValueError):
        downsample(bad)
    with pytest.raises(ValueError):
        reconstruct(bad, np.zeros((2, 2, 3), dtype="<u2"))


@pytest.mark.parametrize(
    "coarse",
    [
        None,
        np.zeros((1, 1, 3), dtype="<u2"),
        np.zeros((2, 2, 3), dtype=">u2"),
        np.full((2, 2, 3), 16384, dtype="<u2"),
    ],
)
def test_rejects_invalid_parent(coarse):
    with pytest.raises(ValueError):
        reconstruct(np.zeros((3, 3, 3), dtype="<u2"), coarse)
