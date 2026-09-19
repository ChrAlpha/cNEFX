"""CML integer color-space boundaries and independent polar references."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from renderer_lch import lab_to_lch, lch_to_lab  # noqa: E402


def test_forward_neutral_axes_and_diagonals():
    pixels = np.array(
        [
            [0, 32768, 32768],
            [7, 32769, 32768],
            [8, 32768, 32769],
            [9, 32767, 32768],
            [10, 32768, 32767],
            [11, 65535, 65535],
            [12, 0, 0],
        ],
        dtype="<u2",
    )
    np.testing.assert_array_equal(
        lab_to_lch(pixels),
        [
            [0, 0, 0],
            [7, 1, 0],
            [8, 1, 16384],
            [9, 1, 32768],
            [10, 1, 49152],
            [11, 46340, 8192],
            [12, 46341, 40960],
        ],
    )


def test_inverse_axes_and_hue_dependent_gamut_limit():
    pixels = np.array(
        [
            [0, 0, 0],
            [1, 32768, 0],
            [2, 32768, 16384],
            [3, 32768, 32768],
            [4, 32768, 49152],
            [5, 65535, 8192],
            [6, 65535, 40960],
        ],
        dtype="<u2",
    )
    np.testing.assert_array_equal(
        lch_to_lab(pixels),
        [
            [0, 32768, 32768],
            [1, 65535, 32768],
            [2, 32768, 65535],
            [3, 0, 32768],
            [4, 32768, 0],
            [5, 65534, 65534],
            [6, 2, 2],
        ],
    )


def test_forward_against_continuous_polar_reference():
    pixels = np.random.default_rng(2919).integers(0, 65536, (16384, 3), dtype="<u2")
    a, b = pixels[:, 1:].astype(np.float64).T - 32768
    expected_chroma = np.hypot(a, b)
    expected_hue = np.arctan2(b, a) * (65536 / (2 * np.pi))
    actual = lab_to_lch(pixels)
    assert np.max(np.abs(actual[:, 1] - expected_chroma)) < 0.502
    hue_error = (actual[:, 2] - expected_hue + 32768) % 65536 - 32768
    assert np.max(np.abs(hue_error)) < 1.15
    np.testing.assert_array_equal(actual[:, 0], pixels[:, 0])


def test_inverse_all_hues_against_continuous_polar_reference():
    hue = np.arange(65536)
    pixels = np.column_stack((hue, np.full(65536, 10000), hue)).astype("<u2")
    radians = hue * (2 * np.pi / 65536)
    expected = np.column_stack((np.cos(radians), np.sin(radians))) * 10000 + 32768
    actual = lch_to_lab(pixels)
    assert np.max(np.abs(actual[:, 1:].astype(np.float64) - expected)) < 1.31
    np.testing.assert_array_equal(actual[:, 0], pixels[:, 0])


@pytest.mark.parametrize("function", [lab_to_lch, lch_to_lab])
def test_immutable_input_and_batch_tail_independence(function):
    pixels = np.random.default_rng(19).integers(0, 65536, (208, 3), dtype="<u2")
    pixels.flags.writeable = False
    before = pixels.copy()
    actual = function(pixels)
    assert actual.dtype == np.dtype("<u2") and not np.shares_memory(actual, pixels)
    for width in range(1, 10):
        batched = np.concatenate(
            [function(pixels[i : i + width]) for i in range(0, len(pixels), width)]
        )
        np.testing.assert_array_equal(actual, batched)
    np.testing.assert_array_equal(pixels, before)


@pytest.mark.parametrize("function", [lab_to_lch, lch_to_lab])
@pytest.mark.parametrize(
    "pixels",
    [
        None,
        [[0, 0, 0]],
        np.zeros(3, dtype="<u2"),
        np.zeros((0, 3), dtype="<u2"),
        np.zeros((2, 2), dtype="<u2"),
        np.zeros((2, 3), dtype="<f4"),
        np.zeros((2, 3), dtype=">u2"),
    ],
)
def test_reject_incompatible_pixels(function, pixels):
    with pytest.raises(ValueError, match="packed"):
        function(pixels)
