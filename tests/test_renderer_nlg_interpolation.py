"""Checks for the three-point, eight-phase Nlg interpolation leaf."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from renderer_nlg_interpolation import interpolate  # noqa: E402


def test_basis_samples_recover_each_phase_weight():
    samples = np.eye(3, dtype="<u2") * 512
    expected = [
        [225, 167, 119, 81, 49, 23, 7, 1],
        [286, 338, 370, 382, 382, 370, 338, 286],
        [1, 7, 23, 49, 81, 119, 167, 225],
    ]
    np.testing.assert_array_equal(interpolate(samples), expected)


@pytest.mark.parametrize("value", [0, 1, 32767, 32768, 65534, 65535])
def test_constant_extrema_are_preserved_without_overflow(value):
    actual = interpolate(np.full((3,), value, dtype="<u2"))
    np.testing.assert_array_equal(actual, np.full(8, value, dtype="<u2"))
    assert actual.dtype == np.dtype("<u2")


def test_fractional_results_are_truncated_without_rounding_bias():
    np.testing.assert_array_equal(interpolate(np.array([0, 1, 0], dtype="<u2")), 0)
    np.testing.assert_array_equal(
        interpolate(np.array([65535, 0, 0], dtype="<u2")),
        [28799, 21375, 15231, 10367, 6271, 2943, 895, 127],
    )


def test_separable_passes_keep_the_intermediate_truncation():
    grid = np.zeros((3, 3), dtype="<u2")
    grid[1, 1] = 2
    horizontal = interpolate(grid)
    actual = interpolate(horizontal.T)
    assert actual[3, 3] == 0
    assert (2 * 382 * 382) // (512 * 512) == 1


def test_leading_dimensions_noncontiguous_readonly_input_and_phase_symmetry():
    samples = np.arange(72, dtype="<u2").reshape(2, 4, 9)[:, ::-1, ::3]
    samples.setflags(write=False)
    before = samples.copy()
    actual = interpolate(samples)
    assert actual.shape == (2, 4, 8)
    assert not np.shares_memory(actual, samples)
    np.testing.assert_array_equal(samples, before)
    np.testing.assert_array_equal(interpolate(samples[..., ::-1]), actual[..., ::-1])


def test_maximum_batch_size_and_empty_leading_dimension():
    samples = np.full((256, 256, 3), 65535, dtype="<u2")
    actual = interpolate(samples)
    assert actual.shape == (256, 256, 8)
    assert np.all(actual == 65535)
    assert interpolate(np.empty((2, 0, 3), dtype="<u2")).shape == (2, 0, 8)


@pytest.mark.parametrize(
    "samples",
    [
        None,
        [0, 1, 2],
        np.array(0, dtype="<u2"),
        np.zeros((2,), dtype="<u2"),
        np.zeros((3, 2), dtype="<u2"),
        np.zeros((3,), dtype=">u2"),
        np.zeros((3,), dtype=np.int16),
        np.zeros((3,), dtype=np.uint32),
        np.zeros((3,), dtype=np.float32),
        np.zeros((3,), dtype=object),
        np.zeros((3,), dtype=bool),
        np.zeros((65537, 3), dtype="<u2"),
    ],
)
def test_reject_malformed_or_oversized_samples(samples):
    with pytest.raises(ValueError, match="little-endian uint16"):
        interpolate(samples)
