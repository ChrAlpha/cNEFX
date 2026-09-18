"""Synthetic checks for the bounded CML tetrahedral consumer."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from cml_interpolation import transform  # noqa: E402


def identity_luts():
    return np.tile(np.arange(65536, dtype="<u2"), (3, 1))


def test_all_six_tetrahedra_use_q14_weights_and_truncate():
    grid = np.zeros((32, 32, 32, 4), dtype="<u2")
    for r in range(2):
        for g in range(2):
            for b in range(2):
                grid[r, g, b, :3] = (4 * r + 2 * g + b) ** 2 * 1000
    grid[..., 3] = 65535
    pixels = np.array(
        [
            [1024, 512, 256],
            [1024, 256, 512],
            [512, 1024, 256],
            [256, 1024, 512],
            [512, 256, 1024],
            [256, 512, 1024],
        ],
        dtype="<u2",
    )
    # Q14 fractions are7936,3968,1984; weights8448,3968,1984,1984.
    expected = np.repeat(
        np.array([14167, 12835, 11261, 7992, 9203, 7265])[:, None], 3, axis=1
    )
    actual = transform(pixels, grid, identity_luts(), identity_luts())
    np.testing.assert_array_equal(actual, expected)


def test_full_range_constant_grid_preserves_endpoints_and_channel_order():
    grid = np.empty((32, 32, 32, 4), dtype="<u2")
    grid[...] = [0, 32768, 65535, 13579]
    ramp = np.repeat(np.arange(65536, dtype="<u2")[:, None], 3, axis=1)
    lut = identity_luts()
    actual = transform(ramp, grid, lut, lut)
    np.testing.assert_array_equal(actual, np.tile([0, 32768, 65535], (65536, 1)))


def test_last_grid_node_is_in_bounds_and_output_lut_is_applied():
    grid = np.zeros((32, 32, 32, 4), dtype="<u2")
    grid[31, 31, 31, :3] = [100, 200, 300]
    lut = identity_luts()
    actual = transform(np.array([[65535] * 3], dtype="<u2"), grid, lut, 65535 - lut)
    np.testing.assert_array_equal(actual, [[65435, 65335, 65235]])


@pytest.mark.parametrize("target", ["pixels", "grid", "input_lut", "output_lut"])
def test_unsupported_shapes_are_rejected(target):
    arguments = {
        "pixels": np.zeros((1, 3), dtype="<u2"),
        "grid": np.zeros((32, 32, 32, 4), dtype="<u2"),
        "input_lut": identity_luts(),
        "output_lut": identity_luts(),
    }
    arguments[target] = np.zeros((1,), dtype="<u2")
    with pytest.raises(ValueError):
        transform(**arguments)
