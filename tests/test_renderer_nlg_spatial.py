"""Checks for the observed scale32 Nlg spatial-grid sampling."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from renderer_nlg import sample_spatial


def reflect(index, extent):
    if extent == 1:
        return 0
    while index < 0 or index >= extent:
        index = -index if index < 0 else 2 * extent - index - 2
    return index


@pytest.mark.parametrize("shape", [(1, 1), (1, 4), (4, 1), (8, 8), (688, 1032)])
@pytest.mark.parametrize("origin,width", [((0, 0), 1), ((5, 7), 17), ((208, 171), 50)])
def test_matches_reflected_integer_coordinates(shape, origin, width):
    grid = np.arange(np.prod(shape), dtype=np.uint32).reshape(shape) % 4096
    grid = grid.astype("<u2")
    left, y = origin
    count = ((left % 8 + width + 7) // 8) + 2
    expected = [
        [
            grid[
                reflect(y // 8 * 32 - 16 + row * 32, shape[0]),
                reflect(left // 8 * 32 - 16 + column * 32, shape[1]),
            ]
            for column in range(count)
        ]
        for row in range(3)
    ]
    np.testing.assert_array_equal(sample_spatial(grid, origin, width), expected)


def test_uses_neighboring_grid_cells_across_tile_edges():
    grid = np.arange(688 * 1032, dtype=np.uint32).reshape(688, 1032) % 4096
    grid = grid.astype("<u2")
    whole = sample_spatial(grid, (0, 85), 258)
    first = sample_spatial(grid, (0, 85), 208)
    last = sample_spatial(grid, (208, 85), 50)
    np.testing.assert_array_equal(first, whole[:, :28])
    np.testing.assert_array_equal(last, whole[:, 26:])


def test_readonly_noncontiguous_grid_and_bounded_allocation():
    grid = np.broadcast_to(np.array([1234], dtype="<u2"), (2048, 2048))
    result = sample_spatial(grid, (15000, 16383), 1024)
    assert result.shape == (3, 130)
    assert result.nbytes == 780
    assert np.all(result == 1234)
    assert not np.shares_memory(result, grid)


@pytest.mark.parametrize(
    "field,value",
    [
        ("grid", None),
        ("grid", np.zeros((0, 3), dtype="<u2")),
        ("grid", np.zeros((2049, 1), dtype="<u2")),
        ("grid", np.zeros((1, 1, 1), dtype="<u2")),
        ("grid", np.zeros((3, 3), dtype=np.float32)),
        ("grid", np.full((3, 3), 4096, dtype="<u2")),
        ("origin", [0, 0]),
        ("origin", (False, 0)),
        ("origin", (-1, 0)),
        ("origin", (0, 16384)),
        ("origin", (16384, 0)),
        ("width", True),
        ("width", 0),
        ("width", 1025),
    ],
)
def test_rejects_unsupported_grid_geometry(field, value):
    args = dict(grid=np.zeros((3, 3), dtype="<u2"), origin=(0, 0), width=1)
    args[field] = value
    with pytest.raises(ValueError):
        sample_spatial(**args)


def test_rejects_row_beyond_image_bound():
    with pytest.raises(ValueError):
        sample_spatial(np.zeros((3, 3), dtype="<u2"), (16380, 0), 5)
