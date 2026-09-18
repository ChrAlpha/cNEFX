"""Matrix arithmetic and acceptance bounds for native CML composition."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from cml_grid import build_grid, inverse_matrix  # noqa: E402


def test_inverse_matrix_preserves_column_order():
    matrix = np.array([1, 0, 0, 2, 1, 0, 3, 4, 1], dtype=np.float64)
    np.testing.assert_array_equal(inverse_matrix(matrix), [1, 0, 0, -2, 1, 0, 5, -4, 1])
    np.testing.assert_array_equal(
        inverse_matrix(np.diag([2.0, 4.0, 8.0]).reshape(-1)),
        np.diag([0.5, 0.25, 0.125]).reshape(-1),
    )


@pytest.mark.parametrize("diagonal", [0, 1 / 65537, -1 / 65537])
def test_inverse_rejects_vendor_singularity_threshold(diagonal):
    with pytest.raises(ValueError, match="singular"):
        inverse_matrix(np.diag([diagonal, 1, 1]).reshape(-1))


def test_inverse_accepts_exact_singularity_boundary():
    np.testing.assert_array_equal(
        inverse_matrix(np.diag([1 / 65536, 1, 1]).reshape(-1)),
        np.diag([65536, 1, 1]).reshape(-1),
    )


@pytest.mark.parametrize("matrix", [np.ones(8), np.full(9, np.inf), np.full(9, np.nan)])
def test_invalid_matrices_are_rejected(matrix):
    with pytest.raises(ValueError, match="finite"):
        inverse_matrix(matrix)


def test_grid_rejects_non_icc_input():
    with pytest.raises(ValueError, match="ICC"):
        build_grid(bytes(132), bytes(132))
