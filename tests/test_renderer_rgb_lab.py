"""Independent CIE and contract checks for the bounded RGB/Lab matrix branch."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from renderer_rgb_lab import transform  # noqa: E402


@pytest.fixture(scope="module")
def curves():
    return np.tile(np.arange(65536, dtype="<f4") / np.float32(65535), (3, 1))


def test_d50_black_white_and_saturation(curves):
    source = np.array([[0, 0, 0], [65535, 65535, 65535]] * 2, dtype="<u2")
    matrix = np.diag(np.array([0.9642, 1, 0.8249], dtype="<f4"))
    result = transform(source, matrix, curves, 4)
    np.testing.assert_array_equal(
        result, [[0, 32768, 32768], [65535, 32768, 32768]] * 2
    )
    saturated = transform(source, np.full((3, 3), 16, dtype="<f4"), curves, 1)
    assert saturated[1, 0] == 65535
    assert result.dtype == np.dtype("<u2")
    np.testing.assert_array_equal(source, [[0, 0, 0], [65535, 65535, 65535]] * 2)


@pytest.mark.parametrize("width", [1, 2, 3, 4, 5, 7, 8, 9, 50, 208])
def test_agrees_with_independent_cie_reference_to_quantization(width, curves):
    source = np.random.default_rng(413).integers(0, 65536, (width * 5, 3), dtype="<u2")
    matrix = np.diag(np.array([0.9642, 1, 0.8249], dtype="<f4"))
    actual = transform(source, matrix, curves, width)
    normalized = source.astype(np.float64) / 65535
    f = np.where(
        normalized > (6 / 29) ** 3,
        np.cbrt(normalized),
        normalized / (3 * (6 / 29) ** 2) + 4 / 29,
    )
    expected = np.column_stack(
        (
            (116 * f[:, 1] - 16) * 655.35,
            (f[:, 0] - f[:, 1]) * 128000 + 32768,
            (f[:, 1] - f[:, 2]) * 51200 + 32768,
        )
    )
    expected = np.rint(np.clip(expected, 0, 65535)).astype(np.int32)
    assert np.max(np.abs(actual.astype(np.int32) - expected)) <= 1


def test_input_curves_are_channel_specific_and_not_modified(curves):
    distinct = curves.copy()
    distinct[0] = 0
    distinct[1] = 1
    distinct[2] = np.float32(0.5)
    source = np.array([[0, 65535, 12345], [65535, 0, 54321]], dtype="<u2")
    before = distinct.copy()
    result = transform(source, np.eye(3, dtype="<f4"), distinct, 1)
    np.testing.assert_array_equal(result[0], result[1])
    np.testing.assert_array_equal(distinct, before)
    assert result[0, 0] == 65535 and result[0, 1] == 0


def test_matrix_is_row_major_and_results_are_fresh(curves):
    source = np.array([[12000, 24000, 36000]], dtype="<u2")
    permute = np.eye(3, dtype="<f4")[[2, 0, 1]]
    permuted = transform(source, permute, curves, 1)
    expected = transform(source[:, [2, 0, 1]], np.eye(3, dtype="<f4"), curves, 1)
    np.testing.assert_array_equal(permuted, expected)
    assert not np.shares_memory(source, permuted)


@pytest.mark.parametrize("width", [0, -1, True, 1.0, "1", None])
def test_reject_invalid_width(width, curves):
    with pytest.raises(ValueError):
        transform(np.zeros((1, 3), dtype="<u2"), np.eye(3, dtype="<f4"), curves, width)


@pytest.mark.parametrize(
    "source",
    [
        [],
        np.zeros((0, 3), dtype="<u2"),
        np.zeros((3,), dtype="<u2"),
        np.zeros((1, 4), dtype="<u2"),
        np.zeros((1, 3), dtype=">u2"),
        np.zeros((1, 3), dtype="<f4"),
    ],
)
def test_reject_invalid_pixels(source, curves):
    with pytest.raises(ValueError):
        transform(source, np.eye(3, dtype="<f4"), curves, 1)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, 17])
def test_reject_invalid_matrix(value, curves):
    matrix = np.eye(3, dtype="<f4")
    matrix[0, 0] = value
    with pytest.raises(ValueError):
        transform(np.zeros((1, 3), dtype="<u2"), matrix, curves, 1)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, 2, 0])
def test_reject_invalid_curve_values_or_order(value, curves):
    bad = curves.copy()
    bad[1, 500] = value
    with pytest.raises(ValueError):
        transform(np.zeros((1, 3), dtype="<u2"), np.eye(3, dtype="<f4"), bad, 1)


def test_reject_partial_rows_and_parameter_shapes(curves):
    source = np.zeros((5, 3), dtype="<u2")
    matrix = np.eye(3, dtype="<f4")
    for m, c, width in (
        (matrix, curves, 2),
        (matrix[:2], curves, 1),
        (matrix.astype("<f8"), curves, 1),
        (matrix, curves[:, :-1], 1),
        (matrix, curves.astype("<f8"), 1),
    ):
        with pytest.raises(ValueError):
            transform(source, m, c, width)
