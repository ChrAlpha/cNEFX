"""Synthetic boundaries for the recovered Lab16-to-RGB16 renderer stage."""

import math
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from renderer_lab_rgb import transform  # noqa: E402


@pytest.fixture(scope="module")
def curves():
    return np.tile(np.arange(65536, dtype="<f4"), (3, 1))


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def scalar_reference(pixel, matrix, curves):
    # CIE Lab with D50, evaluated independently in scalar double precision.
    lightness = int(pixel[0]) * 100 / 65535
    a = (int(pixel[1]) - 32768) / 256
    b = (int(pixel[2]) - 32768) / 256
    fy = (lightness + 16) / 116

    def inverse(value):
        return value**3 if value > 6 / 29 else (value - 4 / 29) * 108 / 841

    xyz = [
        f32(0.9642 * inverse(fy + a / 500)),
        f32(inverse(fy)),
        f32(0.8249 * inverse(fy - b / 200)),
    ]
    result = []
    for row, curve in zip(matrix, curves, strict=True):
        products = [f32(value * float(gain)) for value, gain in zip(xyz, row)]
        value = f32(f32(products[0] + products[1]) + products[2])
        coordinate = min(65535, max(0, f32(value * 65535)))
        index = math.floor(coordinate)
        fraction = f32(coordinate - index)
        value = f32(
            f32(float(curve[index]) * f32(1 - fraction))
            + f32(float(curve[min(index + 1, 65535)]) * fraction)
        )
        result.append(round(value))
    return result


@pytest.mark.parametrize("width", [1, 2, 3])
def test_scalar_rows_match_independent_cie_reference(width, curves):
    pixels = np.array(
        [
            [0, 32768, 32768],
            [65535, 32768, 32768],
            [32768, 40000, 20000],
            [1000, 32768, 32768],
            [5242, 32768, 32768],
            [5243, 32768, 32768],
        ],
        dtype="<u2",
    )
    matrix = np.array([[1.2, -0.2, 0.1], [0.1, 0.9, 0], [0, 0.2, 1.1]], dtype="<f4")
    expected = [scalar_reference(pixel, matrix, curves) for pixel in pixels]
    np.testing.assert_array_equal(transform(pixels, matrix, curves, width), expected)


@pytest.mark.parametrize("width", [1, 4, 5])
def test_black_white_and_chromatic_endpoints(width, curves):
    colors = np.array(
        [[0, 32768, 32768], [65535, 32768, 32768], [32768, 65535, 0]],
        dtype="<u2",
    )
    pixels = np.repeat(colors, width, axis=0)
    expected = np.repeat(
        [[0, 0, 0], [63189, 65535, 54060], [35477, 12071, 65535]], width, axis=0
    )
    actual = transform(pixels, np.eye(3, dtype="<f4"), curves, width)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.dtype("<u2")


def test_negative_matrix_values_clip_before_curve_lookup(curves):
    pixels = np.tile(np.array([65535, 32768, 32768], dtype="<u2"), (5, 1))
    matrix = np.array([[-1, 0, 0], [0, 16, 0], [0, 0, 0]], dtype="<f4")
    expected = np.tile([0, 65535, 0], (5, 1))
    np.testing.assert_array_equal(transform(pixels, matrix, curves, 5), expected)


def test_nearest_even_quantization_in_groups_and_row_tails():
    curves = np.repeat(np.array([[0.5], [1.5], [2.5]], dtype="<f4"), 65536, axis=1)
    pixels = np.tile(np.array([32768, 32768, 32768], dtype="<u2"), (10, 1))
    actual = transform(pixels, np.eye(3, dtype="<f4"), curves, 5)
    np.testing.assert_array_equal(actual, np.tile([0, 2, 2], (10, 1)))


@pytest.mark.parametrize("width", range(1, 10))
def test_each_row_preserves_vector_and_scalar_precision(width, curves):
    pixels = np.tile(np.array([55921, 36651, 27229], dtype="<u2"), (width * 2, 1))
    expected_row = [[46662, 43683, 51149]] * (width // 4 * 4)
    expected_row += [[46661, 43683, 51149]] * (width % 4)
    np.testing.assert_array_equal(
        transform(pixels, np.eye(3, dtype="<f4"), curves, width), expected_row * 2
    )


def test_row_width_changes_precision_for_the_same_flat_buffer(curves):
    pixels = np.tile(np.array([55921, 36651, 27229], dtype="<u2"), (10, 1))
    matrix = np.eye(3, dtype="<f4")
    one_row = transform(pixels, matrix, curves, 10)
    two_rows = transform(pixels, matrix, curves, 5)
    np.testing.assert_array_equal(one_row[:, 0], [46662] * 8 + [46661] * 2)
    np.testing.assert_array_equal(two_rows[:, 0], ([46662] * 4 + [46661]) * 2)


def test_channel_curves_and_inputs_are_preserved(curves):
    curves = curves.copy()
    curves[1] *= np.float32(0.5)
    curves[2] *= np.float32(0.25)
    pixels = np.array([[0, 32768, 32768], [65535, 32768, 32768]], dtype="<u2")
    matrix = np.eye(3, dtype="<f4")
    originals = [value.copy() for value in (pixels, matrix, curves)]
    actual = transform(pixels, matrix, curves, 2)
    np.testing.assert_array_equal(actual, [[0, 0, 0], [63189, 32768, 13515]])
    for value, original in zip((pixels, matrix, curves), originals, strict=True):
        np.testing.assert_array_equal(value, original)
        assert not np.shares_memory(actual, value)


def test_nonlinear_curves_match_independent_scalar_interpolation(curves):
    curves = curves.copy()
    curves[0] = np.sqrt(curves[0] / np.float32(65535)) * np.float32(65535)
    curves[1] = curves[1] ** 2 / np.float32(65535)
    curves[2, :30000] *= np.float32(0.5)
    pixels = np.array(
        [[22111, 31234, 29000], [49877, 41000, 34567], [10500, 35555, 30222]],
        dtype="<u2",
    )
    matrix = np.array([[1.1, -0.1, 0.1], [0, 0.9, 0.1], [0.2, 0, 0.7]], dtype="<f4")
    expected = [scalar_reference(pixel, matrix, curves) for pixel in pixels]
    np.testing.assert_array_equal(transform(pixels, matrix, curves, 3), expected)


@pytest.mark.parametrize("width", [0, -1, 2, 4, 1.0, True, "3", np.int64(3)])
def test_invalid_width_is_rejected(width, curves):
    with pytest.raises(ValueError, match="width"):
        transform(np.zeros((3, 3), dtype="<u2"), np.eye(3, dtype="<f4"), curves, width)


@pytest.mark.parametrize(
    "pixels",
    [
        np.zeros((0, 3), dtype="<u2"),
        np.zeros((1, 1, 3), dtype="<u2"),
        np.zeros((1, 4), dtype="<u2"),
        np.zeros((1, 3), dtype=">u2"),
        np.zeros((1, 3), dtype="<i2"),
        np.zeros((1, 3), dtype="<f4"),
        [[0, 0, 0]],
    ],
)
def test_invalid_pixels_are_rejected(pixels, curves):
    with pytest.raises(ValueError):
        transform(pixels, np.eye(3, dtype="<f4"), curves, 1)


@pytest.mark.parametrize(
    "matrix",
    [
        np.eye(2, dtype="<f4"),
        np.zeros((9,), dtype="<f4"),
        np.eye(3, dtype="<f8"),
        np.eye(3, dtype=">f4"),
        np.full((3, 3), np.nan, dtype="<f4"),
        np.full((3, 3), np.inf, dtype="<f4"),
        np.full((3, 3), -np.inf, dtype="<f4"),
        np.full((3, 3), 16.01, dtype="<f4"),
        np.full((3, 3), -16.01, dtype="<f4"),
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    ],
)
def test_invalid_matrix_is_rejected(matrix, curves):
    with pytest.raises(ValueError, match="matrix"):
        transform(np.zeros((1, 3), dtype="<u2"), matrix, curves, 1)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, -1, 65536, 99])
def test_invalid_curve_values_are_rejected(value, curves):
    changed = curves.copy()
    changed[1, 123] = value
    with pytest.raises(ValueError, match="curves"):
        transform(np.zeros((1, 3), dtype="<u2"), np.eye(3, dtype="<f4"), changed, 1)


@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((3, 65535), "<f4"),
        ((65536, 3), "<f4"),
        ((3, 65536), "<f8"),
        ((3, 65536), ">f4"),
    ],
)
def test_invalid_curve_layout_is_rejected(shape, dtype):
    with pytest.raises(ValueError, match="curves"):
        transform(
            np.zeros((1, 3), dtype="<u2"),
            np.eye(3, dtype="<f4"),
            np.zeros(shape, dtype=dtype),
            1,
        )
