"""Independent checks for the bounded mode5 radial vignette branch."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from renderer_vignette import radial_table, transform  # noqa: E402


def test_zero_coefficients_preserve_extremes_and_leave_input_untouched():
    image = np.array([[[0, 1, 65535], [12345, 32768, 65534]]], dtype="<u2")
    image.setflags(write=False)
    actual = transform(image, (2, 2), (0, 0), (0.0, 0.0), (0.0,) * 4)
    np.testing.assert_array_equal(actual, image)
    assert not np.shares_memory(actual, image)


def test_small_table_uses_float32_polynomial_and_truncation():
    actual = radial_table((4, 3), (0.0, 0.0), (0.0, 0.0, 0.0, 0.5))
    expected = []
    for index in range(11):
        radius = np.float32(index) * np.float32(np.float32(1) / np.float32(11))
        square = np.float32(radius * radius)
        value = np.float32(np.float32(0.5) * square)
        expected.append(int(float(value) * 4096))
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.dtype("<u2")


def test_expanded_table_extends_with_last_capped_coefficient():
    table = radial_table((4, 3), (3.0, -2.0), (0.0, 0.0, 0.0, 0.5))
    assert len(table) == 1 + int(2 * math.sqrt(7 * 7 + 5 * 5))
    assert np.all(table[11:] == 2048)
    assert table[10] < table[11]


def test_negative_polynomial_is_clipped_to_zero():
    table = radial_table((12, 10), (0.0, 0.0), (-1.0, -0.1, -0.2, -0.5))
    assert np.all(table == 0)


def test_pixel_gain_matches_scalar_positive_integer_pipeline():
    image = np.random.default_rng(91).integers(0, 65536, (3, 4, 3), dtype="<u2")
    table = radial_table((4, 3), (0.0, 0.0), (0.0, 0.0, 0.0, 0.5))
    actual = transform(image, (4, 3), (0, 0), (0.0, 0.0), (0.0, 0.0, 0.0, 0.5))
    expected = np.empty_like(image)
    for y in range(3):
        for x in range(4):
            index = min(
                int(math.sqrt((x - 1.5) ** 2 + (y - 1) ** 2) * 4), len(table) - 1
            )
            for channel in range(3):
                value = int(image[y, x, channel])
                addition = min(value * int(table[index]) // 4096, 65535)
                expected[y, x, channel] = min(value + addition * 3276 // 8192, 65535)
    np.testing.assert_array_equal(actual, expected)
    assert np.any(actual != image)


def test_saturation_noncontiguous_pixels_and_tile_coordinates():
    image = np.full((12, 16, 3), 60000, dtype="<u2")[:, ::-1]
    coefficients = (0.0, 0.0, 0.0, 0.8)
    full = transform(image, (16, 12), (0, 0), (0.0, 0.0), coefficients)
    tile = transform(image[4:8, 6:10], (16, 12), (6, 4), (0.0, 0.0), coefficients)
    np.testing.assert_array_equal(tile, full[4:8, 6:10])
    assert full.max() == 65535
    assert np.all(image == 60000)


@pytest.mark.parametrize(
    "size", [None, (0, 4), (4, 0), (16385, 2), (True, 4), (4.0, 3)]
)
def test_reject_invalid_image_size(size):
    with pytest.raises(ValueError):
        radial_table(size, (0.0, 0.0), (0.0,) * 4)


@pytest.mark.parametrize(
    "offset", [None, (0.0,), (float("nan"), 0), (0, float("inf")), (65, 0)]
)
def test_reject_invalid_center_offset(offset):
    with pytest.raises(ValueError):
        radial_table((4, 4), offset, (0.0,) * 4)


@pytest.mark.parametrize(
    "coefficients", [None, (0.0,), (0.0, 0.0, 0.0, float("nan")), (0.0, 0.0, 0.0, 17)]
)
def test_reject_invalid_polynomial(coefficients):
    with pytest.raises(ValueError):
        radial_table((4, 4), (0.0, 0.0), coefficients)


def test_reject_unverified_packed_exponent_branch():
    with pytest.raises(ValueError, match="unit"):
        radial_table((4, 4), (0.0, 0.0), (0.0, 0.0, 0.0, 2.0))


@pytest.mark.parametrize(
    "origin", [None, (0,), (True, 0), (-1, 0), (4, 0), (0, 4), (0.0, 0)]
)
def test_reject_invalid_tile_origin(origin):
    with pytest.raises(ValueError):
        transform(
            np.zeros((2, 2, 3), dtype="<u2"), (4, 4), origin, (0.0, 0.0), (0.0,) * 4
        )


@pytest.mark.parametrize(
    "pixels",
    [
        None,
        np.zeros((2, 2), dtype="<u2"),
        np.zeros((0, 2, 3), dtype="<u2"),
        np.zeros((2, 2, 4), dtype="<u2"),
        np.zeros((2, 2, 3), dtype=">u2"),
        np.zeros((2, 2, 3), dtype=np.float32),
        np.zeros((1025, 2, 3), dtype="<u2"),
    ],
)
def test_reject_invalid_pixels(pixels):
    with pytest.raises(ValueError):
        transform(pixels, (2048, 2048), (0, 0), (0.0, 0.0), (0.0,) * 4)
