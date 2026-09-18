"""Numerical boundary checks for the recovered selector-0 conversion."""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / "tools/kaiji_color.py"
SPEC = importlib.util.spec_from_file_location("nxc_kaiji_color", PATH)
assert SPEC and SPEC.loader
color = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(color)

OUTPUT_SPEC = importlib.util.spec_from_file_location(
    "nxc_kaiji_output", PATH.with_name("kaiji_output.py")
)
assert OUTPUT_SPEC and OUTPUT_SPEC.loader
output = importlib.util.module_from_spec(OUTPUT_SPEC)
OUTPUT_SPEC.loader.exec_module(output)


def test_encoding_endpoints_and_neutral() -> None:
    assert color.rgb_to_lab((0, 0, 0)) == (0, 32768, 32768)
    assert color.rgb_to_lab((1, 1, 1)) == (65280, 32768, 32768)
    assert color.rgb_to_lab((0.5, 0.5, 0.5)) == (34852, 32768, 32767)
    assert color.rgb_to_lab((-1, -1, -1)) == color.rgb_to_lab((0, 0, 0))
    assert color.rgb_to_lab((2, 2, 2)) == color.rgb_to_lab((1, 1, 1))


def test_near_black_primary_and_quantization_boundaries() -> None:
    cases = [(0, 0, 0), (1, 1, 1), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    for value in (0.04045, 0.5, 0.99999):
        for c in (math.nextafter(value, 0), value, math.nextafter(value, 1)):
            cases.extend(((c, c, c), (c, 0, 1), (0, 1, c)))
    for rgb in cases:
        codes = [int(c * 65280 + 0.5) for c in rgb]
        low, high = color.quantized_rgb_lab_bounds(codes)
        actual = color.rgb_to_lab(rgb)
        assert all(a <= b <= c for a, b, c in zip(low, actual, high, strict=True))


@pytest.mark.parametrize("rgb", [(0, 1), (0, 0, float("nan")), (0, 0, float("inf"))])
def test_reject_nonfinite_or_wrong_shape(rgb) -> None:
    with pytest.raises(ValueError):
        color.rgb_to_lab(rgb)


@pytest.mark.parametrize("rgb", [(0, 1), (0, -1, 0), (0, 65281, 0), (0, True, 0)])
def test_reject_invalid_codes(rgb) -> None:
    with pytest.raises(ValueError):
        color.quantized_rgb_lab_bounds(rgb)


def test_vectorized_conversion_matches_scalar_reference() -> None:
    values = np.random.default_rng(20260919).uniform(0, 1, (2048, 3))
    expected = np.array([color.rgb_to_lab(row) for row in values], dtype=np.uint16)
    assert np.array_equal(output.rgb_to_icc_lab(values), expected)


@pytest.mark.parametrize("selector", [-1, 2, 3, True, 1.0, None])
def test_output_selector_rejects_unrecovered_paths(selector):
    with pytest.raises(ValueError, match="selector"):
        output.rgb_to_icc_lab([[0, 0, 0]], selector=selector)


def test_selector1_neutral_and_linear_floor():
    values = np.array([[0, 0, 0], [1, 1, 1], [0.01, 0.01, 0.01]])
    expected = np.array([[0, 32768, 32768], [65280, 32768, 32768], [184, 32768, 32768]])
    np.testing.assert_array_equal(output.rgb_to_icc_lab(values, selector=1), expected)


@pytest.mark.parametrize(
    "values",
    [
        [],
        [0, 1],
        [[0, 1]],
        [0, 0, float("nan")],
        [0, 0, float("inf")],
        [0, 0, -1],
        [0, 0, 2],
        [True, False, True],
        ["0", "0", "0"],
    ],
)
def test_vectorized_conversion_rejects_invalid_input(values) -> None:
    with pytest.raises(ValueError):
        output.rgb_to_icc_lab(values)
