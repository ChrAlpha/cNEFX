"""Numerical boundary checks for the recovered selector-0 conversion."""

import importlib.util
import math
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "tools/kaiji_color.py"
SPEC = importlib.util.spec_from_file_location("nxc_kaiji_color", PATH)
assert SPEC and SPEC.loader
color = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(color)


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
