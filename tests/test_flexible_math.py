"""Integer edge cases for the recovered Flexible Color research helpers."""

import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "tools/flexible_math.py"
SPEC = importlib.util.spec_from_file_location("nxc_flexible_math", PATH)
assert SPEC and SPEC.loader
math = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(math)


@pytest.mark.parametrize(
    ("numerator", "denominator", "expected"),
    [(1, 2, 1), (-1, 2, -1), (4, 3, 1), (-4, 3, -1), (5, 3, 2), (-5, 3, -2)],
)
def test_division_signed_ties(numerator: int, denominator: int, expected: int) -> None:
    assert math.round_divide32(numerator, denominator) == expected


def test_division_preserves_machine_addition_wrap() -> None:
    assert math.round_divide32(2**31 - 1, 2) == -(2**30)
    assert math.round_divide32(-(2**31), 2) == 2**30 - 1
    assert math.round_divide32(-(2**31), 1) == -(2**31)


def test_interpolation_rounds_delta_in_both_directions() -> None:
    assert math.interpolate([0, 1], 8, interval_shift=4) == 1
    assert math.interpolate([1, 0], 8, interval_shift=4) == 0
    assert math.interpolate([-2, 1], 8, interval_shift=4, signed=True) == 0
    assert math.interpolate([1, -2], 8, interval_shift=4, signed=True) == -1


@pytest.mark.parametrize("shift", [4, 6, 7])
def test_segment_boundaries_and_last_valid_index(shift: int) -> None:
    width = 1 << shift
    assert math.interpolate([10, 20, 30], width, interval_shift=shift) == 20
    assert (
        math.interpolate([0, width, width * 2], width * 2 - 1, interval_shift=shift)
        == width * 2 - 1
    )
    with pytest.raises(ValueError, match="index"):
        math.interpolate([10, 20, 30], width * 2, interval_shift=shift)


def test_signed_and_unsigned_extrema_do_not_clip_to_each_other() -> None:
    assert math.interpolate([65535, 0], 0, interval_shift=4) == 65535
    assert math.interpolate([-32768, 32767], 0, interval_shift=4, signed=True) == -32768
    assert math.interpolate([-32768, 32767], 8, interval_shift=4, signed=True) == 0
    assert math.interpolate([32767, -32768], 8, interval_shift=4, signed=True) == -1


@pytest.mark.parametrize(
    "value,expected", [(-(2**31), -4), (-4, -4), (0, 0), (9, 9), (2**31 - 1, 9)]
)
def test_clamp_includes_endpoints(value: int, expected: int) -> None:
    assert math.clamp32(value, -4, 9) == expected


@pytest.mark.parametrize(
    "numerator,denominator",
    [(0, 0), (0, -1), (2**31, 1), (-(2**31) - 1, 1), (0, 2**31), (True, 1)],
)
def test_division_rejects_unverified_inputs(numerator: int, denominator: int) -> None:
    with pytest.raises(ValueError):
        math.round_divide32(numerator, denominator)


@pytest.mark.parametrize(
    "samples,index,shift,signed",
    [
        ([], 0, 4, False),
        ([1], 0, 4, False),
        ([0, 1], -1, 4, False),
        ([0, 1], 2**31, 4, False),
        ([0, 1], 0, 5, False),
        ([0, 1], 0, 7, True),
        ([-1, 1], 0, 4, False),
        ([0, 65536], 0, 4, False),
        ([0, 32768], 0, 4, True),
        ([-32769, 0], 0, 4, True),
        ([True, 0], 0, 4, False),
    ],
)
def test_interpolation_rejects_unverified_inputs(samples, index, shift, signed) -> None:
    with pytest.raises(ValueError):
        math.interpolate(samples, index, interval_shift=shift, signed=signed)


def test_clamp_rejects_inverted_bounds_and_non_int32() -> None:
    with pytest.raises(ValueError):
        math.clamp32(0, 4, -4)
    with pytest.raises(ValueError):
        math.clamp32(2**31, -4, 4)
