"""Analytic and input-boundary checks for native NP3 curve generation."""

import math

import pytest

from nx_ctudio.picture_curve import generate_curve, generate_table


def curve(
    points: tuple[tuple[int, int], ...] = ((0, 0), (255, 255)),
    settings: tuple[int, ...] = (0, 255, 0, 255, 1, 0),
) -> bytes:
    result = bytearray(578)
    result[:9] = b"I0" + bytes(settings) + bytes([len(points)])
    result[9 : 9 + 2 * len(points)] = bytes(
        value for point in points for value in point
    )
    return bytes(result)


def test_identity_includes_both_endpoints_and_half_up_quantization() -> None:
    actual = generate_table(curve())
    assert actual == tuple(int(32767 * i / 256 + 0.5) for i in range(257))
    assert actual[128] == 16384
    assert actual[129] == 16511


def test_curve_name_is_the_primary_api_with_compatibility_alias() -> None:
    payload = curve()
    assert generate_curve(payload) == generate_table(payload)


def test_rendering_curve_uses_12_bit_quantization() -> None:
    payload = curve(settings=(0, 255, 0, 255, 2, 20))
    expected = tuple(int(4095 * math.pow(i / 256, 1 / 2.2) + 0.5) for i in range(257))
    assert generate_curve(payload, sample_max=4095) == expected


@pytest.mark.parametrize("maximum", [0, 255, 4096, 32768, True, 4095.0])
def test_rejects_unverified_sample_ranges(maximum) -> None:
    with pytest.raises(ValueError, match="12-bit and 15-bit"):
        generate_curve(curve(), sample_max=maximum)


def test_decreasing_two_point_curve_is_linear() -> None:
    assert generate_table(curve(((0, 255), (255, 0)))) == tuple(
        int(32767 * (1 - i / 256) + 0.5) for i in range(257)
    )


def test_three_point_natural_spline_matches_analytic_solution() -> None:
    table = generate_table(curve(((0, 0), (85, 255), (170, 0))))
    for index in (8, 32, 64, 100, 128, 160):
        x = index / 256
        t = 3 * min(x, 2 / 3 - x)
        expected = int(32767 * (1.5 * t - 0.5 * t**3) + 0.5)
        assert table[index] == expected
    assert table[171:] == (0,) * 86


@pytest.mark.parametrize("integer,fraction", [(0, 50), (1, 25), (2, 20)])
def test_gamma_on_identity_is_analytic(integer: int, fraction: int) -> None:
    table = generate_table(curve(settings=(0, 255, 0, 255, integer, fraction)))
    gamma = integer + fraction * 0.01
    assert table == tuple(
        int(32767 * math.pow(i / 256, 1 / gamma) + 0.5) for i in range(257)
    )


@pytest.mark.parametrize("encoded,clamped", [(0, 0.05), (25, 20.0)])
def test_gamma_clamps_to_recovered_bounds(encoded: int, clamped: float) -> None:
    table = generate_table(curve(settings=(0, 255, 0, 255, encoded, 0)))
    assert table[64] == int(32767 * math.pow(0.25, 1 / clamped) + 0.5)


def test_levels_and_short_domain_hold_endpoint_values() -> None:
    table = generate_table(curve(((30, 40), (220, 210)), (20, 240, 15, 230, 1, 0)))
    low = int(32767 * ((40 / 255) * (215 / 255) + 15 / 255) + 0.5)
    high = int(32767 * ((210 / 255) * (215 / 255) + 15 / 255) + 0.5)
    assert table[0] == table[40] == low
    assert table[230] == table[256] == high


def test_spline_overshoot_clips_before_output_level_mapping() -> None:
    table = generate_table(
        curve(((0, 0), (85, 255), (170, 255), (255, 0)), (0, 255, 32, 224, 1, 0))
    )
    low, high = (int(32767 * (value / 255) + 0.5) for value in (32, 224))
    assert table[0] == table[-1] == low
    assert table[128] == max(table) == high
    assert min(table) == low


def test_twenty_points_are_supported() -> None:
    points = tuple((i * 13, (i * 47) % 256) for i in range(20))
    table = generate_table(curve(points))
    assert len(table) == 257
    assert all(type(value) is int and 0 <= value <= 32767 for value in table)


def test_stored_table_is_not_an_input_and_record_is_not_mutated() -> None:
    source = curve(((0, 0), (100, 170), (255, 255)))
    changed = source[:64] + b"\xff" * 514
    assert generate_table(source) == generate_table(changed)
    assert source == curve(((0, 0), (100, 170), (255, 255)))


@pytest.mark.parametrize("payload", [b"", bytes(577), bytes(579), None, bytearray(578)])
def test_rejects_unknown_record_type_and_size(payload: bytes) -> None:
    with pytest.raises(ValueError, match="578-byte"):
        generate_table(payload)


@pytest.mark.parametrize(
    "offset,value,reason",
    [
        (0, ord("J"), "marker"),
        (8, 0, "2 to 20"),
        (8, 1, "2 to 20"),
        (8, 21, "2 to 20"),
        (11, 0, "increasing"),
        (3, 0, "level"),
        (2, 255, "level"),
        (5, 0, "level"),
        (4, 255, "level"),
        (7, 100, "hundredths"),
        (63, 1, "padding"),
    ],
)
def test_rejects_unverified_boundaries(offset: int, value: int, reason: str) -> None:
    payload = bytearray(curve())
    payload[offset] = value
    with pytest.raises(ValueError, match=reason):
        generate_table(bytes(payload))


def test_rejects_unsorted_points_instead_of_reordering() -> None:
    with pytest.raises(ValueError, match="increasing"):
        generate_table(curve(((0, 0), (180, 150), (120, 200), (255, 255))))
