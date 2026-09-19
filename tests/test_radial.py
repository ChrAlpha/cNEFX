"""Crop-aware radial sampling and numerical boundaries for pixel-shift synthesis."""

import numpy as np
import pytest

from nx_ctudio.radial import RadialCorrection, correct_crop


@pytest.mark.parametrize("scale", [1, 2])
def test_zero_coefficients_preserve_crop_and_do_not_mutate_source(scale: int) -> None:
    shape = (10, 12)
    pixels = np.random.default_rng(11).integers(
        0, 16384, (9 * scale, 11 * scale, 3), dtype=np.uint16
    )
    pixels.flags.writeable = False
    result = correct_crop(
        pixels, shape, scale, (2, 3, 6, 5), RadialCorrection((0, 0), ((0, 0, 0),) * 2)
    )
    np.testing.assert_array_equal(
        result, pixels[2 * scale : 7 * scale, scale : 7 * scale]
    )
    assert not np.shares_memory(result, pixels)


@pytest.mark.parametrize("scale", [1, 2])
def test_crop_result_matches_independent_full_plane_reference(
    scale: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    from verify_radial import correct

    shape = (68, 74)
    pixels = np.random.default_rng(12).integers(
        0, 16384, (67 * scale, 73 * scale, 3), dtype=np.uint16
    )
    centers = (192, -128)
    coefficients = ((-120, 350, -90), (280, -180, 70))
    expected = correct(pixels, shape, scale, centers, coefficients)
    result = correct_crop(
        pixels, shape, scale, (2, 2, 70, 65), RadialCorrection(centers, coefficients)
    )
    np.testing.assert_array_equal(
        result, expected[scale : 66 * scale, scale : 71 * scale]
    )


def test_outward_mapping_uses_samples_beyond_the_final_crop() -> None:
    pixels = np.zeros((9, 11, 3), dtype=np.uint16)
    pixels[:, -2:, (0, 2)] = 10000
    actual = correct_crop(
        pixels,
        (10, 12),
        1,
        (2, 2, 7, 6),
        RadialCorrection((0, 0), ((1048576, 0, 0),) * 2),
    )
    assert np.any(actual[:, -1, 0])
    assert not np.any(actual[..., 1])


def test_border_taps_are_not_renormalized_and_negative_coordinates_truncate() -> None:
    pixels = np.full((9, 11, 3), 1000, dtype=np.uint16)
    result = correct_crop(
        pixels,
        (10, 12),
        1,
        (1, 1, 11, 9),
        RadialCorrection((0, 0), ((65536, 0, 0),) * 2),
    )

    # Top-left maps to (-0.28125, -0.21875), losing out-of-domain taps.
    def weight(distance: float) -> float:
        return (
            1 - 2 * distance**2 + distance**3
            if distance <= 1
            else (4 - 8 * distance + 5 * distance**2 - distance**3)
        )

    total = 0.0
    for y in (0, 1):
        for x in (0, 1):
            total += 1000 * weight(x + 0.28125) * weight(y + 0.21875)
    assert result[0, 0, 0] == int(total)
    assert result[0, 0, 0] < 1000
    np.testing.assert_array_equal(result[..., 1], pixels[..., 1])


def test_extreme_signed_metadata_does_not_overflow_coordinate_casts() -> None:
    pixels = np.full((9, 11, 3), 1000, dtype=np.uint16)
    with np.errstate(all="raise"):
        actual = correct_crop(
            pixels,
            (10, 12),
            1,
            (1, 1, 11, 9),
            RadialCorrection((2**31 - 1, -(2**31)), ((2**31 - 1,) * 3,) * 2),
        )
    assert not np.any(actual[..., (0, 2)])
    assert np.all(actual[..., 1] == 1000)


@pytest.mark.parametrize("value", [2**31, -(2**31) - 1, 0.5, True])
def test_parameters_reject_values_outside_signed_metadata_domain(value: object) -> None:
    with pytest.raises(ValueError, match="signed 32-bit"):
        RadialCorrection((value, 0), ((0, 0, 0),) * 2)


@pytest.mark.parametrize("crop", [(0, 1, 4, 4), (1, 1, 12, 9), (1, 1, 4, 0)])
def test_crop_outside_reconstructed_domain_fails(
    crop: tuple[int, int, int, int],
) -> None:
    with pytest.raises(ValueError, match="Crop"):
        correct_crop(
            np.zeros((9, 11, 3), np.uint16),
            (10, 12),
            1,
            crop,
            RadialCorrection((0, 0), ((0, 0, 0),) * 2),
        )
