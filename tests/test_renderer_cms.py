"""Independent CIE references for the bounded CMS0 composition."""

import sys
from pathlib import Path

import numpy as np
import pytest
from test_renderer_lut import make_profile
from test_renderer_profile import profile as matrix_profile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from renderer_cms import convert, source_profile, source_table  # noqa: E402


@pytest.mark.parametrize("width", [1, 3, 4, 5, 50, 208])
@pytest.mark.parametrize(
    "codes", [(0, 32768, 32768), (32640, 40000, 24000), (65280, 32768, 32768)]
)
def test_constant_lab_lattice_composes_to_linear_xyz(width, codes):
    grid = np.empty((65, 65, 65, 3), dtype="<u2")
    grid[:] = codes
    source = make_profile(grid)
    destination = matrix_profile(words=(0, 65535))
    pixels = np.random.default_rng(20260919).integers(
        0, 65536, size=(width * 2, 3), dtype="<u2"
    )
    before = pixels.copy()
    # Independent CIE Lab conversion, including the renderer's L16 normalization.
    lightness = ((codes[0] * 65535 + 32640) // 65280) * 100 / 65535
    a, b = (codes[1] - 32768) / 256, (codes[2] - 32768) / 256
    fy = (lightness + 16) / 116

    def inverse(value):
        return value**3 if value > 6 / 29 else 3 * (6 / 29) ** 2 * (value - 4 / 29)

    xyz = np.array([inverse(fy + a / 500), inverse(fy), inverse(fy - b / 200)])
    expected = np.rint(np.clip(xyz * [0.9642, 1, 0.8249], 0, 1) * 65535)
    actual = convert(source, destination, pixels, width)
    assert actual.shape == pixels.shape and actual.dtype == np.dtype("<u2")
    assert np.max(np.abs(actual.astype(np.int32) - expected)) <= 1
    np.testing.assert_array_equal(pixels, before)


@pytest.mark.parametrize("which", ["source", "destination"])
def test_unsupported_profile_is_rejected(which):
    source, destination = make_profile(), matrix_profile()
    if which == "source":
        source = destination
    else:
        destination = source
    with pytest.raises(ValueError):
        convert(source, destination, np.zeros((4, 3), dtype="<u2"), 4)


def test_composition_preserves_each_rows_scalar_tail():
    random = np.random.default_rng(19)
    grid = random.integers(0, 65536, (65, 65, 65, 3), dtype="<u2")
    pixels = random.integers(0, 65536, (10, 3), dtype="<u2")
    source, destination = make_profile(grid), matrix_profile(words=(0, 65535))
    expected = np.concatenate(
        [convert(source, destination, row, 5) for row in np.split(pixels, 2)]
    )
    np.testing.assert_array_equal(convert(source, destination, pixels, 5), expected)
    assert np.any(convert(source, destination, pixels, 10) != expected)


@pytest.mark.parametrize("width", [0, True, 3])
def test_incomplete_or_invalid_rows_are_rejected(width):
    with pytest.raises(ValueError):
        convert(make_profile(), matrix_profile(), np.zeros((4, 3), dtype="<u2"), width)


def test_source_metadata_and_table_must_be_complete():
    with pytest.raises(ValueError):
        source_table(
            b"invalid",
            b"invalid",
            coefficients=b"",
            modes=b"",
            blend_calibration=b"",
            decoding_table=b"",
        )
    with pytest.raises(ValueError):
        source_profile(b"", b"", description="NX Ctudio", copyright="")
