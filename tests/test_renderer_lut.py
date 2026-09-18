"""Synthetic boundaries for the recovered grid65 LUTRGBLab renderer stage."""

import itertools
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from renderer_lut import convert, normalized_grid, transform  # noqa: E402


def make_profile(grid=None):
    if grid is None:
        grid = np.zeros((65, 65, 65, 3), dtype="<u2")
        grid[..., :] = [32640, 12345, 54321]
    tag = bytearray(52)
    tag[:12] = b"mft2\0\0\0\0\x03\x03\x41\0"
    struct.pack_into(">9i", tag, 12, 65536, 0, 0, 0, 65536, 0, 0, 0, 65536)
    struct.pack_into(">HH", tag, 48, 4096, 3)
    curve = (np.arange(4096) * 65535 // 4095).astype(">u2").tobytes()
    tag.extend(curve * 3)
    tag.extend(grid.astype(">u2").tobytes())
    tag.extend(struct.pack(">3H", 0, 32767, 65535) * 3)
    profile = bytearray(144)
    profile[8:24] = b"\x02\x20\0\0scnrRGB Lab "
    profile[36:40] = b"acsp"
    struct.pack_into(">3i", profile, 68, 63190, 65536, 54061)
    struct.pack_into(">I4sII", profile, 128, 1, b"A2B0", 144, len(tag))
    profile.extend(tag)
    struct.pack_into(">I", profile, 0, len(profile))
    return bytes(profile)


@pytest.fixture(scope="module")
def profile():
    return make_profile()


@pytest.fixture(scope="module")
def ramp_grid():
    grid = np.empty((65, 65, 65, 3), dtype="<u2")
    grid[:] = np.arange(65, dtype="<u2")[:, None, None, None]
    return grid


def test_profile_normalizes_lightness_only_with_saturation():
    grid = np.empty((65, 65, 65, 3), dtype="<u2")
    grid[:] = [32640, 12345, 54321]
    lightness = [0, 1, 32640, 65279, 65280, 65535]
    grid[0, 0, :6, 0] = lightness
    profile = make_profile(grid)
    result = normalized_grid(profile)
    assert result.shape == grid.shape and result.dtype == np.dtype("<u2")
    np.testing.assert_array_equal(
        result[0, 0, :6, 0], [0, 1, 32768, 65534, 65535, 65535]
    )
    np.testing.assert_array_equal(result[..., 1:], grid[..., 1:])
    assert not np.shares_memory(result, np.frombuffer(profile, dtype=np.uint8))


@pytest.mark.parametrize("width", range(1, 10))
def test_vector_groups_floor_and_each_row_tail_rounds(width, ramp_grid):
    # RGB 512 maps to Q14 8192, exactly midway between lattice values 0 and 1.
    pixels = np.tile(np.array([512, 0, 0], dtype="<u2"), (width * 3, 1))
    expected_row = [0] * (width // 4 * 4) + [1] * (width % 4)
    expected = np.repeat(np.tile(expected_row, 3)[:, None], 3, axis=1)
    np.testing.assert_array_equal(transform(pixels, ramp_grid, width), expected)


def test_row_boundaries_change_results_even_for_the_same_flat_buffer(ramp_grid):
    pixels = np.tile(np.array([512, 0, 0], dtype="<u2"), (10, 1))
    one_row = transform(pixels, ramp_grid, 10)
    two_rows = transform(pixels, ramp_grid, 5)
    np.testing.assert_array_equal(one_row[:, 0], [0] * 8 + [1, 1])
    np.testing.assert_array_equal(two_rows[:, 0], [0, 0, 0, 0, 1] * 2)


def test_float32_coordinate_halfway_cases_round_to_nearest_even():
    grid = np.zeros((65, 65, 65, 3), dtype="<u2")
    grid[1::2] = 65535
    pixels = np.zeros((4, 3), dtype="<u2")
    pixels[:, 0] = [2048, 6144, 10240, 14336]
    # Float32 Q14 coordinates end in 0.5, 1.5, 2.5, 3.5 respectively.
    expected = np.repeat(np.array([0, 7, 7, 15])[:, None], 3, axis=1)
    np.testing.assert_array_equal(transform(pixels, grid, 4), expected)


def test_all_six_tetrahedra_use_four_vertices_instead_of_trilinear_weights():
    grid = np.zeros((65, 65, 65, 3), dtype="<u2")
    for r, g, b in itertools.product(range(2), repeat=3):
        grid[r, g, b] = (4 * r + 2 * g + b) ** 2 * 1000
    pixels = np.array(list(itertools.permutations([512, 256, 128])), dtype="<u2")
    # Fractions 8192,4096,2048 give vertex weights 1/2,1/4,1/8,1/8.
    expected = np.repeat(
        np.array([14625, 13250, 11625, 9500, 8250, 7500])[:, None], 3, axis=1
    )
    actual = transform(pixels, grid, 6)
    np.testing.assert_array_equal(actual, expected)
    assert actual[0, 0] != 11750  # Eight-vertex trilinear result for this input.


def test_endpoints_saturation_ties_and_input_immutability():
    grid = np.empty((65, 65, 65, 3), dtype="<u2")
    grid[:] = [65535, 65535, 65535]
    grid[0, 0, 0] = [0, 12345, 65535]
    grid[-1, -1, -1] = [65535, 54321, 0]
    pixels = np.array([[0, 0, 0], [65535] * 3, [32768] * 3], dtype="<u2")
    original_pixels, original_grid = pixels.copy(), grid.copy()
    actual = transform(pixels, grid, 3)
    np.testing.assert_array_equal(
        actual, [[0, 12345, 65535], [65535, 54321, 0], [65535] * 3]
    )
    np.testing.assert_array_equal(pixels, original_pixels)
    np.testing.assert_array_equal(grid, original_grid)


def test_complete_profile_conversion_preserves_channel_order(profile):
    pixels = np.array([[0, 65535, 12345], [65535, 0, 54321]], dtype="<u2")
    np.testing.assert_array_equal(
        convert(profile, pixels, 2), [[32768, 12345, 54321]] * 2
    )


@pytest.mark.parametrize("curve_offset,count", [(52, 4096), (1672378, 3)])
@pytest.mark.parametrize("delta", [-2, 2])
def test_identity_curve_tolerance_is_inclusive(profile, curve_offset, count, delta):
    changed = bytearray(profile)
    offset = 144 + curve_offset + (count // 2) * 2
    struct.pack_into(
        ">H", changed, offset, struct.unpack_from(">H", changed, offset)[0] + delta
    )
    np.testing.assert_array_equal(
        normalized_grid(bytes(changed)), normalized_grid(profile)
    )


@pytest.mark.parametrize("curve_offset,count", [(52, 4096), (1672378, 3)])
@pytest.mark.parametrize("channel", range(3))
@pytest.mark.parametrize("delta", [-3, 3])
def test_non_identity_curve_is_rejected(profile, curve_offset, count, channel, delta):
    changed = bytearray(profile)
    offset = 144 + curve_offset + (channel * count + count // 2) * 2
    struct.pack_into(
        ">H", changed, offset, struct.unpack_from(">H", changed, offset)[0] + delta
    )
    with pytest.raises(ValueError, match="identity"):
        normalized_grid(bytes(changed))


@pytest.mark.parametrize(
    "offset,replacement",
    [
        (0, bytes(4)),
        (8, b"\x04\x20\0\0"),
        (12, b"mntr"),
        (16, b"CMYK"),
        (20, b"XYZ "),
        (36, b"fake"),
        (68, bytes(12)),
        (128, struct.pack(">I", 2)),
        (132, b"B2A0"),
        (136, struct.pack(">I", 143)),
        (140, struct.pack(">I", 1672395)),
        (144, b"mAB "),
        (148, b"\0\0\0\1"),
        (152, b"\x04"),
        (153, b"\x04"),
        (154, b"\x21"),
        (155, b"\x01"),
        (156, bytes(36)),
        (192, struct.pack(">H", 0)),
        (192, struct.pack(">H", 65535)),
        (194, struct.pack(">H", 2)),
    ],
)
def test_unsupported_profile_layout_is_rejected(profile, offset, replacement):
    changed = bytearray(profile)
    changed[offset : offset + len(replacement)] = replacement
    with pytest.raises(ValueError):
        normalized_grid(bytes(changed))


@pytest.mark.parametrize("size", [0, 128, 144, 160, 1000, 1672539])
def test_truncated_profiles_are_rejected(profile, size):
    with pytest.raises(ValueError):
        normalized_grid(profile[:size])


@pytest.mark.parametrize("width", [0, -1, 2, 4, 1.0, True, "3"])
def test_invalid_row_dimensions_are_rejected(ramp_grid, width):
    with pytest.raises(ValueError, match="width"):
        transform(np.zeros((3, 3), dtype="<u2"), ramp_grid, width)


@pytest.mark.parametrize(
    "pixels",
    [
        np.zeros((0, 3), dtype="<u2"),
        np.zeros((1, 1, 3), dtype="<u2"),
        np.zeros((1, 4), dtype="<u2"),
        np.zeros((1, 3), dtype=">u2"),
        np.zeros((1, 3), dtype=np.float64),
    ],
)
def test_invalid_pixel_shape_or_type_is_rejected(ramp_grid, pixels):
    with pytest.raises(ValueError):
        transform(pixels, ramp_grid, 1)


@pytest.mark.parametrize(
    "grid",
    [
        np.zeros((33, 33, 33, 3), dtype="<u2"),
        np.zeros((65, 65, 65, 4), dtype="<u2"),
        np.zeros((65, 65, 65, 3), dtype=">u2"),
        np.zeros((65, 65, 65, 3), dtype=np.float64),
    ],
)
def test_unsupported_grid_shape_or_type_is_rejected(grid):
    with pytest.raises(ValueError, match="grid65"):
        transform(np.zeros((1, 3), dtype="<u2"), grid, 1)
