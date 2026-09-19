"""Independent scalar checks for the bounded spatial highlight branch."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from renderer_highlight import recover, transfer_tables, transform  # noqa: E402


def _encode(value):
    return min((math.isqrt(int(value) << 18) + 1) // 2, 65535)


def _decode(value):
    return (value * value + 32768) // 65536


def _scalar_recovery(padded, *, truncate_each_pass=True):
    """Compute local neighborhoods without sparse planes or vectorized filtering."""
    threshold = _encode(15974)
    weights = (1, 4, 6, 4, 1)
    offsets = (-4, -2, 0, 2, 4)

    def smoothed(y, x, channel):
        total = 0
        for dy, wy in zip(offsets, weights):
            row = sum(
                wx * max(_encode(padded[y + dy, x + dx, channel]) - threshold, 0)
                for dx, wx in zip(offsets, weights)
            )
            total += wy * (row // 16 if truncate_each_pass else row)
        return total // (16 if truncate_each_pass else 256)

    height, width = padded.shape[:2]
    result = np.empty((height - 12, width - 12, 3), dtype="<u2")
    for y in range(6, height - 6):
        for x in range(6, width - 6):
            pixel = [_encode(value) for value in padded[y, x]]
            for channel, phase in ((0, 0), (2, 1)):
                if pixel[channel] <= pixel[1]:
                    continue
                rows = (y,) if y % 2 == phase else (y - 1, y + 1)
                columns = (x,) if x % 2 == phase else (x - 1, x + 1)
                excess = sum(
                    smoothed(row, column, channel) for row in rows for column in columns
                ) // (len(rows) * len(columns))
                pixel[channel] = min(max(pixel[channel] - excess, pixel[1]), threshold)
            result[y - 6, x - 6] = [_decode(value) for value in pixel]
    return result


def test_transfer_direction_and_integer_reference_over_entire_domain():
    encode, decode = transfer_tables()
    assert encode.dtype == decode.dtype == np.dtype("<u2")
    assert encode.shape == decode.shape == (65536,)
    np.testing.assert_array_equal(encode, [_encode(value) for value in range(65536)])
    np.testing.assert_array_equal(decode, [_decode(value) for value in range(65536)])
    assert (int(encode[15974]), int(decode[65535])) == (32355, 65534)
    np.testing.assert_array_equal(decode[encode[:16385]], np.arange(16385))
    assert np.count_nonzero(encode[decode[:16385]] != np.arange(16385)) == 12288


@pytest.mark.parametrize("shape", [(14, 14, 3), (16, 18, 3), (20, 24, 3)])
@pytest.mark.parametrize("bounds", [(0, 15974), (15000, 18000), (0, 65536)])
def test_recovery_matches_independent_local_scalar_reference(shape, bounds):
    padded = np.random.default_rng(3051).integers(*bounds, shape, dtype="<u2")
    expected = _scalar_recovery(padded)
    actual = recover(padded)
    assert actual.shape == (shape[0] - 12, shape[1] - 12, 3)
    assert actual.dtype == np.dtype("<u2")
    np.testing.assert_array_equal(actual, expected)


def test_each_separable_filter_pass_truncates_before_interpolation():
    padded = np.random.default_rng(820).integers(15974, 20000, (18, 22, 3), dtype="<u2")
    padded[:, :, 1] = 8000
    expected = _scalar_recovery(padded)
    one_truncation = _scalar_recovery(padded, truncate_each_pass=False)
    assert np.count_nonzero(expected != one_truncation) > 0
    np.testing.assert_array_equal(recover(padded), expected)


def test_green_reference_is_preserved_and_corrections_have_both_clamps():
    padded = np.empty((14, 14, 3), dtype="<u2")
    padded[:] = [65535, 1000, 65535]
    padded[6:8, 6:8] = [10000, 8000, 9000]
    recovered = recover(padded)
    np.testing.assert_array_equal(recovered, np.full((2, 2, 3), 8000, dtype="<u2"))

    padded[:] = [0, 0, 0]
    padded[6:8, 6:8] = [16000, 1000, 1000]
    assert recover(padded)[0, 1, 0] == 15974

    padded[:] = [5000, 10000, 8000]
    np.testing.assert_array_equal(recover(padded), padded[6:-6, 6:-6])


def test_eligibility_is_inclusive_and_selects_the_whole_source_pixel():
    padded = np.empty((14, 14, 3), dtype="<u2")
    padded[:] = [65535, 1000, 65535]
    padded[6:8, 6:8] = [
        [[16384, 1000, 15000], [16385, 1000, 15000]],
        [[15000, 16385, 16384], [15000, 1000, 16385]],
    ]
    recovered = _scalar_recovery(padded)
    actual = transform(padded)
    np.testing.assert_array_equal(actual[0, 0], recovered[0, 0].astype(np.int64) * 4)
    assert actual[0, 0, 0] < 65535
    np.testing.assert_array_equal(actual[0, 1], [65535, 4000, 60000])
    np.testing.assert_array_equal(actual[1, 0], [60000, 65535, 65535])
    np.testing.assert_array_equal(actual[1, 1], [60000, 4000, 65535])
    assert recovered[0, 1, 2] != padded[6, 7, 2]
    assert recovered[1, 1, 0] != padded[7, 7, 0]


@pytest.mark.parametrize("channel,position", [(0, (2, 6)), (2, (1, 1))])
def test_caller_supplied_halo_influences_only_the_matching_color(channel, position):
    padded = np.empty((32, 32, 3), dtype="<u2")
    padded[:] = [15000, 5000, 14000]
    unchanged = recover(padded)
    padded[position[0], position[1], channel] = 65535
    changed = recover(padded)
    assert changed[0, 0, channel] < unchanged[0, 0, channel]
    other_channels = [index for index in range(3) if index != channel]
    np.testing.assert_array_equal(
        changed[:, :, other_channels], unchanged[:, :, other_channels]
    )
    np.testing.assert_array_equal(changed[10:, 10:], unchanged[10:, 10:])
    np.testing.assert_array_equal(changed, _scalar_recovery(padded))


@pytest.mark.parametrize("function", [recover, transform])
def test_readonly_noncontiguous_input_is_preserved_and_output_is_independent(function):
    storage = np.random.default_rng(992).integers(0, 20000, (32, 36, 3), dtype="<u2")
    padded = storage[::2, ::-2]
    before = storage.copy()
    padded.flags.writeable = False
    actual = function(padded)
    assert actual.dtype == np.dtype("<u2")
    assert not np.shares_memory(actual, padded)
    np.testing.assert_array_equal(actual, function(padded.copy()))
    np.testing.assert_array_equal(storage, before)


@pytest.mark.parametrize("function", [recover, transform])
@pytest.mark.parametrize(
    "padded",
    [
        None,
        [[[0, 0, 0]]],
        np.zeros((14, 14), dtype="<u2"),
        np.zeros((14, 14, 4), dtype="<u2"),
        np.zeros((14, 14, 3, 1), dtype="<u2"),
        np.zeros((0, 14, 3), dtype="<u2"),
        np.zeros((12, 14, 3), dtype="<u2"),
        np.zeros((14, 12, 3), dtype="<u2"),
        np.zeros((15, 14, 3), dtype="<u2"),
        np.zeros((14, 15, 3), dtype="<u2"),
        np.zeros((1038, 14, 3), dtype="<u2"),
        np.zeros((14, 1038, 3), dtype="<u2"),
        np.zeros((14, 14, 3), dtype="<u4"),
        np.zeros((14, 14, 3), dtype="<f4"),
        np.zeros((14, 14, 3), dtype=">u2"),
    ],
)
def test_reject_incompatible_or_unsupported_tiles(function, padded):
    with pytest.raises(ValueError, match="even RGB16 tile dimensions"):
        function(padded)
