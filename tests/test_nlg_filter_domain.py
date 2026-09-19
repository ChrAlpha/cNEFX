"""Integer Nlg filter-domain conversion, rounding and zero-strength limits."""

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from nlg_filter_domain import from_domain, to_domain, zero_strength


def signed_half(value):
    return ((abs(value) + 1) // 2) * (-1 if value < 0 else 1)


def inverse_reference(samples):
    result = np.empty(samples.shape, dtype="<u2")
    for index in np.ndindex(samples.shape[:2]):
        y, cb, cr = map(int, samples[index])
        r, b = signed_half(y + cr), signed_half(y + cb)
        g = y - signed_half(r + b)
        result[index] = [
            min(max(c, 0), hi) for c, hi in zip((r, g, b), (16383, 4095, 16383))
        ]
    return result


def test_primaries_odd_samples_and_channel_order():
    rgb = np.array(
        [[[0, 0, 0], [16383, 0, 0], [0, 16383, 0], [0, 0, 16383], [1, 2, 4]]],
        dtype="<u2",
    )
    expected = [
        [
            [0, 0, 0],
            [8192, -8192, 24574],
            [16383, -16383, -16383],
            [8192, 24574, -8192],
            [5, 3, -3],
        ]
    ]
    actual = to_domain(rgb)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.dtype("<i4")


@pytest.mark.parametrize("shape", [(1, 1), (1, 17), (19, 1), (7, 8), (8, 7), (63, 64)])
def test_unfiltered_roundtrip_and_green_clipping(shape):
    rgb = np.random.default_rng(sum(shape)).integers(0, 16384, (*shape, 3), dtype="<u2")
    expected = rgb.copy()
    expected[..., 1] = np.minimum(expected[..., 1], 4095)
    np.testing.assert_array_equal(from_domain(to_domain(rgb)), expected)


def test_inverse_independent_domain_extrema_and_signed_ties():
    y = (0, 1, 2, 4095, 8190, 32767, 32768, 65535)
    chroma = (-32768, -32767, -3, -1, 0, 1, 3, 32767)
    samples = np.array(list(itertools.product(y, chroma, chroma)), dtype="<i4")[None]
    actual = from_domain(samples)
    np.testing.assert_array_equal(actual, inverse_reference(samples))
    assert actual.dtype == np.dtype("<u2")


def test_green_uses_unclipped_red_and_blue():
    samples = np.array([[[0, 0, -3], [8190, 32767, 32767]]], dtype="<i4")
    np.testing.assert_array_equal(
        from_domain(samples), [[[0, 1, 0], [16383, 0, 16383]]]
    )


def test_zero_strength_still_clamps_internal_channels():
    samples = np.array(
        [[[65535, -32768, 32767], [8190, -6143, 6142], [8191, -6144, 6143]]],
        dtype="<i4",
    )
    expected = np.array([[[8190, -6143, 6142]] * 3], dtype="<i4")
    actual = zero_strength(samples)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.dtype("<i4")
    assert not np.shares_memory(samples, actual)


def test_noncontiguous_readonly_inputs_are_preserved():
    rgb = np.arange(5 * 14 * 3, dtype="<u2").reshape(5, 14, 3)[:, ::2]
    rgb.flags.writeable = False
    before = rgb.copy()
    domain = to_domain(rgb)[:, ::-1]
    domain.flags.writeable = False
    domain_before = domain.copy()
    np.testing.assert_array_equal(from_domain(domain), rgb[:, ::-1])
    np.testing.assert_array_equal(zero_strength(domain), domain)
    np.testing.assert_array_equal(rgb, before)
    np.testing.assert_array_equal(domain, domain_before)


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        np.zeros((1, 1)),
        np.zeros((1, 1, 4)),
        np.zeros((0, 1, 3)),
        np.zeros((1, 2049, 3)),
    ],
)
def test_rejects_invalid_geometry(value):
    for function in (to_domain, from_domain, zero_strength):
        with pytest.raises(ValueError):
            function(value)


@pytest.mark.parametrize(
    "value",
    [
        np.full((1, 1, 3), 16384, dtype="<u2"),
        np.zeros((1, 1, 3), dtype=">u2"),
        np.zeros((1, 1, 3), dtype="<i2"),
    ],
)
def test_rejects_invalid_rgb(value):
    with pytest.raises(ValueError):
        to_domain(value)


@pytest.mark.parametrize(
    "channel,value",
    [(0, -1), (0, 65536), (1, -32769), (1, 32768), (2, -32769), (2, 32768)],
)
def test_rejects_out_of_range_domain(channel, value):
    samples = np.zeros((1, 1, 3), dtype="<i4")
    samples[0, 0, channel] = value
    for function in (from_domain, zero_strength):
        with pytest.raises(ValueError):
            function(samples)


@pytest.mark.parametrize("dtype", [">i4", "<i2", "<u4", "<f4"])
def test_rejects_wrong_domain_storage(dtype):
    for function in (from_domain, zero_strength):
        with pytest.raises(ValueError):
            function(np.zeros((1, 1, 3), dtype=dtype))
