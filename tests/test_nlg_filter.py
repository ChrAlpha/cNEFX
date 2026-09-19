"""Guarded vendor vectors and bounded-contract checks for the Nlg filter."""

import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from nlg_filter import _blend_table, _impulse, filter_domain

VECTORS = json.loads(
    (Path(__file__).parent / "fixtures/nlg_filter_vectors.json").read_text()
)
PARAMS = bytes.fromhex(VECTORS["impulses"][0]["params"])


@pytest.mark.parametrize("mode", [0, 1])
@pytest.mark.parametrize("delta", [-15, -14, -13, 13, 14, 15])
def test_intermediate_impulse_selection_distinguishes_threshold_equality(mode, delta):
    window = np.full((1, 1, 5, 5, 3), [4096, 0, 0], dtype=np.int64)
    window[0, 0, 2, 2] += delta
    params = bytearray(PARAMS)
    params[0x14:0x16] = bytes((mode, mode))
    struct.pack_into("<2H", params, 0x16, 7, 7)
    outside = delta < -14 or delta >= 14
    expected = delta if outside == bool(mode) else 0
    actual = _impulse(window, struct.unpack_from("<19H", params), params)
    np.testing.assert_array_equal(actual, [[[expected] * 3]])


@pytest.mark.parametrize("case", VECTORS["isolated_impulses"], ids=lambda c: c["name"])
def test_vendor_isolated_impulses_discriminate_modes_and_thresholds(case):
    mode, delta = case["name"].removeprefix("isolated_impulse_mode").split("_delta")
    source = np.full((7, 9, 3), [4096, 0, 0], dtype="<i4")
    source[3, 4] += int(delta)
    params = bytes.fromhex(case["params"])
    actual = filter_domain(source, params)
    assert hashlib.sha256(actual.tobytes()).hexdigest() == case["expected_sha256"]
    changed = bytearray(params)
    changed[0x14:0x16] = bytes((1 - int(mode),) * 2)
    assert np.any(filter_domain(source, bytes(changed)) != actual)


@pytest.mark.parametrize("index", [0, 1])
def test_vendor_high_blend_weights_with_active_corrections(index):
    case = VECTORS["high_weights"][index]
    rng = np.random.default_rng(20260921)
    y, _ = np.indices((17, 19))
    for _ in range(index + 1):
        noise = rng.integers(-4, 5, (17, 19, 3))
    source = (np.stack((4096 + y * 12, y * 2, -y * 3), axis=-1) + noise).astype("<i4")
    assert hashlib.sha256(source.tobytes()).hexdigest() == case["source_sha256"]
    actual = filter_domain(source, bytes.fromhex(case["params"]))
    assert hashlib.sha256(actual.tobytes()).hexdigest() == case["expected_sha256"]
    assert np.any(actual[..., 1:] != source[..., 1:])


@pytest.mark.parametrize("case", VECTORS["impulses"], ids=lambda c: c["name"])
def test_vendor_impulse_modes_and_signed_threshold_equality(case):
    mode, delta = case["name"].removeprefix("impulse_mode").split("_delta")
    source = np.full((7, 9, 3), [4096, 0, 0], dtype="<i4")
    source[3, 4] += int(delta)
    params = bytes.fromhex(case["params"])
    assert params[0x14:0x16] == bytes((int(mode), int(mode)))
    actual = filter_domain(source, params)
    assert hashlib.sha256(actual.tobytes()).hexdigest() == case["expected_sha256"]


@pytest.mark.parametrize("case", VECTORS["crops"], ids=lambda c: c["name"])
def test_vendor_active_chroma_and_directional_crops(case):
    source = np.array(case["source"], dtype="<i4")
    expected = np.array(case["expected"], dtype="<i4")
    source.flags.writeable = False
    actual = filter_domain(source, bytes.fromhex(case["params"]))
    # A two-pixel halo preserves each saved center pixel's full 5x5 neighborhood.
    np.testing.assert_array_equal(actual[2:-2, 2:-2], expected)
    assert np.count_nonzero(source[2:-2, 2:-2, 1:] != expected[..., 1:]) > 0
    assert actual.dtype == np.dtype("<i4")
    assert not np.shares_memory(actual, source)


@pytest.mark.parametrize("shape", [(1, 1), (1, 9), (7, 1), (16, 17), (17, 19)])
@pytest.mark.parametrize("value", [[0, 0, 0], [8190, -6143, 6142], [4096, -96, 128]])
def test_constant_domain_and_singleton_reflection(shape, value):
    source = np.full((*shape, 3), value, dtype="<i4")
    np.testing.assert_array_equal(filter_domain(source, PARAMS), source)


def test_internal_batch_boundaries_and_noncontiguous_readonly_input():
    rng = np.random.default_rng(20260919)
    source = (rng.integers(-7, 8, (37, 38, 3)) + [4096, -96, 128]).astype("<i4")
    source = source[:, ::2]
    source.flags.writeable = False
    before = source.copy()
    actual = filter_domain(source, PARAMS)
    first = filter_domain(source[:19], PARAMS)[:17]
    second = filter_domain(source[15:], PARAMS)[2:]
    np.testing.assert_array_equal(actual, np.concatenate((first, second)))
    np.testing.assert_array_equal(source, before)


def test_zero_strength_retains_full_domain_clipping():
    params = bytearray(PARAMS)
    params[0x25:0x27] = b"\x00\x00"
    source = np.array([[[65535, -32768, 32767], [0, 0, 0]]], dtype="<i4")
    np.testing.assert_array_equal(
        filter_domain(source, bytes(params)), [[[8190, -6143, 6142], [0, 0, 0]]]
    )


@pytest.mark.parametrize("bounds", [(0, 0), (14, 35), (1024, 256), (65530, 4)])
def test_blend_lookup_uses_first_matching_wrapped_threshold(bounds):
    params = bytearray(PARAMS)
    struct.pack_into("<2H", params, 0x1E, *bounds)
    low, high = bounds
    step = ((high - low) // 8) % 512
    thresholds = [(low + i * step) % 65536 for i in range(8)]
    expected = [
        next((i for i, threshold in enumerate(thresholds) if x <= threshold), 8)
        for x in range(4096)
    ]
    np.testing.assert_array_equal(_blend_table(bytes(params)), expected)


@pytest.mark.parametrize(
    "bad",
    [
        None,
        [],
        np.zeros((2, 3), dtype="<i4"),
        np.zeros((2, 3, 4), dtype="<i4"),
        np.zeros((0, 3, 3), dtype="<i4"),
        np.zeros((2049, 1, 3), dtype="<i4"),
        np.zeros((1, 2049, 3), dtype="<i4"),
        np.zeros((2, 3, 3), dtype=">i4"),
        np.zeros((2, 3, 3), dtype="<u2"),
    ],
)
def test_rejects_invalid_domain_storage(bad):
    with pytest.raises(ValueError):
        filter_domain(bad, PARAMS)


@pytest.mark.parametrize(
    "channel,value", [(0, -1), (0, 8191), (1, -6144), (1, 6143), (2, -6144), (2, 6143)]
)
def test_rejects_nonzero_strength_domain_overflow(channel, value):
    source = np.zeros((2, 3, 3), dtype="<i4")
    source[0, 0, channel] = value
    with pytest.raises(ValueError, match="Nonzero strengths"):
        filter_domain(source, PARAMS)


@pytest.mark.parametrize("params", [None, "", b"", bytes(38), bytes(40), bytearray(39)])
def test_rejects_invalid_parameter_storage(params):
    with pytest.raises(ValueError):
        filter_domain(np.zeros((2, 3, 3), dtype="<i4"), params)


@pytest.mark.parametrize(
    "offset,value", [(0x25, 32), (0x26, 255), (0x1C, 9), (0x24, 31)]
)
def test_rejects_unsupported_strengths_and_left_shifts(offset, value):
    params = bytearray(PARAMS)
    params[offset] = value
    with pytest.raises(ValueError, match="Unsupported filter"):
        filter_domain(np.zeros((2, 3, 3), dtype="<i4"), bytes(params))


@pytest.mark.parametrize("offset,value", [(0x16, 4096), (0x18, 32768)])
def test_rejects_thresholds_requiring_unrecovered_narrowing(offset, value):
    params = bytearray(PARAMS)
    struct.pack_into("<H", params, offset, value)
    with pytest.raises(ValueError, match="Unsupported filter"):
        filter_domain(np.zeros((2, 3, 3), dtype="<i4"), bytes(params))
