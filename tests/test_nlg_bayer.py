"""CFA-aware reduction, normalization and bounded Bayer input contracts."""

import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from nlg_bayer import classification_table, reduce_bayer


def test_vendor_leaf_and_wrapper_corpus():
    vectors = json.loads(
        (Path(__file__).parent / "fixtures/nlg_bayer_vectors.json").read_text()
    )
    rng = np.random.default_rng(vectors["seed"])
    for case in vectors["cases"]:
        height, width = case["source_shape"]
        maximum = 16383 if case["precision"] < 2 else 4095
        y, x = np.indices((height, width), dtype=np.int64)
        kind = case["kind"]
        if kind == "constant":
            source = np.full((height, width), maximum, dtype="<u2")
        elif kind == "ramp":
            source = ((y * 13 + x * 29) % (maximum + 1)).astype("<u2")
        elif kind == "impulse":
            source = np.zeros((height, width), dtype="<u2")
            source[::3, ::5] = maximum
            source[-1, -1] = maximum // 2
        elif kind == "checker":
            source = (((x + y) & 1) * maximum).astype("<u2")
        elif kind == "phase":
            source = (((x & 1) + (y & 1) * 2) * (maximum // 3)).astype("<u2")
        else:
            assert kind == "random"
            source = rng.integers(0, maximum + 1, size=(height, width), dtype="<u2")
        controls = {
            k: case[k] for k in ("pattern", "precision", "monochrome", "low", "high")
        }
        controls["monotonic_shift"] = case["shift"]
        header = struct.pack(
            "<2i4B2H",
            width,
            height,
            case["pattern"],
            case["precision"],
            case["monochrome"],
            case["shift"],
            case["low"],
            case["high"],
        )
        assert (
            hashlib.sha256(header + source.tobytes()).hexdigest()
            == case["input_sha256"]
        )
        rgb, auxiliary = reduce_bayer(source, **controls)
        assert (
            hashlib.sha256(rgb.transpose(0, 2, 1).tobytes()).hexdigest()
            == case["rgb_sha256"]
        )
        assert hashlib.sha256(auxiliary.tobytes()).hexdigest() == case["aux_sha256"]


def test_row_batches_preserve_phase_and_independent_blocks():
    source = np.random.default_rng(37).integers(0, 16384, (262, 258), dtype="<u2")
    rgb, auxiliary = reduce_bayer(source, 3)
    first = reduce_bayer(source[:128], 3)
    second = reduce_bayer(source[128:], 3)
    np.testing.assert_array_equal(rgb, np.concatenate((first[0], second[0])))
    np.testing.assert_array_equal(auxiliary, np.concatenate((first[1], second[1])))


@pytest.mark.parametrize(
    "source,green",
    [
        (
            [
                [0, 3712, 0, 1536],
                [2432, 0, 2944, 0],
                [0, 3584, 0, 128],
                [3456, 0, 128, 0],
            ],
            1120,
        ),
        (
            [
                [0, 1792, 0, 3712],
                [3840, 0, 3584, 0],
                [0, 2176, 0, 2688],
                [896, 0, 2944, 0],
            ],
            1352,
        ),
    ],
    ids=["strict-monotonicity", "first-direction-tie"],
)
def test_auxiliary_requires_strict_monotonicity_and_first_score_tie(source, green):
    rgb, auxiliary = reduce_bayer(np.array(source, dtype="<u2"), 0, low=0, high=0)
    np.testing.assert_array_equal(rgb, [[[0, green, 0]]])
    np.testing.assert_array_equal(auxiliary, [[8]])


@pytest.mark.parametrize("pattern", range(4))
@pytest.mark.parametrize("shape", [(2, 2), (2, 6), (6, 2), (4, 4), (6, 10), (8, 12)])
def test_cfa_channels_and_phase_preserving_partial_blocks(pattern, shape):
    y, x = np.indices(shape)
    ry, rx = divmod(pattern, 2)
    source = np.full(shape, 200, dtype="<u2")
    source[(y % 2 == ry) & (x % 2 == rx)] = 600
    source[(y % 2 != ry) & (x % 2 != rx)] = 1000
    rgb, auxiliary = reduce_bayer(source, pattern)
    expected_shape = ((shape[0] + 2) // 4, (shape[1] + 2) // 4)
    np.testing.assert_array_equal(
        rgb, np.broadcast_to([300, 100, 500], (*expected_shape, 3))
    )
    np.testing.assert_array_equal(auxiliary, np.zeros(expected_shape))
    assert rgb.dtype == np.dtype("<u2")
    assert auxiliary.dtype == np.dtype("u1")


@pytest.mark.parametrize("precision", [0, 1, 2, 3])
def test_precision_normalizes_before_averaging(precision):
    source = np.array([[1, 3], [5, 7]], dtype="<u2")
    expected = [0, 2, 3] if precision < 2 else [2, 8, 14]
    rgb, _ = reduce_bayer(source, 0, precision=precision)
    np.testing.assert_array_equal(rgb, [[expected]])


def test_monochrome_keeps_green_and_auxiliary():
    source = np.arange(64, dtype="<u2").reshape(8, 8) * 200
    color, auxiliary = reduce_bayer(source, 0)
    mono, mono_auxiliary = reduce_bayer(source, 0, monochrome=True)
    expected = np.array([[1350, 1750], [4550, 4950]], dtype="<u2")
    np.testing.assert_array_equal(mono[..., 1], expected)
    np.testing.assert_array_equal(mono[..., (0, 2)], 0)
    np.testing.assert_array_equal(auxiliary, mono_auxiliary)
    assert np.any(color != mono)


@pytest.mark.parametrize("bounds", [(10, 30), (0, 0), (1024, 256), (0, 65535)])
def test_classification_uses_signed_step_and_first_wrapped_threshold(bounds):
    low, high = bounds
    thresholds = [(low + i * ((high - low) // 8)) % 65536 for i in range(8)]
    expected = [
        next((i for i, value in enumerate(thresholds) if x <= value), 8)
        for x in range(1024)
    ]
    np.testing.assert_array_equal(classification_table(low, high), expected)


def test_readonly_noncontiguous_inputs_and_shift_mask():
    source = np.arange(12 * 20, dtype="<u2").reshape(12, 20)[:, ::2] * 23
    source = source[::-1]
    source.flags.writeable = False
    before = source.copy()
    first = reduce_bayer(source, 2, monotonic_shift=0)
    second = reduce_bayer(source, 2, monotonic_shift=32)
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)
        assert not np.shares_memory(source, a)
    np.testing.assert_array_equal(source, before)


@pytest.mark.parametrize(
    "source",
    [
        None,
        [],
        np.zeros((2, 2, 1), dtype="<u2"),
        np.zeros((1, 2), dtype="<u2"),
        np.zeros((4, 5), dtype="<u2"),
        np.zeros((2050, 2), dtype="<u2"),
        np.zeros((2, 2050), dtype="<u2"),
        np.zeros((2, 2), dtype=">u2"),
        np.zeros((2, 2), dtype="<i2"),
        np.full((2, 2), 16384, dtype="<u2"),
    ],
)
def test_rejects_invalid_bayer_inputs(source):
    with pytest.raises(ValueError):
        reduce_bayer(source, 0)


@pytest.mark.parametrize(
    "name,value",
    [
        ("pattern", -1),
        ("pattern", 4),
        ("pattern", True),
        ("precision", 4),
        ("precision", False),
        ("monochrome", 1),
        ("low", -1),
        ("high", 65536),
        ("monotonic_shift", 256),
        ("monotonic_shift", 1.5),
    ],
)
def test_rejects_invalid_controls(name, value):
    controls = dict(pattern=0)
    controls[name] = value
    with pytest.raises(ValueError):
        reduce_bayer(np.zeros((2, 2), dtype="<u2"), **controls)


def test_rejects_12_bit_overflow_before_normalization():
    with pytest.raises(ValueError):
        reduce_bayer(np.full((2, 2), 4096, dtype="<u2"), 0, precision=2)
