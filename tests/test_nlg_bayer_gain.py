"""Vendor gain vectors, exhaustive signed lookup inputs and numeric contracts."""

import dataclasses
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from nlg_bayer_gain import Parameters, decode_parameters, lookup, prepare

VECTORS = json.loads(
    (Path(__file__).parent / "fixtures/nlg_bayer_gain_vectors.json").read_text()
)
SIMPLE = Parameters(0, (512, 512, 512), (0, 0, 0), 0, 100, 0, 256, 512, 16383)


def table():
    values = ((np.arange(65536, dtype=np.uint64) * 73 + 19) & 65535).astype("<u2")
    values[0] = 65535
    return values


@pytest.mark.parametrize("case", VECTORS["cases"], ids=lambda case: case["name"])
def test_guarded_vendor_gain_and_lookup_vectors(case):
    source = np.array(case["samples"], dtype="<u2")
    record = bytes.fromhex(case["parameters_hex"])
    lut = table()
    height, width, _ = source.shape
    payload = (
        struct.pack("<4i", width, height, 0, 0)
        + record
        + lut.tobytes()
        + source.tobytes()
    )
    assert hashlib.sha256(payload).hexdigest() == case["input_sha256"]
    gained = prepare(source, decode_parameters(record))
    np.testing.assert_array_equal(gained, case["gain"])
    np.testing.assert_array_equal(lookup(gained, lut), case["lookup"])
    assert gained.dtype == np.dtype("<i2")


@pytest.mark.parametrize("sweep", VECTORS["lookup_sweeps"])
def test_all_signed16_inputs_in_each_channel_against_vendor(sweep):
    quarter = sweep["quarter"]
    values = np.arange(quarter * 16384, (quarter + 1) * 16384, dtype=np.uint16)
    source = np.stack((values, values[::-1], np.roll(values, 781)), axis=-1)
    source = source.reshape(128, 128, 3).view("<i2")
    output = lookup(source, table())
    assert output.dtype == np.dtype("<u2")
    assert (
        hashlib.sha256(output.transpose(0, 2, 1).tobytes()).hexdigest()
        == sweep["sha256"]
    )


def test_lookup_uses_even_indices_and_preserves_table():
    source = np.array([[[-32768, -1, 0], [1, 8191, 32767]]], dtype="<i2")
    lut = np.zeros(65536, dtype="<u2")
    lut[::2] = 65535
    before = lut.copy()
    lut.flags.writeable = False
    np.testing.assert_array_equal(lookup(source, lut), 4095)
    np.testing.assert_array_equal(lut, before)


def test_decode_parameter_offsets_and_unused_word():
    raw = bytearray(0xB6)
    struct.pack_into("<8H", raw, 0x0C, 1, 2, 3, 4, 5, 6, 7, 65535)
    raw[0x1C] = 255
    struct.pack_into("<4Hh", raw, 0x1E, 8, 9, 10, 11, -12)
    assert decode_parameters(bytes(raw)) == Parameters(
        1, (2, 3, 4), (5, 6, 7), 255, 8, 9, 10, 11, -12
    )


@pytest.mark.parametrize("record", [b"", bytes(181), bytes(183), bytearray(182), None])
def test_rejects_incomplete_or_nonbytes_parameters(record):
    with pytest.raises(ValueError):
        decode_parameters(record)


@pytest.mark.parametrize("nonlinear", [0, 1, 255])
def test_noncontiguous_readonly_inputs_do_not_alias_outputs(nonlinear):
    source = np.arange(18 * 14 * 3, dtype="<u2").reshape(18, 14, 3)[::-1, ::2]
    source.flags.writeable = False
    before = source.copy()
    parameters = dataclasses.replace(SIMPLE, nonlinear=nonlinear)
    output = prepare(source, parameters)
    np.testing.assert_array_equal(output, prepare(before, parameters))
    np.testing.assert_array_equal(source, before)
    assert not np.shares_memory(source, output)


@pytest.mark.parametrize(
    "source",
    [
        None,
        [],
        np.zeros((1, 1), dtype="<u2"),
        np.zeros((1, 1, 4), dtype="<u2"),
        np.zeros((0, 1, 3), dtype="<u2"),
        np.zeros((2049, 1, 3), dtype="<u2"),
        np.zeros((1, 2049, 3), dtype="<u2"),
        np.zeros((1, 1, 3), dtype=">u2"),
        np.zeros((1, 1, 3), dtype="<i2"),
    ],
)
def test_gain_rejects_unsupported_images(source):
    with pytest.raises(ValueError):
        prepare(source, SIMPLE)


@pytest.mark.parametrize(
    "field,value",
    [
        ("black", -1),
        ("gains", (1, 2)),
        ("gains", (1, 2, 65536)),
        ("offsets", [0, 0, 0]),
        ("offsets", (0, 0, True)),
        ("nonlinear", True),
        ("nonlinear", 256),
        ("cutoff_base", 65536),
        ("green_slope_q8", -1),
        ("blue_coupling_q8", 1.5),
        ("red_coupling_q8", 65536),
        ("green_upper", -32769),
        ("green_upper", 32768),
    ],
)
def test_gain_rejects_parameters_outside_stored_widths(field, value):
    with pytest.raises(ValueError):
        prepare(
            np.zeros((1, 1, 3), dtype="<u2"),
            dataclasses.replace(SIMPLE, **{field: value}),
        )


@pytest.mark.parametrize(
    "lut",
    [
        None,
        [],
        np.zeros(65535, dtype="<u2"),
        np.zeros((1, 65536), dtype="<u2"),
        np.zeros(65536, dtype=">u2"),
        np.zeros(65536, dtype="<i2"),
    ],
)
def test_lookup_rejects_wrong_table(lut):
    with pytest.raises(ValueError):
        lookup(np.zeros((1, 1, 3), dtype="<i2"), lut)


@pytest.mark.parametrize("dtype", ["<u2", ">i2", "<i4"])
def test_lookup_requires_signed_little_endian_input(dtype):
    with pytest.raises(ValueError):
        lookup(np.zeros((1, 1, 3), dtype=dtype), table())
