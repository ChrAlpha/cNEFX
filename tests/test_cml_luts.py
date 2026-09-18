"""Synthetic curve and container boundaries for the recovered CML LUTs."""

import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from cml_luts import (  # noqa: E402
    icc_tags,
    input_luts,
    inverse_curve,
    output_luts,
    profile_curves,
)


def profile(tags, pcs=b"XYZ "):
    data = bytearray(132 + 12 * len(tags))
    data[16:24], data[36:40] = b"RGB " + pcs, b"acsp"
    struct.pack_into(">I", data, 128, len(tags))
    for index, (signature, content) in enumerate(tags.items()):
        struct.pack_into(
            ">4sII", data, 132 + 12 * index, signature, len(data), len(content)
        )
        data.extend(content)
    struct.pack_into(">I", data, 0, len(data))
    return bytes(data)


def trc(words):
    return (
        b"curv"
        + bytes(4)
        + struct.pack(">I", len(words))
        + struct.pack(f">{len(words)}H", *words)
    )


def identity_profile():
    return profile({key: trc([0, 65535]) for key in (b"rTRC", b"gTRC", b"bTRC")})


def test_inverse_identity_covers_every_code_without_wrapping():
    words = np.array([0, 65535], dtype="<u2")
    np.testing.assert_array_equal(inverse_curve(words), np.arange(65536))


def test_inverse_piecewise_curve_quantizes_at_the_halfway_knot():
    inverse = inverse_curve(np.array([0, 16384, 65535], dtype="<u2"))
    np.testing.assert_array_equal(
        inverse[[0, 8192, 16384, 65535]], [0, 16384, 32768, 65535]
    )


@pytest.mark.parametrize(
    "words", [[0], [1, 65535], [0, 65534], [0, 1, 1, 65535], [0, 9, 8, 65535]]
)
def test_unrecovered_curve_domains_are_rejected(words):
    with pytest.raises(ValueError):
        inverse_curve(np.array(words, dtype="<u2"))


def test_output_lut_clamps_the_internal_range_and_keeps_channels():
    lut = output_luts(identity_profile())
    assert lut.shape == (3, 65536) and lut.dtype == np.dtype("<u2")
    assert np.all(lut[:, :8193] == 0)
    assert np.all(lut[:, 57343:] == 65535)
    np.testing.assert_array_equal(lut[:, [8193, 16384]], [[1, 10923]] * 3)
    assert np.all(np.diff(lut.astype(np.int32)) >= 0)


def test_input_lut_expands_synthetic_identity_mft2_curves():
    tag = bytearray(240268)
    tag[:12] = b"mft2\0\0\0\0\x03\x03\x21\0"
    struct.pack_into(">9i", tag, 12, 65536, 0, 0, 0, 65536, 0, 0, 0, 65536)
    struct.pack_into(">HH", tag, 48, 4096, 3)
    curve = np.rint(np.arange(4096) * (65535 / 4095)).astype(">u2").tobytes()
    tag[52:24628] = curve * 3
    lut = input_luts(profile({b"A2B0": tag}, b"Lab "))
    assert lut.shape == (3, 65536)
    assert np.max(np.abs(lut.astype(np.int32) - np.arange(65536))) <= 1
    tag[12] = 1
    with pytest.raises(ValueError, match="layout"):
        input_luts(profile({b"A2B0": tag}, b"Lab "))


@pytest.mark.parametrize("offset,value", [(0, 1), (128, 999), (136, 1), (140, 999999)])
def test_malformed_icc_bounds_fail_closed(offset, value):
    data = bytearray(identity_profile())
    struct.pack_into(">I", data, offset, value)
    with pytest.raises(ValueError):
        icc_tags(bytes(data))


def test_missing_duplicate_and_non_sampled_trcs_fail_closed():
    with pytest.raises(ValueError, match="sampled"):
        profile_curves(profile({}))
    data = bytearray(identity_profile())
    data[144:148] = b"rTRC"
    with pytest.raises(ValueError, match="duplicate"):
        icc_tags(bytes(data))
    with pytest.raises(ValueError, match="sampled"):
        profile_curves(profile({b"rTRC": b"para" + bytes(20)}))
