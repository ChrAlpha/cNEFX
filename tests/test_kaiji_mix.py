"""Check signed narrowing and rounding at blend-selector boundaries."""

import hashlib
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import kaiji_mix  # noqa: E402
from kaiji_mix import _rounded_short, blend_mix  # noqa: E402


@pytest.mark.parametrize(
    "n,d,expected",
    [
        (3, 2, 2),
        (-3, 2, -2),
        (3, -2, -1),
        (-3, -2, 1),
        (32768, 1, -32768),
        (-32769, 1, 32767),
        (2**31 + 3, 16384, 1),
        (2**30, 32767, -32767),
    ],
)
def test_vendor_rounding_and_signed_narrowing(n, d, expected):
    assert _rounded_short(n, d) == expected


def test_unpinned_calibration_is_rejected_even_for_disabled_mode():
    with pytest.raises(ValueError, match="pinned"):
        blend_mix(bytes(3162), [0] * 9, 0, 0, 21, 0)


def calibration(monkeypatch):
    data = bytearray(3162)
    struct.pack_into("<9h", data, 0, 16384, 0, 0, 0, 16384, 0, 0, 0, 16384)
    for i in range(81):
        struct.pack_into("<2h", data, 18 + i * 4, i * 16 - 320, i % 7)
    struct.pack_into("<1408h", data, 342, *([1] * 1408))
    struct.pack_into("<2h", data, 3158, 2048, 2048)
    data = bytes(data)
    monkeypatch.setattr(kaiji_mix, "CALIBRATION_HASH", hashlib.sha256(data).hexdigest())
    return data


@pytest.mark.parametrize("value,expected", [(0, 5), (64, 6), (68, 1), (-1284, 1)])
def test_threshold_intervals_include_upper_endpoint(monkeypatch, value, expected):
    resource = calibration(monkeypatch)
    identity = [16384, 0, 0, 0, 16384, 0, 0, 0, 16384]
    assert blend_mix(resource, identity, 512, 512, 17, value) == expected
    assert blend_mix(resource, identity, 512, 512, 1, value) == 1
    assert blend_mix(resource, identity, 512, 512, 21, value) == 0


def test_zero_transformed_red_bypasses_threshold_lookup(monkeypatch):
    assert blend_mix(calibration(monkeypatch), [0] * 9, 512, 512, 17, 64) == 0
