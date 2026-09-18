"""Reject unsafe camera-selector arguments before resource indexing."""

import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import kaiji_index  # noqa: E402
from kaiji_index import BOUNDS, camera_index  # noqa: E402


@pytest.fixture
def pinned_inputs(monkeypatch):
    coefficients = bytearray(422416)
    coefficients[0xA8C + 5] = 7
    coefficients[0xBA3 + 12 * 8] = 11
    coefficients[0xC6B + 20] = 13
    coefficients[0xC94 + 12] = 17
    coefficients = bytes(coefficients)
    mode_table = bytes([6]) * 76
    monkeypatch.setattr(
        kaiji_index, "COEFFICIENT_HASH", hashlib.sha256(coefficients).hexdigest()
    )
    monkeypatch.setattr(
        kaiji_index, "MODE_TABLE_HASH", hashlib.sha256(mode_table).hexdigest()
    )
    return coefficients, mode_table


@pytest.mark.parametrize("index", range(10))
@pytest.mark.parametrize("side", ("low", "high"))
def test_outside_abi_range(index, side):
    values = [0, 0, 14, 6, 12, 0, 20, 0, 0, 1]
    low, high = BOUNDS[index]
    values[index] = low - 1 if side == "low" else high + 1
    with pytest.raises(ValueError, match=f"argument {index}"):
        camera_index(b"", b"", values)


@pytest.mark.parametrize("value", (True, 1.0, "1", None))
def test_wrong_argument_type(value):
    values = [value, 0, 14, 6, 12, 0, 20, 0, 0, 1]
    with pytest.raises(ValueError, match="argument 0"):
        camera_index(b"", b"", values)


def test_unverified_processing_selector():
    with pytest.raises(ValueError, match="processing selectors"):
        camera_index(b"", b"", [0, 0, 14, 6, 12, 0, 20, 0, 1, 1])


def test_wrong_argument_count():
    with pytest.raises(ValueError, match="ten camera selector arguments"):
        camera_index(b"", b"", [0] * 9)


def test_unpinned_coefficients():
    with pytest.raises(ValueError, match="pinned Z8"):
        camera_index(bytes(422416), bytes(76), [0, 0, 14, 6, 12, 0, 20, 0, 0, 1])


@pytest.mark.parametrize("offset", [0, 1, 5, 6, 107, 108])
def test_processing_zero_ignores_mode_including_initializer_default(
    pinned_inputs, offset
):
    for mode in range(5):
        values = [0, 0, 14, 6, 12, 0, 20, offset, 0, mode]
        assert camera_index(*pinned_inputs, values) == min(108, 48 + offset)


def test_processing_two_rejects_zero_mode_before_table_access():
    with pytest.raises(ValueError, match="mode"):
        camera_index(b"", b"", [0, 0, 14, 6, 12, 0, 20, 0, 2, 0])


@pytest.mark.parametrize("mode", [1, 2, 3, 4])
def test_processing_two_still_applies_correction(pinned_inputs, mode):
    values = [0, 0, 14, 6, 12, 0, 20, 1, 2, mode]
    assert camera_index(*pinned_inputs, values) == 55
