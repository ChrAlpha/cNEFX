"""Reject unsafe camera-selector arguments before resource indexing."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from kaiji_index import BOUNDS, camera_index  # noqa: E402


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
