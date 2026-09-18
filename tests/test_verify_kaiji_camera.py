"""Enforce camera replay coverage without checking in vendor tables."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from verify_kaiji_camera import EXPECTED_LAYOUT, validate_ranges  # noqa: E402


def ranges():
    return [(offset, bytes(size)) for offset, size in EXPECTED_LAYOUT.items()]


def test_complete_layout():
    validate_ranges(ranges())


@pytest.mark.parametrize(
    "fault",
    (
        "empty",
        "missing",
        "duplicate",
        "short",
        "overlap",
        "overflow",
        "bool",
        "mutable",
    ),
)
def test_incomplete_or_invalid_layout_is_rejected(fault):
    actual = ranges()
    if fault == "empty":
        actual = []
    elif fault == "missing":
        actual.pop()
    elif fault == "duplicate":
        actual.append(actual[0])
    elif fault == "short":
        actual[0] = (actual[0][0], actual[0][1][:-1])
    elif fault == "overlap":
        actual.append((actual[0][0] + 1, b"x"))
    elif fault == "overflow":
        actual.append((0x34370, b"x"))
    elif fault == "bool":
        actual.append((True, b"x"))
    else:
        actual[0] = (actual[0][0], bytearray(actual[0][1]))
    with pytest.raises(ValueError):
        validate_ranges(actual)
