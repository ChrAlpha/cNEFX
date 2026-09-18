"""Require complete, nonoverlapping native composition without vendor fixtures."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import kaiji_table  # noqa: E402


def sections():
    return [
        (start, bytes([0x55]) * (end - start))
        for start, end in (
            (0x48, 0x61F6),
            (0x61F8, 0x21756),
            (0x21758, 0x26042),
            (0x26044, 0x3436C),
        )
    ]


def setup_sections(monkeypatch, ranges):
    monkeypatch.setattr(kaiji_table, "parameter_ranges", lambda *_: list(ranges))
    monkeypatch.setattr(kaiji_table, "camera_ranges", lambda *_, **__: [])


def test_composition_preserves_declared_bytes_and_zero_fields(monkeypatch):
    setup_sections(monkeypatch, sections())
    table = kaiji_table.generate_table(b"", [0] * 39 + [50, 0])
    assert len(table) == 213872
    assert table.count(0x55) == 213790
    assert table.count(0) == 82
    assert table[:72] == bytes(72)


@pytest.mark.parametrize(
    "fault", ("empty", "missing", "overlap", "padding", "negative", "overflow")
)
def test_invalid_composition_is_rejected(monkeypatch, fault):
    ranges = sections()
    if fault == "empty":
        ranges = []
    elif fault == "missing":
        ranges.pop()
    elif fault == "overlap":
        ranges.append((0x48, b"x"))
    elif fault == "padding":
        ranges.append((0, b"x"))
    elif fault == "negative":
        ranges.append((-1, b"x"))
    else:
        ranges.append((0x34370, b"x"))
    setup_sections(monkeypatch, ranges)
    with pytest.raises(ValueError):
        kaiji_table.generate_table(b"", [0] * 39 + [50, 0])
