"""Require complete, nonoverlapping native composition without vendor fixtures."""

import struct
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
    monkeypatch.setattr(kaiji_table, "parameter_ranges", lambda *_, **__: list(ranges))
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


def test_explicit_curve_bypasses_four_tone_controls(monkeypatch):
    monkeypatch.setattr(
        kaiji_table, "grading_tables", lambda *_: (((0,) * 65,) * 3, ((0,) * 4096,) * 3)
    )
    monkeypatch.setattr(kaiji_table, "input_matrix", lambda *_: (0,) * 9)
    monkeypatch.setattr(
        kaiji_table, "blender_tables", lambda *_, **__: ((0,) * 32,) * 6
    )
    monkeypatch.setattr(
        kaiji_table,
        "contrast_tables",
        lambda *_, **__: (((0,) * 257,) * 4, ((0,) * 4096,) * 2),
    )

    def reject_tone(*_):
        raise AssertionError("Explicit curve must bypass tone generation.")

    monkeypatch.setattr(kaiji_table, "tone_tables", reject_tone)
    payload = bytearray(578)
    payload[:13] = b"I0" + bytes((0, 255, 0, 255, 1, 0, 2, 0, 64, 255, 64))
    values = [0, 100, -100, 100, -100] + [0] * 34 + [50, 0]
    ranges = dict(
        kaiji_table.parameter_ranges(b"", values, curve_payload=bytes(payload))
    )
    assert struct.unpack("<257H", ranges[0x2E044]) == (1028,) * 257
    assert struct.unpack("<4096H", ranges[0x2E246]) == (1028,) * 4096
