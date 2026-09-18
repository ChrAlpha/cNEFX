"""Require mixed curve fixtures to retain every requested normalized control."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from inspect_np3 import PARAMETERS, Record, parse, serialize  # noqa: E402
from np3_parameters import normalize_flexible, parameter_blocks  # noqa: E402
from verify_kaiji_composition import controls  # noqa: E402
from verify_kaiji_curve import mixed_snapshot  # noqa: E402


def base_snapshot():
    return serialize(
        (
            Record(0x100, b"0310"),
            Record(0x200, b"test" + bytes(16)),
            Record(0x300, b"\0\x20"),
            *(Record(i, bytes((128, scale))) for i, (_, scale) in PARAMETERS.items()),
            Record(0x1F00, bytes([128] * 24 + [1, 1, 1, 0])),
            Record(
                0x2000,
                b"\x80\0\x80\x80" * 3 + bytes((1, 1, 1, 0, 178, 1, 128, 1)),
            ),
            Record(0xABCDEF, b"preserved opaque field"),
        )
    )


@pytest.mark.parametrize("values", controls())
def test_roundtrip_preserves_all_controls(values):
    changed = parse(mixed_snapshot(base_snapshot(), values))
    returned, actual = normalize_flexible(parameter_blocks(changed)["flexible"])
    assert actual == values
    assert returned == 128 + values[0]
    assert changed[-1].payload == b"preserved opaque field"


def test_missing_parameter_record_cannot_silently_drop_requested_contrast():
    snapshot = serialize(
        tuple(r for r in parse(base_snapshot()) if r.identifier != 0x1900)
    )
    with pytest.raises(ValueError):
        mixed_snapshot(snapshot, controls()[1])
