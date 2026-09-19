"""Reject unsupported spatial-plane parameters and incomplete replay evidence."""

import copy
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import verify_nlg_pyramid as verifier


def test_parameter_order_and_first_level():
    raw = bytes(0xAF) + bytes([1, 205, 125, 0, 0, 1, 0])
    assert verifier.level_parameters(raw, True) == (205, 0, 125)
    raw = bytes(0xAF) + bytes([1, 205, 35, 71, 1, 1, 0])
    assert verifier.level_parameters(raw, False) == (205, 71, 35)


@pytest.mark.parametrize("offset", [0xAF, 0xB3, 0xB4, 0xB5])
def test_rejects_unrecovered_branches(offset):
    raw = bytearray(bytes(0xAF) + bytes([1, 205, 35, 35, 1, 1, 0]))
    raw[offset] ^= 1
    with pytest.raises(ValueError):
        verifier.level_parameters(bytes(raw), False)


@pytest.mark.parametrize("size", [0, 181, 183])
def test_rejects_truncated_and_extended_parameters(size):
    with pytest.raises(ValueError):
        verifier.level_parameters(bytes(size), True)


def fixture(channels=3):
    event = dict(
        width=5, height=3, channels=channels, saved=True, bounded=True, valid=True
    )
    descriptor = bytearray(0x88)
    struct.pack_into("<Q", descriptor, 0x28, 3 * channels)
    struct.pack_into("<3I", descriptor, 0x60, 5, 3, 32)
    struct.pack_into("<I", descriptor, 0x78, 2)
    pixels = np.arange(3 * channels * 5, dtype="<u2").reshape(3, channels, 5)
    return event, descriptor, pixels


@pytest.mark.parametrize("channels", [1, 3])
def test_plane_layout_already_has_logical_row_order(channels):
    event, descriptor, pixels = fixture(channels)
    actual = verifier.plane_values(event, descriptor, pixels.tobytes())
    expected = pixels.transpose(0, 2, 1)
    np.testing.assert_array_equal(
        actual, expected[:, :, 0] if channels == 1 else expected
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("width", True),
        ("width", 0),
        ("width", 2049),
        ("height", 2049),
        ("channels", 2),
        ("saved", False),
        ("bounded", False),
        ("valid", False),
    ],
)
def test_rejects_invalid_plane_event(field, value):
    event, descriptor, pixels = fixture()
    event[field] = value
    with pytest.raises(ValueError):
        verifier.plane_values(event, descriptor, pixels.tobytes())


@pytest.mark.parametrize(
    "offset,value", [(0x28, 8), (0x60, 6), (0x64, 4), (0x68, 4), (0x78, 3)]
)
def test_rejects_invalid_descriptor(offset, value):
    event, descriptor, pixels = fixture()
    struct.pack_into("<I", descriptor, offset, value)
    with pytest.raises(ValueError):
        verifier.plane_values(event, descriptor, pixels.tobytes())


@pytest.mark.parametrize("change", [-2, 2])
def test_rejects_incomplete_plane_payload(change):
    event, descriptor, pixels = fixture()
    raw = pixels.tobytes()
    with pytest.raises(ValueError):
        verifier.plane_values(
            event, descriptor, raw[:change] if change < 0 else raw + bytes(change)
        )


@pytest.mark.parametrize(
    "events", [[], [{"event": "nlg_pyramid", "index": 1, "result": 0}]]
)
def test_rejects_missing_or_failed_pyramid_calls(events):
    with pytest.raises(ValueError, match="sequence"):
        verifier.pyramid_events(events)


def arguments(tmp_path, monkeypatch, output):
    argv = ["verify_nlg_pyramid.py", "--output", str(output)]
    for name in ("capture", "vendor", "wine-prefix"):
        argv += ["--" + name, str(tmp_path / name)]
    monkeypatch.setattr(sys, "argv", argv)


@pytest.mark.parametrize("error,code", [(ValueError, 1), (KeyboardInterrupt, 130)])
def test_partial_evidence_survives_failure(tmp_path, monkeypatch, capsys, error, code):
    output = tmp_path / "output"
    arguments(tmp_path, monkeypatch, output)
    partial = dict(index=7, slot=14, different_words=1)

    def fail(args, report):
        report["levels"].append(copy.deepcopy(partial))
        raise error("Injected failure")

    monkeypatch.setattr(verifier, "verify", fail)
    assert verifier.main() == code
    result = json.loads((output / "report.json").read_text())
    assert result["passed"] is False
    assert result["levels"] == [partial]
    assert json.loads(capsys.readouterr().out)["passed"] is False


@pytest.mark.parametrize("location", ["existing", "repository", "link"])
def test_no_overwrite_or_repository_output(tmp_path, monkeypatch, location):
    output = tmp_path / "output"
    if location == "existing":
        output.mkdir()
    elif location == "repository":
        output = Path(verifier.__file__).parent / "uncreated-pyramid-output"
    else:
        output.symlink_to(tmp_path / "missing")
    arguments(tmp_path, monkeypatch, output)
    with pytest.raises(SystemExit) as exc:
        verifier.main()
    assert exc.value.code == 2
