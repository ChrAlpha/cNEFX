"""Failure accounting and strict contracts for saved highlight replay."""

import copy
import json
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import verify_renderer_highlight as verifier  # noqa: E402


def captures():
    return [
        {
            "event": "highlight_buffer",
            "index": index,
            "rect": list(verifier.RECTANGLES[(index - 1) % 2]),
            "channel": 1,
            "bayer": [1, 0],
            "error": False,
            "saved": [1, 1, 1],
        }
        for index in range(1, 5)
    ]


def test_buffer_identity_allows_thread_reordering_only():
    events = captures()[::-1]
    assert set(verifier.buffer_events(events)) == {1, 2, 3, 4}
    for changed in (events[:-1], events + events[:1]):
        with pytest.raises(ValueError, match="buffer capture"):
            verifier.buffer_events(changed)
    for key, value in (
        ("index", True),
        ("index", 5),
        ("rect", [1, 0, 209, 172]),
        ("rect", [0, 0, 208, 174]),
        ("channel", 0),
        ("bayer", [0, 1]),
        ("error", True),
        ("error", 0),
        ("saved", [1, 0, 1]),
        ("saved", [True, True, True]),
    ):
        changed = copy.deepcopy(events)
        changed[0][key] = value
        with pytest.raises(ValueError, match="buffer capture"):
            verifier.buffer_events(changed)


def valid_state():
    state = bytearray(0x590)
    struct.pack_into("<f", state, 0x214, 4.0)
    struct.pack_into("<3I", state, 0x224, 4096, 4096, 4096)
    state[0x3F2] = 1
    struct.pack_into("<2I", state, 0x550, 1, 0)
    for offset in (0x250, 0x290, 0x320, 0x360):
        struct.pack_into("<Q", state, offset + 0x38, 65536)
    return state


@pytest.mark.parametrize(
    "offset",
    [
        0x1FC,
        0x200,
        0x20C,
        0x214,
        0x224,
        0x228,
        0x22C,
        0x3F2,
        0x3F4,
        0x550,
        0x554,
        0x288,
        0x2C8,
        0x358,
        0x398,
    ],
)
def test_state_rejects_mode_gain_exposure_bayer_and_table_drift(offset):
    state = valid_state()
    verifier.state_contract(state)
    state[offset] ^= 1
    with pytest.raises(ValueError, match="highlight state"):
        verifier.state_contract(state)


def test_state_rejects_truncation_and_ignores_relocated_pointers():
    state = valid_state()
    with pytest.raises(ValueError, match="highlight state"):
        verifier.state_contract(state[:-1])
    struct.pack_into("<Q", state, 0, 0x180B11D10)
    verifier.state_contract(state)


def arguments(tmp_path, monkeypatch, output):
    argv = ["verify_renderer_highlight.py", "--output", str(output)]
    for name in ("capture", "vendor", "wine-prefix"):
        argv.extend([f"--{name}", str(tmp_path / name)])
    monkeypatch.setattr(sys, "argv", argv)


@pytest.mark.parametrize("exception,code", [(KeyboardInterrupt, 130), (ValueError, 1)])
def test_partial_evidence_survives_failure(
    tmp_path, monkeypatch, capsys, exception, code
):
    output = tmp_path / "output"
    arguments(tmp_path, monkeypatch, output)

    def fail(args, report):
        report["replays"].append({"buffer": 1, "different_words": 3})
        raise exception("Injected failure")

    monkeypatch.setattr(verifier, "verify", fail)
    assert verifier.main() == code
    report = json.loads((output / "report.json").read_text())
    assert report["passed"] is False
    assert report["replays"] == [{"buffer": 1, "different_words": 3}]
    assert report.get("cancelled", False) == (code == 130)
    assert json.loads(capsys.readouterr().out)["passed"] is False


@pytest.mark.parametrize("location", ["existing", "repository", "dangling_link"])
def test_reject_unsafe_output(tmp_path, monkeypatch, location):
    output = tmp_path / "output"
    if location == "existing":
        output.mkdir()
    elif location == "repository":
        output = Path(verifier.__file__).resolve().parent / "uncreated-highlight-report"
    else:
        output.symlink_to(tmp_path / "missing")
    arguments(tmp_path, monkeypatch, output)
    with pytest.raises(SystemExit) as exc:
        verifier.main()
    assert exc.value.code == 2
