"""Strict mode boundaries and persistent failure evidence for vignette replay."""

import json
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import verify_renderer_vignette as verifier  # noqa: E402


def state():
    result = bytearray(0xD30)
    struct.pack_into("<I", result, 0x260, 5)
    struct.pack_into("<2d", result, 0x430, 0.4, 1.0)
    struct.pack_into("<2f", result, 0x6D8, 1.0, 1.0)
    struct.pack_into("<2f", result, 0x6A8, -0.5, 0.25)
    for offset in (0x2A0, 0x2E0):
        struct.pack_into("<Q", result, offset, 65536)
    for offset, value in zip((0x6B0, 0x6B8, 0x6C0, 0x6C8), (0.0, 0.125, 0.25, 0.5)):
        struct.pack_into("<f", result, offset, value)
    return result


def test_extracts_numeric_parameters_without_pointer_inputs():
    original = state()
    expected = ((-0.5, 0.25), (0.0, 0.125, 0.25, 0.5))
    assert verifier.parameters(original) == expected
    for offset in (0, 0x240, 0x290, 0x2D0):
        struct.pack_into("<Q", original, offset, 0xDEADBEEF)
    assert verifier.parameters(original) == expected


@pytest.mark.parametrize(
    "offset",
    [
        0x260,
        0x2E8,
        0x430,
        0x438,
        0x6D8,
        0x6DC,
        0x6E0,
        0xD28,
        0xD29,
        0xD2A,
        0x2A0,
        0x2E0,
    ],
)
def test_rejects_unsupported_modes_gains_normalization_and_corrections(offset):
    original = state()
    original[offset] ^= 1
    with pytest.raises(ValueError, match="mode5"):
        verifier.parameters(original)


def test_rejects_incomplete_state():
    with pytest.raises(ValueError, match="mode5"):
        verifier.parameters(state()[:-1])


def test_auxiliary_events_require_actual_v11_calls_and_success():
    events = [
        {
            "event": "highlight_buffer",
            "index": index,
            "rect": rectangle,
            "channel": 1,
            "bayer": [1, 0],
            "error": False,
            "saved": [1, 1, 1],
        }
        for index, rectangle in (
            (1, [0, 0, 208, 172]),
            (2, [208, 0, 258, 172]),
            (3, [208, 0, 258, 172]),
        )
    ]
    verifier.auxiliary_events(events[::-1])
    for changed in (events[:-1], events + events[:1]):
        with pytest.raises(ValueError, match="auxiliary"):
            verifier.auxiliary_events(changed)
    for field, value in (
        ("error", True),
        ("saved", [1, 0, 1]),
        ("rect", [0, 0, 208, 172]),
        ("channel", True),
    ):
        previous = events[-1][field]
        events[-1][field] = value
        with pytest.raises(ValueError, match="auxiliary"):
            verifier.auxiliary_events(events)
        events[-1][field] = previous


def arguments(tmp_path, monkeypatch, output):
    argv = ["verify_renderer_vignette.py", "--output", str(output)]
    for name in ("capture", "vendor", "wine-prefix"):
        argv.extend([f"--{name}", str(tmp_path / name)])
    monkeypatch.setattr(sys, "argv", argv)


@pytest.mark.parametrize("exception,code", [(KeyboardInterrupt, 130), (ValueError, 1)])
def test_failure_and_cancellation_keep_partial_report(
    tmp_path, monkeypatch, capsys, exception, code
):
    output = tmp_path / "output"
    arguments(tmp_path, monkeypatch, output)

    def fail(args, report):
        report["tables"].append({"kind": "radial", "different_words": 2})
        raise exception("Injected failure")

    monkeypatch.setattr(verifier, "verify", fail)
    assert verifier.main() == code
    report = json.loads((output / "report.json").read_text())
    assert report["passed"] is False
    assert report["tables"] == [{"kind": "radial", "different_words": 2}]
    assert report.get("cancelled", False) == (code == 130)
    assert json.loads(capsys.readouterr().out)["passed"] is False


@pytest.mark.parametrize("location", ["existing", "repository", "dangling_link"])
def test_output_must_be_new_external_directory(tmp_path, monkeypatch, location):
    output = tmp_path / "output"
    if location == "existing":
        output.mkdir()
    elif location == "repository":
        output = Path(verifier.__file__).resolve().parent / "uncreated-vignette-report"
    else:
        output.symlink_to(tmp_path / "missing")
    arguments(tmp_path, monkeypatch, output)
    with pytest.raises(SystemExit) as exc:
        verifier.main()
    assert exc.value.code == 2
