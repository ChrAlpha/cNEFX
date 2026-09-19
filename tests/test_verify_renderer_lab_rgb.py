"""Reject incomplete live evidence and excluded failed Lab/RGB captures."""

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import verify_renderer_lab_rgb as verifier  # noqa: E402
from verify_np3_header import windows  # noqa: E402
from verify_renderer_lab_rgb import (  # noqa: E402
    KERNEL_RVA,
    capture_events,
    validate_events,
)


def lifecycle(dll):
    return [
        {"event": "provider", "path_hex": windows(dll).encode().hex()},
        {
            "event": "ready",
            "kernel_rva": KERNEL_RVA,
            "width": 5,
            "height": 2,
            "read_only_inputs": True,
            "mxcsr": 0x1F80,
        },
        {
            "event": "complete",
            "rows": 2,
            "bytes": 60,
            "mxcsr": [0x1F80, 0x1FA0],
            "inputs_and_guards_intact": True,
            "saved": True,
            "released": True,
        },
    ]


@pytest.mark.parametrize(
    "field,value",
    [
        ("rows", True),
        ("bytes", 59),
        ("inputs_and_guards_intact", False),
        ("saved", 1),
        ("released", False),
        ("mxcsr", [0x1F80, 0x3F80]),
        ("mxcsr", [0x1F80, 0x1F81]),
        ("mxcsr", [False, 0x1FA0]),
    ],
)
def test_reject_invalid_lifecycle(tmp_path, field, value):
    dll = tmp_path / "image_render.n5m"
    events = lifecycle(dll)
    validate_events(events, dll, 5, 2)
    events[-1][field] = value
    with pytest.raises(ValueError):
        validate_events(events, dll, 5, 2)


def test_reject_wrong_provider_and_reordered_events(tmp_path):
    dll = tmp_path / "image_render.n5m"
    events = lifecycle(dll)
    for bad in [events[::-1], events[:-1], events + [events[-1]], [None] * 3]:
        with pytest.raises(ValueError):
            validate_events(bad, dll, 5, 2)
    with pytest.raises(ValueError, match="provider"):
        validate_events(events, tmp_path / "other.n5m", 5, 2)


def captures():
    return [
        {
            "event": "inner_pixels",
            "index": 1,
            "object": 256,
            "format": 0x10060204,
            "width": 208,
            "height": 172,
            "kernel_rva": KERNEL_RVA,
            "saved": [1, 1],
            "error": False,
        },
        {
            "event": "inner_pixels",
            "index": 2,
            "object": 512,
            "format": 0x10060201,
            "width": 208,
            "height": 172,
            "kernel_rva": 0x53B640,
            "saved": [1, 1],
            "error": False,
        },
    ]


def test_selects_only_observed_lab_rgb_branch():
    events = captures()
    assert capture_events(events) == events[:1]


@pytest.mark.parametrize("position", [0, 1])
@pytest.mark.parametrize(
    "field,value",
    [
        ("error", True),
        ("error", 0),
        ("saved", [1, 0]),
        ("saved", [True, True]),
        ("width", True),
        ("height", 1),
        ("width", 4097),
        ("object", 0),
        ("format", 0),
        ("kernel_rva", 0),
        ("index", -1),
    ],
)
def test_reject_failed_or_invalid_calls_including_excluded_branch(
    position, field, value
):
    events = captures()
    events[position][field] = value
    with pytest.raises(ValueError):
        capture_events(events)


def test_reject_duplicate_or_absent_selected_capture():
    events = captures()
    for bad in ([], events[1:], events + [copy.deepcopy(events[0])]):
        with pytest.raises(ValueError):
            capture_events(bad)


def test_reverse_branch_selection_and_black_point_exception_flags(tmp_path):
    events = captures()
    assert capture_events(events, rgb_lab=True) == events[1:]
    dll = tmp_path / "image_render.n5m"
    recorded = lifecycle(dll)
    recorded[1]["kernel_rva"] = 0x53B640
    recorded[2]["mxcsr"][1] = 0x1FA1
    validate_events(recorded, dll, 5, 2, rgb_lab=True)
    with pytest.raises(ValueError):
        validate_events(recorded, dll, 5, 2)
    recorded[1]["mxcsr"] = recorded[2]["mxcsr"][0] = 0x1FA1
    with pytest.raises(ValueError):
        validate_events(recorded, dll, 5, 2, rgb_lab=True)


def test_cancellation_persists_report_and_returns_130(tmp_path, monkeypatch, capsys):
    output = tmp_path / "output"
    dll, probe, capture = (
        tmp_path / name for name in ("image.n5m", "probe.exe", "capture.json")
    )
    dll.write_bytes(b"test provider")
    probe.write_bytes(b"test probe")
    capture.write_bytes(b"{}")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_renderer_lab_rgb.py",
            "--dll",
            str(dll),
            "--probe",
            str(probe),
            "--capture",
            str(capture),
            "--wine-prefix",
            str(tmp_path / "wine"),
            "--output",
            str(output),
        ],
    )
    monkeypatch.setattr(verifier, "digest", lambda _: verifier.RENDER_HASH)
    read = Path.read_bytes

    def interrupt(path):
        if path == capture:
            raise KeyboardInterrupt
        return read(path)

    monkeypatch.setattr(Path, "read_bytes", interrupt)
    assert verifier.main() == 130
    report = json.loads((output / "report.json").read_text())
    assert report["passed"] is False and report["cancelled"] is True
    assert len(report["input_hashes"]) == 5
    assert "cancelled" in report["error"]
    assert json.loads(capsys.readouterr().out)["passed"] is False
