"""Require complete, immutable source-window evidence and exact native results."""

import copy
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import verify_nlg_bayer as verifier


def events():
    result = []
    for call in (1, 2):
        result.append(
            dict(
                event="nlg_bayer_before",
                call=call,
                valid=True,
                saved=True,
                pattern=0,
                precision=0,
                monochrome=0,
                low=10,
                high=30,
                shift=1,
                clamp=[0, 0],
                source_shape=[5504, 8256],
                output_shape=[1376, 2064],
                cursors=[0, 0, 0],
                mxcsr=8123,
            )
        )
        for index, origin in enumerate(verifier.ORIGINS):
            result.append(
                dict(
                    event="nlg_bayer_window",
                    call=call,
                    window=index,
                    origin=list(origin),
                    source_shape=[4, 1024],
                    output_shape=[1, 256],
                    valid=True,
                    input_unchanged=True,
                    saved=True,
                )
            )
        result.append(
            dict(
                event="nlg_bayer_after",
                call=call,
                valid=True,
                saved=True,
                descriptors_unchanged=True,
                mxcsr=[8123, 8123],
                capture_bytes=call * 120656,
            )
        )
    return result


def test_events_accept_preserved_exception_status_bits():
    calls = verifier.bayer_events(events())
    assert [call["call"] for call in calls] == [1, 2]


@pytest.mark.parametrize(
    "index,key,value",
    [
        (0, "pattern", 4),
        (0, "precision", True),
        (0, "clamp", [1, 0]),
        (0, "cursors", [1, 0, 0]),
        (0, "mxcsr", 0x3F80),
        (1, "window", 1),
        (1, "origin", [0, 1]),
        (1, "input_unchanged", False),
        (1, "saved", False),
        (13, "descriptors_unchanged", False),
        (13, "mxcsr", [8123, 8096]),
        (13, "capture_bytes", 0),
        (14, "call", 1),
    ],
)
def test_rejects_changed_capture_contract(index, key, value):
    captured = events()
    captured[index][key] = value
    with pytest.raises(ValueError):
        verifier.bayer_events(captured)


@pytest.mark.parametrize("change", ["omit", "duplicate", "reorder"])
def test_rejects_incomplete_or_reordered_events(change):
    captured = events()
    if change == "omit":
        captured.pop()
    elif change == "duplicate":
        captured.append(captured[0])
    else:
        captured[1], captured[2] = captured[2], captured[1]
    with pytest.raises(ValueError):
        verifier.bayer_events(captured)


def descriptor(width, height, channels, halo_x=0, halo_y=0):
    raw = bytearray(0x88)
    struct.pack_into("<Q", raw, 0x28, (height + 2 * halo_y) * channels)
    struct.pack_into("<3I", raw, 0x60, width, height, width + 2 * halo_x)
    return bytes(raw)


def snapshots():
    payloads = {}
    for call in (1, 2):
        base = f"case0.nlg-bayer-{call}-window"
        controls = bytearray(32)
        struct.pack_into("<5I", controls, 0, 0, 8256, 5504, 2064, 1376)
        controls[22] = 1
        struct.pack_into("<2H", controls, 26, 10, 30)
        payloads[f"{base}-1-controls.bin"] = bytes(controls)
        for name, dims in (
            ("source", (8256, 5504, 1)),
            ("rgb", (2064, 1376, 3, 2, 1)),
            ("aux", (2064, 1376, 1)),
        ):
            for phase in ("before", "after"):
                payloads[f"{base}-1-{name}-descriptor-{phase}.bin"] = descriptor(*dims)
        for index in range(12):
            payloads[f"{base}{index}-input.bin"] = np.full(
                (4, 1024), 100, dtype="<u2"
            ).tobytes()
            payloads[f"{base}{index}-rgb.bin"] = np.full(
                (3, 256), 50, dtype="<u2"
            ).tobytes()
            payloads[f"{base}{index}-aux.bin"] = bytes(256)
    return payloads


def replay(payloads):
    report = dict(windows=[], sources_unchanged=True, capture_providers=[None] * 28)
    verifier.replay_windows(
        verifier.bayer_events(events()), payloads.__getitem__, report
    )
    verifier.summarize(report)
    return report


def test_complete_replay_compares_both_planes_and_all_windows():
    payloads = snapshots()
    before = copy.deepcopy(payloads)
    report = replay(payloads)
    assert report["passed"] is True
    assert report["rgb_words"] == 18432
    assert report["aux_bytes"] == 6144
    assert len(report["windows"]) == 24
    assert payloads == before


@pytest.mark.parametrize("suffix", ["rgb", "aux"])
def test_pixel_mismatch_retains_all_comparisons(suffix):
    payloads = snapshots()
    key = f"case0.nlg-bayer-1-window0-{suffix}.bin"
    payloads[key] = bytes([payloads[key][0] ^ 1]) + payloads[key][1:]
    report = replay(payloads)
    assert report["passed"] is False
    assert len(report["windows"]) == 24
    assert report["windows"][0][suffix]["different_words"] == 1


@pytest.mark.parametrize("suffix", ["input", "rgb", "aux"])
def test_rejects_truncated_pixel_payload(suffix):
    payloads = snapshots()
    key = f"case0.nlg-bayer-1-window0-{suffix}.bin"
    payloads[key] = payloads[key][:-1]
    with pytest.raises(ValueError):
        replay(payloads)


@pytest.mark.parametrize(
    "suffix,offset",
    [
        ("controls", 22),
        ("controls", 25),
        ("rgb-descriptor-before", 0x28),
        ("source-descriptor-before", 0x60),
        ("aux-descriptor-before", 0x78),
        ("source-descriptor-after", 0),
    ],
)
def test_rejects_changed_control_or_descriptor(suffix, offset):
    payloads = snapshots()
    key = f"case0.nlg-bayer-1-window-1-{suffix}.bin"
    changed = bytearray(payloads[key])
    changed[offset] ^= 1
    payloads[key] = bytes(changed)
    with pytest.raises(ValueError):
        replay(payloads)


@pytest.mark.parametrize(
    "change", ["missing", "duplicate", "hash", "provider", "count"]
)
def test_summary_requires_complete_evidence(change):
    report = replay(snapshots())
    if change == "missing":
        report["windows"].pop()
    elif change == "duplicate":
        report["windows"][-1] = report["windows"][0]
    elif change == "hash":
        report["sources_unchanged"] = False
    elif change == "provider":
        report["capture_providers"].pop()
    else:
        report["windows"][0]["aux"]["compared_words"] = 1
    verifier.summarize(report)
    assert report["passed"] is False


@pytest.mark.parametrize("cancelled", [False, True])
def test_command_retains_failed_or_cancelled_partial_report(
    tmp_path, monkeypatch, cancelled
):
    output = tmp_path / "report"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_nlg_bayer",
            "--capture",
            str(tmp_path / "capture.json"),
            "--vendor",
            str(tmp_path / "vendor"),
            "--wine-prefix",
            str(tmp_path / "wine"),
            "--output",
            str(output),
        ],
    )

    def fail(args, report):
        report["windows"].append({"call": 1, "window": 0})
        if cancelled:
            raise KeyboardInterrupt
        raise ValueError("Truncated saved window.")

    monkeypatch.setattr(verifier, "verify", fail)
    assert verifier.main() == (130 if cancelled else 1)
    saved = (output / "report.json").read_bytes()
    report = json.loads(saved)
    assert report["passed"] is False
    assert report["windows"] == [{"call": 1, "window": 0}]
    if cancelled:
        assert report["cancelled"] is True
    else:
        assert report["error"] == dict(
            type="ValueError", message="Truncated saved window."
        )
    with pytest.raises(SystemExit) as exc:
        verifier.main()
    assert exc.value.code == 2
    assert (output / "report.json").read_bytes() == saved
