"""Reject malformed row-oracle evidence and incomplete renderer captures."""

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from verify_np3_header import windows  # noqa: E402
from verify_renderer_lut import (  # noqa: E402
    KERNEL_RVA,
    WIDTHS,
    pixel_inputs,
    validate_capture_report,
    validate_events,
)


def events(dll):
    return [
        {"event": "provider", "path_hex": windows(dll).encode().hex()},
        {
            "event": "ready",
            "kernel_rva": KERNEL_RVA,
            "nodes": 65,
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


def capture():
    return {
        "evidence_kind": "instrumented_small_render_pilot",
        "artifacts": {"case0.rgb-25-grid.bin": "a" * 64},
        "cases": [
            {
                "index": 0,
                "returncode": 0,
                "events": [
                    {
                        "event": "cml_profile",
                        "index": 7,
                        "size": 1672998,
                        "saved": 1,
                        "result": 0,
                    },
                    {"event": "rgb_kernel", "index": 25, "rva": 0x4F2690},
                    {
                        "event": "rgb_lab",
                        "index": 25,
                        "format": 0x10060201,
                        "width": 5,
                        "height": 2,
                        "strides": [30, 6, 2, 30, 6, 2],
                        "saved": [1, 1],
                        "error": False,
                    },
                ],
            }
        ],
    }


@pytest.mark.parametrize("width", WIDTHS)
def test_points_cover_boundaries_and_remainders_deterministically(width):
    points = pixel_inputs(width)
    assert points.dtype == np.dtype("<u2")
    assert points.shape[1] == 3 and len(points) % width == 0
    assert len(points) >= 4500
    assert np.array_equal(points, pixel_inputs(width))
    assert np.any(np.all(points == (65535, 65535, 65535), axis=1))
    assert np.any(np.all(points == (0, 0, 0), axis=1))
    assert np.any(np.all(points == (32767, 32768, 65534), axis=1))


@pytest.mark.parametrize("width", (0, True, 1.0, 10, -1, None))
def test_rejects_unbounded_widths(width):
    with pytest.raises(ValueError):
        pixel_inputs(width)


def test_accepts_only_complete_guarded_lifecycle(tmp_path):
    dll = tmp_path / "image_render.n5m"
    validate_events(events(dll), dll, 5, 2)
    for key in ("inputs_and_guards_intact", "saved", "released"):
        altered = events(dll)
        altered[-1][key] = False
        with pytest.raises(ValueError):
            validate_events(altered, dll, 5, 2)


@pytest.mark.parametrize(
    "field,value",
    [
        ("rows", True),
        ("bytes", 61),
        ("mxcsr", [0x1F80, 0x1F81]),
        ("mxcsr", [0x1F80, 0x3F80]),
        ("mxcsr", [0x1F80]),
        ("mxcsr", None),
        ("mxcsr", [False, 0x1F80]),
        ("extra", 0),
    ],
)
def test_rejects_corrupt_completion(tmp_path, field, value):
    dll = tmp_path / "image_render.n5m"
    altered = events(dll)
    altered[-1][field] = value
    with pytest.raises(ValueError):
        validate_events(altered, dll, 5, 2)


@pytest.mark.parametrize("value", (None, 42, "zz", "ff", b"00", "5a3a2f736861646f77"))
def test_rejects_invalid_provider_paths(tmp_path, value):
    dll = tmp_path / "image_render.n5m"
    altered = events(dll)
    altered[0]["path_hex"] = value
    with pytest.raises(ValueError):
        validate_events(altered, dll, 5, 2)


def test_rejects_missing_reordered_or_unstructured_events(tmp_path):
    dll = tmp_path / "image_render.n5m"
    original = events(dll)
    for altered in (
        None,
        [],
        original[:-1],
        original[::-1],
        [None] * 3,
        original + [original[-1]],
    ):
        with pytest.raises(ValueError):
            validate_events(altered, dll, 5, 2)
    original[1]["kernel_rva"] += 1
    with pytest.raises(ValueError):
        validate_events(original, dll, 5, 2)


def test_valid_capture_selects_matching_profile_and_kernel():
    selected = validate_capture_report(capture())
    assert len(selected) == 1
    assert selected[0][:2] == (0, 7)
    assert selected[0][2]["index"] == 25


@pytest.mark.parametrize("error", [True, None, 0])
def test_unsaved_failed_block_is_rejected(error):
    report = capture()
    event = copy.deepcopy(report["cases"][0]["events"][-1])
    event.update(index=26, saved=[0, 0], error=error)
    report["cases"][0]["events"].append(event)
    with pytest.raises(ValueError):
        validate_capture_report(report)


def test_unsaved_selected_block_cannot_be_treated_as_omitted():
    report = capture()
    report["cases"][0]["events"][-1]["saved"] = [0, 0]
    with pytest.raises(ValueError, match="missing"):
        validate_capture_report(report)


def test_successful_unsaved_later_block_is_allowed_as_partial_capture():
    report = capture()
    event = copy.deepcopy(report["cases"][0]["events"][-1])
    event.update(index=26, saved=[0, 0])
    report["cases"][0]["events"].append(event)
    assert len(validate_capture_report(report)) == 1


def test_successful_unsaved_setup_call_is_validated():
    report = capture()
    event = copy.deepcopy(report["cases"][0]["events"][-1])
    event.update(index=24, width=1, height=1, saved=[0, 0])
    event["strides"] = [1, 6, 2, 1, 6, 2]
    report["cases"][0]["events"].append(event)
    assert len(validate_capture_report(report)) == 1
    event["error"] = True
    with pytest.raises(ValueError):
        validate_capture_report(report)


@pytest.mark.parametrize(
    "name,value",
    [
        ("../grid.bin", "a" * 64),
        ("/grid.bin", "a" * 64),
        ("case0.rgb-25-grid.bin", "bad"),
        ("case0.rgb-25-grid.bin", True),
        ("case0.rgb-25-grid.bin", "A" * 64),
        ("case0/../grid.bin", "a" * 64),
    ],
)
def test_rejects_unsafe_or_unpinned_artifacts(name, value):
    report = capture()
    report["artifacts"] = {name: value}
    with pytest.raises(ValueError):
        validate_capture_report(report)


@pytest.mark.parametrize(
    "field,value",
    [
        ("index", True),
        ("returncode", False),
        ("returncode", 6),
        ("events", None),
        ("events", [False]),
    ],
)
def test_rejects_invalid_capture_case(field, value):
    report = capture()
    report["cases"][0][field] = value
    with pytest.raises(ValueError):
        validate_capture_report(report)


@pytest.mark.parametrize(
    "field,value",
    [
        ("width", True),
        ("height", 0),
        ("width", 2048),
        ("format", 0),
        ("saved", [True, True]),
        ("error", 0),
        ("strides", [29, 6, 2, 30, 6, 2]),
        ("strides", [30, 6, True, 30, 6, 2]),
        ("strides", None),
    ],
)
def test_rejects_invalid_capture_blocks(field, value):
    report = capture()
    report["cases"][0]["events"][-1][field] = value
    with pytest.raises(ValueError):
        validate_capture_report(report)


def test_rejects_ambiguous_profile_and_kernel_selection():
    for position in (0, 1, 2):
        report = capture()
        report["cases"][0]["events"].append(
            copy.deepcopy(report["cases"][0]["events"][position])
        )
        with pytest.raises(ValueError):
            validate_capture_report(report)
    report = capture()
    report["cases"][0]["events"][0]["index"] = 26
    with pytest.raises(ValueError):
        validate_capture_report(report)
