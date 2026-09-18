"""Separate valid CML evidence from numeric equivalence."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from verify_cml import metrics, pixel_inputs, validate_events  # noqa: E402
from verify_np3_header import windows  # noqa: E402


def test_metrics_do_not_wrap_unsigned_differences_or_call_near_matches_equal():
    native = np.array([[65535, 0, 32768]], dtype="<u2")
    vendor = np.array([[0, 1, 32768]], dtype="<u2")
    result = metrics(native, vendor)
    assert result["different_words"] == 2
    assert result["max_absolute_error"] == 65535
    assert result["equivalent"] is False
    assert metrics(native, native)["equivalent"] is True


def test_pixel_inputs_include_boundaries_and_are_reproducible():
    first = pixel_inputs()
    assert first.shape == (8464, 3)
    assert first.dtype == np.dtype("<u2")
    assert first.min() == 0 and first.max() == 65535
    assert np.array_equal(first, pixel_inputs())


def test_lifecycle_requires_intact_inputs_and_release(tmp_path):
    dll = tmp_path / "CML10.dll"
    events = [{"event": "provider", "path_hex": windows(dll).encode().hex()}]
    events += [
        {"event": name, "result": 0}
        for name in (
            "xform_created",
            "input_profile",
            "output_profile",
            "intent",
            "proof",
            "transformer_created",
            "transform_set",
            "input_bits",
            "output_bits",
            "processed",
        )
    ]
    events.append(
        {
            "event": "complete",
            "pixels": 8464,
            "saved": True,
            "inputs_and_guards_intact": True,
            "released": True,
        }
    )
    validate_events(events, dll, 8464)
    for key in ("saved", "inputs_and_guards_intact", "released"):
        events[-1][key] = False
        with pytest.raises(ValueError):
            validate_events(events, dll, 8464)
        events[-1][key] = True
    events[1]["result"] = False
    with pytest.raises(ValueError):
        validate_events(events, dll, 8464)
    events[0]["path_hex"] = b"Z:/shadow/CML10.dll".hex()
    with pytest.raises(ValueError, match="provider"):
        validate_events(events, dll, 8464)


def test_snapshot_lifecycle_rejects_other_kernels_and_nested_booleans(tmp_path):
    dll = tmp_path / "CML10.dll"
    events = [{"event": "provider", "path_hex": windows(dll).encode().hex()}]
    events += [
        {"event": name, "result": 0}
        for name in (
            "xform_created",
            "input_profile",
            "output_profile",
            "intent",
            "proof",
            "transformer_created",
            "transform_set",
            "input_bits",
            "output_bits",
            "processed",
        )
    ]
    state = {
        "event": "cml_state",
        "dimensions": [32, 32, 32],
        "channels": [3, 3],
        "mode": 1,
        "entry_rva": 0x1F320,
        "inner_rva": 0x245C0,
        "flags": [1, 0, 0, 0],
        "identity_masks": [7, 0],
        "mxcsr_rounding": [0, 0],
    }
    events.extend(
        [
            state,
            {"event": "snapshots", "result": 0},
            {
                "event": "complete",
                "pixels": 8464,
                "saved": True,
                "inputs_and_guards_intact": True,
                "released": True,
            },
        ]
    )
    validate_events(events, dll, 8464, True)
    state["mode"] = 0
    with pytest.raises(ValueError):
        validate_events(events, dll, 8464, True)
    state["mode"] = 1
    state["flags"][0] = True
    with pytest.raises(ValueError):
        validate_events(events, dll, 8464, True)
