"""Reject incomplete or ambiguous SDK render evidence."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from verify_image_render import record_process, validate_render_events  # noqa: E402
from verify_image_sdk import PINNED, provider_paths  # noqa: E402
from verify_np3_header import windows  # noqa: E402


def test_truncated_process_output_retains_diagnostics():
    stdout = '{"event":"initialized","result":0}\n{"event":"render_'
    result = subprocess.CompletedProcess([], 11, stdout, "render process crashed")
    case = {}
    with pytest.raises(json.JSONDecodeError):
        record_process(case, result)
    assert case == {
        "stdout": stdout,
        "stderr": "render process crashed",
        "returncode": 11,
        "events": [{"event": "initialized", "result": 0}],
    }


def events(profile):
    values = [0] * 41
    result = []
    for name in (
        "initialized",
        "created",
        "services",
        "metadata_service",
        "metadata_start",
        "metadata_complete",
        "picture_imported",
        "render_service",
        "composer_service",
        "render_composed",
        "render_setup",
        "render_profile",
        "render_profile_path",
        "render_scale",
        "render_bits",
        "render_size",
        "render_target",
        "render_dependencies_released",
        "render_started",
        "render_waited",
        "render_released",
        "render_complete",
        "closed",
    ):
        event = {"event": name}
        if name not in (
            "closed",
            "render_released",
            "render_dependencies_released",
            "render_complete",
            "render_profile_path",
        ):
            event["result"] = 0
        if name in ("created", "services", "metadata_service"):
            event["present"] = True
        if name == "metadata_complete":
            event.update(complete=True, steps=17)
        if name == "picture_imported":
            event.update(guards_intact=True, input_unchanged=True, values=values)
        if name == "render_profile_path":
            event["path_hex"] = windows(profile).encode().hex()
        if name == "render_complete":
            event.update(
                width=258,
                height=172,
                format=1,
                bits=16,
                scale=0.03125,
                bytes=258 * 172 * 6,
                calls=3,
                covered=258 * 172,
                missing=0,
                overlaps=0,
                rejected=0,
                target_released=1,
                target_destroyed=0,
                guards_intact=True,
                metadata_intact=True,
                saved=True,
            )
        result.append(event)
    return result, {"normalized_values": values}


def test_complete_render_evidence(tmp_path):
    profile = tmp_path / "NKsRGB.icm"
    output, picture = events(profile)
    assert validate_render_events(output, picture, profile)["bytes"] == 266256


@pytest.mark.parametrize(
    "field,value",
    [
        ("covered", 44375),
        ("missing", 1),
        ("overlaps", 1),
        ("rejected", 1),
        ("calls", 0),
        ("calls", 44377),
        ("target_released", 0),
        ("target_released", 2),
        ("target_destroyed", 1),
        ("guards_intact", 1),
        ("metadata_intact", False),
        ("saved", False),
        ("width", True),
        ("height", 1025),
        ("bytes", 1),
        ("format", True),
        ("bits", 8),
        ("scale", 1),
    ],
)
def test_invalid_coverage_or_target_metadata_fails(tmp_path, field, value):
    profile = tmp_path / "NKsRGB.icm"
    output, picture = events(profile)
    output[-2][field] = value
    with pytest.raises(ValueError):
        validate_render_events(output, picture, profile)


@pytest.mark.parametrize("index", [5, 6, 16, 17, 18, 19, 20, 22])
def test_missing_lifecycle_event_fails(tmp_path, index):
    profile = tmp_path / "NKsRGB.icm"
    output, picture = events(profile)
    del output[index]
    with pytest.raises(ValueError, match="lifecycle"):
        validate_render_events(output, picture, profile)


def test_nested_boolean_and_wrong_profile_fail(tmp_path):
    profile = tmp_path / "NKsRGB.icm"
    output, picture = events(profile)
    damaged = copy.deepcopy(output)
    damaged[6]["values"][0] = False
    with pytest.raises(ValueError):
        validate_render_events(damaged, picture, profile)
    output[12]["path_hex"] = b"Z:/shadow/NKsRGB.icm".hex()
    with pytest.raises(ValueError, match="profile"):
        validate_render_events(output, picture, profile)


def test_failed_diagnostic_and_boolean_status_fail(tmp_path):
    profile = tmp_path / "NKsRGB.icm"
    output, picture = events(profile)
    with pytest.raises(ValueError, match="diagnostic"):
        validate_render_events(
            output + [{"event": "sdk_log", "code": 12}], picture, profile
        )
    output[18]["result"] = False
    with pytest.raises(ValueError):
        validate_render_events(output, picture, profile)


def test_render_provider_is_required_and_cannot_be_shadowed(tmp_path):
    vendor, wine = tmp_path / "vendor", tmp_path / "wine"
    hashes = {str(vendor / name): value for name, value in PINNED.items()}
    output = [
        {"event": "provider", "path_hex": windows(Path(path)).encode().hex()}
        for path in hashes
    ]
    provider_paths(output, hashes, vendor, wine)
    with pytest.raises(ValueError, match="Missing"):
        provider_paths(output[:-1], hashes, vendor, wine)
    output[-1]["path_hex"] = b"Z:/shadow/image_render.n5m".hex()
    with pytest.raises(ValueError, match="Unexpected"):
        provider_paths(output, hashes, vendor, wine)
