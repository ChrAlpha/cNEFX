"""Check metadata arithmetic and the source/active TOPParam bank boundary."""

import hashlib
import struct
import sys
from pathlib import Path

import pytest
from test_np3 import flexible_records, np3

from nx_ctudio import _tiff

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import image_controls  # noqa: E402
from image_controls import source_controls, top_controls, verify_top  # noqa: E402
from verify_image_sdk import (  # noqa: E402
    PINNED,
    picture_controls,
    provider_paths,
    validate_events,
)
from verify_np3_header import windows  # noqa: E402


def source():
    maker = _tiff.Directory({})
    maker.set(12, 5, struct.pack("<8I", 977, 512, 787, 512, 512, 512, 512, 512))
    maker.set(59, 5, struct.pack("<8I", *([512, 512] * 4)))
    maker.set(0xB0, 7, b"0103" + bytes(12))
    exif = _tiff.Directory({37500: _tiff.Tag(37500, 7, 0, b"", [maker])})
    root = _tiff.Directory({34665: _tiff.Tag(34665, 4, 1, bytes(4), [exif])})
    root.set(272, 2, b"NIKON Z 8\0")
    return root


@pytest.mark.parametrize(
    "iso,offset,bucket",
    [
        (0, 108, 6),
        (72, 0, 6),
        (76, 1, 6),
        (80, 2, 0),
        (83, 2, 0),
        (84, 3, 1),
        (104, 8, 6),
        (231, 39, 6),
        (1096, 0, 6),
        (1104, 2, 0),
    ],
)
def test_iso_truncates_to_byte_before_clamping(monkeypatch, iso, offset, bucket):
    coefficients = bytearray(422416)
    coefficients[0x830 : 0x830 + 109] = bytes(range(109))
    coefficients = bytes(coefficients)
    monkeypatch.setattr(
        image_controls, "COEFFICIENT_HASH", hashlib.sha256(coefficients).hexdigest()
    )
    decoded = bytearray(b"0803" + bytes(162))
    struct.pack_into("<H", decoded, 0x8A, iso)
    result = source_controls(source(), bytes(decoded), coefficients)
    assert result["iso_offset"] == offset
    assert result["iso_bucket"] == bucket
    assert result["wb_levels"] == [977, 787]
    assert result["processing"] == result["mode"] == 0


def test_unpinned_resource_is_rejected():
    with pytest.raises(ValueError, match="pinned"):
        source_controls(source(), b"0803" + bytes(162), bytes(422416))


def test_bank_offsets_preserve_shared_metadata():
    data = bytearray(0x4D0)
    struct.pack_into("<3d", data, 0x178, 2.0, 1.0, 3.0)
    struct.pack_into("<2H", data, 0x1D0, 1024, 1536)
    struct.pack_into("<H", data, 0x198, 104)
    for bank, value in ((0, 1), (0x250, 2)):
        data[bank + 0x200] = value
        data[bank + 0x23C : bank + 0x23F] = bytes([value + 3, 6, value + 4])
        data[bank + 0x246] = 20
        struct.pack_into("<h", data, bank + 0x248, -128)
        data[bank + 0x24A : bank + 0x24C] = bytes([6, 12])
        struct.pack_into("<I", data, bank + 0x208, value + 10)
    for active, value in ((False, 1), (True, 2)):
        result = top_controls(bytes(data), active=active)
        assert result["rgb"] == [2, 1, 3]
        assert result["iso_code"] == 104
        assert result["profile_row"] == value + 10
        assert result["camera_index"] == value + 4
        assert result["selector_inputs"] == [
            1024,
            1536,
            value,
            6,
            12,
            -128,
            20,
            value + 3,
            0,
            0,
        ]


@pytest.mark.parametrize("data", [b"", bytes(0x4CF), bytes(0x4D1), bytearray(0x4D0)])
def test_top_snapshot_size_and_type_are_strict(data):
    with pytest.raises(ValueError):
        top_controls(data, active=False)


def test_wrong_metadata_does_not_reach_selector():
    with pytest.raises(ValueError, match="Metadata"):
        verify_top(bytes(0x4D0), {"iso_code": 104}, b"", b"", active=True)


def lifecycle():
    return [
        {"event": "initialized", "result": 0},
        {"event": "created", "result": 0, "present": True},
        {"event": "services", "result": 0, "present": True},
        {"event": "metadata_service", "result": 0, "present": True},
        {"event": "metadata_start", "result": 0},
        {"event": "metadata_complete", "result": 0, "complete": True, "steps": 17},
        {"event": "top_initialized", "result": 0},
        {"event": "top_applied", "result": 0},
        {"event": "top_guards", "intact": True, "saved": True},
        {"event": "closed"},
    ]


def test_sdk_lifecycle_requires_completion_and_cleanup():
    validate_events(lifecycle())
    for field in ("complete", "result"):
        events = lifecycle()
        events[5][field] = False if field == "complete" else 1
        with pytest.raises(ValueError):
            validate_events(events)
    with pytest.raises(ValueError):
        validate_events(lifecycle()[:-1])
    with pytest.raises(ValueError):
        validate_events(lifecycle() + [{"event": "sdk_log", "code": 12}])


def test_guards_must_be_boolean():
    events = lifecycle()
    events[8]["intact"] = 1
    with pytest.raises(ValueError):
        validate_events(events)


def test_providers_require_loaded_paths_and_handle_wine_zlib(tmp_path):
    vendor, wine = tmp_path / "vendor", tmp_path / "wine"
    hashes = {str(vendor / name): value for name, value in PINNED.items()}
    zlib = wine / "drive_c/windows/system32/zlib1.dll"
    hashes[str(zlib)] = "0" * 64
    events = [
        {"event": "provider", "path_hex": windows(Path(path)).encode().hex()}
        for path in hashes
        if path != str(zlib)
    ]
    events.append(
        {"event": "provider", "path_hex": b"C:/windows/system32/zlib1.dll".hex()}
    )
    assert str(zlib) in provider_paths(events, hashes, vendor, wine)
    with pytest.raises(ValueError, match="Missing"):
        provider_paths(events[1:], hashes, vendor, wine)
    events[0]["path_hex"] = b"Z:/shadow/Rome2.dll".hex()
    with pytest.raises(ValueError, match="Unexpected"):
        provider_paths(events, hashes, vendor, wine)


def picture(curve=False):
    overrides = {0x1900: bytes((119, 4)), 0x1C00: bytes((137, 3))}
    if curve:
        payload = bytearray(578)
        payload[:13] = b"I0" + bytes([0, 255, 0, 255, 1, 0, 2, 0, 0, 255, 255])
        overrides[2] = bytes(payload)
    return np3.serialize(flexible_records(overrides))


@pytest.mark.parametrize("curve", [False, True])
def test_picture_expectations_use_imported_profile_and_curve_presence(curve):
    result = picture_controls(picture(curve))
    assert result["picture_control_id"] == 32
    assert result["profile_row"] == 14
    assert result["profile_family"] == 12
    assert result["normalized_values"][:6] == [-2, 0, 0, 3, 0, 0]
    assert result["normalized_values"][-2:] == [50, 0]
    assert result["custom_curve"] is curve


def test_picture_lifecycle_requires_guarded_import_before_top():
    expected = picture_controls(picture())
    event = {
        "event": "picture_imported",
        "result": 0,
        "guards_intact": True,
        "input_unchanged": True,
        "values": expected["normalized_values"],
    }
    events = lifecycle()
    events.insert(6, event)
    validate_events(events, expected)
    for field in ("input_unchanged", "guards_intact"):
        with pytest.raises(ValueError):
            validate_events(
                events[:6] + [dict(event, **{field: False})] + events[7:], expected
            )
    with pytest.raises(ValueError):
        validate_events(lifecycle(), expected)
    event["values"] = event["values"][:-1] + [True]
    with pytest.raises(ValueError):
        validate_events(events, expected)


def test_top_picture_values_and_curve_flag_use_the_selected_bank():
    data = bytearray(0x4D0)
    source_values = list(range(41))
    working_values = list(range(-20, 21))
    struct.pack_into("<41i", data, 0xCC, *source_values)
    struct.pack_into("<41i", data, 0x31C, *working_values)
    data[0x170], data[0x3C0] = 0, 1
    source_result = top_controls(bytes(data), active=False)
    active_result = top_controls(bytes(data), active=True)
    assert source_result["normalized_values"] == source_values
    assert source_result["custom_curve"] is False
    assert active_result["normalized_values"] == working_values
    assert active_result["custom_curve"] is True
