"""Capture identity and exact exposure metadata requirements."""

from types import SimpleNamespace

import pytest

from nx_ctudio.raw import _capture_metadata


def exif() -> dict:
    return {
        "BodySerialNumber": "test-body",
        "DateTimeOriginal": "2026:09:18 10:20:30",
        "SubsecTimeOriginal": "125",
        "ExposureTime": (10, 200),
        "FNumber": (800, 100),
        "ExposureBiasValue": (0, 1),
    }


def test_equivalent_rationals_and_consecutive_shutters_normalize() -> None:
    metadata = exif()
    first = _capture_metadata(
        metadata, {0xA7: SimpleNamespace(dtype=4, count=1, value=100)}, 1
    )
    metadata["ExposureTime"] = (1, 20)
    second = _capture_metadata(
        metadata, {0xA7: SimpleNamespace(dtype=4, count=1, value=101)}, 2
    )
    assert first == second
    assert first["capture_time"] == "2026-09-18T10:20:30.125000"
    assert first["exposure_time"] == (1, 20)
    assert first["shutter_sequence_anchor"] == 99


@pytest.mark.parametrize(
    "key", ["BodySerialNumber", "DateTimeOriginal", "ExposureTime", "FNumber"]
)
def test_missing_identity_or_exposure_fails(key: str) -> None:
    metadata = exif()
    del metadata[key]
    with pytest.raises(ValueError):
        _capture_metadata(
            metadata, {0xA7: SimpleNamespace(dtype=4, count=1, value=100)}, 1
        )


def test_camera_identity_does_not_publish_serial() -> None:
    metadata = exif()
    tags = {0xA7: SimpleNamespace(dtype=4, count=1, value=100)}
    first = _capture_metadata(metadata, tags, 1)
    metadata["BodySerialNumber"] = "other-body"
    second = _capture_metadata(metadata, tags, 1)
    assert first["camera_identity_sha256"] != second["camera_identity_sha256"]
    assert "test-body" not in str(first)
