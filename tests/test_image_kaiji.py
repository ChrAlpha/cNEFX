"""Verify application contrast suppression and factory acceptance boundaries."""

import struct
import sys
from pathlib import Path

import pytest
from test_image_controls import lifecycle, picture

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from image_kaiji import application_settings, verify_state  # noqa: E402
from verify_image_sdk import picture_controls, validate_events  # noqa: E402


@pytest.mark.parametrize("curve,contrast", [(False, 159), (True, 0)])
def test_curve_suppresses_effective_contrast_but_retains_other_settings(
    curve, contrast
):
    values = [62, -75, -11, -65, 67, 45] + [0] * 35
    before = values.copy()
    assert application_settings(values, curve) == {
        "contrast": contrast,
        "highlights": -90,
        "shadows": -13,
        "white": -65,
        "black": 67,
        "saturation": 45,
        "saturation_bias": 23,
        "custom_curve": int(curve),
    }
    assert values == before


@pytest.mark.parametrize("value,expected", [(-100, -256), (0, 0), (100, 256)])
def test_manual_contrast_endpoints(value, expected):
    values = [value] + [0] * 40
    assert application_settings(values, False)["contrast"] == expected
    assert application_settings(values, True)["contrast"] == 0


@pytest.mark.parametrize("values", [[0] * 40, [True] + [0] * 40, [101] + [0] * 40])
def test_invalid_normalized_controls_are_rejected(values):
    with pytest.raises(ValueError, match="normalized"):
        application_settings(values, False)


def test_wrong_parameter_layout_is_rejected_before_table_generation():
    with pytest.raises(ValueError, match="snapshot"):
        verify_state(bytes(0x17F7), [0] * 41, {}, 8)


@pytest.mark.parametrize(
    "offset,code,value,name",
    [
        (0xE3C, "I", 256, "log_mode"),
        (0x153E, "h", 511, "raw153e"),
        (0x1528, "H", 256, "blend mix"),
    ],
)
def test_parameter_guards_check_full_storage_width(offset, code, value, name):
    data = bytearray(0x17F8)
    for at, kind, initial in [
        (0xC, "I", 1071),
        (0x1518, "I", 14),
        (0x151C, "I", 12),
        (0x1520, "H", 256),
        (0x1522, "H", 12),
        (0x1526, "H", 16),
        (0x152A, "H", 8),
        (0x1538, "H", 128),
        (0x153E, "h", 255),
        (0x154A, "B", 6),
        (0x17E8, "H", 50),
    ]:
        struct.pack_into("<" + kind, data, at, initial)
    values = [0] * 39 + [50, 0]
    settings = application_settings(values, False)
    verify_state(bytes(data), values, settings, 8)
    struct.pack_into("<" + code, data, offset, value)
    with pytest.raises(ValueError, match=name):
        verify_state(bytes(data), values, settings, 8)


def test_factory_requires_cleanup_and_a_supported_engine():
    expected = picture_controls(picture())
    events = lifecycle()
    events.insert(
        6,
        {
            "event": "picture_imported",
            "result": 0,
            "guards_intact": True,
            "input_unchanged": True,
            "values": expected["normalized_values"],
        },
    )
    factory = [
        {"event": "kaiji_cms", "result": 0, "present": True},
        {"event": "kaiji_config", "result": 0},
        {"event": "kaiji_engine", "value": 9},
        {"event": "kaiji_tables", "guards_intact": True, "saved": True},
        {"event": "kaiji_released", "guards_intact": True},
    ]
    events[-1:-1] = factory
    validate_events(events, expected, kaiji=True)
    with pytest.raises(ValueError):
        validate_events(events[:-2] + events[-1:], expected, kaiji=True)
    for index, key, value in (
        (1, "result", 1),
        (2, "value", 8),
        (3, "saved", False),
        (4, "guards_intact", False),
    ):
        prior = factory[index][key]
        factory[index][key] = value
        with pytest.raises(ValueError):
            validate_events(events, expected, kaiji=True)
        factory[index][key] = prior
    with pytest.raises(ValueError, match="requires"):
        validate_events(events, kaiji=True)
