"""Bound the Z8 source metadata import and reject misleading oracle evidence."""

import copy
import struct
import sys
from pathlib import Path

import pytest

from nx_ctudio import _tiff

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from color_inputs import (  # noqa: E402
    rational_gains,
    source_blend_inputs,
    source_white_balance,
    white_balance_levels,
)
from verify_colorbalance import validate_color_events  # noqa: E402
from verify_np3_header import windows  # noqa: E402


def source():
    maker = _tiff.Directory({})
    maker.set(12, 5, struct.pack("<8I", 1536, 512, 2048, 512, 768, 512, 1024, 512))
    maker.set(59, 5, struct.pack("<8I", *([512, 512] * 4)))
    maker.set(0xB0, 7, b"0103" + bytes(12))
    exif = _tiff.Directory({37500: _tiff.Tag(37500, 7, 0, b"", [maker])})
    root = _tiff.Directory({34665: _tiff.Tag(34665, 4, 1, bytes(4), [exif])})
    root.set(272, 2, b"NIKON Z 8\0")
    return root, maker


def events(rome):
    return [
        {
            "event": "white_balance",
            "rome_path": windows(rome),
            "status": 1,
            "guards_intact": True,
            "values": [3.0, 1.75, 4.0],
        },
        *[
            {
                "event": "color_tag",
                "key": key,
                "success": True,
                "guards_intact": True,
                "hex": struct.pack("<4d", *values).hex(),
            }
            for key, values in (
                (0x8000C, (3, 4, 1.5, 2)),
                (0x8003B, (1, 1, 1, 1)),
                (0x80E17, (3, 4, 1.5, 2)),
            )
        ],
        {
            "event": "color_tag",
            "key": 0x80E2D,
            "success": False,
            "guards_intact": True,
            "hex": "a5" * 16,
        },
    ]


def test_source_gains_preserve_rb_and_average_green():
    root, _ = source()
    assert source_white_balance(root) == (3.0, 1.75, 4.0)
    assert white_balance_levels(source_white_balance(root)) == (1536, 2048)
    assert white_balance_levels((513 / 511, 2, 1.999)) == (514, 1023)


@pytest.mark.parametrize("damage", ["model", "missing", "override", "b0", "fine"])
def test_unsupported_metadata_fails_closed(damage):
    root, maker = source()
    if damage == "model":
        root.set(272, 2, b"NIKON Z 9\0")
    elif damage == "missing":
        del maker.tags[12]
    elif damage == "override":
        maker.set(0xE17, 7, bytes(32))
    elif damage == "b0":
        maker.set(0xB0, 7, b"0100" + bytes(12))
    else:
        maker.set(59, 5, struct.pack("<8I", *([1024, 512] * 4)))
    with pytest.raises(ValueError):
        source_white_balance(root)


@pytest.mark.parametrize(
    "tag",
    [
        _tiff.Tag(12, 4, 8, bytes(32)),
        _tiff.Tag(12, 5, 3, bytes(24)),
        _tiff.Tag(12, 5, 4, bytes(31)),
        _tiff.Tag(12, 5, 4, bytes(32)),
        _tiff.Tag(12, 5, 4, struct.pack("<8I", *([0, 512] * 4))),
        _tiff.Tag(12, 5, 4, struct.pack("<8I", *([65536, 512] * 4))),
    ],
)
def test_bad_rational_gains_are_rejected(tag):
    with pytest.raises(ValueError):
        rational_gains(tag)


@pytest.mark.parametrize("value", [True, 0, -1, 128, float("nan"), float("inf")])
def test_bad_levels_are_rejected(value):
    with pytest.raises(ValueError):
        white_balance_levels((value, 1, 1))


def test_color_event_validation(tmp_path):
    root, _ = source()
    rome = tmp_path / "Rome2.dll"
    result = validate_color_events(events(rome), root, rome)
    assert result["native_rgb"] == (3.0, 1.75, 4.0)
    assert result["native_top_wb_levels"] == (1536, 2048)


@pytest.mark.parametrize(
    "index,field,value",
    [
        (0, "guards_intact", False),
        (0, "status", 0),
        (0, "status", True),
        (0, "values", [3, 1, 4]),
        (0, "rome_path", "Z:/shadow/Rome2.dll"),
        (1, "hex", "00" * 32),
        (2, "success", False),
        (3, "guards_intact", False),
        (3, "guards_intact", 1),
        (4, "success", True),
        (4, "hex", "00" * 16),
    ],
)
def test_invalid_oracle_evidence_is_rejected(tmp_path, index, field, value):
    root, _ = source()
    rome = tmp_path / "Rome2.dll"
    damaged = copy.deepcopy(events(rome))
    damaged[index][field] = value
    with pytest.raises(ValueError):
        validate_color_events(damaged, root, rome)


def test_processing_override_is_not_treated_as_absent(tmp_path):
    root, maker = source()
    maker.set(0xE2D, 7, bytes(16))
    rome = tmp_path / "Rome2.dll"
    with pytest.raises(ValueError, match="processing"):
        validate_color_events(events(rome), root, rome)


@pytest.mark.parametrize(
    "name,mode", [(b"AUTO1       \0", 17), (b"AUTO0       \0", 20)]
)
@pytest.mark.parametrize("value", [-32768, -52, 0, 1834, 32767])
def test_source_blend_fields_preserve_signed_words_and_order(name, mode, value):
    root, maker = source()
    maker.set(5, 2, name)
    decoded = bytearray(b"0803" + bytes(162))
    matrix = [-32768, -2, -1, 0, 1, 2, 4, 16384, 32767]
    struct.pack_into("<h", decoded, 0x3A, value)
    struct.pack_into("<9h", decoded, 0x90, *matrix)
    assert source_blend_inputs(root, bytes(decoded)) == {
        "correction_matrix": matrix,
        "wb_levels_signed": [1536, 2048],
        "wb_mode": mode,
        "wb_value": value,
    }


@pytest.mark.parametrize(
    "name,kind", [(b"AUTO2       \0", 2), (b"AUTO1\0", 2), (b"AUTO1       \0", 7)]
)
def test_unknown_or_malformed_blend_mode_is_rejected(name, kind):
    root, maker = source()
    maker.set(5, kind, name)
    with pytest.raises(ValueError, match="AUTO1/AUTO0"):
        source_blend_inputs(root, b"0803" + bytes(162))
