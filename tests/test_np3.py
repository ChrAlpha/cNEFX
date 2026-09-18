"""Bounded NP3 record parsing independent of external reference samples."""

import importlib.util
import struct
import sys
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "tools/inspect_np3.py"
SPEC = importlib.util.spec_from_file_location("nxc_np3_research", PATH)
assert SPEC and SPEC.loader
np3 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = np3
SPEC.loader.exec_module(np3)


def container(*records: tuple[int, bytes]) -> bytes:
    return (
        b"NCP\0"
        + b"".join(
            struct.pack(">II", identifier, len(payload)) + payload
            for identifier, payload in records
        )
        + bytes(4)
    )


def test_unknown_records_order_and_payload_are_preserved() -> None:
    data = container(
        (0x100, b"0310"), (0xABCDEF, b"opaque\0payload"), (0x800, b"\xff\x04")
    )
    records = np3.parse(data)
    assert [r.identifier for r in records] == [0x100, 0xABCDEF, 0x800]
    assert np3.serialize(records) == data


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"NP3\0" + bytes(4),
        b"NCP\0\0",
        b"NCP\0" + struct.pack(">I", 3),
        b"NCP\0" + struct.pack(">II", 3, 0xFFFFFFFF),
        container((1, b"x"))[:-4],
        container((1, b"x")) + b"trailing",
    ],
)
def test_malformed_containers_fail_closed(data: bytes) -> None:
    with pytest.raises(ValueError):
        np3.parse(data)


def test_record_budget_prevents_unbounded_empty_fields() -> None:
    data = container(*[(7, b"")] * (np3.MAX_RECORDS + 1))
    with pytest.raises(ValueError, match="Too many"):
        np3.parse(data)


def test_duplicate_semantics_are_rejected_without_losing_raw_roundtrip() -> None:
    data = container((0x100, b"0310"), (0x100, b"0310"))
    records = np3.parse(data)
    assert np3.serialize(records) == data
    with pytest.raises(ValueError, match="Duplicate"):
        np3.describe(records)


def test_interpretation_requires_observed_version_and_flexible_profile() -> None:
    with pytest.raises(ValueError, match="0310"):
        np3.describe(np3.parse(container((0x100, b"0300"))))
    with pytest.raises(ValueError, match="Flexible Color"):
        np3.describe(np3.parse(container((0x100, b"0310"), (0x300, b"\0\1"))))


def test_comment_decoder_requires_valid_terminated_utf8() -> None:
    assert np3._text(b"hello\0\0", "utf-8") == "hello"
    for payload in (b"hello", b"hello\0x", b"\xff\0"):
        with pytest.raises(ValueError):
            np3._text(payload, "utf-8")


def flexible_records(overrides: dict[int, bytes]) -> tuple:
    fields = {
        0x100: b"0310",
        0x200: b"test" + bytes(16),
        0x300: b"\0\x20",
        **{
            identifier: bytes((128, scale))
            for identifier, (_, scale) in np3.PARAMETERS.items()
        },
        0x1F00: bytes([128] * 24 + [1, 1, 1, 0]),
        0x2000: b"\x80\0\x80\x80" * 3 + bytes((1, 1, 1, 0, 178, 1, 128, 1)),
    }
    fields.update(overrides)
    return tuple(
        np3.Record(identifier, payload) for identifier, payload in fields.items()
    )


@pytest.mark.parametrize("encoded", [0, 1, 255])
def test_inspector_preserves_special_values_as_sentinels(encoded: int) -> None:
    report = np3.describe(flexible_records({0x1900: bytes((encoded, 0))}))
    assert report["parameters"]["contrast"] == {
        "encoded_value": encoded,
        "scale_byte": 0,
        "value": None,
        "state": "sentinel",
    }


def test_inspector_honors_flexible_scales_and_signed_hues() -> None:
    records = flexible_records(
        {
            0x1900: bytes((119, 4)),
            0x1F00: bytes((119, 137, 129)) * 8 + bytes((4, 4, 3, 0)),
            0x2000: b"\x7f\xf7\x89\x77" * 3 + bytes((4, 4, 4, 0, 137, 4, 119, 4)),
        }
    )
    report = np3.describe(records)
    assert report["parameters"]["contrast"]["value"] == -2
    assert report["color_blender"]["red"] == {"hue": -2, "chroma": 2, "brightness": 0}
    assert report["color_grading"]["highlights"] == {
        "hue": -2,
        "chroma": 2,
        "brightness": -2,
    }
    assert report["color_grading"]["blending"] == 2
    assert report["color_grading"]["balance"] == -2


def test_unverified_base_scale_has_no_invented_numeric_value() -> None:
    report = np3.describe(flexible_records({0x600: bytes((136, 255))}))
    assert report["parameters"]["sharpening"]["value"] is None
    assert report["parameters"]["sharpening"]["state"] == "unverified_scale"


def test_zero_scale_does_not_prevent_raw_roundtrip() -> None:
    records = flexible_records({0x1900: bytes((137, 0))})
    assert np3.parse(np3.serialize(records)) == records
    with pytest.raises(ValueError, match="zero scale"):
        np3.describe(records)
