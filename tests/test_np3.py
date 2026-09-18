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
