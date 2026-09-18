"""Tests for bounded NP3 extraction and the public curve command."""

import json
import struct
from pathlib import Path

import pytest

from nx_ctudio.cli import main
from nx_ctudio.picture_control import extract_curve, parse_np3


def container(*records: tuple[int, bytes]) -> bytes:
    return (
        b"NCP\0"
        + b"".join(
            struct.pack(">II", identifier, len(payload)) + payload
            for identifier, payload in records
        )
        + bytes(4)
    )


def curve_record() -> bytes:
    payload = bytearray(578)
    payload[:9] = b"I0\0\xff\0\xff\1\0\2"
    payload[9:13] = bytes((0, 0, 255, 255))
    return bytes(payload)


def sample() -> bytes:
    return container(
        (0x100, b"0310"),
        (0x200, b"nxc-test" + bytes(12)),
        (0x300, b"\0\x20"),
        (0xABCDEF, b"opaque"),
        (2, curve_record()),
    )


def test_unknown_records_survive_bounded_parse() -> None:
    records = parse_np3(sample())
    assert [record.identifier for record in records] == [
        0x100,
        0x200,
        0x300,
        0xABCDEF,
        2,
    ]


def test_extract_curve_returns_native_table() -> None:
    report = extract_curve(sample())
    assert report["version"] == "0310"
    assert report["profile_id"] == 32
    assert report["name"] == "nxc-test"
    assert len(report["curve_samples"]) == 257
    assert report["curve_samples"][128] == 16384


def test_extract_curve_rejects_duplicate_or_missing_semantic_records() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        extract_curve(sample()[:-4] + struct.pack(">II", 0x100, 4) + b"0310" + bytes(4))
    missing_name = container((0x100, b"0310"), (0x300, b"\0\x20"), (2, curve_record()))
    with pytest.raises(ValueError, match="name"):
        extract_curve(missing_name)


@pytest.mark.parametrize("data", [b"", b"NCP\0\0", sample()[:-4], sample() + b"tail"])
def test_malformed_container_fails_closed(data: bytes) -> None:
    with pytest.raises(ValueError):
        parse_np3(data)


def test_cli_json_curve_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "sample.NP3"
    path.write_bytes(sample())
    assert main(["picture-curve", str(path), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "ok"
    assert report["picture_control"]["profile_id"] == 32


def test_cli_curve_failure_is_machine_readable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "broken.NP3"
    path.write_bytes(b"bad")
    assert main(["picture-curve", str(path), "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "error"
    assert report["error"]["code"] == "picture_control_error"
