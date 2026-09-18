"""Tests for bounded NP3 extraction and the public curve command."""

import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from nx_ctudio import picture_control
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


@pytest.mark.parametrize("version,profile", [(b"0100", 32), (b"0310", 65535)])
def test_extraction_rejects_unevidenced_version_and_profile(
    version: bytes, profile: int
) -> None:
    payload = container(
        (0x100, version),
        (0x200, b"nxc-test" + bytes(12)),
        (0x300, struct.pack(">H", profile)),
        (2, curve_record()),
    )
    with pytest.raises(ValueError, match="0310|profile"):
        extract_curve(payload)


def test_cli_rejects_a_device_as_nonregular(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["picture-curve", os.devnull, "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "error"
    assert "regular file" in report["error"]["message"]


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO support required")
def test_cli_rejects_fifo_without_waiting_for_a_writer(tmp_path: Path) -> None:
    path = tmp_path / "fifo.NP3"
    os.mkfifo(path)
    result = subprocess.run(
        [sys.executable, "-m", "nx_ctudio.cli", "picture-curve", str(path), "--json"],
        capture_output=True,
        text=True,
        timeout=3,
        check=False,
    )
    assert result.returncode == 1
    assert "regular file" in json.loads(result.stdout)["error"]["message"]


def test_bounded_reader_rejects_growth_and_closes_its_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "growing.NP3"
    path.write_bytes(sample())
    monkeypatch.setattr(picture_control, "MAX_BYTES", len(sample()))
    original_fstat = os.fstat
    descriptors: list[int] = []

    def grow_after_stat(descriptor: int) -> os.stat_result:
        status = original_fstat(descriptor)
        if not descriptors:
            descriptors.append(descriptor)
            with path.open("ab") as writer:
                writer.write(b"extra data")
        return status

    monkeypatch.setattr(os, "fstat", grow_after_stat)
    with pytest.raises(ValueError, match="limit"):
        picture_control.read_np3(path)
    with pytest.raises(OSError):
        original_fstat(descriptors[0])


@pytest.mark.parametrize("same_size", [False, True])
def test_reader_rejects_mutation_within_the_size_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, same_size: bool
) -> None:
    path = tmp_path / "changing.NP3"
    path.write_bytes(sample())
    original_fstat = os.fstat
    modified = False

    def grow_after_stat(descriptor: int) -> os.stat_result:
        nonlocal modified
        status = original_fstat(descriptor)
        if not modified:
            modified = True
            if same_size:
                with path.open("r+b") as writer:
                    writer.write(b"X")
                os.utime(path, ns=(status.st_atime_ns, status.st_mtime_ns + 1))
            else:
                with path.open("ab") as writer:
                    writer.write(b"extra data")
        return status

    monkeypatch.setattr(os, "fstat", grow_after_stat)
    with pytest.raises(ValueError, match="changed"):
        picture_control.read_np3(path)


def test_reader_accepts_regular_snapshot_at_exact_size_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "exact.NP3"
    payload = sample()
    path.write_bytes(payload)
    monkeypatch.setattr(picture_control, "MAX_BYTES", len(payload))
    assert picture_control.read_np3(path) == payload


def test_reader_rejects_an_oversized_regular_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "oversized.NP3"
    payload = sample()
    path.write_bytes(payload)
    monkeypatch.setattr(picture_control, "MAX_BYTES", len(payload) - 1)
    with pytest.raises(ValueError, match="limit"):
        picture_control.read_np3(path)
