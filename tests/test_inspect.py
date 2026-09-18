"""Exercise the inspection contract without implying pixel-shift reconstruction."""

import hashlib
import json
import shutil
import struct
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from nx_ctudio.cli import main
from nx_ctudio.inspect import inspect_file


def minimal_tiff(path: Path) -> bytes:
    """Create a tiny uncompressed TIFF, independent of the reader under test."""
    entries = [
        (256, 4, 1, 2),
        (257, 4, 1, 2),
        (258, 3, 1, 8),
        (259, 3, 1, 1),
        (262, 3, 1, 1),
        (273, 4, 1, 122),
        (277, 3, 1, 1),
        (278, 4, 1, 2),
        (279, 4, 1, 4),
    ]
    data = b"II\x2a\x00\x08\x00\x00\x00" + struct.pack("<H", len(entries))
    data += b"".join(struct.pack("<HHII", *entry) for entry in entries)
    data += b"\x00\x00\x00\x00\x01\x02\x03\x04"
    path.write_bytes(data)
    return data


@pytest.fixture
def exiftool() -> str:
    executable = shutil.which("exiftool")
    assert executable, "Install ExifTool to run the reader integration tests."
    return executable


def test_real_reader_is_read_only_and_preserves_reported_metadata(
    tmp_path: Path, exiftool: str
) -> None:
    path = tmp_path / "-input with spaces.NEF"
    original = minimal_tiff(path)
    result = inspect_file(path, exiftool)
    assert result["status"] == "ok"
    assert result["sha256"] == hashlib.sha256(original).hexdigest()
    # ExifTool labels TIFF containers using the suffix. This is metadata evidence,
    # not proof of a valid NEF; the CLI explicitly leaves RAW validation pending.
    assert result["metadata"]["File:FileType"] == "NEF"
    assert result["metadata"]["IFD0:ImageWidth"] == 2
    assert path.read_bytes() == original
    assert "Nikon:PixelShiftActive" not in result["metadata"]


def test_cli_preserves_input_order_and_reports_partial_failure(
    tmp_path: Path, exiftool: str, capsys: pytest.CaptureFixture[str]
) -> None:
    good = tmp_path / "valid.NEF"
    bad = tmp_path / "broken.NEF"
    minimal_tiff(good)
    bad.write_bytes(b"not a raw image")
    assert main(["inspect", str(good), str(bad), "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["schema_version"] == 1
    assert report["tool"]["name"] == "NX Ctudio"
    assert [item["path"] for item in report["files"]] == [str(good), str(bad)]
    assert [item["status"] for item in report["files"]] == ["ok", "error"]
    assert report["sequence_validation"] == "not_performed"
    assert report["raw_decode"] == "not_performed"


def test_missing_file_is_a_structured_failure(tmp_path: Path, exiftool: str) -> None:
    result = inspect_file(tmp_path / "missing.NEF", exiftool)
    assert result["status"] == "error"
    assert result["error"]["code"] == "input_error"


def test_directory_is_not_read_as_a_sequence(tmp_path: Path, exiftool: str) -> None:
    result = inspect_file(tmp_path, exiftool)
    assert result["status"] == "error"
    assert result["error"]["code"] == "input_error"


@pytest.mark.parametrize(
    "payload",
    ["not json", "{}", "[]", "[null]", "[{}, {}]", '[{"File:FileType": ["NEF"]}]'],
)
def test_invalid_tool_output_fails_closed(
    tmp_path: Path, exiftool: str, payload: str
) -> None:
    path = tmp_path / "input.NEF"
    minimal_tiff(path)
    with patch("nx_ctudio.inspect.subprocess.run") as run:
        run.return_value = subprocess.CompletedProcess([], 0, payload, "")
        result = inspect_file(path, exiftool)
    assert result["status"] == "error"
    assert result["error"]["code"] == "metadata_error"


def test_timeout_is_reported(tmp_path: Path, exiftool: str) -> None:
    path = tmp_path / "input.NEF"
    minimal_tiff(path)
    with patch("nx_ctudio.inspect.subprocess.run") as run:
        run.side_effect = subprocess.TimeoutExpired("exiftool", 30)
        result = inspect_file(path, exiftool)
    assert result["error"]["code"] == "metadata_timeout"


def test_changed_input_cannot_produce_success(tmp_path: Path, exiftool: str) -> None:
    path = tmp_path / "input.NEF"
    minimal_tiff(path)

    def change_input(
        *args: object, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        path.write_bytes(b"changed during inspection")
        return subprocess.CompletedProcess([], 0, '[{"File:FileType": "TIFF"}]', "")

    with patch("nx_ctudio.inspect.subprocess.run", side_effect=change_input):
        result = inspect_file(path, exiftool)
    assert result["error"]["code"] == "input_changed"


def test_missing_exiftool_is_machine_readable(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with patch("nx_ctudio.cli.shutil.which", return_value=None):
        assert main(["inspect", "input.NEF", "--json"]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["error"]["code"] == "dependency_missing"


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.startswith("NX Ctudio ")


def test_cancellation_preserves_completed_human_output(
    exiftool: str, capsys: pytest.CaptureFixture[str]
) -> None:
    completed = {"path": "first.NEF", "status": "ok", "sha256": "abc", "metadata": {}}
    with patch(
        "nx_ctudio.cli.inspect_file", side_effect=[completed, KeyboardInterrupt]
    ):
        assert main(["inspect", "first.NEF", "second.NEF"]) == 130
    output = capsys.readouterr()
    assert "ok: first.NEF" in output.out
    assert "Inspection cancelled." in output.err
