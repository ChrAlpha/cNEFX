"""Sequence rejection and output publication boundaries."""

from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest

from nx_ctudio.cli import main
from nx_ctudio.merge import CONSISTENT, merge, validate_sequence


def sequence() -> list[dict]:
    first = dict.fromkeys(CONSISTENT, 1)
    first.update(expected_count=4, shot_number=1)
    return [dict(first, shot_number=i) for i in range(1, 5)]


def test_consecutive_sequence_does_not_require_a_fixed_numbering_origin() -> None:
    metadata = sequence()
    for item in metadata:
        item["shot_number"] += 100
    validate_sequence(metadata)


@pytest.mark.parametrize("numbers", [[1, 2, 2, 4], [1, 2, 4, 5], [4, 3, 2, 1]])
def test_invalid_order_and_missing_frames_fail(numbers: list[int]) -> None:
    metadata = sequence()
    for item, number in zip(metadata, numbers, strict=True):
        item["shot_number"] = number
    with pytest.raises(ValueError, match="consecutive"):
        validate_sequence(metadata)


@pytest.mark.parametrize("field", CONSISTENT)
def test_inconsistent_sequence_metadata_fails(field: str) -> None:
    metadata = deepcopy(sequence())
    metadata[2][field] = 9
    with pytest.raises(ValueError, match=field):
        validate_sequence(metadata)


def test_existing_output_prevents_any_decode(tmp_path: Path) -> None:
    output = tmp_path / "output.nefx"
    output.write_bytes(b"existing")
    with patch("nx_ctudio.merge.read_frame") as read:
        with pytest.raises(FileExistsError):
            merge([tmp_path / "missing.NEF"] * 4, output)
        read.assert_not_called()
    assert output.read_bytes() == b"existing"


def test_duplicate_file_prevents_any_decode(tmp_path: Path) -> None:
    source = tmp_path / "input.NEF"
    source.write_bytes(b"input")
    with patch("nx_ctudio.merge.read_frame") as read:
        with pytest.raises(ValueError, match="Duplicate"):
            merge([source] * 4, tmp_path / "output.nefx")
        read.assert_not_called()


def test_cli_error_is_json(capsys: pytest.CaptureFixture[str]) -> None:
    import json

    assert main(["merge", "missing.NEF", "--output", "out.nefx", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "error"
    assert report["backend"] == "native"
    assert report["error"]["code"] == "merge_failed"
