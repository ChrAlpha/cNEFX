"""Sequence rejection and output publication boundaries."""

import hashlib
import weakref
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from nx_ctudio.cli import main
from nx_ctudio.merge import CONSISTENT, merge, validate_sequence


def sequence() -> list[dict]:
    first = dict.fromkeys(CONSISTENT, 1)
    first.update(
        expected_count=4, shot_number=1, capture_time="2026-09-18T00:00:00.000000"
    )
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


def test_capture_timestamps_cannot_move_backwards() -> None:
    metadata = sequence()
    metadata[1]["capture_time"] = "2026-09-17T00:00:00.000000"
    with pytest.raises(ValueError, match="timestamps"):
        validate_sequence(metadata)


def test_cli_error_is_json(capsys: pytest.CaptureFixture[str]) -> None:
    import json

    assert main(["merge", "missing.NEF", "--output", "out.nefx", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "error"
    assert report["backend"] == "native"
    assert report["error"]["code"] == "merge_failed"


def test_cli_forwards_chromatic_aberration_option(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with patch("nx_ctudio.merge.merge", return_value={"status": "ok"}) as synthesize:
        assert (
            main(
                [
                    "merge",
                    "1.NEF",
                    "2.NEF",
                    "3.NEF",
                    "4.NEF",
                    "--output",
                    "out.nefx",
                    "--chromatic-aberration-correction",
                    "--json",
                ]
            )
            == 0
        )
    assert synthesize.call_args.kwargs == {"chromatic_aberration_correction": True}
    assert '"status": "ok"' in capsys.readouterr().out


@pytest.mark.parametrize("failure", ["missing", "different"])
def test_correction_rejects_unusable_or_inconsistent_parameters_before_write(
    tmp_path: Path, failure: str
) -> None:
    sources = [tmp_path / f"{i}.NEF" for i in range(4)]
    for path in sources:
        path.write_bytes(b"source")
    metadata = sequence()
    for item in metadata:
        item.update(
            crop=(2, 2, 4, 4),
            raw_shape=(8, 8),
            selector=bytes(24),
            iso=3201,
            radial={
                "available": True,
                "center_q8": [0, 0],
                "coefficients_q20": [[0, 0, 0], [0, 0, 0]],
            },
        )
    if failure == "missing":
        metadata[0]["radial"] = {"available": False, "reason": "Missing test block."}
    else:
        metadata[2]["radial"]["coefficients_q20"][0][0] = 50

    def read(path: Path) -> tuple[np.ndarray, dict]:
        return np.zeros((8, 8), np.uint16), metadata[sources.index(path)]

    with (
        patch("nx_ctudio.merge.read_frame", side_effect=read),
        patch("nx_ctudio.merge.write_nefx") as write,
        pytest.raises(ValueError, match="correction|radial"),
    ):
        merge(sources, tmp_path / "output.nefx", chromatic_aberration_correction=True)
    write.assert_not_called()
    assert set(tmp_path.iterdir()) == set(sources)


@pytest.mark.parametrize("count", [4, 8, 16, 32])
@pytest.mark.parametrize("failure", ["writer", "metadata", "decode", "cancel"])
def test_grouped_merge_releases_sources_and_never_publishes_on_failure(
    tmp_path: Path, count: int, failure: str
) -> None:
    sources = []
    for index in range(count):
        path = tmp_path / f"frame-{index}.NEF"
        path.write_bytes(f"synthetic input {index}".encode("ascii"))
        sources.append(path)
    reads, references = [], []
    first = sequence()[0]
    first.update(
        expected_count=count,
        crop=(2, 2, 4, 4),
        raw_shape=(8, 8),
        selector=bytes(24),
        iso=3201,
    )

    def read(path: Path) -> tuple[np.ndarray, dict]:
        index = sources.index(path)
        assert index not in reads
        reads.append(index)
        if index == count - 1:
            if failure == "decode":
                raise OSError("late decode failure")
            if failure == "cancel":
                raise KeyboardInterrupt
        information = dict(
            first,
            shot_number=index + 1,
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        if index == count - 1 and failure == "metadata":
            information["shot_number"] += 1
        samples = np.full((8, 8), index, np.uint16)
        references.append(weakref.ref(samples))
        assert sum(ref() is not None for ref in references) <= 4
        return samples, information

    output = tmp_path / "output.nefx"
    error = {
        "writer": (RuntimeError, "writer reached"),
        "metadata": (ValueError, "consecutive"),
        "decode": (OSError, "late decode failure"),
        "cancel": (KeyboardInterrupt, "^$"),
    }[failure]
    with (
        patch("nx_ctudio.merge.read_frame", side_effect=read),
        patch(
            "nx_ctudio.merge.write_nefx", side_effect=RuntimeError("writer reached")
        ) as writer,
        pytest.raises(error[0], match=error[1]),
    ):
        merge(sources, output)
    if failure == "writer":
        writer.assert_called_once()
    else:
        writer.assert_not_called()
    assert sorted(reads) == list(range(count))
    assert not output.exists()
    assert set(tmp_path.iterdir()) == set(sources)


@pytest.mark.parametrize("failure", ["final_partial_block", "shape"])
def test_decoded_output_mismatch_is_rejected_before_publication(
    tmp_path: Path, failure: str
) -> None:
    sources = [tmp_path / f"{i}.NEF" for i in range(4)]
    for path in sources:
        path.write_bytes(b"source")
    metadata = sequence()
    for item in metadata:
        item.update(
            crop=(2, 2, 4, 257),
            raw_shape=(260, 8),
            selector=bytes(24),
            iso=3201,
            source_sha256=hashlib.sha256(b"source").hexdigest(),
        )

    def read(path: Path) -> tuple[np.ndarray, dict]:
        return np.full((260, 8), 100, np.uint16), metadata[sources.index(path)]

    def write(path: Path, *args: object) -> list[str]:
        path.write_bytes(b"candidate")
        return []

    actual = np.full((257, 4, 3), 100, np.uint16)
    if failure == "shape":
        actual = actual[:-1]
    else:
        actual[-1, -1, -1] += 1
    with (
        patch("nx_ctudio.merge.read_frame", side_effect=read),
        patch("nx_ctudio.merge.write_nefx", side_effect=write),
        patch("nx_ctudio.merge.rawpy.imread") as decode,
    ):
        decode.return_value.__enter__.return_value.raw_image = actual
        with pytest.raises(ValueError, match="exact pixel verification"):
            merge(sources, tmp_path / "output.nefx")
    assert set(tmp_path.iterdir()) == set(sources)
