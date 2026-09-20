"""Complete-sequence validation and all-or-nothing multi-output publication."""

import hashlib
import json
import os
import sys
import weakref
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import rawpy

from nx_ctudio._tiff import read as read_tiff
from nx_ctudio._tiff import serialize
from nx_ctudio.cli import main
from nx_ctudio.merge import CONSISTENT, merge_outputs
from nx_ctudio.pixel_shift import output_groups, reconstruct

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from ci_fixture import pixels
from ci_fixture import sequence as camera_sequence


def inputs(tmp_path: Path, count: int) -> list[Path]:
    paths = [tmp_path / f"frame-{i:02d}.NEF" for i in range(count)]
    for i, path in enumerate(paths):
        path.write_bytes(f"source-{i}".encode("ascii"))
    return paths


@pytest.mark.parametrize(
    "model", ["NIKON Z 8", "NIKON Z f", "NIKON Z6_3", "NIKON Z5_2"]
)
@pytest.mark.parametrize("count, outputs", [(8, 2), (16, 4), (32, 2), (32, 4), (32, 8)])
@pytest.mark.parametrize("corrected", [False, True])
def test_camera_output_sets(
    tmp_path: Path, model: str, count: int, outputs: int, corrected: bool
) -> None:
    sources = camera_sequence(tmp_path, count, model=model)
    target = tmp_path / "result"
    report = merge_outputs(
        sources, target, outputs, chromatic_aberration_correction=corrected
    )
    assert report["camera"]["model"] == model
    assert json.loads((target / "manifest.json").read_text()) == report
    groups = output_groups(count, outputs)
    for output, group in zip(report["outputs"], groups, strict=True):
        assert output["input_indices"] == list(group)
        expected = reconstruct(
            [pixels(i) for i in group], (4, 4, 56, 40), b"0100" + bytes(24), 100
        )
        with rawpy.imread(output["output"]) as decoded:
            np.testing.assert_array_equal(decoded.raw_image[..., :3], expected)
        tree = read_tiff(Path(output["output"]).read_bytes())
        maker = tree.tags[34665].children[0].tags[37500].children[0]
        assert maker.tags[0xA7].integers() == (group[0] + 101,)
        assert tree.tags[272].payload == model.encode("ascii") + b"\0"
    assert report["samples_verified"] == sum(
        item["samples_verified"] for item in report["outputs"]
    )
    assert set(tmp_path.iterdir()) == {*sources, target}


def test_late_mixed_camera_never_publishes_partial_set(tmp_path: Path) -> None:
    sources = camera_sequence(tmp_path, 8, model="NIKON Z f")
    tree = read_tiff(sources[-1].read_bytes())
    tree.set(272, 2, b"NIKON Z6_3\0")
    sources[-1].write_bytes(serialize(tree))
    with pytest.raises(ValueError, match="Sequence metadata differs: model"):
        merge_outputs(sources, tmp_path / "result", 2)
    assert set(tmp_path.iterdir()) == set(sources)


@contextmanager
def pipeline(paths: list[Path], target: Path, failure: str | None = None):
    reads, writes, references = [], [], []
    decoded = {}

    def read(path: Path):
        i = paths.index(path)
        assert i not in reads
        reads.append(i)
        if failure == "decode" and i == len(paths) - 1:
            raise OSError("Late decode failure.")
        info = dict.fromkeys(CONSISTENT, 1)
        info.update(
            expected_count=len(paths),
            shot_number=i + 1,
            capture_time="2026-09-20T00:00:00",
            crop=(2, 2, 4, 4),
            raw_shape=(8, 8),
            selector=bytes(24),
            iso=3201,
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            radial={
                "available": True,
                "center_q8": [0, 0],
                "coefficients_q20": [[0, 0, 0], [0, 0, 0]],
            },
        )
        if failure == "metadata" and i == len(paths) - 1:
            info["burst_key"] = 2
        samples = np.full((8, 8), 100 + i, np.uint16)
        references.append(weakref.ref(samples))
        assert sum(ref() is not None for ref in references) <= 4
        return samples, info

    def write(path: Path, source: bytes, pixels: np.ndarray, count: int):
        writes.append((source, count))
        assert not target.exists()
        if len(writes) == 2:
            if failure == "writer":
                raise OSError("Late write failure.")
            if failure == "cancel":
                raise KeyboardInterrupt
            if failure == "mutation":
                paths[0].write_bytes(b"changed")
            if failure == "race":
                target.mkdir()
        path.write_bytes(source)
        decoded[str(path)] = pixels.copy()
        if failure == "verification" and len(writes) == 2:
            decoded[str(path)][-1, -1, -1] += 1
        if failure == "cleanup" and len(writes) == 2:
            return ["Private writer cleanup failed."]
        return []

    @contextmanager
    def decode(path: str):
        class Image:
            raw_image = decoded[path]

        yield Image()

    with (
        patch("nx_ctudio.merge.read_frame", side_effect=read),
        patch("nx_ctudio.merge.write_nefx", side_effect=write),
        patch("nx_ctudio.merge.rawpy.imread", side_effect=decode),
    ):
        yield reads, writes, references


@pytest.mark.parametrize("corrected", [False, True])
@pytest.mark.parametrize(
    "count, outputs",
    [(4, 1), (8, 1), (8, 2), (16, 1), (16, 4), (32, 1), (32, 2), (32, 4), (32, 8)],
)
def test_all_modes_publish_complete_manifests_and_release_sources(
    tmp_path: Path, count: int, outputs: int, corrected: bool
) -> None:
    paths = inputs(tmp_path, count)
    target = tmp_path / "merged"
    with pipeline(paths, target) as (reads, writes, references):
        report = merge_outputs(
            paths, target, outputs, chromatic_aberration_correction=corrected
        )
    assert report["output_count"] == outputs
    assert report["chromatic_aberration_correction"] is corrected
    assert json.loads((target / "manifest.json").read_text()) == report
    assert len(list(target.iterdir())) == outputs + 1
    assert sorted(reads) == list(range(count))
    assert all(ref() is None for ref in references)
    for item, (source, used_count) in zip(report["outputs"], writes, strict=True):
        first = item["input_indices"][0]
        assert source == paths[first].read_bytes()
        assert used_count == len(item["input_indices"]) == count // outputs
        assert (
            item["sha256"]
            == hashlib.sha256(Path(item["output"]).read_bytes()).hexdigest()
        )
    assert set(tmp_path.iterdir()) == {*paths, target}


@pytest.mark.parametrize("corrected", [False, True])
@pytest.mark.parametrize(
    "failure",
    [
        "metadata",
        "decode",
        "writer",
        "verification",
        "mutation",
        "cancel",
        "race",
        "cleanup",
    ],
)
def test_late_failures_do_not_publish_partial_sets(
    tmp_path: Path, corrected: bool, failure: str
) -> None:
    paths = inputs(tmp_path, 8)
    target = tmp_path / "merged"
    with pipeline(paths, target, failure) as (_, writes, _):
        with pytest.raises((ValueError, OSError, KeyboardInterrupt)):
            merge_outputs(paths, target, 2, chromatic_aberration_correction=corrected)
    assert writes
    if failure == "race":
        assert target.is_dir()
        assert list(target.iterdir()) == []
        assert set(tmp_path.iterdir()) == {*paths, target}
    else:
        assert not target.exists()
        assert set(tmp_path.iterdir()) == set(paths)


def test_existing_output_directory_prevents_decoding(tmp_path: Path) -> None:
    target = tmp_path / "merged"
    target.mkdir()
    with patch("nx_ctudio.merge.read_frame") as read:
        with pytest.raises(FileExistsError):
            merge_outputs([tmp_path / "missing.NEF"] * 8, target, 2)
    read.assert_not_called()


def test_invalid_mode_prevents_decoding(tmp_path: Path) -> None:
    with patch("nx_ctudio.merge.read_frame") as read:
        with pytest.raises(ValueError, match="combination"):
            merge_outputs([tmp_path / "missing.NEF"] * 16, tmp_path / "merged", 2)
    read.assert_not_called()


def test_cli_requires_a_directory_for_multiple_outputs() -> None:
    with patch("nx_ctudio.merge.merge_outputs") as merge:
        with pytest.raises(SystemExit) as exc:
            main(["merge", "source.NEF", "-o", "out.nefx", "--outputs", "2"])
    assert exc.value.code == 2
    merge.assert_not_called()


def test_cli_reports_each_output_and_total_verified_samples(
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = {
        "outputs": [
            {"output": "batch/merged-01.nefx", "sha256": "one"},
            {"output": "batch/merged-02.nefx", "sha256": "two"},
        ],
        "samples_verified": 96,
        "limitations": [],
        "warnings": [],
    }
    with patch("nx_ctudio.merge.merge_outputs", return_value=report) as merge:
        assert (
            main(
                ["merge", "source.NEF", "--output-directory", "batch", "--outputs", "2"]
            )
            == 0
        )
    assert merge.call_args.args == ([Path("source.NEF")], Path("batch"), 2)
    text = capsys.readouterr().out
    assert "batch/merged-01.nefx" in text and "batch/merged-02.nefx" in text
    assert "Samples verified: 96" in text


def test_parent_sync_failure_reports_published_batch_without_deleting_outputs(
    tmp_path: Path,
) -> None:
    paths = inputs(tmp_path, 8)
    target = tmp_path / "merged"
    parent_inode = tmp_path.stat().st_ino
    sync = os.fsync

    def fail_parent(descriptor: int) -> None:
        if os.fstat(descriptor).st_ino == parent_inode:
            raise OSError("Parent sync unavailable.")
        sync(descriptor)

    with (
        pipeline(paths, target),
        patch("nx_ctudio._atomic.os.fsync", side_effect=fail_parent),
    ):
        report = merge_outputs(paths, target, 2)
    assert report["status"] == "ok"
    assert len(report["warnings"]) == 1
    assert "parent directory sync failed" in report["warnings"][0]
    for item in report["outputs"]:
        assert (
            hashlib.sha256(Path(item["output"]).read_bytes()).hexdigest()
            == item["sha256"]
        )
    assert len(list(target.iterdir())) == 3
