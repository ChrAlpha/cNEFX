"""Publication and cleanup outcomes are separate observable states."""

from pathlib import Path
from unittest.mock import patch

import pytest

from nx_ctudio._atomic import publication


@pytest.mark.parametrize("error", [OSError("busy"), KeyboardInterrupt()])
def test_cleanup_error_after_publish_retains_success_and_output(
    tmp_path: Path, error: BaseException
) -> None:
    target = tmp_path / "final.nefx"
    cleanup = patch("tempfile.TemporaryDirectory.cleanup", side_effect=error)
    try:
        with publication(target) as pending:
            pending.temporary.write_bytes(b"complete output")
            pending.publish()
            cleanup.start()
    finally:
        cleanup.stop()
    assert pending.published
    assert target.read_bytes() == b"complete output"
    assert len(pending.warnings) == 1
    assert "Output published" in pending.warnings[0]
    assert (str(error) or type(error).__name__) in pending.warnings[0]


def test_racing_destination_is_preserved(tmp_path: Path) -> None:
    target = tmp_path / "final.nefx"
    with publication(target) as pending:
        pending.temporary.write_bytes(b"new output")
        target.write_bytes(b"other writer")
        with pytest.raises(FileExistsError):
            pending.publish()
    assert not pending.published
    assert target.read_bytes() == b"other writer"
    assert list(tmp_path.iterdir()) == [target]


def test_cancel_before_publish_cleans_temporary(tmp_path: Path) -> None:
    with pytest.raises(KeyboardInterrupt):
        with publication(tmp_path / "final.nefx") as pending:
            pending.temporary.write_bytes(b"partial")
            raise KeyboardInterrupt
    assert list(tmp_path.iterdir()) == []


def test_cleanup_cancel_before_publish_propagates(tmp_path: Path) -> None:
    target = tmp_path / "final.nefx"
    with patch("tempfile.TemporaryDirectory.cleanup", side_effect=KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):
            with publication(target) as pending:
                pending.temporary.write_bytes(b"unpublished output")
    assert not pending.published
    assert not target.exists()


def test_directory_is_published_as_one_complete_set(tmp_path: Path) -> None:
    target = tmp_path / "merged"
    with publication(target, directory=True) as pending:
        (pending.temporary / "first.nefx").write_bytes(b"first")
        (pending.temporary / "second.nefx").write_bytes(b"second")
        assert not target.exists()
        pending.publish()
    assert sorted(path.name for path in target.iterdir()) == [
        "first.nefx",
        "second.nefx",
    ]
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("kind", ["empty_directory", "directory", "file", "symlink"])
def test_directory_publication_never_replaces_racing_destination(
    tmp_path: Path, kind: str
) -> None:
    target = tmp_path / "merged"
    with publication(target, directory=True) as pending:
        (pending.temporary / "new.nefx").write_bytes(b"new")
        if kind in ("empty_directory", "directory"):
            target.mkdir()
            if kind == "directory":
                (target / "other").write_bytes(b"existing")
        elif kind == "file":
            target.write_bytes(b"existing")
        else:
            target.symlink_to(tmp_path / "missing")
        identity = target.lstat().st_ino
        with pytest.raises(FileExistsError):
            pending.publish()
    assert target.lstat().st_ino == identity
    assert not pending.published
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("error", [OSError("Disk full"), KeyboardInterrupt()])
def test_directory_failure_or_cancellation_removes_all_private_outputs(
    tmp_path: Path, error: BaseException
) -> None:
    with pytest.raises(type(error)):
        with publication(tmp_path / "merged", directory=True) as pending:
            (pending.temporary / "first.nefx").write_bytes(b"first")
            raise error
    assert list(tmp_path.iterdir()) == []


def test_directory_fsync_failure_does_not_publish(tmp_path: Path) -> None:
    with pytest.raises(OSError, match="Disk full"):
        with publication(tmp_path / "merged", directory=True) as pending:
            (pending.temporary / "first.nefx").write_bytes(b"first")
            with patch("nx_ctudio._atomic.os.fsync", side_effect=OSError("Disk full")):
                pending.publish()
    assert list(tmp_path.iterdir()) == []


def test_directory_backend_failure_cleans_staging(tmp_path: Path) -> None:
    with pytest.raises(OSError, match="unsupported"):
        with publication(tmp_path / "merged", directory=True) as pending:
            (pending.temporary / "first.nefx").write_bytes(b"first")
            pending.directory_rename = lambda *_: (_ for _ in ()).throw(
                OSError("unsupported filesystem")
            )
            pending.publish()
    assert list(tmp_path.iterdir()) == []


def test_directory_cleanup_failure_retains_published_files(tmp_path: Path) -> None:
    target = tmp_path / "merged"
    cleanup = patch("tempfile.TemporaryDirectory.cleanup", side_effect=OSError("busy"))
    try:
        with publication(target, directory=True) as pending:
            (pending.temporary / "first.nefx").write_bytes(b"first")
            pending.publish()
            cleanup.start()
    finally:
        cleanup.stop()
    assert pending.published
    assert (target / "first.nefx").read_bytes() == b"first"
    assert len(pending.warnings) == 1
