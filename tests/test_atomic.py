"""Publication and cleanup outcomes are separate observable states."""

from pathlib import Path
from unittest.mock import patch

import pytest

from nx_ctudio._atomic import publication


def test_cleanup_error_after_publish_retains_success_and_output(tmp_path: Path) -> None:
    target = tmp_path / "final.nefx"
    cleanup = patch("tempfile.TemporaryDirectory.cleanup", side_effect=OSError("busy"))
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
