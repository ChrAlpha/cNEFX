"""No-overwrite file publication with observable post-publication cleanup."""

from __future__ import annotations

import ctypes
import os
import sys
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path


def _directory_rename() -> Callable[[Path, Path], None]:
    if sys.platform != "linux":
        raise NotImplementedError("Atomic output directories currently require Linux.")
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, "renameat2", None)
    if rename is None:
        raise NotImplementedError("Atomic output directories require renameat2.")
    rename.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    rename.restype = ctypes.c_int

    def publish(source: Path, destination: Path) -> None:
        # AT_FDCWD and RENAME_NOREPLACE prevent replacing even an empty directory.
        if rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1):
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code), str(destination))

    return publish


@dataclass
class Publication:
    temporary: Path
    destination: Path
    published: bool = False
    warnings: list[str] = field(default_factory=list)
    directory_rename: Callable[[Path, Path], None] | None = None

    def publish(self) -> None:
        if self.directory_rename is None:
            # Windows fsync requires a writable handle, even after writes close.
            with self.temporary.open("rb+") as stream:
                os.fsync(stream.fileno())
            os.link(self.temporary, self.destination)
        else:
            descriptor = os.open(self.temporary, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            self.directory_rename(self.temporary, self.destination)
        self.published = True
        try:
            descriptor = os.open(
                self.destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            )
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except (OSError, KeyboardInterrupt) as exc:
            reason = str(exc) or type(exc).__name__
            self.warnings.append(
                "Output published; parent directory sync failed at "
                f"{self.destination.parent}: {reason}"
            )


@contextmanager
def publication(destination: Path, *, directory: bool = False) -> Iterator[Publication]:
    if os.path.lexists(destination):
        raise FileExistsError("Output already exists; overwriting is not supported.")
    rename = _directory_rename() if directory else None
    staging = tempfile.TemporaryDirectory(prefix=".nxc-", dir=destination.parent)
    pending = Publication(
        Path(staging.name) / ("outputs" if directory else "output.nefx"),
        destination,
        directory_rename=rename,
    )
    try:
        if directory:
            pending.temporary.mkdir()
        yield pending
    finally:
        try:
            staging.cleanup()
        except (OSError, KeyboardInterrupt) as exc:
            if not pending.published:
                raise
            reason = str(exc) or type(exc).__name__
            pending.warnings.append(
                "Output published; temporary cleanup failed at "
                f"{staging.name}: {reason}"
            )
