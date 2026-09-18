"""No-overwrite file publication with observable post-publication cleanup."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Publication:
    temporary: Path
    destination: Path
    published: bool = False
    warnings: list[str] = field(default_factory=list)

    def publish(self) -> None:
        with self.temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.link(self.temporary, self.destination)
        self.published = True


@contextmanager
def publication(destination: Path) -> Iterator[Publication]:
    if os.path.lexists(destination):
        raise FileExistsError("Output already exists; overwriting is not supported.")
    directory = tempfile.TemporaryDirectory(prefix=".nxc-", dir=destination.parent)
    pending = Publication(Path(directory.name) / "output.nefx", destination)
    try:
        yield pending
    finally:
        try:
            directory.cleanup()
        except (OSError, KeyboardInterrupt) as exc:
            if not pending.published:
                raise
            reason = str(exc) or type(exc).__name__
            pending.warnings.append(
                "Output published; temporary cleanup failed at "
                f"{directory.name}: {reason}"
            )
