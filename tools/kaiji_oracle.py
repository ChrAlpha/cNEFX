"""Pin the external coefficient dependencies consumed by Kaiji vendor probes."""

from __future__ import annotations

from pathlib import Path

from verify_np3_header import digest, windows

MOON_HASH = "f02f7499564c14f18933ee2b0bda961c25889f0240a30bfe042fa165ba2e8da0"
RESOURCE_HASH = "8d6f1bf219561f3c7c11c61b9b9e3317a369c9b5be5089ddffd82db96ca7f707"


def dependency_hashes(dll: Path, probe: Path) -> dict[str, str]:
    """Moon locates prm.bin relative to the executable, not relative to Rome2."""
    paths = {
        dll.resolve().parent / "Moon2.dll": MOON_HASH,
        probe.resolve().parent / "prm.bin": RESOURCE_HASH,
    }
    for path, expected in paths.items():
        if digest(path) != expected:
            raise ValueError(f"Unexpected vendor dependency hash: {path.name}.")
    return {str(path): expected for path, expected in paths.items()}


def loaded_provider_matches(event: dict, dll: Path) -> bool:
    actual = event.get("moon_path")
    expected = windows(dll.resolve().parent / "Moon2.dll")
    return (
        isinstance(actual, str)
        and actual.replace("\\", "/").casefold()
        == expected.replace("\\", "/").casefold()
    )
