"""Read-only camera layout evidence; ordinary NEFs do not validate synthesis."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import logging
import struct
import subprocess
from pathlib import Path

import rawpy
import tifffile


def inspect(path: Path) -> dict:
    before = path.stat()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    result = {"path": str(path.resolve()), "sha256": digest}
    with tifffile.TiffFile(path) as image:
        root = image.pages[0]
        exif = root.tags[34665].value
        result.update(
            make=root.tags[271].value,
            model=root.tags[272].value,
            byteorder=image.byteorder,
            subifds=len(root.pages),
        )
        maker = exif["MakerNote"]
        result["maker_header_hex"] = maker[:10].hex()
        with tifffile.TiffFile(io.BytesIO(maker[10:])) as notes:
            tags = notes.pages[0].tags
            result["blocks"] = {
                f"{code:#06x}": {
                    "size": tags[code].count,
                    "version": tags[code].value[:4].decode("ascii"),
                }
                for code in (0x51, 0x54, 0x56, 0x5E, 0x97)
            }
            result.update(
                crop=tags[0x45].value,
                compression=struct.unpack_from("<H", tags[0x51].value, 10)[0],
                capture_active=tags[0x56].value[12],
                pixel_shift_mode=tags[0x5E].value[4],
            )
        raw = root.pages[1]
        result.update(
            raw_shape=raw.shape,
            bits=raw.bitspersample,
            cfa=list(raw.tags[33422].value),
            strips=len(raw.dataoffsets),
        )
        info = raw.tags[0xC7D5].value
        result["nefinfo_header_hex"] = info[:10].hex()
        with tifffile.TiffFile(io.BytesIO(info[10:])) as notes:
            result["nefinfo_blocks"] = {
                f"{code:#06x}": {
                    "size": notes.pages[0].tags[code].count,
                    "version": notes.pages[0].tags[code].value[:4].decode("ascii"),
                }
                for code in (7, 8, 11, 13)
            }
    with rawpy.imread(str(path)) as decoded:
        decoded.unpack()
        result["decode"] = {
            "raw_shape": decoded.raw_image.shape,
            "sizes": decoded.sizes._asdict(),
            "pattern": decoded.raw_pattern.tolist(),
            "black_levels": decoded.black_level_per_channel,
            "white_level": decoded.white_level,
            "gains": decoded.camera_whitebalance,
            "xyz_to_camera_matrix": decoded.rgb_xyz_matrix[:3].tolist(),
        }
    with path.open("rb") as stream:
        after_digest = hashlib.file_digest(stream, "sha256").hexdigest()
    after = path.stat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if digest != after_digest or any(
        getattr(before, key) != getattr(after, key) for key in fields
    ):
        raise ValueError("Source changed during inspection.")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.ERROR)
    results = []
    for path in args.files:
        try:
            results.append({"status": "ok", **inspect(path)})
        except Exception as exc:
            results.append({"status": "error", "path": str(path), "error": str(exc)})
    report = {
        "schema_version": 1,
        "evidence_kind": "source_layout_and_libraw_decode_only",
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, timeout=10
        ).strip(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "rawpy", "tifffile")
        },
        "libraw_version": rawpy.libraw_version,
        "files": results,
        "limitations": [
            "No source metadata was patched; no synthesis or vendor comparison ran.",
            "Ordinary RAW compatibility is not authentic pixel-shift acceptance.",
        ],
    }
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return int(any(item["status"] != "ok" for item in results))


if __name__ == "__main__":
    raise SystemExit(main())
