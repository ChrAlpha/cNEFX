# /// script
# requires-python = ">=3.11"
# dependencies = ["tifffile==2026.9.15"]
# ///
"""Compare TIFF metadata payloads without publishing photograph metadata values."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import struct
from pathlib import Path

import tifffile

POINTERS = {273, 324, 330, 513, 34665, 34853, 40965}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def snapshot(path: Path) -> dict[str, dict]:
    data = path.read_bytes()
    result: dict[str, dict] = {}
    visits = 0

    def walk(blob: bytes, offset: int, prefix: str, order: str) -> None:
        nonlocal visits
        visits += 1
        if visits > 32:
            raise ValueError("Too many TIFF directories.")
        with tifffile.TiffFile(io.BytesIO(blob)) as image:
            count = struct.unpack_from(order + "H", blob, offset)[0]
            for index in range(count):
                tag = tifffile.TiffTag.fromfile(image, offset=offset + 2 + 12 * index)
                key = f"{prefix}/0x{tag.code:04x}"
                payload = blob[tag.valueoffset : tag.valueoffset + tag.valuebytecount]
                item = {"type": int(tag.dtype), "count": tag.count}
                if tag.code in POINTERS or (
                    tag.code == 0x11 and prefix.endswith("/0x927c")
                ):
                    item["storage_pointer"] = True
                else:
                    item["payload_sha256"] = digest(payload)
                result[key] = item
                if tag.code in (34665, 34853, 40965):
                    walk(blob, int(tag.valueoffset), key, order)
                elif tag.code == 330:
                    for child, child_offset in enumerate(tag.value):
                        walk(blob, int(child_offset), f"{key}[{child}]", order)
                elif tag.code in (37500, 0xC7D5) and payload[:6] == b"Nikon\0":
                    inner = payload[10:]
                    inner_order = "<" if inner[:2] == b"II" else ">"
                    inner_offset = struct.unpack_from(inner_order + "I", inner, 4)[0]
                    walk(inner, inner_offset, key, inner_order)

    order = "<" if data[:2] == b"II" else ">"
    if struct.unpack_from(order + "H", data, 2)[0] != 42:
        raise ValueError("Expected classic TIFF.")
    walk(data, struct.unpack_from(order + "I", data, 4)[0], "IFD0", order)
    return result


def compare(source: Path, output: Path) -> dict:
    before, after = snapshot(source), snapshot(output)
    shared = before.keys() & after.keys()
    changed = [k for k in sorted(shared) if before[k] != after[k]]
    return {
        "source_sha256": digest(source.read_bytes()),
        "output_sha256": digest(output.read_bytes()),
        "payload_values": (
            "Only hashes are retained; values may contain personal metadata."
        ),
        "scope": (
            "IFD0, SubIFDs, EXIF, GPS, interoperability, Nikon MakerNote and "
            "NikonNEFInfo. Next-IFD chains, Nikon preview IFD and image payloads "
            "are not traversed."
        ),
        "storage_pointer_tags": [
            k for k in sorted(shared) if before[k].get("storage_pointer")
        ],
        "preserved_payloads": [
            k
            for k in sorted(shared)
            if before[k] == after[k] and not before[k].get("storage_pointer")
        ],
        "changed": {k: {"before": before[k], "after": after[k]} for k in changed},
        "removed": {k: before[k] for k in sorted(before.keys() - after.keys())},
        "added": {k: after[k] for k in sorted(after.keys() - before.keys())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
