# /// script
# requires-python = ">=3.11"
# dependencies = ["tifffile==2026.9.15"]
# ///
"""Inspect classic Nikon TIFF offsets and payload hashes, never metadata values."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import struct
from pathlib import Path

import tifffile


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def inspect(path: Path) -> dict:
    data = path.read_bytes()
    directories: list[dict] = []
    payloads: dict[str, dict] = {}
    seen: set[tuple[int, int]] = set()

    def block(offset: int, size: int) -> bytes:
        if offset < 0 or size < 0 or offset + size > len(data):
            raise ValueError("TIFF region exceeds file bounds.")
        return data[offset : offset + size]

    def header(base: int) -> tuple[str, int]:
        raw = block(base, 8)
        if raw[:4] not in (b"II*\0", b"MM\0*"):
            raise ValueError("Expected classic TIFF header.")
        order = "<" if raw[:2] == b"II" else ">"
        return order, struct.unpack_from(order + "I", raw, 4)[0]

    def walk(base: int, relative: int, name: str, kind: str) -> None:
        if (base, relative) in seen:
            raise ValueError("Repeated TIFF directory reference.")
        if len(seen) >= 64:
            raise ValueError("Too many TIFF directories.")
        seen.add((base, relative))
        order, _ = header(base)
        offset = base + relative
        count = struct.unpack(order + "H", block(offset, 2))[0]
        raw = block(offset, 2 + count * 12 + 4)
        following = struct.unpack_from(order + "I", raw, 2 + count * 12)[0]
        entry = {
            "path": name,
            "kind": kind,
            "base": base,
            "offset": offset,
            "entries": count,
            "next_ifd": base + following if following else 0,
            "tags": [],
        }
        directories.append(entry)
        values = {}
        # Keep the full backing file: Nikon preview offsets can leave MakerNote's
        # declared payload extent while still using its embedded TIFF origin.
        with tifffile.TiffFile(io.BytesIO(data[base:])) as image:
            for index in range(count):
                tag = tifffile.TiffTag.fromfile(image, offset=relative + 2 + index * 12)
                values[tag.code] = tag.value
                payload_offset = base + tag.valueoffset
                size = tag.valuebytecount
                block(payload_offset, size)
                entry["tags"].append(
                    {
                        "code": tag.code,
                        "type": int(tag.dtype),
                        "count": tag.count,
                        "size": size,
                        "offset": payload_offset,
                        "inline": size <= 4 and tag.code not in (34665, 34853, 40965),
                    }
                )
                if tag.code in (34665, 34853, 40965):
                    walk(
                        base, int(tag.valueoffset), f"{name}/0x{tag.code:04x}", "linked"
                    )
                elif tag.code == 330:
                    offsets = (
                        tag.value if isinstance(tag.value, tuple) else (tag.value,)
                    )
                    for child, child_offset in enumerate(offsets):
                        walk(
                            base, int(child_offset), f"{name}/SubIFD[{child}]", "subifd"
                        )
                elif tag.code in (37500, 0xC7D5):
                    if block(payload_offset, 6) == b"Nikon\0":
                        child_base = payload_offset + 10
                        _, child_offset = header(child_base)
                        walk(
                            child_base,
                            child_offset,
                            f"{name}/0x{tag.code:04x}",
                            "nikon",
                        )
                elif kind == "nikon" and tag.code == 0x11 and name.endswith("/0x927c"):
                    walk(base, int(tag.value), f"{name}/PreviewIFD", "preview")
        for offsets_tag, lengths_tag, label in (
            (513, 514, "JPEG"),
            (273, 279, "strips"),
            (324, 325, "tiles"),
        ):
            if offsets_tag not in values:
                continue
            offsets, lengths = values[offsets_tag], values[lengths_tag]
            offsets = offsets if isinstance(offsets, tuple) else (offsets,)
            lengths = lengths if isinstance(lengths, tuple) else (lengths,)
            if len(offsets) != len(lengths):
                raise ValueError("Mismatched payload offsets and lengths.")
            for index, (start, size) in enumerate(zip(offsets, lengths, strict=True)):
                content = block(base + start, size)
                payloads[f"{name}/{label}[{index}]"] = {
                    "offset": base + start,
                    "size": size,
                    "sha256": digest(content),
                    "jpeg_soi": content[:2] == b"\xff\xd8",
                }
        if following:
            walk(base, following, f"{name}/NextIFD", "next")

    _, first = header(0)
    walk(0, first, "IFD0", "root")
    return {
        "sha256": digest(data),
        "size": len(data),
        "directories": directories,
        "payloads": payloads,
    }


def compare(source: Path, output: Path) -> dict:
    before, after = inspect(source), inspect(output)
    differences = []
    for key, payload in after["payloads"].items():
        old = before["payloads"].get(key)
        differences.append(
            {
                "path": key,
                **payload,
                "source_sha256": old["sha256"] if old else None,
                "preserved": old is not None and old["sha256"] == payload["sha256"],
            }
        )
    return {
        "source": str(source),
        "output": str(output),
        "source_sha256": before["sha256"],
        "output_sha256": after["sha256"],
        "directories": after["directories"],
        "payloads": differences,
        "removed_payloads": sorted(
            before["payloads"].keys() - after["payloads"].keys()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
