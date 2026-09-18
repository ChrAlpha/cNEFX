"""Bounded classic little-endian Nikon TIFF graph relocation."""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

WIDTHS = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8}
LINKS = {34665, 34853, 40965, 330}
IMAGES = ((513, 514), (273, 279), (324, 325))


@dataclass
class Tag:
    code: int
    kind: int
    count: int
    payload: bytes
    children: list[Directory] = field(default_factory=list)
    nikon_header: bytes = b""

    def integers(self) -> tuple[int, ...]:
        if self.kind not in (3, 4):
            raise ValueError("Unsupported TIFF pointer or length type.")
        return struct.unpack(
            "<" + ("H" if self.kind == 3 else "I") * self.count, self.payload
        )


@dataclass
class Directory:
    tags: dict[int, Tag]
    images: dict[int, list[bytes]] = field(default_factory=dict)

    def set(self, code: int, kind: int, payload: bytes) -> None:
        if len(payload) % WIDTHS[kind]:
            raise ValueError("Invalid TIFF value width.")
        self.tags[code] = Tag(code, kind, len(payload) // WIDTHS[kind], payload)

    def numbers(self, code: int, kind: int, *values: int) -> None:
        self.set(
            code, kind, struct.pack("<" + {3: "H", 4: "I"}[kind] * len(values), *values)
        )


def read(data: bytes) -> Directory:
    seen: set[tuple[int, int]] = set()

    def block(offset: int, length: int) -> bytes:
        if offset < 0 or length < 0 or offset + length > len(data):
            raise ValueError("TIFF region exceeds file bounds.")
        return data[offset : offset + length]

    def first(base: int) -> int:
        header = block(base, 8)
        if header[:4] != b"II*\0":
            raise ValueError("Only classic little-endian TIFF is supported.")
        return struct.unpack_from("<I", header, 4)[0]

    def walk(base: int, relative: int, maker: bool = False) -> Directory:
        if (base, relative) in seen or len(seen) >= 64:
            raise ValueError("Repeated or excessive TIFF directory references.")
        seen.add((base, relative))
        offset = base + relative
        count = struct.unpack("<H", block(offset, 2))[0]
        entries = block(offset + 2, count * 12 + 4)
        if struct.unpack_from("<I", entries, count * 12)[0]:
            raise ValueError("Nonzero TIFF next-directory chains are unsupported.")
        directory = Directory({})
        for index in range(count):
            code, kind, size, value = struct.unpack_from("<HHII", entries, index * 12)
            if kind not in WIDTHS or code in directory.tags:
                raise ValueError("Unsupported TIFF type or duplicate tag.")
            length = size * WIDTHS[kind]
            position = offset + 2 + index * 12 + 8 if length <= 4 else base + value
            payload = block(position, length)
            tag = Tag(code, kind, size, payload)
            directory.tags[code] = tag
            if code in LINKS or (maker and code == 17):
                if kind != 4 or not 1 <= size <= 16:
                    raise ValueError("Unsupported linked TIFF directory layout.")
                tag.children = [walk(base, pointer) for pointer in tag.integers()]
            elif code in (37500, 0xC7D5):
                if kind != 7 or payload[:6] != b"Nikon\0" or len(payload) < 18:
                    raise ValueError("Unsupported embedded Nikon TIFF.")
                tag.nikon_header = payload[:10]
                child_base = position + 10
                tag.children = [walk(child_base, first(child_base), code == 37500)]
        for offset_tag, length_tag in IMAGES:
            if offset_tag not in directory.tags:
                continue
            if length_tag not in directory.tags:
                raise ValueError("Missing TIFF image payload length.")
            offsets = directory.tags[offset_tag].integers()
            lengths = directory.tags[length_tag].integers()
            if len(offsets) != len(lengths) or len(offsets) > 32:
                raise ValueError("Unsupported TIFF strip or tile array.")
            directory.images[offset_tag] = [
                block(base + start, length)
                for start, length in zip(offsets, lengths, strict=True)
            ]
        return directory

    return walk(0, first(0))


def serialize(root: Directory) -> bytes:
    output = bytearray(b"II*\0\x08\0\0\0")

    def allocate(size: int, align: bool = True) -> int:
        offset = (len(output) + 15) & ~15 if align else len(output)
        if size < 0 or offset + size >= 1 << 32:
            raise ValueError("Output exceeds classic TIFF offset limits.")
        output.extend(bytes(offset + size - len(output)))
        return offset

    def write(payload: bytes) -> int:
        offset = allocate(len(payload))
        output[offset : offset + len(payload)] = payload
        return offset

    def walk(directory: Directory, base: int, start: int | None = None) -> int:
        tags = [directory.tags[code] for code in sorted(directory.tags)]
        if len(tags) > 65535:
            raise ValueError("Too many TIFF tags.")
        size = 2 + len(tags) * 12 + 4
        if start is None:
            start = allocate(size)
        else:
            if start != len(output):
                raise ValueError("Invalid directory placement.")
            allocate(size, align=False)
        struct.pack_into("<H", output, start, len(tags))
        image_values = {}
        for offsets_tag, lengths_tag in IMAGES:
            if offsets_tag not in directory.images:
                continue
            offsets, lengths = [], []
            for payload in directory.images[offsets_tag]:
                offsets.append(write(payload) - base if payload else 0)
                lengths.append(len(payload))
            image_values[offsets_tag] = offsets
            image_values[lengths_tag] = lengths
        for index, tag in enumerate(tags):
            kind, count, payload = tag.kind, tag.count, tag.payload
            if tag.code in image_values:
                values = image_values[tag.code]
                kind, count = 4, len(values)
                payload = struct.pack("<" + "I" * count, *values)
            if tag.nikon_header:
                position = allocate(18)
                output[position : position + 18] = tag.nikon_header + b"II*\0\x08\0\0\0"
                walk(tag.children[0], position + 10, position + 18)
                count = len(output) - position
                kind = 7
                value = struct.pack("<I", position - base)
            else:
                if tag.children:
                    values = [walk(child, base) - base for child in tag.children]
                    kind, count = 4, len(values)
                    payload = struct.pack("<" + "I" * count, *values)
                if len(payload) != count * WIDTHS[kind]:
                    raise ValueError("TIFF value length disagrees with its type.")
                value = (
                    struct.pack("<I", write(payload) - base)
                    if len(payload) > 4
                    else payload.ljust(4, b"\0")
                )
            entry = struct.pack("<HHI", tag.code, kind, count) + value
            output[start + 2 + index * 12 : start + 14 + index * 12] = entry
        return start

    walk(root, 0, 8)
    return bytes(output)
