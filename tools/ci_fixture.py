"""Authored packed14 Nikon fixtures for CI, never authentic-capture evidence."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from nx_ctudio._tiff import Directory, Tag, serialize


def pixels(index: int, shape: tuple[int, int] = (48, 64)) -> np.ndarray:
    return np.full(shape, 1400 + index * 4, dtype=np.uint16)


def sequence(
    directory: Path,
    count: int = 4,
    *,
    model: str = "NIKON Z 8",
    shape: tuple[int, int] = (48, 64),
) -> list[Path]:
    height, width = shape
    row_bytes = width * 7 // 4
    stride = (row_bytes + 15) & ~15
    paths = []
    for index in range(count):
        maker = Directory({})
        maker.numbers(0x45, 3, 4, 4, width - 8, height - 8)
        maker.set(0x51, 7, b"0102" + bytes(6) + b"\x0a\0" + bytes(12))
        maker.set(0x54, 7, b"0100" + struct.pack("<4i", 32, 256, 24, 256))
        maker.set(0x56, 7, b"0100" + struct.pack("<III", 123, index + 1, 1))
        mode = (4, 8, 16, 32).index(count) + 1
        maker.set(0x5E, 7, b"0100" + bytes([mode]) + bytes(23))
        # LibRaw reads this header before selecting its packed14 decoder by size.
        maker.set(0x96, 7, b"\x46\x30" + bytes(10))
        maker.set(0x97, 7, b"0803" + bytes(162))
        maker.numbers(0xA7, 4, index + 101)
        maker.set(0x0C, 5, struct.pack("<8I", 2, 1, 1, 1, 1, 1, 1, 1))
        info = Directory({})
        for code in (8, 11, 13):
            info.set(code, 7, b"0100" + bytes(18))
        coefficients = bytearray(b"0100" + bytes(100))
        for offset in (44, 28, 20, 80, 64, 56):
            struct.pack_into("<I", coefficients, offset + 4, 1048576)
        info.set(7, 7, bytes(coefficients))
        header = b"Nikon\0\x02\x11\0\0"
        raw = Directory({})
        for code, kind, values in (
            (256, 4, (width,)),
            (257, 4, (height,)),
            (258, 3, (14,)),
            (259, 3, (34713,)),
            (262, 3, (32803,)),
            (273, 4, (0,)),
            (277, 3, (1,)),
            (278, 4, (height,)),
            (279, 4, (height * stride,)),
            (33421, 3, (2, 2)),
        ):
            raw.numbers(code, kind, *values)
        raw.set(33422, 1, b"\0\1\1\2")
        raw.tags[0xC7D5] = Tag(0xC7D5, 7, 0, b"", [info], b"Nikon\0\x01\x03\0\0")
        samples = pixels(index, shape)
        # Independent scalar packing keeps the input separate from the NEFX writer.
        raw.images[273] = [
            b"".join(
                b"".join(
                    sum(int(v) << (14 * i) for i, v in enumerate(group)).to_bytes(
                        7, "little"
                    )
                    for group in row.reshape(-1, 4)
                )
                + bytes(stride - row_bytes)
                for row in samples
            )
        ]
        exif = Directory({37500: Tag(37500, 7, 0, b"", [maker], header)})
        exif.set(42033, 2, b"ci-synthetic-body\0")
        exif.set(36867, 2, b"2026:09:20 00:00:00\0")
        exif.set(33434, 5, struct.pack("<II", 1, 100))
        exif.set(33437, 5, struct.pack("<II", 8, 1))
        exif.set(37380, 10, struct.pack("<ii", 0, 1))
        exif.numbers(34855, 3, 100)
        root = Directory({34665: Tag(34665, 4, 1, bytes(4), [exif])})
        root.set(271, 2, b"NIKON CORPORATION\0")
        root.set(272, 2, model.encode("ascii") + b"\0")
        root.numbers(274, 3, 1)
        images = [Directory({}) for _ in range(6)]
        images[1] = raw
        root.tags[330] = Tag(330, 4, 6, bytes(24), images)
        path = directory / f"synthetic-{index:02d}.NEF"
        path.write_bytes(serialize(root))
        paths.append(path)
    return paths
