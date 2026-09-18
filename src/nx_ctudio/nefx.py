"""Native NEFX serialization for the observed little-endian Z8 layout."""

from __future__ import annotations

import binascii
import struct
from datetime import datetime
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from . import __version__
from ._tiff import Directory, read, serialize


def preview(samples: NDArray[np.uint16]) -> NDArray[np.uint16]:
    """Reproduce Nikon's private RGB preview using bounded row accumulators."""
    height, width = samples.shape[:2]
    xs = np.trunc(np.arange(641, dtype=np.float64) / (640 / width)).astype(np.int64)
    ys = np.trunc(np.arange(425, dtype=np.float64) / (424 / height)).astype(np.int64)
    x0, x1 = xs[:-1], np.minimum(xs[1:], width - 1)
    result = np.zeros((424, 640, 3), dtype="<u2")
    for row in range(424):
        y0, y1 = ys[row], min(ys[row + 1], height - 1)
        if y1 <= y0:
            continue
        columns = samples[y0:y1].sum(axis=0, dtype=np.uint64)
        integral = np.pad(columns.cumsum(axis=0), ((1, 0), (0, 0)))
        totals = (integral[x1] - integral[x0]) & 0xFFFFFFFF
        area = (x1 - x0) * (y1 - y0)
        valid = area > 0
        result[row, valid] = (totals[valid] // area[valid, None]).astype(np.uint16)
    return result


def pack14(samples: NDArray[np.uint16]) -> bytes:
    height, width, _ = samples.shape
    if width % 4:
        raise ValueError("The observed NEFX packing requires width divisible by four.")
    length = width * 3 * 7 // 4
    stride = (length + 15) & ~15
    output = np.zeros((height, stride), dtype=np.uint8)
    shifts = np.arange(7, dtype=np.uint64) * 8
    for start in range(0, height, 64):
        block = samples[start : start + 64]
        pixels = block.astype(np.uint64).reshape(len(block), -1, 4)
        words = pixels[..., 0] | (pixels[..., 1] << 14)
        words |= (pixels[..., 2] << 28) | (pixels[..., 3] << 42)
        output[start : start + len(block), :length] = (
            ((words[..., None] >> shifts) & 255)
            .astype(np.uint8)
            .reshape(len(block), -1)
        )
    return output.tobytes()


def _patch(
    directory: Directory,
    code: int,
    updates: list[tuple[int, bytes]],
    *,
    crc: bool = False,
) -> None:
    tag = directory.tags[code]
    data = bytearray(tag.payload)
    if tag.kind != 7 or data[:4] != b"0100":
        raise ValueError(f"Unsupported Nikon metadata block {code:#06x}.")
    for offset, value in updates:
        if offset + len(value) > len(data) - (2 if crc else 0):
            raise ValueError("Truncated Nikon metadata update block.")
        data[offset : offset + len(value)] = value
    if crc:
        data[-2:] = struct.pack("<H", binascii.crc_hqx(data[:-2], 65535))
    directory.set(code, tag.kind, bytes(data))


def encode(source: bytes, samples: NDArray[np.uint16], count: int) -> bytes:
    """Preserve source metadata and JPEGs; Adobe XMP defaults remain unchanged."""
    if count not in (4, 8, 16, 32):
        raise ValueError("Unsupported pixel-shift input count.")
    if (
        samples.ndim != 3
        or samples.shape[2] != 3
        or samples.dtype != np.uint16
        or not samples.size
        or int(samples.max()) > 16383
    ):
        raise ValueError("Expected nonempty RGB uint16 samples in the 14-bit range.")
    root = read(source)
    subimages = root.tags[330].children
    if len(subimages) != 6:
        raise ValueError("Expected the observed six-SubIFD Nikon layout.")
    raw = subimages[1]
    if (
        raw.tags[262].integers() != (32803,)
        or set(raw.images) != {273}
        or len(raw.images[273]) != 1
    ):
        raise ValueError("Expected the observed single-strip source Bayer RAW.")
    maker = root.tags[34665].children[0].tags[37500].children[0]
    info = raw.tags[0xC7D5].children[0]
    scale = 2 if count in (16, 32) else 1
    crop = maker.tags[0x45].integers()
    height, width = samples.shape[:2]
    if (width, height) != (crop[2] * scale, crop[3] * scale):
        raise ValueError("RGB dimensions do not match the source crop and mode.")
    for code, kind, values in (
        (256, 4, (width,)),
        (257, 4, (height,)),
        (258, 3, (14, 14, 14)),
        (259, 3, (34713,)),
        (262, 3, (2,)),
        (277, 3, (3,)),
        (278, 4, (height,)),
        (284, 3, (1,)),
    ):
        raw.numbers(code, kind, *values)
    raw.images[273] = [pack14(samples)]
    root.set(305, 2, f"NX Ctudio {__version__}\0".encode("ascii"))
    root.set(306, 2, datetime.now().strftime("%Y:%m:%d %H:%M:%S\0").encode("ascii"))
    if 0xCD41 in root.tags:
        root.set(0xCD41, root.tags[0xCD41].kind, b"")
    maker.numbers(0x45, 3, 0, 0, width, height)
    compression = bytearray(maker.tags[0x51].payload)
    if len(compression) != 24 or compression[:4] != b"0102":
        raise ValueError("Unsupported Nikon compression metadata.")
    compression[10:12] = b"\x0a\0"
    maker.set(0x51, 7, bytes(compression))
    for code in (0x58, 0x96):
        maker.tags.pop(code, None)
    if 0x68 in maker.tags:
        _patch(maker, 0x68, [(4, b"\0")])
    if scale == 2:
        for node, code, crc in ((maker, 0x54, False), (info, 8, True)):
            data = node.tags[code].payload
            updates = [
                (
                    offset,
                    struct.pack("<i", struct.unpack_from("<i", data, offset)[0] * 2),
                )
                for offset in (4, 12)
            ]
            _patch(node, code, updates, crc=crc)
        if 0xB7 in maker.tags:
            tag = maker.tags[0xB7]
            if len(tag.payload) < 8:
                raise ValueError("Truncated autofocus metadata.")
            maker.set(0xB7, tag.kind, tag.payload[:8] + bytes(len(tag.payload) - 8))
    mode = maker.tags[0x5E].payload[4]
    marker = b"0100\x02\0" + bytes((mode, (4, 8, 16, 32).index(count) + 1)) + bytes(8)
    maker.set(0xE2D, 7, marker)
    header = struct.pack("<4sIHH4B", b"0100", 16, 640, 424, 3, 14, 14, 14)
    maker.set(0xE2E, 7, header + preview(samples).tobytes())
    for code in (7, 15):
        info.tags.pop(code, None)
    _patch(info, 11, [(4, struct.pack("<4H", 0, 0, width, height))], crc=True)
    _patch(info, 13, [(4, b"\x0a\0")], crc=True)
    return serialize(root)


def write_nefx(
    path: Path, source: bytes, samples: NDArray[np.uint16], count: int
) -> None:
    data = encode(source, samples, count)
    with path.open("xb") as stream:
        stream.write(data)
