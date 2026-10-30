"""Authored Nikon HE/HE* (TicoRAW) fixtures for CI, never authentic-capture evidence.

``codestream`` encodes seeded quantized coefficients through every entropy path of
the camera profile: zero and vertical GCLI prediction, significance flags,
truncation levels from precinct quantization and refinement, partial code groups,
several slices and zero precinct padding. ``mosaic`` reconstructs the samples
those coefficients decode to. The golden file pins an independent decoder's
output for each fixture, which checks both.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import ci_fixture
from tiff_fixture import read, serialize

CAP = b"CONTACT_INTOPIX_\xef\xc0" + bytes(14)
# Nc, Ng, Ss, Bw, Fq/Br, Fslc/Ppoc/Cpih, NLx/NLy, Lh/Rl/Qpih/Fs/Rm.
PROFILE = bytes((4, 4, 8, 18, 0x44, 0x10, 0x51, 0x14))
VENDOR = bytes.fromhex("50887083f01523d149cd3f7f07")
CDT = bytes((14, 0x11, 0x51, 14, 0x11, 0x51, 14, 0x11, 0, 14, 0x11, 0x51))
# Gain and priority per band, in packet order.
WEIGHTS = bytes(
    (
        3,
        2,
        3,
        19,
        2,
        9,
        2,
        22,
        1,
        10,
        1,
        21,
        2,
        5,
        1,
        3,
        1,
        17,
        0,
        1,
        0,
        7,
        0,
        13,
        0,
        11,
    )
    + (2, 6, 1, 4, 1, 16, 0, 0, 0, 8, 0, 14, 1, 20, 0, 18, 0, 15, 0, 23, 0, 12, 0, 24)
)
SLICE_HEIGHT = 16


class Bits:
    def __init__(self) -> None:
        self.bits: list[int] = []

    def write(self, value: int, count: int) -> None:
        self.bits += [(value >> (count - 1 - i)) & 1 for i in range(count)]

    def unary(self, value: int) -> None:
        self.bits += [1] * value + [0]

    def bytes(self) -> bytes:
        padded = self.bits + [0] * (-len(self.bits) % 8)
        return np.packbits(np.array(padded, dtype=np.uint8)).tobytes()


def layout(width: int) -> tuple[list[tuple[int, int, int, int, int]], list, list[int]]:
    """Bands (component, x, width, row, lines), packets and weight indices."""
    lows = [width]
    for _ in range(5):
        lows.append(lows[-1] - lows[-1] // 2)
    bands = []
    for c in range(4):
        if c == 2:
            bands.append((c, 0, width, 0, 2))
            continue
        bands.append((c, 0, lows[5], 0, 1))
        bands += [(c, lows[i], lows[i - 1] - lows[i], 0, 1) for i in range(5, 0, -1)]
        bands += [(c, 0, lows[1], 1, 1), (c, lows[1], width - lows[1], 1, 1)]
    packets, weight = [], [0] * len(bands)
    order = 0
    for half in (0, 1):
        for c in range(4):
            packet = []
            for b, (component, _, _, row, lines) in enumerate(bands):
                if component == c and (lines == 2 or row == half):
                    packet.append((b, half if lines == 2 else 0))
                    if half == 0 or lines == 1:
                        weight[b] = order
                        order += 1
            packets.append(packet)
    return bands, packets, weight


def codestream(width: int, height: int, seed: int) -> tuple[bytes, list[np.ndarray]]:
    """Encode seeded coefficients; return the codestream and the scaled planes."""
    rng = np.random.default_rng(seed)
    w, h = width // 2, height // 2
    bands, packets, weight = layout(w)
    planes = [np.zeros((h, w), dtype=np.int64) for _ in range(4)]
    precincts = h // 2
    body = bytearray()
    for slice_index, first in enumerate(range(0, precincts, SLICE_HEIGHT)):
        body += b"\xff\x20\x00\x04" + slice_index.to_bytes(2, "big")
        top: dict[int, tuple[list[int], int]] = {}
        for p in range(first, min(precincts, first + SLICE_HEIGHT)):
            q, r = int(rng.integers(0, 7)), int(rng.integers(0, 26))
            modes = [int(m) for m in rng.integers(0, 4, len(bands))]
            gtli = []
            for b in range(len(bands)):
                gain, priority = WEIGHTS[2 * weight[b]], WEIGHTS[2 * weight[b] + 1]
                pump = int(priority < r)
                gtli.append(0 if q < gain + pump else min(q - gain - pump, 15))
            data = bytearray()
            for packet in packets:
                significance, codes, payload, signs = Bits(), Bits(), Bits(), Bits()
                for b, line in packet:
                    c, x0, band_width, row, _ = bands[b]
                    groups, t = -(-band_width // 4), gtli[b]
                    vertical, coded = modes[b] & 1, modes[b] & 2
                    above, above_t = top.get(b, ([0] * groups, t))
                    predicted = [max(v, above_t, t) for v in above]
                    target = predicted if vertical else [t] * groups
                    values = [
                        t + int(rng.integers(0, min(6, 18 - t) + 1))
                        for _ in range(groups)
                    ]
                    flags = []
                    for start in range(0, groups, 8):
                        span = range(start, min(groups, start + 8))
                        skip = coded and rng.random() < 0.3
                        skip = skip and all(target[i] - t <= 15 for i in span)
                        if skip:
                            for i in span:
                                values[i] = target[i]
                        if coded:
                            flags.append(int(skip))
                            significance.write(int(skip), 1)
                    for i, value in enumerate(values):
                        if coded and flags[i // 8]:
                            continue
                        if not vertical:
                            codes.unary(value - t)
                            continue
                        threshold = predicted[i] - t
                        delta = value - predicted[i]
                        codes.unary(
                            delta + threshold
                            if delta > threshold
                            else 2 * delta
                            if delta >= 0
                            else -2 * delta - 1
                        )
                    top[b] = (values, t)
                    coefficients = np.zeros(groups * 4, dtype=np.int64)
                    for i, value in enumerate(values):
                        planes_coded = value - t
                        if planes_coded <= 0:
                            continue
                        nibbles = rng.integers(0, 16, planes_coded)
                        nibbles[0] = rng.integers(1, 16)
                        for nibble in nibbles:
                            payload.write(int(nibble), 4)
                        for j in range(4):
                            v = 0
                            for nibble in nibbles:
                                v = v << 1 | (int(nibble) >> (3 - j) & 1)
                            v <<= t
                            if t:
                                total = 0
                                while v:
                                    total += v
                                    v >>= planes_coded + 1
                                v = total
                            coefficients[i * 4 + j] = v
                    for k in np.flatnonzero(coefficients):
                        negative = int(rng.integers(0, 2))
                        signs.write(negative, 1)
                        if negative:
                            coefficients[k] = -coefficients[k]
                    planes[c][2 * p + row + line, x0 : x0 + band_width] = (
                        coefficients[:band_width] * 16
                    )
                parts = [s.bytes() for s in (significance, codes, payload, signs)]
                header = Bits()
                header.write(0, 1)
                header.write(len(parts[2]), 20)
                header.write(len(parts[1]), 20)
                header.write(len(parts[3]), 15)
                data += header.bytes() + b"".join(parts)
            data += bytes(int(rng.integers(0, 4)))
            head = Bits()
            head.write(len(data), 24)
            head.write(q, 8)
            head.write(r, 8)
            for mode in modes:
                head.write(mode, 2)
            body += head.bytes() + data
    pih_tail = (
        (0).to_bytes(4, "big")
        + width.to_bytes(2, "big")
        + height.to_bytes(2, "big")
        + bytes(2)
        + SLICE_HEIGHT.to_bytes(2, "big")
        + PROFILE
        + VENDOR
    )
    main = (
        b"\xff\x10\xff\x50\x00\x22"
        + CAP
        + b"\xff\x12\x00\x27"
        + b"LCOD"
        + pih_tail
        + b"\xff\x13\x00\x0e"
        + CDT
        + b"\xff\x14\x00\x34"
        + WEIGHTS
    )
    stream = bytearray(main + body + b"\xff\x11")
    offset = main.index(b"LCOD")
    stream[offset : offset + 4] = len(stream).to_bytes(4, "big")
    return bytes(stream), planes


def synthesize(low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Inverse 5/3 lifting along the last axis with symmetric extension."""
    nl, nh = low.shape[-1], high.shape[-1]
    k = np.arange(nl)
    left, right = np.clip(k - 1, 0, nh - 1), np.clip(k, 0, nh - 1)
    even = low - ((high[..., left] + high[..., right] + 2) >> 2)
    k = np.arange(nh)
    odd = high + ((even[..., k] + even[..., np.where(k + 1 < nl, k + 1, k)]) >> 1)
    out = np.empty(low.shape[:-1] + (nl + nh,), dtype=np.int64)
    out[..., 0::2], out[..., 1::2] = even, odd
    return out


def mosaic(planes: list[np.ndarray]) -> np.ndarray:
    y, red, delta, blue = (p.copy() for p in planes)
    height, width = y.shape
    lows = [width]
    for _ in range(5):
        lows.append(lows[-1] - lows[-1] // 2)
    for plane in (y, red, blue):
        for level in range(5, 0, -1):
            n = lows[level - 1]
            plane[0::2, :n] = synthesize(
                plane[0::2, : lows[level]], plane[0::2, lows[level] : n]
            )
        plane[1::2] = synthesize(plane[1::2, : lows[1]], plane[1::2, lows[1] :])
        plane[:] = synthesize(plane[0::2].T, plane[1::2].T).T

    def at(plane: np.ndarray, dy: int, dx: int) -> np.ndarray:
        rows = np.clip(np.arange(height) + dy, 0, height - 1)
        cols = np.clip(np.arange(width) + dx, 0, width - 1)
        return plane[rows][:, cols]

    y -= (at(delta, -1, 0) + at(delta, -1, 1) + delta + at(delta, 0, 1)) >> 3
    delta += (at(y, 0, -1) + y + at(y, 1, -1) + at(y, 1, 0)) >> 2
    y -= (red + at(red, 0, 1) + at(blue, -1, 0) + blue) >> 3
    delta -= (at(blue, 0, -1) + blue + red + at(red, 1, 0)) >> 3
    red += (at(y, 0, -1) + y + at(delta, -1, 0) + delta) >> 2
    blue += (delta + at(delta, 0, 1) + y + at(y, 1, 0)) >> 2
    t = VENDOR
    black = (t[3] & 0x0F) << 8 | t[4]
    gain = (t[7] & 0x03) << 16 | t[8] << 8 | t[9]
    knee = t[10] << 8 | t[11]

    def curve(v: np.ndarray) -> np.ndarray:
        d = np.clip(v + 32768, 0, 65535) - knee
        scaled = (d * gain + 32768) >> 16
        q = np.where(d < 0, -(d * d), scaled * scaled)
        return np.clip(black + ((q + 163840) >> 18), 0, 16383)

    result = np.empty((2 * height, 2 * width), dtype=np.uint16)
    result[0::2, 0::2], result[0::2, 1::2] = curve(red), curve(y)
    result[1::2, 0::2], result[1::2, 1::2] = curve(delta), curve(blue)
    return result


def sequence(
    directory: Path,
    count: int,
    *,
    model: str,
    code: int,
    seed: int,
    shape: tuple[int, int] = (72, 68),
) -> tuple[list[Path], list[bytes], list[np.ndarray]]:
    """HE sources, their codestreams and the mosaics they decode to."""
    paths = ci_fixture.sequence(directory, count, model=model, shape=shape)
    streams, mosaics = [], []
    for index, path in enumerate(paths):
        stream, planes = codestream(shape[1], shape[0], seed + index)
        tree = read(path.read_bytes())
        maker = tree.tags[34665].children[0].tags[37500].children[0]
        data = bytearray(maker.tags[0x51].payload)
        data[10:12] = code.to_bytes(2, "little")
        maker.set(0x51, 7, bytes(data))
        # HE captures carry no lossless linearization table.
        del maker.tags[0x96]
        raw = tree.tags[330].children[1]
        raw.numbers(279, 4, len(stream))
        raw.images[273] = [stream]
        # Captures carry large previews. LibRaw 0.22 reads a lossless curve of up to
        # 32 KiB near the start of a Z5II HE file, which would run past a tiny file.
        preview = tree.tags[330].children[0]
        preview.numbers(513, 4, 0)
        preview.numbers(514, 4, 36864)
        preview.images[513] = [bytes(36864)]
        path.write_bytes(serialize(tree))
        streams.append(stream)
        mosaics.append(mosaic(planes))
    return paths, streams, mosaics
