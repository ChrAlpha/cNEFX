"""Native Kaiji scalar RGB evaluation using an externally supplied generated table."""

from __future__ import annotations

import struct
from collections.abc import Sequence


def _clip(value: int, low: int = 0, high: int = 4095) -> int:
    return max(low, min(high, value))


def _signed(value: int) -> int:
    return (value + 2**31) % 2**32 - 2**31


def _trunc(value: int, shift: int) -> int:
    return (value + ((1 << shift) - 1 if value < 0 else 0)) >> shift


def _round(value: int, shift: int) -> int:
    value = _signed(value)
    value = _signed(value + ((1 << (shift - 1)) if value >= 0 else -(1 << (shift - 1))))
    return _trunc(value, shift)


def _sector(first: int, second: int, scale: int) -> tuple[int, int, int]:
    a, b = abs(first), abs(second)
    if scale == 512:
        a, b = min(4095, a), min(4095, b)
    fraction = min(scale - 1, (b * scale + (a + b) // 2) // (a + b)) if a + b else 0
    quadrant = (3 if first >= 0 else 2) if second > 0 else (1 if first > 0 else 0)
    part, remainder = divmod(fraction, scale // 8)
    starts = (
        (0, 1, 2, 3, 4, 5, 6, 7),
        (8, 9, 10, 11, 12, 13, 14, 15),
        (0, 17, 18, 19, 20, 21, 22, 23),
        (8, 25, 26, 27, 28, 29, 30, 31),
    )
    left = starts[quadrant][part]
    right = starts[quadrant][part + 1] if part < 7 else (16 if quadrant < 2 else 24)
    return left, right, remainder


class ScalarTable:
    """Research evaluator for recovered mode0/2 RGB output, selector-independent."""

    def __init__(self, data: bytes):
        if type(data) is not bytes or len(data) != 0x34370:
            raise ValueError("Expected a 0x34370-byte generated Kaiji table.")
        self.data = data

    def u(self, offset: int) -> int:
        return struct.unpack_from("<H", self.data, offset)[0]

    def s(self, offset: int) -> int:
        return struct.unpack_from("<h", self.data, offset)[0]

    def i(self, offset: int) -> int:
        return struct.unpack_from("<i", self.data, offset)[0]

    def evaluate(self, indices: Sequence[int], grid: int = 33) -> tuple[int, int, int]:
        if type(grid) is not int or grid != 33:
            raise ValueError("Only the validated 33-node grid is supported.")
        if len(indices) != 3 or any(
            type(v) is not int or not 0 <= v < grid for v in indices
        ):
            raise ValueError("Expected three in-grid integer indices.")
        codes = [
            min(4095, int(min(4095.0, v * (1.0 / (grid - 1)) * 4095.0) + 0.5))
            for v in indices
        ]
        rgb = [
            self.u(offset + code * 2)
            for offset, code in zip((0x1D4, 0x21D4, 0x41D4), codes, strict=True)
        ]
        floor = -self.s(0x4C)
        rgb = [max(floor, value) if value < 4096 else 4095 for value in rgb]
        transformed = [
            _round(
                sum(
                    _round(self.s(0x61D4 + row * 6 + col * 2) * rgb[col], 6)
                    for col in range(3)
                ),
                3,
            )
            for row in range(3)
        ]
        y = _clip(transformed[0])
        u, v = (_clip(value, -4096) for value in transformed[1:])
        left, right, fraction = _sector(u, v, 512)

        def sector(offset: int, shift: int = 6, scale: int = 1) -> int:
            a, b = self.s(offset + left * 2), self.s(offset + right * 2)
            return a * scale + _round((b - a) * fraction, shift)

        blend = [_clip(sector(0x213C6 + i * 64, 4, 4), -2048, 2047) for i in range(6)]
        weights = [
            _clip(sector(offset), -512, 511)
            for offset in (
                0x65AA,
                0x65EA,
                0x662A,
                0x666A,
                0x66AA,
                0x66EA,
                0x672A,
                0x676A,
                0x6528,
                0x67AA,
            )
        ]
        au, av = min(abs(u), 4095), min(abs(v), 4095)
        chroma = _clip(_round(abs(av - au) + 2 * (av + au), 2), 0, 32767)
        domains = [chroma] * 12
        domains[5] = domains[6] = y
        domains[10] = 4095 - y
        lookup = [
            _clip(_signed(domain * self.i(0x61F8 + index * 4)))
            for index, domain in enumerate(domains)
        ]

        def lut(bank: int) -> int:
            return self.u(0x6BC6 + bank * 8192 + lookup[bank] * 2)

        def factor(bank: int, weight: int) -> int:
            return _clip(256 - weight + _round(lut(bank) * weight, 8), 0, 2047)

        saturation = _clip(
            _round(factor(0, weights[0]) * factor(1, weights[1]) * lut(2), 16), 0, 2047
        )
        luminance = _clip(
            _round(factor(3, weights[2]) * factor(4, weights[3]), 8), 0, 2047
        )
        y_factor = _clip(
            _round(factor(5, weights[4]) * factor(6, weights[5]), 8), 0, 2047
        )
        rotation = (
            _clip(_round(lut(7) * weights[6], 8), -512, 511)
            + _clip(_round(lut(8) * weights[7], 8), -512, 511)
        ) >> 1
        rotation = _clip(_clip(rotation, -256, 255) + 256, 0, 511)
        cosine, sine = self.s(0x20BC6 + rotation * 2), self.s(0x20FC6 + rotation * 2)
        correction = _clip(256 - _clip(_round(lut(9) * lut(10), 8), 0, 1023), 0, 1023)
        correction = _clip(
            256 - weights[8] + _round(correction * weights[8], 8), 0, 1023
        )
        y_offset = _round(self.s(0x1EBC6 + y * 2) * weights[9], 8)
        y_offset = _clip(y_offset, -lut(11), lut(11))
        y = _clip(
            _round(
                _trunc(_signed(blend[5] * v + blend[4] * u), 2)
                + luminance * y
                + y_offset * 256,
                8,
            )
        )
        saturation = _clip(_signed(y_factor * saturation * correction) >> 16)
        a = _clip(
            _round((_signed(blend[1] * v + blend[0] * u) >> 8) * saturation, 10), -4096
        )
        b = _clip(
            _round((_signed(blend[3] * v + blend[2] * u) >> 8) * saturation, 10), -4096
        )
        u = _clip(_round(a * cosine - b * sine, 10), -4096)
        v = _clip(_round(a * sine + b * cosine, 10), -4096)
        y = self.u(0x21758 + y * 2)
        u = _clip(
            u + _trunc(_signed((self.s(0x2154C) - u) * self.i(0x23758)), 10), -4096
        )
        v = _clip(
            v + _trunc(_signed((self.s(0x2154A) - v) * self.i(0x23758)), 10), -4096
        )
        transformed = []
        for row in range(3):
            value = sum(
                _trunc(_signed(self.s(0x2375C + row * 6 + col * 2) * channel), 7)
                for col, channel in enumerate((y, u, v))
            )
            transformed.append(_trunc(value, 3) + self.s(0x2376E + row * 2))
        y = _clip(transformed[0])
        u, v = (_clip(value, -4096) for value in transformed[1:])
        left, right, fraction = _sector(u, v, 128)
        base = self.u(0x23774 + left * 2)
        delta = _clip(
            _round((self.u(0x23774 + right * 2) - base) * fraction + base * 16, 4)
        )
        gain = self.u(0x23836 + y * 2)
        u, v = _clip(gain * u >> 10, -4096), _clip(gain * v >> 10, -4096)
        au, av = min(abs(u), 4095), min(abs(v), 4095)
        chroma = _clip(
            _round(min(abs(au - av), 4095) + min(au + av, 4095) + au + av, 2)
        )
        index = _clip((chroma * delta + y * 1024) >> 10)
        y = _clip(self.i(0x26044 + index * 4) * y >> 20)
        gain = self.i(0x2A044 + index * 4)
        u, v = _clip(u * gain >> 20, -4096), _clip(v * gain >> 20, -4096)
        y = self.u(0x2E246 + y * 2)
        matrix_rgb = [
            _clip(
                sum(
                    self.s(0x30246 + row * 6 + col * 2) * channel >> 10
                    for col, channel in enumerate((y, u, v))
                )
                + self.s(0x30258 + row * 2),
                -8192,
                8191,
            )
            for row in range(3)
        ]
        maximum, minimum = _clip(max(matrix_rgb), 0, 8191), min(matrix_rgb)
        shift = self.u(0x30262) & 31
        hi_index = _clip(_signed((maximum - self.u(0x30260)) << shift))
        lo_index = _clip(_signed((self.u(0x3025E) - minimum) << shift))
        hi, lo = self.u(0x3036C + hi_index * 2), self.u(0x3236C + lo_index * 2)
        correction = (
            _signed(
                _trunc(_signed((4095 - y) * (2048 - lo)), 10) * self.u(0x30266)
                + _trunc(_signed((hi - 2048) * y), 10) * self.u(0x30264)
            )
            >> 8
        )
        y = _clip(y + _clip(correction, -4096))
        gain = _signed(lo * hi) >> 11
        u, v = (
            _clip(_signed(gain * u) >> 11, -2048, 2047),
            _clip(_signed(gain * v) >> 11, -2048, 2047),
        )
        luminance = (y << 4) * 1.5259021896696422e-05
        r = (v << 4) * 2.139467460135805e-05 + luminance
        b = (u << 4) * 2.7026779583428704e-05 + luminance
        g = (luminance - (u << 4) * 5.271992065308613e-06) - (
            v << 4
        ) * 1.0904097047379264e-05
        return tuple(
            int(max(0.0, min(1.0, value)) * 65280.0 + 0.5) for value in (r, g, b)
        )
