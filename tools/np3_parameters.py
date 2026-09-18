"""Recovered NP3 0310 record-to-Nikon-parameter-buffer mapping."""

from __future__ import annotations

import struct

from inspect_np3 import Record, describe

BASE_PAIRS = {
    0x600: 57,
    0x700: 61,
    0x800: 63,
    0x900: 65,
    0xA00: 67,
    0xB00: 69,
    0xE00: 73,
    0xF00: 90,
    0x1000: 94,
    0x1100: 96,
    0x1200: 98,
    0x1300: 92,
    0x1400: 53,
    0x1500: 55,
    0x1600: 59,
    0x1700: 75,
    0x1800: 77,
}
BASE_SINGLE = {0x400: 52, 0xC00: 71, 0xD00: 72}


def parameter_blocks(records: tuple[Record, ...]) -> dict[str, bytes]:
    """Predict buffers from Rome2 RVA0x137080 for the interpreted Flexible subset."""
    describe(records)
    fields = {record.identifier: record.payload for record in records}
    base, flexible, curve = bytearray(108), bytearray(88), bytearray(578)
    base[:8] = fields[0x100] * 2
    base[8:28] = fields[0x200]
    base[48:50] = fields[0x300][::-1]
    flexible[:4] = b"0100"
    for identifier, offset in BASE_PAIRS.items():
        if identifier in fields:
            payload = fields[identifier]
            if len(payload) != 2:
                raise ValueError("Unsupported base parameter record size.")
            base[offset : offset + 2] = payload
    for identifier, offset in BASE_SINGLE.items():
        if identifier in fields:
            payload = fields[identifier]
            if len(payload) != 2:
                raise ValueError("Unsupported single-value parameter record size.")
            base[offset] = payload[0]
    for identifier in range(0x1900, 0x1F00, 0x100):
        offset = 12 + (identifier - 0x1900) // 0x100 * 2
        flexible[offset : offset + 2] = fields[identifier]
    flexible[24:52] = fields[0x1F00]
    grading = fields[0x2000]
    flexible[52:72] = grading
    for offset in (52, 56, 60):
        flexible[offset : offset + 2] = flexible[offset : offset + 2][::-1]
    if 2 in fields:
        curve[:] = fields[2]
        for offset in range(64, 578, 2):
            curve[offset : offset + 2] = curve[offset : offset + 2][::-1]
    comment = fields.get(0x10100, b"")
    extra = fields.get(0x10200, b"")
    if len(comment) > 2000 or len(extra) > 256:
        raise ValueError("Comment exceeds the observed Nikon output buffer.")
    result = {
        "base": bytes(base),
        "flexible": bytes(flexible),
        "curve": bytes(curve),
        "comment": comment.ljust(2000, b"\0"),
        "extra": extra.ljust(256, b"\0"),
    }
    if struct.unpack_from("<H", result["base"], 48)[0] != 32:
        raise ValueError("Unexpected profile mapping.")
    return result
