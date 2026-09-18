"""Bounded native import of decrypted Z8 ColorBalance 0803 records."""

from __future__ import annotations

import struct

# Asteroid11 imports packed fields into an aligned tagStrColorBalance union.
FIELDS = (
    (0x10, 4),
    (0x14, 36),
    (0x38, 20),
    (0x4C, 15),
    (0x5C, 2),
    (0x60, 16),
    (0x70, 4),
    (0x74, 24),
    (0x8C, 1),
    (0x90, 8),
    (0x98, 10),
    (0xA2, 2),
    (0xA4, 2),
    (0xA6, 18),
    (0xB8, 4),
)


def import_decoded(data: bytes) -> bytes:
    """Reconstruct type0x109 from an already decrypted little-endian record.

    Decryption and its metadata keys are the caller's responsibility. This does
    not accept the encrypted MakerNote payload directly or infer EXIF ISO.
    """
    if type(data) is not bytes or len(data) != 166 or data[:4] != b"0803":
        raise ValueError("Expected a decoded 166-byte ColorBalance 0803 record.")
    output = bytearray(0xD08)
    output[:4] = data[:4]
    source = 0
    for target, length in FIELDS:
        output[target : target + length] = data[source : source + length]
        source += length
    return bytes(output)


def iso_code(data: bytes) -> int:
    """Read property0x100016b after importing the decoded packed record."""
    return struct.unpack_from("<H", import_decoded(data), 0xA0)[0]
