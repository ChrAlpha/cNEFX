"""Bounded native decryption and import of Z8 ColorBalance 0803 records."""

from __future__ import annotations

import hashlib
import struct

from nx_ctudio import _tiff

DECRYPT_TABLE_HASH = "79f0e716581a40357e6e6872aa64459f91c5cb572b14576a74773fb74bdfd3c4"


def decrypt(data: bytes, serial: int, count: int, table: bytes) -> bytes:
    """Decode 0803 using the public Nikon stream recurrence and external tables.

    The first four version bytes stay plaintext. This reversible transform has
    no authentication; successful decoding cannot verify supplied metadata keys.
    """
    if type(data) is not bytes or len(data) != 166 or data[:4] != b"0803":
        raise ValueError("Expected an encrypted 166-byte ColorBalance 0803 record.")
    if any(type(v) is not int or not 0 <= v <= 0xFFFFFFFF for v in (serial, count)):
        raise ValueError("Serial and shutter count must be unsigned 32-bit integers.")
    if (
        type(table) is not bytes
        or len(table) != 512
        or hashlib.sha256(table).hexdigest() != DECRYPT_TABLE_HASH
    ):
        raise ValueError("Only the pinned external Nikon decoding table is supported.")
    key = 0
    for shift in (0, 8, 16, 24):
        key ^= (count >> shift) & 255
    ci, cj, ck = table[serial & 255], table[256 + key], 0x60
    output = bytearray(data)
    for index in range(4, len(output)):
        cj = (cj + ci * ck) & 255
        ck = (ck + 1) & 255
        output[index] ^= cj
    return bytes(output)


def decrypt_source(root: _tiff.Directory, table: bytes) -> bytes:
    """Decode the bounded Z8 MakerNote record using its own serial/count tags."""
    try:
        if root.tags[272].payload != b"NIKON Z 8\0":
            raise ValueError("Only Nikon Z8 source decryption is validated.")
        maker = root.tags[34665].children[0].tags[37500].children[0]
        raw, serial, count = (maker.tags[k] for k in (0x97, 0x1D, 0xA7))
        if (
            raw.kind != 7
            or raw.count != 166
            or serial.kind != 2
            or not 2 <= serial.count <= 11
            or serial.count != len(serial.payload)
            or not serial.payload.endswith(b"\0")
            or not serial.payload[:-1].isdigit()
            or count.kind != 4
            or count.count != 1
            or len(count.payload) != 4
        ):
            raise ValueError("Unsupported ColorBalance or decryption-key tag layout.")
        return decrypt(
            raw.payload, int(serial.payload[:-1]), count.integers()[0], table
        )
    except (KeyError, IndexError) as exc:
        raise ValueError(
            "Missing required Z8 ColorBalance decryption metadata."
        ) from exc


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
