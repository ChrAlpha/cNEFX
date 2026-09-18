"""Bounded NP3 container parsing and native Picture Control curve extraction."""

from __future__ import annotations

import os
import stat
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .picture_curve import generate_curve

MAX_BYTES = 4 * 1024 * 1024
MAX_RECORDS = 4096


@dataclass(frozen=True)
class PictureControlRecord:
    identifier: int
    payload: bytes


def read_np3(path: Path) -> bytes:
    """Read one bounded regular-file snapshot, rejecting observable mutations."""

    def open_nonblocking(name: str, flags: int) -> int:
        # A FIFO must not block before its opened descriptor can be inspected.
        return os.open(name, flags | getattr(os, "O_NONBLOCK", 0))

    with open(path, "rb", opener=open_nonblocking) as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("NP3 input must be a regular file.")
        if before.st_size > MAX_BYTES:
            raise ValueError("NP3 input exceeds the 4 MiB limit.")
        data = stream.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise ValueError("NP3 input exceeds the 4 MiB limit.")
        after = os.fstat(stream.fileno())
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if len(data) != before.st_size or any(
            getattr(before, field) != getattr(after, field) for field in fields
        ):
            raise ValueError("NP3 input changed during the read.")
    return data


def parse_np3(data: bytes) -> tuple[PictureControlRecord, ...]:
    """Parse an NP3/NCP record stream without interpreting unknown records."""
    if (
        type(data) is not bytes
        or not 8 <= len(data) <= MAX_BYTES
        or data[:4] != b"NCP\0"
    ):
        raise ValueError("Expected a bounded NCP Picture Control container.")
    offset = 4
    records: list[PictureControlRecord] = []
    while offset < len(data):
        if offset + 4 > len(data):
            raise ValueError("Truncated NP3 record identifier.")
        identifier = struct.unpack_from(">I", data, offset)[0]
        offset += 4
        if identifier == 0:
            if offset != len(data):
                raise ValueError("Unexpected bytes after the NP3 terminator.")
            return tuple(records)
        if len(records) >= MAX_RECORDS or offset + 4 > len(data):
            raise ValueError("Too many NP3 records or truncated record length.")
        length = struct.unpack_from(">I", data, offset)[0]
        offset += 4
        if length > len(data) - offset:
            raise ValueError("NP3 record payload exceeds the file bounds.")
        records.append(PictureControlRecord(identifier, data[offset : offset + length]))
        offset += length
    raise ValueError("Missing NP3 container terminator.")


def _ascii_name(payload: bytes) -> str:
    if len(payload) != 20 or b"\0" not in payload:
        raise ValueError("NP3 name record must be 20-byte NUL-terminated ASCII.")
    content, padding = payload.split(b"\0", 1)
    if any(padding):
        raise ValueError("NP3 name padding must be zero.")
    return content.decode("ascii")


def extract_curve(data: bytes) -> dict[str, Any]:
    """Return metadata and the regenerated 257-entry native curve table."""
    records = parse_np3(data)
    if len({record.identifier for record in records}) != len(records):
        raise ValueError("Duplicate NP3 records have unverified precedence.")
    fields = {record.identifier: record.payload for record in records}
    if fields.get(0x100) != b"0310":
        raise ValueError("Only observed NP3 container version 0310 is supported.")
    profile = fields.get(0x300)
    if profile is None or len(profile) != 2:
        raise ValueError("NP3 profile record is missing or malformed.")
    if profile != b"\0\x20":
        raise ValueError("Only observed Flexible Color profile 32 is supported.")
    name = fields.get(0x200)
    if name is None:
        raise ValueError("NP3 name record is missing.")
    curve = fields.get(0x2)
    if curve is None:
        raise ValueError("NP3 tone-curve record 0x0002 is missing.")
    return {
        "version": fields[0x100].decode("ascii"),
        "profile_id": struct.unpack(">H", profile)[0],
        "name": _ascii_name(name),
        "record_count": len(records),
        "curve_samples": list(generate_curve(curve)),
        "limitations": [
            "Curve table generation only; full Picture Control rendering is not "
            "performed.",
            "Camera import and Lightroom acceptance are not established by this "
            "command.",
        ],
    }
