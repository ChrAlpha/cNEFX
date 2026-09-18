"""Read a locally supplied Nikon resource container without executing vendor code.

Recovered from PixelShiftCombine.dll RVAs 0x33e790 and 0x33e090.
Prints record metadata and optional decoded hashes, never proprietary payloads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import zlib
from pathlib import Path

HEADER = struct.Struct("<8I")


def records(data: bytes) -> list[dict[str, int]]:
    if len(data) < HEADER.size:
        raise ValueError("Truncated resource header.")
    if HEADER.unpack_from(data)[:2] != (0x4E4F4B4E, 0x01010000):
        raise ValueError("Unsupported resource signature or version.")
    entries = []
    offset = HEADER.size
    while offset < len(data):
        if offset + HEADER.size > len(data):
            raise ValueError("Truncated record header.")
        kind, key, alias, flags, packed, unpacked, _, _ = HEADER.unpack_from(
            data, offset
        )
        if offset + HEADER.size + packed > len(data):
            raise ValueError("Record payload exceeds container size.")
        entries.append(
            dict(
                offset=offset,
                type=kind,
                id=key,
                alias_id=alias,
                flags=flags,
                stored_bytes=packed,
                decoded_bytes=unpacked,
            )
        )
        offset += HEADER.size + packed
    return entries


def decode(data: bytes, record: dict[str, int]) -> bytes:
    packed = record["stored_bytes"]
    unpacked = record["decoded_bytes"]
    if packed > 8 * 1024**2 or unpacked > 16 * 1024**2:
        raise ValueError("Resource exceeds the observed vendor size limits.")
    if record["flags"] & 3 != 3 or record["alias_id"]:
        raise ValueError("Expected a direct record with both transform flags set.")
    start = record["offset"] + HEADER.size
    payload = bytearray(data[start : start + packed])
    state = 0x10DCD
    for index in range(len(payload)):
        state = (state * 0x6C078965 + 1) & 0xFFFFFFFF
        payload[index] ^= (state >> 13) & 255
    inflater = zlib.decompressobj()
    decoded = inflater.decompress(payload, unpacked + 1)
    if len(decoded) != unpacked or not inflater.eof or inflater.unused_data:
        raise ValueError("Decoded resource size or stream boundary is invalid.")
    return decoded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path)
    parser.add_argument("--type", type=lambda value: int(value, 0))
    parser.add_argument("--id", type=lambda value: int(value, 0))
    args = parser.parse_args()
    if (args.type is None) != (args.id is None):
        parser.error("--type and --id must be supplied together.")
    try:
        data = args.file.read_bytes()
        entries = records(data)
        report: dict[str, object] = {
            "schema_version": 1,
            "evidence_kind": "resource_container_inspection",
            "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
            "record_count": len(entries),
        }
        if args.id is None:
            report["records"] = entries
        else:
            key = args.id
            visited: set[int] = set()
            while True:
                if key in visited:
                    raise ValueError("Resource alias cycle.")
                visited.add(key)
                matches = [
                    r for r in entries if r["type"] == args.type and r["id"] == key
                ]
                if len(matches) != 1:
                    raise ValueError("Resource lookup must select exactly one record.")
                record = matches[0]
                if record["alias_id"] == 0:
                    break
                key = record["alias_id"]
            payload = decode(data, record)
            report["record"] = record
            report["decoded_sha256"] = hashlib.sha256(payload).hexdigest()
            report["alias_chain"] = sorted(visited)
    except (OSError, ValueError, zlib.error) as exc:
        parser.exit(1, f"inspect_prm: {exc}\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
