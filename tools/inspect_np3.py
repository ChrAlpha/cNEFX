"""Inspect observed NP3 record structure without applying Picture Control colors.

Parameter names cross-reference ssssota/nikon-flexible-color-picture-control,
commit ed0397fbf1e4d4f2bc11ab3070522c613667d359 (MIT). File structure and
roundtrips are checked independently; Nikon acceptance remains a separate gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

MAX_BYTES = 4 * 1024 * 1024
MAX_RECORDS = 4096
PARAMETERS = {
    0x600: ("sharpening", 4),
    0x700: ("clarity", 4),
    0x1600: ("mid_range_sharpening", 4),
    0x1900: ("contrast", 1),
    0x1A00: ("highlights", 1),
    0x1B00: ("shadows", 1),
    0x1C00: ("white_level", 1),
    0x1D00: ("black_level", 1),
    0x1E00: ("saturation", 1),
}
COLORS = ("red", "orange", "yellow", "green", "cyan", "blue", "purple", "magenta")


@dataclass(frozen=True)
class Record:
    identifier: int
    payload: bytes


def parse(data: bytes) -> tuple[Record, ...]:
    if not 8 <= len(data) <= MAX_BYTES or data[:4] != b"NCP\0":
        raise ValueError("Expected a bounded NCP-signature record container.")
    offset, records = 4, []
    while offset < len(data):
        if offset + 4 > len(data):
            raise ValueError("Truncated record identifier.")
        identifier = struct.unpack_from(">I", data, offset)[0]
        offset += 4
        if identifier == 0:
            if offset != len(data):
                raise ValueError("Unexpected data after the container terminator.")
            return tuple(records)
        if len(records) >= MAX_RECORDS or offset + 4 > len(data):
            raise ValueError("Too many records or truncated record length.")
        length = struct.unpack_from(">I", data, offset)[0]
        offset += 4
        if length > len(data) - offset:
            raise ValueError("Record payload exceeds the file bounds.")
        records.append(Record(identifier, data[offset : offset + length]))
        offset += length
    raise ValueError("Missing container terminator.")


def serialize(records: tuple[Record, ...]) -> bytes:
    if len(records) > MAX_RECORDS:
        raise ValueError("Too many records.")
    result = bytearray(b"NCP\0")
    for record in records:
        if not 0 < record.identifier <= 0xFFFFFFFF:
            raise ValueError("Invalid record identifier.")
        if len(result) + 8 + len(record.payload) + 4 > MAX_BYTES:
            raise ValueError("Serialized container exceeds the supported size.")
        result.extend(struct.pack(">II", record.identifier, len(record.payload)))
        result.extend(record.payload)
    result.extend(bytes(4))
    return bytes(result)


def _text(data: bytes, encoding: str) -> str:
    content, separator, padding = data.partition(b"\0")
    if not separator or any(padding):
        raise ValueError("Text record lacks a terminator or contains nonzero padding.")
    return content.decode(encoding, errors="strict")


def _scaled(value: int, scale: int, *, word: bool = False) -> int | None:
    if value in ((65535,) if word else (0, 1, 255)):
        return None
    if scale == 0:
        raise ValueError("Ordinary Flexible Color value has a zero scale.")
    return math.trunc((value - (32768 if word else 128)) / scale)


def describe(records: tuple[Record, ...]) -> dict:
    identifiers = [record.identifier for record in records]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Duplicate records have unverified semantic precedence.")
    fields = {record.identifier: record.payload for record in records}
    if not records or records[0] != Record(0x100, b"0310"):
        raise ValueError("Only the observed NP3 version 0310 is interpreted.")
    if fields.get(0x300) != b"\0\x20":
        raise ValueError("Only the observed Flexible Color profile is interpreted.")
    name = fields.get(0x200, b"")
    if len(name) != 20:
        raise ValueError("Expected a 20-byte Picture Control name record.")
    result = {"version": "0310", "profile_id": 32, "name": _text(name, "ascii")}
    parameters = {}
    for identifier, (label, divisor) in PARAMETERS.items():
        value = fields.get(identifier, b"")
        if len(value) != 2:
            raise ValueError(f"Missing or malformed parameter {label}.")
        sentinel = value[0] in (0, 1, 255)
        supported = identifier >= 0x1900 or value[1] == divisor
        decoded = None
        if not sentinel and supported:
            decoded = (
                _scaled(*value) if identifier >= 0x1900 else (value[0] - 128) / divisor
            )
        parameters[label] = {
            "encoded_value": value[0],
            "scale_byte": value[1],
            "value": decoded,
            "state": "sentinel"
            if sentinel
            else "numeric"
            if supported
            else "unverified_scale",
        }
    result["parameters"] = parameters
    blender = fields.get(0x1F00, b"")
    grading = fields.get(0x2000, b"")
    if len(blender) != 28 or len(grading) != 20:
        raise ValueError("Missing or malformed Flexible Color adjustment records.")
    result["color_blender"] = {
        color: dict(
            zip(
                ("hue", "chroma", "brightness"),
                (
                    _scaled(v, blender[24 + channel])
                    for channel, v in enumerate(blender[i * 3 : i * 3 + 3])
                ),
                strict=True,
            )
        )
        for i, color in enumerate(COLORS)
    }
    result["color_blender_suffix_hex"] = blender[24:].hex()
    result["color_grading"] = {
        region: {
            "hue": _scaled(
                struct.unpack_from(">H", grading, i * 4)[0], grading[12], word=True
            ),
            "chroma": _scaled(grading[i * 4 + 2], grading[13]),
            "brightness": _scaled(grading[i * 4 + 3], grading[14]),
        }
        for i, region in enumerate(("highlights", "mid_tone", "shadows"))
    }
    result["color_grading"].update(
        blending=_scaled(grading[16], grading[17]),
        balance=_scaled(grading[18], grading[19]),
    )
    result["sentinel_policy"] = (
        "Null means an encoded sentinel, not an ordinary numeric value. "
        "Nikon normalizes Flexible Color sentinels to zero, except blending to 50. "
        "Base sharpening/clarity sentinel behavior is not established here."
    )
    result["color_grading_suffix_hex"] = grading[12:].hex()
    if 0x10100 in fields:
        comment = fields[0x10100]
        if not comment or len(comment) % 2:
            raise ValueError("Expected nonempty even-sized UTF-8 comment storage.")
        result["comment"] = _text(comment, "utf-8")
    if 2 in fields:
        curve = fields[2]
        if len(curve) != 578 or curve[:2] != b"I0" or curve[8] > 20:
            raise ValueError("Unsupported tone-curve record layout.")
        result["tone_curve"] = {
            "points": [list(curve[9 + 2 * i : 11 + 2 * i]) for i in range(curve[8])],
            "samples": list(struct.unpack_from(">257H", curve, 64)),
            "header_hex": curve[:8].hex(),
        }
    result["record_count"] = len(records)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.report.exists() or args.report.resolve().is_relative_to(root):
        parser.error("Report must be a new path outside the repository.")
    report = {
        "evidence_kind": "public_np3_record_analysis",
        "files": [],
        "passed": False,
        "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "limitations": [
            "No Nikon parser or renderer execution.",
            "Unknown records are retained; their semantics are not inferred.",
        ],
    }
    for path in args.files:
        item = {"path": str(path), "passed": False}
        report["files"].append(item)
        try:
            with path.open("rb") as stream:
                data = stream.read(MAX_BYTES + 1)
            records = parse(data)
            item.update(
                sha256=hashlib.sha256(data).hexdigest(),
                size_bytes=len(data),
                semantics=describe(records),
                roundtrip_equal=serialize(records) == data,
                records=[
                    {
                        "id": hex(r.identifier),
                        "bytes": len(r.payload),
                        "sha256": hashlib.sha256(r.payload).hexdigest(),
                    }
                    for r in records
                ],
            )
            item["passed"] = item["roundtrip_equal"]
        except (ValueError, OSError) as exc:
            item["error"] = str(exc)
    report["passed"] = all(item["passed"] for item in report["files"])
    with args.report.open("x") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=True)
        stream.write("\n")
    print(json.dumps({"report": str(args.report), "passed": report["passed"]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
