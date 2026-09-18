"""Inspect observed NP3 record structure without applying Picture Control colors.

Parameter names cross-reference ssssota/nikon-flexible-color-picture-control,
commit ed0397fbf1e4d4f2bc11ab3070522c613667d359 (MIT). File structure and
roundtrips are checked independently; Nikon acceptance remains a separate gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
        parameters[label] = {
            "encoded_value": value[0],
            "scale_byte": value[1],
            "value": None if value[0] == 1 else (value[0] - 128) / divisor,
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
                (v - 128 for v in blender[i * 3 : i * 3 + 3]),
                strict=True,
            )
        )
        for i, color in enumerate(COLORS)
    }
    result["color_blender_suffix_hex"] = blender[24:].hex()
    result["color_grading"] = {
        region: {
            "hue": ((grading[i * 4] & 15) << 8) | grading[i * 4 + 1],
            "chroma": grading[i * 4 + 2] - 128,
            "brightness": grading[i * 4 + 3] - 128,
        }
        for i, region in enumerate(("highlights", "mid_tone", "shadows"))
    }
    result["color_grading"].update(
        blending=grading[16] - 128, balance=grading[18] - 128
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
