# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.4.6", "rawpy==0.27.1", "tifffile==2026.9.15"]
# ///
"""Verify saved NEFX payload copying, alignment and exact packed-14 encoding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rawpy
from inspect_nefx_layout import compare, digest


def verify(source: Path, output: Path) -> dict:
    layout = compare(source, output)
    raw_key = "IFD0/SubIFD[1]/strips[0]"
    raw_payload = next(item for item in layout["payloads"] if item["path"] == raw_key)
    checks = {
        "only_raw_image_payload_changes": all(
            item["preserved"] for item in layout["payloads"] if item["path"] != raw_key
        ),
        "no_removed_image_payloads": not layout["removed_payloads"],
        "payload_alignment": all(
            item["offset"] % 16 == 0 for item in layout["payloads"]
        ),
        "zero_next_ifd": all(d["next_ifd"] == 0 for d in layout["directories"]),
        "directory_alignment": all(
            d["offset"] % 16 == 0
            for d in layout["directories"]
            if d["kind"] not in ("root", "nikon")
        ),
        "tag_payload_alignment": all(
            t["offset"] % 16 == 0
            for d in layout["directories"]
            for t in d["tags"]
            if not t["inline"]
        ),
    }
    with rawpy.imread(str(output)) as image:
        image.unpack()
        samples = image.raw_image[..., :3]
        height, width = samples.shape[:2]
        if width * 3 % 4:
            raise ValueError("This verifier requires sample rows divisible by four.")
        row_bytes = width * 3 * 7 // 4
        stride = (row_bytes + 15) // 16 * 16
        differences = padding_nonzero = 0
        with output.open("rb") as stream:
            stream.seek(raw_payload["offset"])
            for row in samples:
                p = row.reshape(-1, 4).astype(np.uint64)
                words = p[:, 0] | (p[:, 1] << 14) | (p[:, 2] << 28) | (p[:, 3] << 42)
                expected = (
                    ((words[:, None] >> (np.arange(7, dtype=np.uint64) * 8)) & 255)
                    .astype(np.uint8)
                    .ravel()
                )
                data = stream.read(stride)
                if len(data) != stride:
                    raise ValueError("Truncated packed-14 strip.")
                actual = np.frombuffer(data, dtype=np.uint8)
                differences += int(np.count_nonzero(expected != actual[:row_bytes]))
                padding_nonzero += int(np.count_nonzero(actual[row_bytes:]))
    checks.update(
        raw_payload_size=raw_payload["size"] == height * stride,
        packed_bytes_equal=differences == 0,
        zero_row_padding=padding_nonzero == 0,
    )
    return {
        "source": str(source),
        "output": str(output),
        "source_sha256": layout["source_sha256"],
        "output_sha256": layout["output_sha256"],
        "checks": checks,
        "passed": all(checks.values()),
        "directories": layout["directories"],
        "payloads": layout["payloads"],
        "packed_bytes_compared": height * row_bytes,
        "packed_byte_differences": differences,
        "row_padding_bytes": height * (stride - row_bytes),
        "nonzero_row_padding_bytes": padding_nonzero,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Report already exists.")
    report = {
        "evidence_kind": "saved_synthetic_nefx_serialization_verification",
        "tool_sha256": digest(Path(__file__).read_bytes()),
        "inspector_sha256": digest(
            Path(__file__).with_name("inspect_nefx_layout.py").read_bytes()
        ),
        **verify(args.source, args.output),
    }
    with args.report.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("directories", "payloads")}
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
