# /// script
# requires-python = ">=3.11"
# dependencies = ["tifffile==2026.9.15"]
# ///
"""Validate recovered NEFX writer rules against saved synthetic pipeline results."""

from __future__ import annotations

import argparse
import binascii
import io
import json
import struct
from pathlib import Path

import tifffile
from inspect_nefx_tags import compare, digest


def directories(path: Path) -> tuple[dict, dict, dict, dict]:
    with tifffile.TiffFile(path) as image:
        root = {tag.code: tag.value for tag in image.pages[0].tags.values()}
        raw = {tag.code: tag.value for tag in image.pages[0].pages[1].tags.values()}
        maker_bytes = root[34665]["MakerNote"]
    with tifffile.TiffFile(io.BytesIO(maker_bytes[10:])) as image:
        maker = {tag.code: tag.value for tag in image.pages[0].tags.values()}
    with tifffile.TiffFile(io.BytesIO(raw[0xC7D5][10:])) as image:
        nef = {tag.code: tag.value for tag in image.pages[0].tags.values()}
    return root, raw, maker, nef


def validate(source: Path, output: Path, count: int) -> dict:
    before_root, before_raw, before_maker, before_nef = directories(source)
    root, raw, maker, nef = directories(output)
    scale = 2 if count in (16, 32) else 1
    width, height = (v * scale for v in before_maker[0x45][2:4])
    checks = {
        "raw_width": raw[256] == width,
        "raw_height": raw[257] == height,
        "bits_per_sample": tuple(raw[258]) == (14, 14, 14),
        "compression": raw[259] == 34713,
        "rgb_photometric": raw[262] == 2,
        "samples_per_pixel": raw[277] == 3,
        "rows_per_strip": raw[278] == height,
        "planar_configuration": raw[284] == 1,
        "crop": tuple(maker[0x45]) == (0, 0, width, height),
        "nikon_compression": maker[0x51][10:12] == b"\x0a\x00",
        "maker_deletions": all(tag not in maker for tag in (0x58, 0x96)),
        "nefinfo_deletions": all(tag not in nef for tag in (7, 15)),
        "software": root[305] == "NX Ctudio probe",
        "selector_preserved": maker[0x5E] == before_maker[0x5E],
        "capture_group_preserved": maker[0x56] == before_maker[0x56],
        "colorbalance_preserved": maker[0x97] == before_maker[0x97],
        "cfa_tags_preserved": all(
            raw[tag] == before_raw[tag] for tag in (33421, 33422)
        ),
    }
    marker = (
        b"0100\x02\x00"
        + bytes([before_maker[0x5E][4], (4, 8, 16, 32).index(count) + 1])
        + bytes(8)
    )
    checks["pixelshift_marker"] = maker[0xE2D] == marker
    preview = maker[0xE2E]
    checks["preview_header"] = preview[:16] == struct.pack(
        "<4sIHH4B", b"0100", 16, 640, 424, 3, 14, 14, 14
    )
    checks["preview_size"] = len(preview) == 16 + 640 * 424 * 6
    for tag, prefix in ((8, "center"), (11, "crop"), (13, "compression")):
        if tag in nef:
            payload = nef[tag]
            checks[f"nefinfo_{prefix}_crc"] = struct.unpack("<H", payload[-2:])[
                0
            ] == binascii.crc_hqx(payload[:-2], 65535)
    checks["nefinfo_crop"] = struct.unpack_from("<4H", nef[11], 4) == (
        0,
        0,
        width,
        height,
    )
    checks["nefinfo_compression"] = nef[13][4:6] == b"\x0a\x00"
    if scale == 2:
        for offset in (4, 12):
            checks[f"maker_center_{offset}_scaled"] = (
                struct.unpack_from("<i", maker[0x54], offset)[0]
                == 2 * struct.unpack_from("<i", before_maker[0x54], offset)[0]
            )
            checks[f"nefinfo_center_{offset}_scaled"] = (
                struct.unpack_from("<i", nef[8], offset)[0]
                == 2 * struct.unpack_from("<i", before_nef[8], offset)[0]
            )
        if 0xB7 in maker:
            checks["af_metadata_cleared"] = maker[0xB7][:8] == before_maker[0xB7][
                :8
            ] and not any(maker[0xB7][8:])
    if count in (4, 16):
        checks["xmp_preserved"] = root[700] == before_root[700]
    return {
        "input_count": count,
        "source": str(source),
        "output": str(output),
        "checks": checks,
        "passed": all(checks.values()),
        "tag_comparison": compare(source, output),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists.")
    result: dict = {
        "evidence_kind": "saved_synthetic_nefx_metadata_verification",
        "cases": [],
        "reports": [],
    }
    for report_path in args.reports:
        report = json.loads(report_path.read_text())
        if report["evidence_kind"] != "live_full_file_vendor_synthetic_sparse_pipeline":
            parser.error("Unexpected pipeline report.")
        result["reports"].append(
            {"path": str(report_path), "sha256": digest(report_path.read_bytes())}
        )
        for case in report["cases"]:
            source = Path(case["fixtures"][0]["path"])
            if digest(source.read_bytes()) != case["fixtures"][0]["sha256"]:
                raise ValueError("Input hash mismatch.")
            for item in case["outputs"]:
                output = Path(item["path"])
                if digest(output.read_bytes()) != item["sha256"]:
                    raise ValueError("Output hash mismatch.")
                result["cases"].append(validate(source, output, case["input_count"]))
    result["passed"] = all(case["passed"] for case in result["cases"])
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "cases": len(result["cases"]),
                "failed_checks": [
                    {
                        "output": c["output"],
                        "checks": [k for k, v in c["checks"].items() if not v],
                    }
                    for c in result["cases"]
                    if not c["passed"]
                ],
            }
        )
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
