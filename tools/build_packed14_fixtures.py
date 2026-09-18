# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.4.3", "rawpy==0.27.1", "tifffile==2026.9.15"]
# ///
"""Construct synthetic Nikon packed-14 fixtures without executing vendor code."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import struct
from pathlib import Path

import numpy as np
import rawpy
import tifffile

SOURCE = Path("/workspaces/LumaForge/test-images/SGL_1606.NEF")
SOURCE_SHA256 = "0f44ec054c311c4a6c4c162ba6b956495d690dca7a237400c3ae54773393bdbb"
DEFAULT_OUTPUT = Path("/workspaces/LumaForge/nxc-data/scratch/packed14-distinct-v2")
WIDTH, HEIGHT = 640, 432
CROP = (4, 4, 632, 424)
PATTERN = "Independent PCG64(seed=20260918+frame_index) uint16 samples in [0,16384)"


def sha256(data: bytes | bytearray) -> str:
    return hashlib.sha256(data).hexdigest()


def values_for_frame(frame_index: int) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(20260918 + frame_index))
    values = rng.integers(0, 16384, size=(HEIGHT, WIDTH), dtype=np.uint16)
    values[8, 8:16] = (0, 1, 1007, 1008, 1009, 8191, 16382, 16383)
    return values


def pack_nikon14(values: np.ndarray) -> bytes:
    assert values.shape == (HEIGHT, WIDTH) and WIDTH % 4 == 0
    pixels = values.astype(np.uint64).reshape(HEIGHT, WIDTH // 4, 4)
    words = pixels[..., 0] | (pixels[..., 1] << 14)
    words |= (pixels[..., 2] << 28) | (pixels[..., 3] << 42)
    shifts = np.arange(7, dtype=np.uint64) * 8
    packed = ((words[..., None] >> shifts) & 255).astype(np.uint8)
    row_bytes = (WIDTH * 7 // 4 + 15) // 16 * 16
    result = np.zeros((HEIGHT, row_bytes), dtype=np.uint8)
    result[:, : WIDTH * 7 // 4] = packed.reshape(HEIGHT, -1)
    return result.tobytes()


def metadata_layout(source: bytes) -> dict[str, object]:
    with tifffile.TiffFile(io.BytesIO(source)) as image:
        assert image.byteorder == "<" and not image.is_bigtiff
        raw = image.pages[0].pages[1]
        assert raw.shape == (5520, 8280)
        assert raw.tags[273].value == (6901248,)
        exif_offset = image.pages[0].tags[34665].valueoffset
        count = struct.unpack_from("<H", source, exif_offset)[0]
        exif_tags = [
            tifffile.TiffTag.fromfile(image, offset=exif_offset + 2 + index * 12)
            for index in range(count)
        ]
        maker = next(tag for tag in exif_tags if tag.code == 0x927C)
        maker_data = source[maker.valueoffset : maker.valueoffset + maker.count]
        assert maker_data[:7] == b"Nikon\0\x02"
        maker_base = maker.valueoffset + 10
        with tifffile.TiffFile(io.BytesIO(maker_data[10:])) as notes:
            assert notes.byteorder == "<"
            wanted = {
                code: notes.pages[0].tags[code] for code in (0x45, 0x51, 0x56, 0x5E)
            }
            assert int(wanted[0x45].dtype) == 3 and wanted[0x45].count == 4
            assert wanted[0x51].count == 24
            assert wanted[0x56].count == 16
            assert wanted[0x5E].count == 28
            maker_offsets = {
                code: maker_base + tag.valueoffset for code, tag in wanted.items()
            }
        return {
            "raw_strip_offset": raw.tags[273].value[0],
            "raw_scalar_offsets": {
                code: raw.tags[code].valueoffset for code in (256, 257, 278, 279)
            },
            "maker_offsets": maker_offsets,
        }


def patch_header(
    source: bytes, layout: dict[str, object], mode: int, count: int, frame: int
) -> tuple[bytearray, list[dict[str, object]]]:
    raw_offset = int(layout["raw_strip_offset"])
    header = bytearray(source[:raw_offset])
    patches: list[dict[str, object]] = []

    def patch(offset: int, replacement: bytes, reason: str) -> None:
        before = bytes(header[offset : offset + len(replacement)])
        header[offset : offset + len(replacement)] = replacement
        patches.append(
            {
                "offset": offset,
                "before_hex": before.hex(),
                "after_hex": replacement.hex(),
                "reason": reason,
            }
        )

    scalars = layout["raw_scalar_offsets"]
    for code, value in (
        (256, WIDTH),
        (257, HEIGHT),
        (278, HEIGHT),
        (279, (WIDTH * 7 // 4 + 15) // 16 * 16 * HEIGHT),
    ):
        patch(scalars[code], struct.pack("<I", value), f"raw TIFF tag {code}")
    notes = layout["maker_offsets"]
    patch(notes[0x45], struct.pack("<4H", *CROP), "MakerNote 0x0045 synthetic CropArea")
    patch(
        notes[0x51] + 10,
        struct.pack("<H", 10),
        "MakerNote 0x0051 packed-14 compression value",
    )
    group_key = (100 << 19) | ((100 + count) << 5) | 2
    patch(
        notes[0x56] + 4,
        struct.pack("<I", group_key),
        "MakerNote 0x0056 synthetic group key",
    )
    patch(
        notes[0x56] + 8,
        struct.pack("<I", frame + 1),
        "MakerNote 0x0056 synthetic one-based frame key",
    )
    patch(
        notes[0x56] + 12, b"\x01", "MakerNote 0x0056 synthetic pixel-shift active flag"
    )
    patch(notes[0x5E] + 4, bytes([mode]), "MakerNote 0x005e synthetic mode")
    reconstructed = bytearray(source[:raw_offset])
    for item in patches:
        offset = item["offset"]
        replacement = bytes.fromhex(item["after_hex"])
        reconstructed[offset : offset + len(replacement)] = replacement
    assert reconstructed == header
    return header, patches


def verify(path: Path, expected: np.ndarray) -> dict[str, object]:
    with rawpy.imread(str(path)) as image:
        image.unpack()
        actual = image.raw_image
        assert actual.shape == expected.shape, (actual.shape, expected.shape)
        differences = int(np.count_nonzero(actual != expected))
        assert differences == 0, differences
        return {
            "decoder": "LibRaw via rawpy",
            "rawpy_version": rawpy.__version__,
            "libraw_version": list(rawpy.libraw_version),
            "raw_shape": list(actual.shape),
            "different_samples": differences,
            "raw_sha256": sha256(actual.astype("<u2").tobytes()),
            "sizes": image.sizes._asdict(),
            "black_level_per_channel": list(image.black_level_per_channel),
            "white_level": int(image.white_level),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--counts", nargs="+", type=int, choices=(4, 8, 16, 32), default=[4, 8, 16, 32]
    )
    args = parser.parse_args()
    logging.getLogger("tifffile").setLevel(logging.ERROR)
    source = SOURCE.read_bytes()
    assert sha256(source) == SOURCE_SHA256
    args.output.mkdir(exist_ok=False)
    layout = metadata_layout(source)
    raw_offset = int(layout["raw_strip_offset"])
    manifest = {
        "evidence_kind": "synthetic_packed14_input_fixture",
        "source": str(SOURCE),
        "source_sha256": SOURCE_SHA256,
        "original_header_sha256": sha256(source[:raw_offset]),
        "raw_strip_offset": raw_offset,
        "width": WIDTH,
        "height": HEIGHT,
        "crop_left_top_width_height": list(CROP),
        "pattern": PATTERN,
        "frame_index_base": 0,
        "limitations": [
            "Synthetic raw values and metadata are not authentic Nikon captures.",
            "Embedded previews and remaining metadata are inherited from the source "
            "and do not describe the synthetic raw scene.",
        ],
        "files": [],
    }
    for count in args.counts:
        mode = (4, 8, 16, 32).index(count) + 1
        destination = args.output / f"mode-{count:02d}"
        destination.mkdir()
        for frame in range(count):
            expected = values_for_frame(frame)
            packed = pack_nikon14(expected)
            header, patches = patch_header(source, layout, mode, count, frame)
            path = destination / f"frame-{frame + 1:02d}.NEF"
            content = header + packed
            with path.open("xb") as stream:
                stream.write(content)
            verification = verify(path, expected)
            manifest["files"].append(
                {
                    "path": str(path),
                    "mode": mode,
                    "input_count": count,
                    "frame_index": frame,
                    "size_bytes": len(content),
                    "sha256": sha256(content),
                    "header_sha256": sha256(header),
                    "packed_payload_sha256": sha256(packed),
                    "patches": patches,
                    "verification": verification,
                }
            )
    assert sha256(SOURCE.read_bytes()) == SOURCE_SHA256
    manifest["original_source_unchanged"] = True
    with (args.output / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "fixture_count": len(manifest["files"]),
                "original_source_unchanged": True,
                "all_raw_samples_match": True,
            }
        )
    )


if __name__ == "__main__":
    main()
