# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.4.6", "rawpy==0.27.1", "tifffile==2026.9.15"]
# ///
"""Compare recovered chromatic correction against a local synthetic NEFX result.

This verifier assumes zero sparse-repair selection fields in its input files.
"""

import argparse
import io
import json
import math
import struct
from pathlib import Path

import numpy as np
import rawpy
import tifffile
from verify_vendor_matrix import reconstruct, sha256


def parameters(path: Path) -> tuple[list[int], list[list[int]]]:
    with tifffile.TiffFile(path) as image:
        maker = image.pages[0].tags[34665].value["MakerNote"]
        nefinfo = image.pages[0].pages[1].tags[0xC7D5].value
    with tifffile.TiffFile(io.BytesIO(maker[10:])) as notes:
        center = notes.pages[0].tags[0x54].value
    with tifffile.TiffFile(io.BytesIO(nefinfo[10:])) as notes:
        coefficients = notes.pages[0].tags[7].value
    return (
        [struct.unpack_from("<i", center, offset)[0] for offset in (4, 12)],
        [
            [struct.unpack_from("<i", coefficients, offset)[0] for offset in offsets]
            for offsets in ((44, 28, 20), (80, 64, 56))
        ],
    )


def weight(distance: np.ndarray) -> np.ndarray:
    distance = np.abs(distance)
    squared = distance * distance
    cubed = squared * distance
    return np.where(
        distance <= 1,
        1 - 2 * squared + cubed,
        np.where(distance <= 2, 4 - 8 * distance + 5 * squared - cubed, 0),
    )


def correct(
    planes: np.ndarray,
    raw_shape: tuple[int, ...],
    scale: int,
    center: list[int],
    coefficients: list[list[int]],
) -> np.ndarray:
    raw_height, raw_width = raw_shape
    height, width = planes.shape[:2]
    yy, xx = np.indices((height, width), dtype=np.float64)
    cx = scale * ((raw_width - 1) / 2 + center[0] / 256 - 1)
    cy = scale * ((raw_height - 1) / 2 + center[1] / 256 - 1)
    dx, dy = xx - cx, yy - cy
    radius = scale * math.sqrt(raw_width * raw_width + raw_height * raw_height) / 2
    normalized = np.sqrt(dx * dx + dy * dy) / radius
    squared = normalized * normalized
    output = planes.copy()
    for channel, triple in zip((0, 2), coefficients, strict=True):
        a, b, c = [value / 1048576 for value in triple]
        factor = squared * b + a + squared * normalized * c + 1
        sx, sy = factor * dx + cx, factor * dy + cy
        ix, iy = np.trunc(sx).astype(np.int64), np.trunc(sy).astype(np.int64)
        accumulated = np.zeros_like(sx)
        for row in (-1, 0, 1, 2):
            y = iy + row
            wy = weight(y - sy)
            for column in (-1, 0, 1, 2):
                x = ix + column
                wx = weight(x - sx)
                valid = (y >= 0) & (y < height) & (x >= 0) & (x < width)
                samples = planes[y.clip(0, height - 1), x.clip(0, width - 1), channel]
                accumulated += np.where(valid, samples.astype(np.float64) * wx * wy, 0)
        output[:, :, channel] = np.clip(accumulated, 0, 16383).astype(np.uint16)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()
    if len(args.inputs) not in (4, 8, 16, 32):
        parser.error("Expected 4, 8, 16, or 32 source files in capture order.")
    arrays = []
    for path in args.inputs:
        with rawpy.imread(str(path)) as image:
            image.unpack()
            arrays.append(image.raw_image.copy())
            if len(arrays) == 1:
                crop = image.sizes
    height, width = arrays[0].shape
    scale = 2 if len(arrays) in (16, 32) else 1
    centers, coefficients = parameters(args.inputs[0])
    planes = reconstruct(arrays, (1, 1, width - 1, height - 1))
    corrected = correct(planes, arrays[0].shape, scale, centers, coefficients)
    top, left = scale * (crop.crop_top_margin - 1), scale * (crop.crop_left_margin - 1)
    expected = corrected[
        top : top + scale * crop.crop_height, left : left + scale * crop.crop_width
    ]
    with rawpy.imread(str(args.output)) as image:
        image.unpack()
        actual = image.raw_image[..., :3]
        if expected.shape != actual.shape:
            raise ValueError("Output geometry differs from the recovered crop.")
        diff = expected.astype(np.int32) - actual.astype(np.int32)
    report = {
        "evidence_kind": "synthetic_radial_correction_comparison",
        "input_sha256": [sha256(path) for path in args.inputs],
        "output_sha256": sha256(args.output),
        "center_q8": centers,
        "coefficients_q20": coefficients,
        "scale": scale,
        "shape": list(expected.shape),
        "samples_compared": int(expected.size),
        "different_samples": int(np.count_nonzero(diff)),
        "max_abs_error": int(np.abs(diff).max()),
        "different_samples_by_channel": [
            int(np.count_nonzero(diff[:, :, c])) for c in range(3)
        ],
    }
    print(json.dumps(report, indent=2))
    return int(report["different_samples"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
