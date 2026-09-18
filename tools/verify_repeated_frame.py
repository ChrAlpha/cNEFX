# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.4.6", "rawpy==0.27.1"]
# ///
"""Compare a repeated-frame probe with the recovered four/eight-input rules.

This does not validate authentic pixel-shift quality or sparse sample repair.
The source must have no selected repair rows, and correction must be disabled.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import rawpy


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compare(source: Path, output: Path, frames: int) -> dict[str, object]:
    with rawpy.imread(str(source)) as image:
        image.unpack()
        raw = image.raw_image.copy()
        crop = image.sizes
    with rawpy.imread(str(output)) as image:
        image.unpack()
        if image.raw_image.ndim != 3 or image.raw_image.shape[2] < 3:
            raise ValueError("Expected a decoded three-channel NEFX.")
        actual = image.raw_image[..., :3].copy()
    height, width = actual.shape[:2]
    if raw.ndim != 2 or (height, width) != (crop.crop_height, crop.crop_width):
        raise ValueError("Source and single-resolution output geometry do not match.")
    if crop.crop_left_margin < 1 or crop.crop_top_margin < 1:
        raise ValueError("The recovered crop requires positive source margins.")
    if frames == 8:
        raw[2:] = ((raw[2:].astype(np.uint32) + raw[:-2]) >> 1).astype(np.uint16)
    phase_x = (0, 1, 1, 0)
    phase_y = (0, 0, 1, 1)
    channels = ((2, 1, 0, 1), (1, 2, 1, 0), (0, 1, 2, 1), (1, 0, 1, 2))
    corners = ((0, 0), (0, 1), (1, 1), (1, 0))
    expected = np.zeros_like(actual)
    for frame in range(4):
        for corner, (py, px) in enumerate(corners):
            x0 = (px - (crop.crop_left_margin - 1)) % 2
            y0 = (py - (crop.crop_top_margin - 1)) % 2
            ys = np.arange(y0, height, 2)
            xs = np.arange(x0, width, 2)
            samples = raw[
                np.ix_(
                    ys + crop.crop_top_margin - phase_y[frame],
                    xs + crop.crop_left_margin - phase_x[frame],
                )
            ]
            channel = channels[frame][corner]
            previous = expected[y0::2, x0::2, channel]
            if channel == 1 and frame >= 2:
                samples = ((samples.astype(np.uint32) + previous) >> 1).astype(
                    np.uint16
                )
            previous[:] = samples
    diff = expected.astype(np.int32) - actual.astype(np.int32)
    differences, counts = np.unique(diff, return_counts=True)
    return {
        "evidence_kind": "synthetic_repeated_frame_comparison",
        "input_count": frames,
        "source_sha256": digest(source),
        "output_sha256": digest(output),
        "decoder": {"rawpy": rawpy.__version__, "libraw": list(rawpy.libraw_version)},
        "shape": list(actual.shape),
        "sample_count": int(actual.size),
        "different_samples": int(np.count_nonzero(diff)),
        "max_abs_error": int(np.abs(diff).max()),
        "signed_difference_counts": {
            str(int(key)): int(value)
            for key, value in zip(differences, counts, strict=True)
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--frames", type=int, choices=(4, 8), default=4)
    args = parser.parse_args()
    report = compare(args.source, args.output, args.frames)
    print(json.dumps(report, indent=2))
    return int(report["different_samples"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
