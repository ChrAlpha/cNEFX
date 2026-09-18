# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.4.6", "rawpy==0.27.1", "tifffile==2026.9.15"]
# ///
"""Compare recovered private-preview averaging with a saved vendor NEFX."""

import argparse
import io
import json
from pathlib import Path

import numpy as np
import rawpy
import tifffile
from inspect_nefx_tags import digest


def preview(samples: np.ndarray) -> np.ndarray:
    height, width = samples.shape[:2]
    xs = np.trunc(np.arange(641, dtype=np.float64) / (640 / width)).astype(np.int64)
    ys = np.trunc(np.arange(425, dtype=np.float64) / (424 / height)).astype(np.int64)
    x0, x1 = xs[:-1], np.minimum(xs[1:], width - 1)
    y0, y1 = ys[:-1], np.minimum(ys[1:], height - 1)
    integral = np.pad(
        samples.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0), (0, 0))
    )
    area = np.maximum(y1 - y0, 0)[:, None] * np.maximum(x1 - x0, 0)[None, :]
    total = (
        integral[y1[:, None], x1]
        - integral[y0[:, None], x1]
        - integral[y1[:, None], x0]
        + integral[y0[:, None], x0]
    ) & 0xFFFFFFFF
    return np.where(
        area[..., None] > 0, total // np.maximum(area[..., None], 1), 0
    ).astype(np.uint16)


def verify(path: Path) -> dict:
    with rawpy.imread(str(path)) as raw:
        raw.unpack()
        expected = preview(raw.raw_image[..., :3])
    with tifffile.TiffFile(path) as image:
        maker = image.pages[0].tags[34665].value["MakerNote"]
    with tifffile.TiffFile(io.BytesIO(maker[10:])) as image:
        data = image.pages[0].tags[0xE2E].value
    actual = np.frombuffer(data, dtype="<u2", offset=16).reshape(424, 640, 3)
    diff = expected.astype(np.int32) - actual.astype(np.int32)
    return {
        "path": str(path),
        "sha256": digest(path.read_bytes()),
        "evidence_kind": "saved_synthetic_nefx_private_preview_comparison",
        "samples": int(actual.size),
        "differences": int(np.count_nonzero(diff)),
        "max_abs_error": int(np.max(np.abs(diff))),
        "zero_pixels": int(np.count_nonzero(np.all(actual == 0, axis=2))),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", type=Path, nargs="+")
    args = parser.parse_args()
    reports = [verify(path) for path in args.files]
    print(json.dumps(reports, indent=2))
    return 0 if all(report["differences"] == 0 for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
