"""Write the experimentally validated three-channel Linear DNG contract."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from numpy.typing import NDArray


def _rationals(values: NDArray[np.float64]) -> tuple[int, ...]:
    result = []
    for value in values.flat:
        fraction = Fraction(float(value)).limit_denominator(1000000)
        if not (
            -(1 << 31) <= fraction.numerator < (1 << 31)
            and 0 < fraction.denominator < (1 << 31)
        ):
            raise ValueError("DNG metadata exceeds the supported rational range.")
        result.extend((fraction.numerator, fraction.denominator))
    return tuple(result)


def write_dng(
    path: Path, samples: NDArray[np.uint16], metadata: dict[str, Any]
) -> None:
    """Create a new Linear DNG without changing samples or normalizing colors."""
    if (
        samples.dtype != np.uint16
        or samples.ndim != 3
        or samples.shape[2] != 3
        or min(samples.shape[:2]) < 1
    ):
        raise ValueError("Expected a nonempty uint16 HxWx3 RGB sample array.")
    if samples.nbytes >= (1 << 32) - (1 << 20):
        raise ValueError("Image exceeds the supported classic TIFF size.")
    crop = metadata["crop"]
    count = metadata["expected_count"]
    if len(crop) != 4 or count not in (4, 8, 16, 32):
        raise ValueError("Missing or unsupported source crop and capture mode.")
    shapes = {(crop[3], crop[2], 3)}
    if count in (16, 32):
        shapes.add((2 * crop[3], 2 * crop[2], 3))
    if samples.shape not in shapes:
        raise ValueError("RGB geometry does not match a recovered output mode.")
    if (metadata["make"], metadata["model"]) != ("NIKON CORPORATION", "NIKON Z 8"):
        raise ValueError("Only the recovered Nikon Z8 color contract is supported.")
    orientation = metadata["orientation"]
    if orientation not in range(1, 9) or metadata["calibration_illuminant"] != 0:
        raise ValueError("Unsupported orientation or calibration illuminant.")
    black = np.asarray(metadata["black_level"], dtype=np.float64)
    source_black = np.asarray(metadata["black_level_per_channel"], dtype=np.float64)
    white = metadata["white_level"]
    green = np.asarray(metadata["green_gains"], dtype=np.float64)
    neutral = np.asarray(metadata["as_shot_neutral"], dtype=np.float64)
    matrix = np.asarray(metadata["color_matrix_xyz_to_camera"], dtype=np.float64)
    if (
        black.shape != (3,)
        or source_black.shape != (4,)
        or not np.all(np.isfinite(black))
        or not np.all(black == black[0])
        or not np.all(source_black == black[0])
        or not 0 <= black[0] < white == 16383
        or int(samples.max()) > white
    ):
        raise ValueError("Unequal black levels or unsupported 14-bit sample range.")
    if (
        green.shape != (2,)
        or not np.all(np.isfinite(green))
        or not np.all(green > 0)
        or green[0] != green[1]
        or neutral.shape != (3,)
        or not np.all(np.isfinite(neutral))
        or not np.all(neutral > 0)
        or neutral[1] != 1
    ):
        raise ValueError("Unsupported white balance or unequal green gains.")
    if (
        matrix.shape != (3, 3)
        or not np.all(np.isfinite(matrix))
        or abs(np.linalg.det(matrix)) < 1e-6
    ):
        raise ValueError("Missing or singular XYZ-to-camera matrix.")
    height, width, _ = samples.shape
    tags = [
        (271, "s", 0, metadata["make"], False),
        (272, "s", 0, metadata["model"], False),
        (274, "H", 1, orientation, False),
        (50706, "B", 4, (1, 4, 0, 0), False),
        (50707, "B", 4, (1, 3, 0, 0), False),
        (50708, "s", 0, metadata["model"], False),
        (50713, "H", 2, (1, 1), False),
        (50714, "2I", 3, _rationals(black), False),
        (50717, "I", 3, (white,) * 3, False),
        (50718, "2I", 2, (1, 1, 1, 1), False),
        (50719, "I", 2, (0, 0), False),
        (50720, "I", 2, (width, height), False),
        (50721, "2i", 9, _rationals(matrix), False),
        (50728, "2I", 3, _rationals(neutral), False),
        (50778, "H", 1, 0, False),
        (50829, "I", 4, (0, 0, height, width), False),
        (50936, "s", 0, "NX Ctudio LibRaw matrix research", False),
    ]
    with path.open("xb") as stream:
        tifffile.imwrite(
            stream,
            samples,
            byteorder="<",
            bigtiff=False,
            photometric=34892,
            planarconfig="contig",
            extrasamples=False,
            compression=None,
            rowsperstrip=min(height, 256),
            subfiletype=0,
            metadata=None,
            software="NX Ctudio research",
            extratags=tags,
        )
