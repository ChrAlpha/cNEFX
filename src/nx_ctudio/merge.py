"""Sequence validation and atomic native NEFX publication."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import rawpy

from ._atomic import publication
from .nefx import write_nefx
from .pixel_shift import reconstruct
from .raw import read_frame

CONSISTENT = (
    "make",
    "model",
    "orientation",
    "raw_shape",
    "crop",
    "iso",
    "mode",
    "expected_count",
    "burst_key",
    "selector",
    "black_level_per_channel",
    "white_level",
    "camera_gains",
    "green_gains",
    "color_matrix_xyz_to_camera",
)
LIMITATIONS = [
    "Experimental: authentic pixel-shift captures and Lightroom are not validated.",
    "Radial correction is disabled; Adobe XMP defaults are preserved without updates.",
    "HE/HE* decoding and Flexible Color/Picture Control rendering are not implemented.",
    "Input order must follow consecutive capture numbers in a single sequence.",
]


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_sequence(metadata: list[dict[str, Any]]) -> None:
    if len(metadata) not in (4, 8, 16, 32):
        raise ValueError("Expected a complete 4, 8, 16 or 32-frame sequence.")
    first = metadata[0]
    if first["expected_count"] != len(metadata):
        raise ValueError("Input count does not match the capture mode.")
    numbers = [item["shot_number"] for item in metadata]
    if numbers != list(range(numbers[0], numbers[0] + len(numbers))):
        raise ValueError("Inputs must have consecutive shot numbers in capture order.")
    for item in metadata[1:]:
        for key in CONSISTENT:
            if item[key] != first[key]:
                raise ValueError(f"Sequence metadata differs: {key}.")


def merge(paths: list[Path], output: Path) -> dict[str, Any]:
    if len(paths) not in (4, 8, 16, 32):
        raise ValueError("Expected a complete 4, 8, 16 or 32-frame sequence.")
    if output.suffix.lower() != ".nefx":
        raise ValueError("Native merge output must have a .nefx extension.")
    output = output.absolute()
    if os.path.lexists(output):
        raise FileExistsError("Output already exists; overwriting is not supported.")
    identities = set()
    resolved = []
    for path in paths:
        path = path.resolve(strict=True)
        identity = (path.stat().st_dev, path.stat().st_ino)
        if identity in identities:
            raise ValueError("Duplicate input file or hard link.")
        identities.add(identity)
        resolved.append(path)
    arrays, metadata = [], []
    for path in resolved:
        samples, information = read_frame(path)
        arrays.append(samples)
        metadata.append(information)
    validate_sequence(metadata)
    first = metadata[0]
    pixels = reconstruct(
        arrays, first["crop"], b"0100" + first["selector"], first["iso"]
    )
    del arrays
    source = resolved[0].read_bytes()
    if hashlib.sha256(source).hexdigest() != first["source_sha256"]:
        raise ValueError("First input changed before metadata copying.")
    with publication(output) as pending:
        candidate = pending.temporary
        warnings = write_nefx(candidate, source, pixels, len(paths))
        with rawpy.imread(str(candidate)) as image:
            actual = image.raw_image[..., :3]
            if actual.shape != pixels.shape or not np.array_equal(actual, pixels):
                raise ValueError("Written NEFX failed exact pixel verification.")
        for path, information in zip(resolved, metadata, strict=True):
            if digest(path) != information["source_sha256"]:
                raise ValueError("Input changed during synthesis.")
        output_hash = digest(candidate)
        pending.publish()
    return {
        "status": "ok",
        "backend": "native",
        "output_format": "NEFX",
        "output": str(output),
        "sha256": output_hash,
        "shape": list(pixels.shape),
        "samples_verified": int(pixels.size),
        "sequence_validation": "complete_consecutive_metadata_consistent",
        "inputs": [
            {
                "path": str(path),
                "sha256": info["source_sha256"],
                "shot_number": info["shot_number"],
            }
            for path, info in zip(resolved, metadata, strict=True)
        ],
        "limitations": LIMITATIONS,
        "warnings": warnings + pending.warnings,
    }
