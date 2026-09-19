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
from .pixel_shift import reconstruct_from_reader
from .radial import RadialCorrection
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
    "camera_identity_sha256",
    "capture_timezone",
    "shutter_sequence_anchor",
    "exposure_time",
    "f_number",
    "exposure_bias",
)
LIMITATIONS = [
    "Experimental: authentic pixel-shift captures and Lightroom are not validated.",
    "Adobe XMP defaults are preserved without updates.",
    "HE/HE* decoding is not implemented; use lossless-compressed 14-bit NEF inputs.",
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
    times = [item["capture_time"] for item in metadata]
    if times != sorted(times):
        raise ValueError("Capture timestamps move backwards within the sequence.")
    for item in metadata[1:]:
        for key in CONSISTENT:
            if item[key] != first[key]:
                raise ValueError(f"Sequence metadata differs: {key}.")


def merge(
    paths: list[Path], output: Path, *, chromatic_aberration_correction: bool = False
) -> dict[str, Any]:
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
    samples, first = read_frame(resolved[0])
    initial = [samples]
    del samples
    by_index = {0: first}
    correction = None
    if chromatic_aberration_correction:
        parameters = first["radial"]
        if not parameters["available"]:
            raise ValueError(
                "Supported chromatic aberration correction metadata is required: "
                + parameters["reason"]
            )
        correction = RadialCorrection(
            tuple(parameters["center_q8"]),
            tuple(tuple(row) for row in parameters["coefficients_q20"]),
        )

    def load(index: int) -> np.ndarray:
        if index == 0:
            return initial.pop()
        samples, information = read_frame(resolved[index])
        if correction is not None and information["radial"] != first["radial"]:
            raise ValueError("Sequence metadata differs: radial correction parameters.")
        by_index[index] = information
        return samples

    pixels = reconstruct_from_reader(
        len(paths),
        load,
        first["crop"],
        b"0100" + first["selector"],
        first["iso"],
        radial=correction,
    )
    metadata = [by_index[i] for i in range(len(paths))]
    validate_sequence(metadata)
    source = resolved[0].read_bytes()
    if hashlib.sha256(source).hexdigest() != first["source_sha256"]:
        raise ValueError("First input changed before metadata copying.")
    with publication(output) as pending:
        candidate = pending.temporary
        warnings = write_nefx(candidate, source, pixels, len(paths))
        with rawpy.imread(str(candidate)) as image:
            actual = image.raw_image[..., :3]
            if actual.shape != pixels.shape or any(
                not np.array_equal(actual[row : row + 128], pixels[row : row + 128])
                for row in range(0, len(pixels), 128)
            ):
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
        "chromatic_aberration_correction": chromatic_aberration_correction,
        "inputs": [
            {
                "path": str(path),
                "sha256": info["source_sha256"],
                "shot_number": info["shot_number"],
            }
            for path, info in zip(resolved, metadata, strict=True)
        ],
        "limitations": LIMITATIONS
        + (
            []
            if chromatic_aberration_correction
            else ["Chromatic aberration correction is disabled."]
        ),
        "warnings": warnings + pending.warnings,
    }
