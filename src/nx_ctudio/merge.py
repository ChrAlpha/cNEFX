"""Sequence validation and atomic native NEFX publication."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import rawpy

from ._atomic import publication
from .nefx import write_nefx
from .pixel_shift import output_groups, reconstruct_from_reader
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
    if output.suffix.lower() != ".nefx":
        raise ValueError("Native merge output must have a .nefx extension.")
    return _merge(paths, output, 1, False, chromatic_aberration_correction)


def merge_outputs(
    paths: list[Path],
    output_directory: Path,
    outputs: int,
    *,
    chromatic_aberration_correction: bool = False,
) -> dict[str, Any]:
    """Publish a complete output set in a new directory, retaining capture identity."""
    return _merge(
        paths, output_directory, outputs, True, chromatic_aberration_correction
    )


def _write_verified(
    candidate: Path,
    source_path: Path,
    information: dict[str, Any],
    pixels: np.ndarray,
    count: int,
) -> dict[str, Any]:
    source = source_path.read_bytes()
    if hashlib.sha256(source).hexdigest() != information["source_sha256"]:
        raise ValueError("First input changed before metadata copying.")
    warnings = write_nefx(candidate, source, pixels, count)
    with rawpy.imread(str(candidate)) as image:
        actual = image.raw_image[..., :3]
        if actual.shape != pixels.shape or any(
            not np.array_equal(actual[row : row + 128], pixels[row : row + 128])
            for row in range(0, len(pixels), 128)
        ):
            raise ValueError("Written NEFX failed exact pixel verification.")
    return {
        "sha256": digest(candidate),
        "shape": list(pixels.shape),
        "samples_verified": int(pixels.size),
        "warnings": warnings,
    }


def _merge(
    paths: list[Path],
    output: Path,
    outputs: int,
    directory: bool,
    chromatic_aberration_correction: bool,
) -> dict[str, Any]:
    groups = output_groups(len(paths), outputs)
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

    results, warnings = [], []
    with publication(output, directory=directory) as pending:
        for index, members in enumerate(groups):
            pixels = reconstruct_from_reader(
                len(members),
                lambda i: load(members[i]),
                first["crop"],
                b"0100" + first["selector"],
                first["iso"],
                radial=correction,
            )
            if len(by_index) == len(paths):
                validate_sequence([by_index[i] for i in range(len(paths))])
            name = f"merged-{index + 1:02d}.nefx"
            candidate = pending.temporary / name if directory else pending.temporary
            result = _write_verified(
                candidate,
                resolved[members[0]],
                by_index[members[0]],
                pixels,
                len(members),
            )
            del pixels
            if directory and result["warnings"]:
                raise OSError("Temporary output publication reported a warning.")
            warnings.extend(result.pop("warnings"))
            result.update(output=str(output / name if directory else output))
            if directory:
                result["input_indices"] = list(members)
            results.append(result)
        metadata = [by_index[i] for i in range(len(paths))]
        for path, information in zip(resolved, metadata, strict=True):
            if digest(path) != information["source_sha256"]:
                raise ValueError("Input changed during synthesis.")
        report = {
            "status": "ok",
            "backend": "native",
            "output_format": "NEFX",
            "camera": {"make": first["make"], "model": first["model"]},
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
                [
                    "LibRaw has no XYZ-to-camera matrix for this input; NEFX "
                    "preserves source color metadata. Color rendering is unvalidated."
                ]
                if first["color_matrix_xyz_to_camera"] is None
                else []
            )
            + (
                []
                if chromatic_aberration_correction
                else ["Chromatic aberration correction is disabled."]
            ),
            "warnings": warnings,
        }
        if directory:
            report.update(
                output_directory=str(output),
                output_count=len(results),
                outputs=results,
                samples_verified=sum(item["samples_verified"] for item in results),
            )
            with (pending.temporary / "manifest.json").open(
                "x", encoding="ascii"
            ) as stream:
                json.dump(report, stream, indent=2, ensure_ascii=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
        else:
            report.update(results[0])
        pending.publish()
    report["warnings"] += pending.warnings
    return report
