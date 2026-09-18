"""Read source evidence without decoding pixels or assigning sequence positions."""

from __future__ import annotations

import hashlib
import json
import stat
import subprocess
from pathlib import Path
from typing import Any

# Preserve group names and numeric values, including Nikon's encoded shot count.
# Source: https://github.com/exiftool/exiftool/blob/master/lib/Image/ExifTool/Nikon.pm
TAGS = (
    "FileType",
    "Make",
    "Model",
    "Software",
    "FirmwareVersion",
    "ImageWidth",
    "ImageHeight",
    "BitsPerSample",
    "SamplesPerPixel",
    "Compression",
    "PhotometricInterpretation",
    "CFARepeatPatternDim",
    "CFAPattern2",
    "NEFCompression",
    "PixelShiftActive",
    "PixelShiftShooting",
    "PixelShiftNumberShots",
    "PixelShiftDelay",
    "PixelShiftInterval",
    "ShutterCount",
    "BurstNumber",
    "BurstShotNumber",
    "SequenceNumber",
    "FocusShiftShooting",
    "FocusShiftNumberShots",
    "DateTimeOriginal",
    "SubSecTimeOriginal",
    "ExposureTime",
    "FNumber",
    "ISO",
    "ExposureCompensation",
    "Orientation",
    "ActiveArea",
    "DefaultCropOrigin",
    "DefaultCropSize",
    "BlackLevel",
    "WhiteLevel",
    "AsShotNeutral",
    "ColorMatrix1",
    "ColorMatrix2",
    "CalibrationIlluminant1",
    "CalibrationIlluminant2",
    "DNGVersion",
    "Warning",
    "Error",
)


class InspectionError(Exception):
    """An observable failure at the external metadata-reader boundary."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def read_metadata(path: Path, executable: str) -> dict[str, Any]:
    # Disable user ExifTool configuration and use absolute paths to prevent a
    # filename from being interpreted as a write option. Never invoke a shell.
    command = [executable, "-config", "", "-j", "-G1", "-n", "-a"]
    command.extend(f"-{tag}" for tag in TAGS)
    command.append(str(path))
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise InspectionError(
            "metadata_timeout", "ExifTool exceeded 30 seconds."
        ) from exc
    if result.returncode:
        raise InspectionError(
            "metadata_error",
            result.stderr.strip() or "ExifTool could not read the input.",
        )
    try:
        rows = json.loads(result.stdout)
        if (
            not isinstance(rows, list)
            or len(rows) != 1
            or not isinstance(rows[0], dict)
        ):
            raise ValueError("Expected one metadata object.")
        metadata = rows[0]
        if not isinstance(metadata.get("File:FileType"), str):
            raise ValueError("Missing or invalid file type in metadata.")
    except (ValueError, TypeError) as exc:
        raise InspectionError(
            "metadata_error", "Invalid ExifTool JSON response."
        ) from exc
    for key, value in metadata.items():
        if key.rsplit(":", 1)[-1] == "Error":
            raise InspectionError("metadata_error", str(value))
    if metadata["File:FileType"] not in {"NEF", "NEFX", "DNG", "TIFF"}:
        raise InspectionError(
            "unsupported_file_type", "Expected a Nikon RAW, DNG, or TIFF container."
        )
    metadata.pop("SourceFile", None)
    if result.stderr.strip():
        metadata["ExifTool:Diagnostics"] = result.stderr.strip()
    return metadata


def inspect_file(path: Path, executable: str) -> dict[str, Any]:
    """Return evidence or a structured failure; leave the input untouched."""
    result: dict[str, Any] = {"path": str(path), "status": "error"}
    try:
        path = path.resolve(strict=True)
        result["path"] = str(path)
        before = path.stat()
        if not stat.S_ISREG(before.st_mode):
            raise InspectionError("input_error", "Input must be a regular file.")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        result.update(size_bytes=before.st_size, sha256=digest)
        metadata = read_metadata(path, executable)
        after = path.stat()
        for field in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"):
            if getattr(before, field) != getattr(after, field):
                raise InspectionError(
                    "input_changed", "Input changed during inspection."
                )
        result.update(status="ok", metadata=metadata)
    except InspectionError as exc:
        result["error"] = {"code": exc.code, "message": str(exc)}
    except (OSError, RuntimeError) as exc:
        result["error"] = {"code": "input_error", "message": str(exc)}
    return result
