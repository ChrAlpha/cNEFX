"""Strict native decoding of the observed Nikon pixel-shift NEF formats."""

from __future__ import annotations

import hashlib
import io
import stat
import struct
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import rawpy
import tifffile
from numpy.typing import NDArray

_MAKER_HEADER = b"Nikon\0\x02\x11\0\0"
_NEFINFO_HEADER = b"Nikon\0\x01\x03\0\0"
SUPPORTED_MODELS = ("NIKON Z 8", "NIKON Z f", "NIKON Z6_3", "NIKON Z5_2")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _block(tags: Any, code: int, version: bytes, length: int) -> bytes:
    tag = tags.get(code)
    _require(tag is not None, f"Missing Nikon metadata tag {code:#06x}.")
    value = tag.value
    _require(
        tag.dtype == 7
        and tag.count == length
        and isinstance(value, bytes)
        and len(value) == length
        and value[:4] == version,
        f"Unsupported Nikon metadata version or size for tag {code:#06x}.",
    )
    return value


def _capture_metadata(exif: dict[str, Any], tags: Any, shot_number: int) -> dict:
    serial = exif.get("BodySerialNumber")
    if not isinstance(serial, str) or not serial.strip():
        serial_tag = tags.get(0x1D)
        serial = serial_tag.value if serial_tag is not None else None
    _require(isinstance(serial, str) and bool(serial.strip()), "Missing camera serial.")
    timestamp = exif.get("DateTimeOriginal")
    _require(isinstance(timestamp, str), "Missing capture timestamp.")
    instant = datetime.strptime(timestamp, "%Y:%m:%d %H:%M:%S")
    subsecond = exif.get("SubsecTimeOriginal", "0")
    _require(
        isinstance(subsecond, str) and subsecond.isascii() and subsecond.isdigit(),
        "Invalid capture subsecond timestamp.",
    )
    instant = instant.replace(microsecond=int((subsecond + "000000")[:6]))
    shutter_tag = tags.get(0xA7)
    _require(
        shutter_tag is not None and shutter_tag.dtype == 4 and shutter_tag.count == 1,
        "Missing Nikon shutter count for sequence identity.",
    )
    shutter = shutter_tag.value
    _require(
        isinstance(shutter, int) and shutter >= shot_number,
        "Shutter count is inconsistent with the shot number.",
    )
    result = {
        "camera_identity_sha256": hashlib.sha256(
            serial.strip().encode("utf-8")
        ).hexdigest(),
        "capture_time": instant.isoformat(timespec="microseconds"),
        "capture_timezone": exif.get("OffsetTimeOriginal"),
        "shutter_sequence_anchor": shutter - shot_number,
    }
    for source_key, target_key in (
        ("ExposureTime", "exposure_time"),
        ("FNumber", "f_number"),
        ("ExposureBiasValue", "exposure_bias"),
    ):
        value = exif.get(source_key)
        _require(
            isinstance(value, tuple)
            and len(value) == 2
            and all(isinstance(n, int) for n in value)
            and value[1] > 0,
            f"Missing or malformed exposure metadata: {source_key}.",
        )
        number = Fraction(*value)
        if source_key != "ExposureBiasValue":
            _require(number > 0, f"Invalid exposure metadata: {source_key}.")
        result[target_key] = (number.numerator, number.denominator)
    return result


def _radial(tags: Any, raw: tifffile.TiffPage) -> dict[str, Any]:
    try:
        center = _block(tags, 0x54, b"0100", 20)
        info_tag = raw.tags.get(0xC7D5)
        _require(info_tag is not None, "Missing Nikon radial coefficient block.")
        info = info_tag.value
        _require(
            isinstance(info, bytes) and info[:10] == _NEFINFO_HEADER,
            "Unsupported Nikon radial coefficient header.",
        )
        with tifffile.TiffFile(io.BytesIO(info[10:])) as notes:
            _require(
                notes.byteorder == "<" and not notes.is_bigtiff,
                "Unsupported Nikon radial coefficient TIFF layout.",
            )
            coefficients = _block(notes.pages[0].tags, 7, b"0100", 104)
        _require(
            all(struct.unpack_from("<I", center, n)[0] == 256 for n in (8, 16))
            and all(
                struct.unpack_from("<I", coefficients, n + 4)[0] == 1048576
                for n in (44, 28, 20, 80, 64, 56)
            ),
            "Unsupported radial coefficient or center fixed-point scale.",
        )
        return {
            "available": True,
            "center_q8": [struct.unpack_from("<i", center, n)[0] for n in (4, 12)],
            "coefficients_q20": [
                [struct.unpack_from("<i", coefficients, n)[0] for n in offsets]
                for offsets in ((44, 28, 20), (80, 64, 56))
            ],
        }
    except (ValueError, KeyError, IndexError, TypeError, struct.error) as exc:
        return {"available": False, "reason": str(exc)}


def _metadata(path: Path) -> dict[str, Any]:
    with tifffile.TiffFile(path) as image:
        _require(
            image.byteorder == "<" and not image.is_bigtiff,
            "Only classic little-endian Nikon TIFF containers are supported.",
        )
        root = image.pages[0]
        _require(50706 not in root.tags, "Expected a source NEF, not DNG.")
        make, model = (root.tags[code].value for code in (271, 272))
        _require(
            make == "NIKON CORPORATION" and model in SUPPORTED_MODELS,
            f"Unsupported camera model {make!r} / {model!r}; "
            "native pixel-shift inputs must be Nikon Z8, Zf, Z6III or Z5II.",
        )
        orientation = int(root.tags[274].value)
        _require(orientation in range(1, 9), "Unsupported TIFF orientation.")
        exif = root.tags[34665].value
        maker = exif["MakerNote"]
        _require(
            isinstance(maker, bytes) and maker[:10] == _MAKER_HEADER,
            "Unsupported Nikon MakerNote header.",
        )
        with tifffile.TiffFile(io.BytesIO(maker[10:])) as notes:
            _require(
                notes.byteorder == "<" and not notes.is_bigtiff,
                "Unsupported Nikon MakerNote TIFF layout.",
            )
            tags = notes.pages[0].tags
            compression = _block(tags, 0x51, b"0102", 24)
            codec = struct.unpack_from("<H", compression, 10)[0]
            if codec in (13, 14, 15):
                raise ValueError(
                    "Nikon HE/HE* RAW compression is not supported by this native "
                    "decoder. Use lossless-compressed 14-bit NEF inputs."
                )
            _require(codec in (3, 10), "Unsupported Nikon RAW compression code.")
            _require(0x0E2D not in tags, "Expected a source NEF, not merged NEFX.")
            _block(tags, 0x97, b"0803", 166)
            capture = _block(tags, 0x56, b"0100", 16)
            shift = _block(tags, 0x5E, b"0100", 28)
            mode = shift[4]
            _require(
                mode in (1, 2, 3, 4) and capture[12] == 1,
                "Expected active 4, 8, 16 or 32-frame pixel-shift metadata.",
            )
            crop_tag = tags.get(0x45)
            _require(
                crop_tag is not None and crop_tag.dtype == 3 and crop_tag.count == 4,
                "Missing or unsupported Nikon crop metadata.",
            )
            crop = tuple(int(value) for value in crop_tag.value)
            candidates = [
                page for page in (root.pages or ()) if int(page.photometric) == 32803
            ]
            _require(len(candidates) == 1, "Expected exactly one Bayer RAW SubIFD.")
            raw = candidates[0]
            _require(
                raw.bitspersample == 14
                and raw.samplesperpixel == 1
                and int(raw.compression) == 34713,
                "Expected a 14-bit Nikon Bayer RAW image.",
            )
            _require(
                tuple(raw.tags[33421].value) == (2, 2)
                and bytes(raw.tags[33422].value) == b"\0\1\1\2",
                "Only the observed RGGB source CFA layout is supported.",
            )
            height, width = raw.shape
            left, top, crop_width, crop_height = crop
            _require(
                height >= 8
                and width >= 8
                and not (height % 2 or width % 2)
                and left >= 1
                and top >= 1
                and crop_width > 0
                and crop_height > 0
                and left + crop_width <= width
                and top + crop_height <= height,
                "RAW geometry or crop is outside the recovered sampling domain.",
            )
            iso = exif.get("ISOSpeedRatings")
            _require(
                isinstance(iso, int) and 0 < iso < 65535,
                "Missing or unsupported EXIF ISO value.",
            )
            return {
                **_capture_metadata(
                    exif, tags, struct.unpack_from("<I", capture, 8)[0]
                ),
                "make": make,
                "model": model,
                "orientation": orientation,
                "raw_shape": (height, width),
                "crop": crop,
                "iso": iso,
                "compression": codec,
                "mode": mode,
                "expected_count": (4, 8, 16, 32)[mode - 1],
                "burst_key": struct.unpack_from("<I", capture, 4)[0],
                "shot_number": struct.unpack_from("<I", capture, 8)[0],
                "pixel_shift_active": True,
                "selector": shift[4:],
                "radial": _radial(tags, raw),
            }


def read_frame(path: Path) -> tuple[NDArray[np.uint16], dict[str, Any]]:
    """Decode an unchanged source frame and return only merge-relevant metadata."""
    path = path.resolve(strict=True)
    before = path.stat()
    _require(stat.S_ISREG(before.st_mode), "Input must be a regular file.")
    source_hash = _digest(path)
    try:
        metadata = _metadata(path)
        with rawpy.imread(str(path)) as image:
            image.unpack()
            _require(
                image.raw_type == rawpy.RawType.Flat
                and image.raw_image.ndim == 2
                and image.raw_image.dtype == np.uint16
                and image.raw_image.shape == metadata["raw_shape"],
                "Decoded RAW geometry disagrees with the source Bayer TIFF.",
            )
            _require(
                image.num_colors == 3
                and image.color_desc == b"RGBG"
                and np.array_equal(image.raw_pattern, [[0, 1], [3, 2]]),
                "Decoded CFA layout is incompatible with the recovered mapping.",
            )
            sizes = image.sizes
            _require(
                sizes.top_margin == 0
                and sizes.left_margin == 0
                and sizes.pixel_aspect == 1
                and (sizes.raw_height, sizes.raw_width) == metadata["raw_shape"]
                and (
                    sizes.crop_left_margin,
                    sizes.crop_top_margin,
                    sizes.crop_width,
                    sizes.crop_height,
                )
                == metadata["crop"],
                "Decoded crop or pixel geometry disagrees with Nikon metadata.",
            )
            black = np.asarray(image.black_level_per_channel, dtype=np.float64)
            gains = np.asarray(image.camera_whitebalance, dtype=np.float64)
            matrix = image.rgb_xyz_matrix[:3].astype(np.float64)
            white = int(image.white_level)
            _require(
                black.shape == (4,)
                and np.all(np.isfinite(black))
                and np.all(black == black[0])
                and 0 <= black[0] < white == 16383,
                "Unequal black levels or unsupported 14-bit sample range.",
            )
            _require(
                gains.shape == (4,)
                and np.all(np.isfinite(gains))
                and np.all(gains > 0)
                and gains[1] == gains[3],
                "Missing white balance or unequal green gains are unsupported.",
            )
            # LibRaw 0.22.1 decodes Z5II but has no color-table entry for it.
            # NEFX synthesis uses sensor samples and preserves source color tags.
            missing_matrix = (
                metadata["model"] == "NIKON Z5_2"
                and matrix.shape == (3, 3)
                and np.all(matrix == 0)
            )
            _require(
                missing_matrix
                or (
                    matrix.shape == (3, 3)
                    and np.all(np.isfinite(matrix))
                    and abs(np.linalg.det(matrix)) >= 1e-6
                ),
                "Missing or singular XYZ-to-camera matrix.",
            )
            samples = image.raw_image.copy()
            _require(int(samples.max()) <= white, "RAW samples exceed 14 bits.")
            metadata.update(
                sizes=sizes._asdict(),
                black_level_per_channel=black.astype(int).tolist(),
                black_level=black[:3].tolist(),
                white_level=white,
                camera_gains=gains[:3].tolist(),
                green_gains=[float(gains[1]), float(gains[3])],
                as_shot_neutral=(gains[1] / gains[:3]).tolist(),
                color_matrix_xyz_to_camera=None if missing_matrix else matrix.tolist(),
                calibration_illuminant=0,
                source_sha256=source_hash,
                source_size_bytes=before.st_size,
            )
    except (KeyError, IndexError, TypeError, struct.error) as exc:
        raise ValueError("Missing or malformed source NEF metadata.") from exc
    after_hash = _digest(path)
    after = path.stat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    _require(
        source_hash == after_hash
        and all(getattr(before, field) == getattr(after, field) for field in fields),
        "Source file changed during decoding.",
    )
    return samples, metadata
