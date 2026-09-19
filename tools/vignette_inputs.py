"""Recover the observed full-sensor Z8 mode5 vignette parameters from metadata."""

from __future__ import annotations

import struct

import numpy as np

from nx_ctudio import _tiff


def _payload(tag: _tiff.Tag, length: int, prefix: bytes) -> bytes:
    if (
        tag.kind != 7
        or tag.count != length
        or len(tag.payload) != length
        or not tag.payload.startswith(prefix)
    ):
        raise ValueError("Unsupported Nikon vignette or center record.")
    return tag.payload


def source_parameters(
    root: _tiff.Directory, size: tuple[int, int]
) -> tuple[tuple[float, float], tuple[float, float, float, float]]:
    """Read mode5 coefficients and rescale the optical center for an image stage.

    This covers the default Normal vignette setting and uncropped Z8 sensor
    geometry. User overrides, other sensors, crops and anisotropic scales reject.
    """
    if (
        type(size) is not tuple
        or len(size) != 2
        or any(type(v) is not int or v < 2 for v in size)
    ):
        raise ValueError("Expected a supported integer stage size.")
    width, height = size
    if (
        8256 % width
        or 5504 % height
        or 8256 // width != 5504 // height
        or 8256 // width not in (1, 2, 4, 8, 16, 32, 64)
    ):
        raise ValueError("Only full-sensor Z8 power-of-two downscales are supported.")
    try:
        if root.tags[272].payload != b"NIKON Z 8\0":
            raise ValueError("Only Nikon Z8 source vignette parameters are verified.")
        subimages = root.tags[330].children
        raw_images = [
            d for d in subimages if 262 in d.tags and d.tags[262].integers() == (32803,)
        ]
        if (
            len(subimages) != 6
            or len(raw_images) != 1
            or raw_images[0] is not subimages[1]
        ):
            raise ValueError("Expected the observed single Z8 RAW SubIFD.")
        raw = raw_images[0]
        maker = root.tags[34665].children[0].tags[37500].children[0]
        if (
            raw.tags[256].integers() != (8280,)
            or raw.tags[257].integers() != (5520,)
            or maker.tags[0x45].integers() != (12, 8, 8256, 5504)
            or maker.tags[0x2A].integers() != (3,)
        ):
            raise ValueError("Unsupported Z8 crop, sensor size or vignette strength.")
        info = raw.tags[0xC7D5].children[0]
        data = _payload(info.tags[6], 116, b"0100\x01\x01\0\0")
        center = _payload(maker.tags[0x54], 20, b"0100")
    except (KeyError, IndexError, AttributeError) as exc:
        raise ValueError("Missing required Z8 vignette metadata.") from exc
    if struct.unpack_from("<I", data, 16)[0] != 8:
        raise ValueError("Unsupported vignette polynomial degree.")
    coefficients = []
    for index in range(9):
        numerator, denominator = struct.unpack_from("<2i", data, 20 + index * 8)
        if denominator <= 0 or abs(numerator / denominator) > 16:
            raise ValueError("Invalid vignette rational coefficient.")
        coefficients.append(np.float32(numerator / denominator))
    if any(coefficients[i] != 0 for i in (1, 3, 5, 7)) or coefficients[8] != 1:
        raise ValueError("Only even vignette powers with a unit constant are verified.")
    offsets = []
    for position, extent, active in ((4, width, 8256), (12, height, 5504)):
        numerator, denominator = struct.unpack_from("<2i", center, position)
        if denominator != 256:
            raise ValueError("Unsupported optical center fixed-point scale.")
        value = np.float32(numerator / denominator)
        scaled = (np.float32(extent) * value) / np.float32(active)
        if not np.isfinite(scaled) or abs(scaled) > 64:
            raise ValueError("Optical center exceeds the verified parameter range.")
        offsets.append(float(scaled))
    return tuple(offsets), tuple(float(coefficients[i]) for i in (0, 2, 4, 6))
