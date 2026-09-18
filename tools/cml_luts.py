"""Native sampled-curve LUTs for the recovered CML RGB16 profile path."""

from __future__ import annotations

import struct

import numpy as np


def icc_tags(profile: bytes) -> dict[bytes, bytes]:
    if (
        len(profile) < 132
        or profile[36:40] != b"acsp"
        or struct.unpack_from(">I", profile)[0] != len(profile)
        or profile[16:20] != b"RGB "
    ):
        raise ValueError("Expected a complete RGB ICC profile.")
    count = struct.unpack_from(">I", profile, 128)[0]
    end = 132 + count * 12
    if end > len(profile):
        raise ValueError("Invalid ICC tag count.")
    tags = {}
    for signature, offset, size in struct.iter_unpack(">4sII", profile[132:end]):
        if offset < end or offset + size > len(profile) or signature in tags:
            raise ValueError("Invalid ICC tag bounds or duplicate tag.")
        tags[signature] = profile[offset : offset + size]
    return tags


def input_table(profile: bytes) -> bytes:
    """Validate the recovered Kaiji RGB/Lab mft2 layout."""
    tag = icc_tags(profile).get(b"A2B0", b"")
    identity = struct.pack(">9i", 65536, 0, 0, 0, 65536, 0, 0, 0, 65536)
    if (
        profile[20:24] != b"Lab "
        or len(tag) != 240268
        or tag[:12] != b"mft2\0\0\0\0\x03\x03\x21\0"
        or tag[12:48] != identity
        or struct.unpack_from(">HH", tag, 48) != (4096, 3)
    ):
        raise ValueError("Expected the recovered grid33 RGB/Lab mft2 layout.")
    return tag


def input_luts(profile: bytes) -> np.ndarray:
    """Expand the Kaiji mft2 sampled input curves, without extra user LUTs."""
    tag = input_table(profile)
    curves = np.frombuffer(tag, dtype=">u2", count=3 * 4096, offset=52)
    curves = curves.reshape(3, 4096).astype(np.float64)
    coordinate = np.arange(65536, dtype=np.float64) * (1.0 / 65535)
    position = coordinate * 4095
    lower = position.astype(np.int64)
    delta = curves[:, np.minimum(lower + 1, 4095)] - curves[:, lower]
    value = (curves[:, lower] + delta * (position - lower)) * (1.0 / 65535)
    biased = value * 65535.0 + 6755399441055744.0
    return (biased.view(np.uint64) & 65535).astype("<u2")


def profile_curves(profile: bytes) -> tuple[np.ndarray, ...]:
    tags = icc_tags(profile)
    if profile[20:24] != b"XYZ ":
        raise ValueError("Expected an RGB/XYZ output profile.")
    result = []
    for signature in (b"rTRC", b"gTRC", b"bTRC"):
        tag = tags.get(signature, b"")
        if len(tag) < 12 or tag[:8] != b"curv\0\0\0\0":
            raise ValueError("Expected a sampled output curve.")
        count = struct.unpack_from(">I", tag, 8)[0]
        if not 2 <= count <= 65536 or 12 + count * 2 != len(tag):
            raise ValueError("Invalid sampled curve count.")
        words = np.frombuffer(tag, dtype=">u2", count=count, offset=12)
        result.append(words.astype("<u2"))
    return tuple(result)


def inverse_curve(words: np.ndarray) -> np.ndarray:
    """CML0x18020: invert a strictly increasing full-domain sampled TRC."""
    if (
        words.ndim != 1
        or not 2 <= len(words) <= 65536
        or words.dtype != np.dtype("<u2")
        or words[0] != 0
        or words[-1] != 65535
        or np.any(np.diff(words.astype(np.int64)) <= 0)
    ):
        raise ValueError("Expected a strictly increasing full-domain uint16 TRC.")
    words = words.astype(np.float64)
    queries = np.arange(65536, dtype=np.float64)
    lower = np.clip(np.searchsorted(words, queries, side="left") - 1, 0, len(words) - 2)
    reciprocal = 1.0 / (words[lower + 1] - words[lower])
    expanded = ((queries - words[lower]) * reciprocal + lower) * (
        65535.0 / (len(words) - 1)
    )
    # Add the float32 rounding bias in double precision before narrowing.
    biased = (expanded + 12582912.0).astype(np.float32)
    result = (biased.view(np.uint32) & 65535).astype("<u2")
    result[-1] = 65535
    return result


def output_luts(profile: bytes) -> np.ndarray:
    """CML0x3160: output range0x2000..0xdfff, no extra LUT or cyclic channel."""
    inverse = np.stack([inverse_curve(w) for w in profile_curves(profile)])
    coordinate = np.arange(49152, dtype=np.float64) * (1.0 / 49151)
    position = coordinate * 65535
    lower = position.astype(np.int64)
    words = inverse.astype(np.float64)
    interpolated = (
        (position - lower) * (words[:, np.minimum(lower + 1, 65535)] - words[:, lower])
        + words[:, lower]
    ) * (1.0 / 65535)
    biased = interpolated * 65535.0 + 6755399441055744.0
    active = (biased.view(np.uint64) & 65535).astype("<u2")
    return np.pad(active, ((0, 0), (8192, 8192)), mode="edge")
