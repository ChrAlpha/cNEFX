"""Bounded native source white balance for the observed Z8 metadata layout."""

from __future__ import annotations

import math
import struct

from colorbalance import import_decoded

from nx_ctudio import _tiff


def rational_gains(tag: _tiff.Tag) -> tuple[float, ...]:
    if tag.kind != 5 or tag.count != 4 or len(tag.payload) != 32:
        raise ValueError("Expected four unsigned rational white-balance gains.")
    pairs = tuple(struct.iter_unpack("<II", tag.payload))
    if any(denominator == 0 for _, denominator in pairs):
        raise ValueError("White-balance denominator must be nonzero.")
    result = tuple(numerator / denominator for numerator, denominator in pairs)
    if any(not 0 < value < 128 for value in result):
        raise ValueError("White-balance gain is outside the supported range.")
    return result


def source_white_balance(root: _tiff.Directory) -> tuple[float, float, float]:
    """Recover Rome2 0x128a00 flags0/1 for the validated Z8 source layout.

    This is source metadata, not application white-balance overrides. Other
    adjustment records and synthesized-tag overrides deliberately fail closed.
    """
    try:
        if root.tags[272].payload != b"NIKON Z 8\0":
            raise ValueError("Only the observed Nikon Z8 source layout is supported.")
        maker = root.tags[34665].children[0].tags[37500].children[0]
        adjustment = maker.tags[0xB0]
        if (
            0xE17 in maker.tags
            or adjustment.kind != 7
            or adjustment.count != 16
            or adjustment.payload != b"0103" + bytes(12)
        ):
            raise ValueError("Unsupported white-balance adjustment metadata.")
        red, blue, green_a, green_b = rational_gains(maker.tags[12])
        if rational_gains(maker.tags[59]) != (1.0, 1.0, 1.0, 1.0):
            raise ValueError("Only unity fine white balance is validated.")
    except (KeyError, IndexError) as exc:
        raise ValueError("Missing required Z8 white-balance metadata.") from exc
    return red, (green_a + green_b) * 0.5, blue


def white_balance_levels(rgb: tuple[float, float, float]) -> tuple[int, int]:
    """Encode TOPParam+1d0/+1d2 without normalizing the green channel."""
    if len(rgb) != 3 or any(
        type(value) not in (int, float)
        or not math.isfinite(value)
        or not 0 < value < 128
        for value in rgb
    ):
        raise ValueError("Expected three finite white-balance gains in (0, 128).")
    return int(rgb[0] * 512), int(rgb[2] * 512)


def source_blend_inputs(root: _tiff.Directory, decoded: bytes) -> dict:
    """Recover source-only Kaiji blend inputs for observed AUTO1/AUTO0 metadata."""
    levels = white_balance_levels(source_white_balance(root))
    typed = import_decoded(decoded)
    maker = root.tags[34665].children[0].tags[37500].children[0]
    modes = {b"AUTO1       \0": 17, b"AUTO0       \0": 20}
    mode = maker.tags.get(5)
    if mode is None or mode.kind != 2 or mode.count != 13 or mode.payload not in modes:
        raise ValueError("Only observed AUTO1/AUTO0 white-balance modes are supported.")
    return {
        "correction_matrix": list(struct.unpack_from("<9h", typed, 0xA6)),
        "wb_levels_signed": [(v + 32768) % 65536 - 32768 for v in levels],
        "wb_mode": modes[mode.payload],
        "wb_value": struct.unpack_from("<h", typed, 0x4A)[0],
    }
