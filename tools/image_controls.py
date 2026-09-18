"""Recover bounded source-metadata controls for the Z8 TOPParam generator."""

from __future__ import annotations

import hashlib
import struct

from color_inputs import source_white_balance, white_balance_levels
from colorbalance import iso_code
from kaiji_grading import COEFFICIENT_HASH
from kaiji_index import camera_index

from nx_ctudio import _tiff


def source_controls(root: _tiff.Directory, decoded: bytes, coefficients: bytes) -> dict:
    if (
        type(coefficients) is not bytes
        or len(coefficients) != 422416
        or hashlib.sha256(coefficients).hexdigest() != COEFFICIENT_HASH
    ):
        raise ValueError("Only the pinned Z8 coefficients are supported.")
    maker = root.tags[34665].children[0].tags[37500].children[0]
    if 0xE2D in maker.tags:
        raise ValueError("Only absent processing metadata is validated.")
    rgb = source_white_balance(root)
    iso = iso_code(decoded)
    return {
        "rgb": list(rgb),
        "wb_levels": list(white_balance_levels(rgb)),
        "iso_code": iso,
        "iso_offset": coefficients[0x830 + min(108, ((iso - 72) >> 2) & 255)],
        "iso_bucket": min(6, ((iso - 80) >> 2) & 255),
        "processing": 0,
        "mode": 0,
    }


def top_controls(data: bytes, *, active: bool) -> dict:
    if type(data) is not bytes or len(data) != 0x4D0:
        raise ValueError("Expected a complete 1232-byte TOPParam snapshot.")
    bank = 0x250 if active else 0

    def at(offset: int, kind: str) -> int:
        return struct.unpack_from("<" + kind, data, offset)[0]

    levels = list(struct.unpack_from("<2H", data, 0x1D0))
    result = {
        "rgb": list(struct.unpack_from("<3d", data, 0x178)),
        "wb_levels": levels,
        "iso_code": at(0x198, "H"),
        "iso_offset": at(bank + 0x23C, "B"),
        "iso_bucket": at(bank + 0x23D, "B"),
        "processing": at(0x14, "I"),
        "mode": at(0x1C, "I"),
        "camera_index": at(bank + 0x23E, "B"),
        "picture_control_id": at(bank + 0xB8, "I"),
        "profile_row": at(bank + 0x208, "I"),
        "profile_family": at(bank + 0x20C, "I"),
        "custom_curve": at(bank + 0x170, "B") != 0,
        "normalized_values": list(struct.unpack_from("<41i", data, bank + 0xCC)),
    }
    result["selector_inputs"] = levels + [
        at(bank + 0x200, "B"),
        at(bank + 0x24A, "B"),
        at(bank + 0x24B, "B"),
        at(bank + 0x248, "h"),
        at(bank + 0x246, "B"),
        result["iso_offset"],
        result["processing"],
        result["mode"],
    ]
    return result


def verify_top(
    data: bytes, expected: dict, coefficients: bytes, mode_table: bytes, *, active: bool
) -> dict:
    observed = top_controls(data, active=active)
    if any(observed[key] != value for key, value in expected.items()):
        raise ValueError("Metadata-derived controls disagree with live TOPParam.")
    index = camera_index(coefficients, mode_table, observed["selector_inputs"])
    if observed["camera_index"] != index:
        raise ValueError("Native selector disagrees with live TOPParam.")
    return dict(observed, native_camera_index=index)
