"""Compare live image factory state with bounded native Flexible Color tables."""

from __future__ import annotations

import hashlib
import itertools
import struct

import numpy as np
from flexible_math import round_divide32
from kaiji_index import camera_index
from kaiji_mix import blend_mix
from kaiji_output import MATRIX, MATRIX_SELECTOR1, rgb_to_icc_lab
from kaiji_scalar import ScalarTable
from kaiji_table import generate_table


def application_settings(values: list[int], custom_curve: bool) -> dict:
    if (
        len(values) != 41
        or any(type(v) is not int for v in values)
        or any(not -100 <= v <= 100 for v in values[:5])
        or not -50 <= values[5] <= 50
        or type(custom_curve) is not bool
    ):
        raise ValueError("Expected normalized Flexible Color settings.")
    return {
        "contrast": 0 if custom_curve else round_divide32(values[0] * 64, 25),
        "highlights": round_divide32(values[1] * 6, 5),
        "shadows": round_divide32(values[2] * 6, 5),
        "white": values[3],
        "black": values[4],
        "saturation": values[5],
        "saturation_bias": round_divide32(values[5] * 6, 25) + 12,
        "custom_curve": int(custom_curve),
    }


def verify_state(
    data: bytes,
    values: list[int],
    settings: dict,
    index: int,
    *,
    output_selector: int = 0,
) -> dict:
    if type(data) is not bytes or len(data) != 0x17F8:
        raise ValueError("Expected a complete 6136-byte KaijiParam snapshot.")
    if type(output_selector) is not int or output_selector not in (0, 1):
        raise ValueError("Unsupported output selector.")
    blend_mix = struct.unpack_from("<H", data, 0x1528)[0]
    if not 0 <= blend_mix <= 10:
        raise ValueError("Live blend mix is outside the validated generator range.")
    layout = {
        "camera": (0xC, "I", 1071),
        "output_selector": (0xE3C, "I", output_selector),
        "profile": (0x1518, "I", 14),
        "family": (0x151C, "I", 12),
        "strength": (0x1520, "H", 256),
        "saturation_bias": (0x1522, "H", settings["saturation_bias"]),
        "matrix": (0x1524, "H", 0),
        "camera_mode": (0x1526, "H", 16),
        "camera_index": (0x152A, "H", index),
        "raw152c": (0x152C, "H", 0),
        "exposure_adjustment": (0x1533, "b", 0),
        "hue": (0x1538, "H", 128),
        "raw153e": (0x153E, "h", 255),
        "contrast": (0x1548, "h", settings["contrast"]),
        "brightness": (0x154A, "B", 6),
        "alternate_tone": (0x174F, "B", 0),
        "custom_curve": (0x1756, "B", settings["custom_curve"]),
        "highlights": (0x1758, "h", settings["highlights"]),
        "shadows": (0x175A, "h", settings["shadows"]),
        "white": (0x175C, "h", settings["white"]),
        "black": (0x175E, "h", settings["black"]),
        "saturation": (0x1760, "h", settings["saturation"]),
        "blending": (0x17E8, "H", values[39]),
        "balance": (0x17EA, "h", values[40]),
    }
    observed = {"blend_mix": blend_mix}
    for name, (offset, code, expected) in layout.items():
        observed[name] = struct.unpack_from("<" + code, data, offset)[0]
        if observed[name] != expected:
            raise ValueError(f"Unsupported or mismatched live Kaiji field: {name}")
    observed["blender"] = list(struct.unpack_from("<24i", data, 0x1764))
    observed["grading"] = list(struct.unpack_from("<9i", data, 0x17C4))
    if observed["blender"] != values[6:30] or observed["grading"] != values[30:39]:
        raise ValueError("Live Kaiji Blender/Grading controls disagree with NP3.")
    return observed


def verify_image_table(
    state: bytes,
    table: bytes,
    picture: dict,
    source: dict,
    top: dict,
    coefficients: bytes,
    modes: bytes,
    curve: bytes | None,
    blend_inputs: dict,
    blend_calibration: bytes,
    *,
    output_selector: int = 0,
) -> dict:
    values = picture["normalized_values"]
    settings = application_settings(values, picture["custom_curve"])
    if (curve is not None) != picture["custom_curve"] or source["iso_bucket"] != 6:
        raise ValueError("Curve activation or source ISO is outside validated scope.")
    selector = source["wb_levels"] + [
        0,
        6,
        settings["saturation_bias"],
        settings["contrast"],
        22,
        source["iso_offset"],
        source["processing"],
        source["mode"],
    ]
    index = camera_index(coefficients, modes, selector)
    if top["selector_inputs"] != selector or top["camera_index"] != index:
        raise ValueError("Native application selector disagrees with working TOP.")
    observed = verify_state(
        state, values, settings, index, output_selector=output_selector
    )
    captured = {
        "correction_matrix": list(struct.unpack_from("<9h", state, 0xB4)),
        "wb_levels_signed": list(struct.unpack_from("<2h", state, 0x152E)),
        "wb_mode": struct.unpack_from("<I", state, 0x66C)[0],
        "wb_value": struct.unpack_from("<i", state, 0x674)[0],
    }
    if captured != blend_inputs:
        raise ValueError("Native source blend inputs disagree with live Kaiji state.")
    native_mix = blend_mix(
        blend_calibration,
        blend_inputs["correction_matrix"],
        *blend_inputs["wb_levels_signed"],
        blend_inputs["wb_mode"],
        blend_inputs["wb_value"],
    )
    if native_mix != observed["blend_mix"]:
        raise ValueError("Native blend selection disagrees with live Kaiji state.")
    effective = values.copy()
    if curve is not None:
        effective[0] = 0
    native = generate_table(
        coefficients,
        effective,
        camera_index=index,
        curve_payload=curve,
        blend_mix=native_mix,
        output_selector=output_selector,
    )
    if len(table) != len(native):
        raise ValueError("Expected a complete 213872-byte Kaiji table.")
    differences = sum(a != b for a, b in zip(native, table, strict=True))
    if differences:
        raise ValueError(f"Native image table differs in {differences} bytes.")
    return {
        "fields": observed,
        "vendor_inputs": {},
        "native_blend_inputs": blend_inputs,
        "native_blend_mix": native_mix,
        "selector_inputs": selector,
        "compared_bytes": len(native),
        "different_bytes": differences,
        "native_sha256": hashlib.sha256(native).hexdigest(),
        "curve_samples_compared": 257 if curve is not None else 0,
    }


def verify_prepared_grid(
    table: bytes,
    prepared: bytes,
    planes: list[bytes],
    *,
    renderer_profile: bool = False,
) -> dict:
    """Compare Lab preparation after the complete source table is verified."""
    if type(renderer_profile) is not bool:
        raise ValueError("Renderer profile selection must be boolean.")
    grid, selector = (65, 1) if renderer_profile else (33, 0)
    matrix = MATRIX_SELECTOR1 if renderer_profile else MATRIX
    if len(table) != 0x34370 or len(prepared) != len(table):
        raise ValueError("Expected complete initial and prepared Kaiji tables.")
    if prepared[:72] != struct.pack("<9d", *matrix.flat) or prepared[72:] != table[72:]:
        raise ValueError("Prepared output matrix or table tail mismatch.")
    if len(planes) != 3 or any(len(plane) != grid**3 * 2 for plane in planes):
        raise ValueError(f"Expected three complete {grid}-node output planes.")
    evaluator = ScalarTable(table)
    rgb = np.array(
        [
            evaluator.evaluate_rgb(point, grid)
            for point in itertools.product(range(grid), repeat=3)
        ]
    )
    native = rgb_to_icc_lab(rgb, selector=selector).T
    expected = np.array([np.frombuffer(plane, dtype="<u2") for plane in planes])
    delta = native.astype(np.int32) - expected.astype(np.int32)
    differences = int(np.count_nonzero(delta))
    maximum = int(np.max(np.abs(delta)))
    if differences:
        raise ValueError(
            f"Prepared Lab grid differs in {differences} words, max {maximum}."
        )
    return {
        "grid": grid,
        "output": f"selector{selector}_icc_lab",
        "compared_words": int(native.size),
        "different_words": differences,
        "max_absolute_error": maximum,
        "native_plane_sha256": [
            hashlib.sha256(row.astype("<u2").tobytes()).hexdigest() for row in native
        ],
    }
