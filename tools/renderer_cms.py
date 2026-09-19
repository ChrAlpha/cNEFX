"""Native source-profile preparation and bounded CMS0 RGB16 composition.

Source photographs supply metadata only. Pixels must already be at the Nlg
output boundary; this module does not decode or preprocess sensor samples.
"""

from __future__ import annotations

import itertools

import numpy as np
from color_inputs import source_blend_inputs
from colorbalance import decrypt_source
from image_controls import source_controls
from image_kaiji import application_settings
from inspect_np3 import parse
from kaiji_index import camera_index
from kaiji_mix import blend_mix
from kaiji_output import rgb_to_icc_lab
from kaiji_profile import build_profile
from kaiji_scalar import ScalarTable
from kaiji_table import generate_table
from np3_parameters import normalize_flexible, parameter_blocks
from renderer_lab_rgb import transform as lab_to_rgb
from renderer_lut import normalized_grid
from renderer_lut import transform as rgb_to_lab
from renderer_profile import parameters

from nx_ctudio import _tiff


def source_table(
    nef: bytes,
    picture_control: bytes,
    *,
    coefficients: bytes,
    modes: bytes,
    blend_calibration: bytes,
    decoding_table: bytes,
) -> bytes:
    """Generate Mode3 tables from bounded Z8 metadata and Flexible Color NP3.

    The four external resources are hash-pinned by their existing consumers.
    Source ISO bucket 6, absent processing metadata and AUTO0/AUTO1 white balance
    retain the established image-factory scope.
    """
    root = _tiff.read(nef)
    decoded = decrypt_source(root, decoding_table)
    source = source_controls(root, decoded, coefficients)
    if source["iso_bucket"] != 6:
        raise ValueError("Source ISO is outside the validated factory scope.")
    blend = source_blend_inputs(root, decoded)
    records = parse(picture_control)
    _, values = normalize_flexible(parameter_blocks(records)["flexible"])
    curve = next((record.payload for record in records if record.identifier == 2), None)
    settings = application_settings(values, curve is not None)
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
    mix = blend_mix(
        blend_calibration,
        blend["correction_matrix"],
        *blend["wb_levels_signed"],
        blend["wb_mode"],
        blend["wb_value"],
    )
    if curve is not None:
        values[0] = 0
    return generate_table(
        coefficients,
        values,
        camera_index=index,
        blend_mix=mix,
        curve_payload=curve,
        output_selector=1,
    )


def source_profile(
    table: bytes, template: bytes, *, description: str, copyright: str
) -> bytes:
    """Generate every grid65 plane natively, then serialize the source ICC."""
    evaluator = ScalarTable(table)
    rgb = np.array(
        [
            evaluator.evaluate_rgb(point, 65)
            for point in itertools.product(range(65), repeat=3)
        ]
    )
    lab = rgb_to_icc_lab(rgb, selector=1)
    return build_profile(
        template,
        [lab[:, channel].astype("<u2").tobytes() for channel in range(3)],
        description=description,
        copyright=copyright,
        grid=65,
    )


def convert(
    source: bytes, destination: bytes, pixels: np.ndarray, width: int
) -> np.ndarray:
    """Apply observed CMS0 mode6: grid65 RGB/Lab, then sampled-matrix Lab/RGB.

    Both stages preserve the supplied tile width, including each row's scalar
    tail. Profiles must satisfy the existing bounded parsers.
    """
    grid = normalized_grid(source)
    matrix, curves = parameters(destination, inverse=True)
    return lab_to_rgb(rgb_to_lab(pixels, grid, width), matrix, curves, width)
