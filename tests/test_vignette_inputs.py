"""Source-derived vignette parameters and unsupported metadata rejection."""

import copy
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

from nx_ctudio._tiff import Directory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from vignette_inputs import source_parameters  # noqa: E402


def source():
    root, exif, maker, raw, info = [Directory({}) for _ in range(5)]
    root.set(272, 2, b"NIKON Z 8\0")
    root.numbers(34665, 4, 0)
    root.tags[34665].children = [exif]
    exif.set(37500, 7, b"")
    exif.tags[37500].children = [maker]
    root.numbers(330, 4, *range(6))
    root.tags[330].children = [Directory({}), raw] + [Directory({}) for _ in range(4)]
    raw.numbers(256, 4, 8280)
    raw.numbers(257, 4, 5520)
    raw.numbers(262, 3, 32803)
    raw.set(0xC7D5, 7, b"")
    raw.tags[0xC7D5].children = [info]
    maker.numbers(0x45, 3, 12, 8, 8256, 5504)
    maker.numbers(0x2A, 3, 3)
    maker.set(0x54, 7, b"0100" + struct.pack("<4i", -512, 256, 256, 256))
    block = bytearray(116)
    block[:8] = b"0100\x01\x01\0\0"
    struct.pack_into("<I", block, 16, 8)
    for index, value in enumerate((0, 0, 131072, 0, 262144, 0, 524288, 0, 1048576)):
        struct.pack_into("<2i", block, 20 + index * 8, value, 1048576)
    info.set(6, 7, bytes(block))
    return root, maker, raw, info


@pytest.mark.parametrize(
    "size,offset",
    [
        ((8256, 5504), (-2.0, 1.0)),
        ((2064, 1376), (-0.5, 0.25)),
        ((258, 172), (-0.0625, 0.03125)),
    ],
)
def test_metadata_coefficients_and_float32_center_scaling(size, offset):
    root, _, _, _ = source()
    before = copy.deepcopy(root)
    actual_offset, actual_coefficients = source_parameters(root, size)
    assert actual_offset == offset
    assert actual_coefficients == (0.0, 0.125, 0.25, 0.5)
    assert root == before


def test_uses_signed_rationals_and_narrows_once_to_float32():
    root, _, _, info = source()
    block = bytearray(info.tags[6].payload)
    struct.pack_into("<2i", block, 36, -1, 3)
    info.set(6, 7, bytes(block))
    assert source_parameters(root, (258, 172))[1][1] == float(np.float32(-1 / 3))


@pytest.mark.parametrize(
    "size",
    [
        None,
        (258,),
        (0, 0),
        (258, 171),
        (300, 200),
        (True, 172),
        (258.0, 172),
        (16512, 11008),
    ],
)
def test_reject_unsupported_scale_and_dimensions(size):
    with pytest.raises(ValueError):
        source_parameters(source()[0], size)


@pytest.mark.parametrize(
    "mutation",
    [
        "model",
        "raw_size",
        "raw_type",
        "crop",
        "strength",
        "center_version",
        "center_scale",
        "center_size",
        "block_version",
        "block_size",
        "degree",
        "denominator",
        "odd_coefficient",
        "constant",
        "missing",
        "duplicate_raw",
    ],
)
def test_reject_unsupported_source_metadata(mutation):
    root, maker, raw, info = source()
    if mutation == "model":
        root.set(272, 2, b"NIKON Z 9\0")
    elif mutation == "raw_size":
        raw.numbers(256, 4, 8256)
    elif mutation == "raw_type":
        raw.numbers(262, 3, 2)
    elif mutation == "crop":
        maker.numbers(0x45, 3, 0, 0, 8256, 5504)
    elif mutation == "strength":
        maker.numbers(0x2A, 3, 1)
    elif mutation.startswith("center"):
        center = bytearray(maker.tags[0x54].payload)
        if mutation == "center_version":
            center[0] = 49
        elif mutation == "center_scale":
            struct.pack_into("<I", center, 8, 0)
        else:
            center = center[:-1]
        maker.set(0x54, 7, bytes(center))
    elif mutation == "missing":
        del info.tags[6]
    elif mutation == "duplicate_raw":
        root.tags[330].children[0] = raw
    else:
        block = bytearray(info.tags[6].payload)
        if mutation == "block_version":
            block[0] = 49
        elif mutation == "block_size":
            block = block[:-1]
        else:
            position, value = {
                "degree": (16, 6),
                "denominator": (24, 0),
                "odd_coefficient": (28, 1),
                "constant": (84, 0),
            }[mutation]
            struct.pack_into("<i", block, position, value)
        info.set(6, 7, bytes(block))
    with pytest.raises(ValueError):
        source_parameters(root, (258, 172))
