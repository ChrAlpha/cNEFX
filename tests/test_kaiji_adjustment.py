"""Analytic controls for signed exposure-dependent tone and chroma branches."""

import hashlib
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import kaiji_camera as camera  # noqa: E402
from verify_kaiji_composition import exposure_configurations  # noqa: E402


@pytest.fixture
def inputs(monkeypatch):
    coefficients = bytearray(422416)
    struct.pack_into("<H", coefficients, 0x2A3A8, 64)
    for i in range(65):
        struct.pack_into("<H", coefficients, 0x2B040 + 14 * 130 + i * 2, 3000)
    tables = struct.pack("<37H", 1000, *([3000] * 35), 2000) * 257
    tables += struct.pack("<36H", *([512] * 35), 2048) * 65
    coefficients = bytes(coefficients)
    monkeypatch.setattr(
        camera, "COEFFICIENT_HASH", hashlib.sha256(coefficients).hexdigest()
    )
    monkeypatch.setattr(camera, "ADJUSTMENT_HASH", hashlib.sha256(tables).hexdigest())
    return coefficients, tables


def generate(inputs, adjustment, **kwargs):
    coefficients, tables = inputs
    return dict(
        camera.camera_ranges(
            coefficients,
            model_index=14,
            selector=0,
            saturation_bias=12,
            exposure_adjustment=adjustment,
            adjustment_tables=tables,
            **kwargs,
        )
    )


@pytest.mark.parametrize("adjustment,tone,gain", [(-36, 750, 1500), (-1, 2750, 2047)])
def test_tone_column_and_gain_before_clipping(inputs, adjustment, tone, gain):
    ranges = generate(inputs, adjustment)
    assert struct.unpack("<257H", ranges[0x21554]) == (tone,) * 257
    assert struct.unpack("<4096H", ranges[0x21758]) == (tone,) * 4096
    assert struct.unpack("<65H", ranges[0x237B4]) == (gain,) * 65
    assert struct.unpack("<4096H", ranges[0x23836]) == (gain,) * 4096


def test_nonnegative_adjustments_bypass_fixed_tables(inputs):
    assert generate(inputs, 0) == generate(inputs, 1) == generate(inputs, 30)
    coefficients, _ = inputs
    without_tables = dict(
        camera.camera_ranges(
            coefficients,
            model_index=14,
            selector=0,
            saturation_bias=12,
            exposure_adjustment=30,
        )
    )
    assert without_tables == generate(inputs, 30)


@pytest.mark.parametrize("adjustment", [-37, 31, True, -1.0])
def test_invalid_adjustment_is_rejected(inputs, adjustment):
    with pytest.raises(ValueError, match="Encoded exposure"):
        generate(inputs, adjustment)


@pytest.mark.parametrize("tables", [None, b"", bytes(23698), bytearray(23698)])
def test_negative_adjustment_requires_pinned_tables(inputs, tables):
    coefficients, _ = inputs
    with pytest.raises(ValueError, match="tables"):
        camera.camera_ranges(
            coefficients,
            model_index=14,
            selector=0,
            saturation_bias=12,
            exposure_adjustment=-1,
            adjustment_tables=tables,
        )


def test_sweep_covers_signed_range_at_three_camera_indices():
    cases = exposure_configurations()
    assert len(cases) == 217
    for index in (0, 54, 108):
        assert {v for _, _, i, v in cases[:201] if i == index} == set(range(-36, 31))
    assert all(v < 0 for _, _, _, v in cases[201:])
    assert len({tuple(values) for values, _, _, _ in cases[201:]}) == 16
