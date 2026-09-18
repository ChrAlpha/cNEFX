"""Integer boundaries in the native Kaiji scalar lattice evaluator."""

import importlib.util
import struct
import sys
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "tools/kaiji_scalar.py"
SPEC = importlib.util.spec_from_file_location("nxc_kaiji_scalar", PATH)
assert SPEC and SPEC.loader
scalar = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scalar)
sys.path.insert(0, str(PATH.parent))
from image_kaiji import verify_prepared_grid  # noqa: E402
from kaiji_output import MATRIX  # noqa: E402


def test_second_sector_keeps_signed_domain_endpoint() -> None:
    assert scalar._sector(-4096, -4032, 128) == (3, 4, 15)
    assert scalar._sector(-4095, -4032, 128) == (4, 5, 0)


def test_first_sector_clips_absolute_chroma() -> None:
    assert scalar._sector(-4096, -4032, 512) == scalar._sector(-4095, -4032, 512)


def test_sector_quadrant_wraps_and_zero_vector() -> None:
    assert scalar._sector(0, 0, 512) == (0, 1, 0)
    assert scalar._sector(0, -4096, 512) == (7, 16, 63)
    assert scalar._sector(0, 4095, 512) == (31, 24, 63)


def test_truncation_and_rounding_are_distinct_for_negative_halves() -> None:
    assert scalar._trunc(-31, 6) == 0
    assert scalar._round(-31, 6) == 0
    assert scalar._trunc(-32, 6) == 0
    assert scalar._round(-32, 6) == -1
    assert scalar._round(32, 6) == 1


@pytest.mark.parametrize(
    "data", [b"", bytes(0x3436F), bytes(0x34371), bytearray(0x34370)]
)
def test_reject_wrong_table_shape(data) -> None:
    with pytest.raises(ValueError):
        scalar.ScalarTable(data)


@pytest.mark.parametrize(
    "point,grid",
    [
        ((0, 0), 33),
        ((0, -1, 0), 33),
        ((33, 0, 0), 33),
        ((True, 0, 0), 33),
        ((0, 0, 0), 32),
    ],
)
def test_reject_invalid_lattice_inputs(point, grid) -> None:
    with pytest.raises(ValueError):
        scalar.ScalarTable(bytes(0x34370)).evaluate(point, grid)
    with pytest.raises(ValueError):
        scalar.ScalarTable(bytes(0x34370)).evaluate_rgb(point, grid)


def test_prepared_grid_rejects_wrong_matrix_and_table_tail():
    table = bytes(0x34370)
    with pytest.raises(ValueError, match="matrix"):
        verify_prepared_grid(table, table, [])
    prepared = bytearray(struct.pack("<9d", *MATRIX.flat) + table[72:])
    prepared[72] = 1
    with pytest.raises(ValueError, match="tail"):
        verify_prepared_grid(table, bytes(prepared), [])


@pytest.mark.parametrize("planes", [[], [b""] * 3, [bytes(33**3 * 2)] * 2])
def test_prepared_grid_rejects_incomplete_planes(planes):
    table = bytes(0x34370)
    prepared = struct.pack("<9d", *MATRIX.flat) + table[72:]
    with pytest.raises(ValueError, match="planes"):
        verify_prepared_grid(table, prepared, planes)
