"""Test the packed/typed ColorBalance boundary without photograph fixtures."""

import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from colorbalance import import_decoded, iso_code  # noqa: E402


def test_import_retains_packed_fields_and_alignment():
    raw = b"0803" + bytes(range(4, 166))
    result = import_decoded(raw)
    assert len(result) == 0xD08
    assert result[:16] == b"0803" + bytes(12)
    assert result[16:91] == raw[:75]
    assert result[91] == 0
    assert result[92:94] == raw[75:77]
    assert result[94:96] == bytes(2)
    assert result[96:141] == raw[77:122]
    assert result[141:144] == bytes(3)
    assert result[144:188] == raw[122:]
    assert result[188:] == bytes(0xD08 - 188)


@pytest.mark.parametrize("value", [0, 1, 80, 104, 231, 32768, 65535])
def test_iso_uses_packed_offset_not_typed_offset(value):
    raw = bytearray(b"0803" + bytes(162))
    struct.pack_into("<H", raw, 0x8A, value)
    struct.pack_into("<H", raw, 0xA0, value ^ 65535)
    assert iso_code(bytes(raw)) == value


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"0803" + bytes(161),
        b"0803" + bytes(163),
        b"0703" + bytes(162),
        bytearray(b"0803" + bytes(162)),
    ],
)
def test_unsupported_records_are_rejected(data):
    with pytest.raises(ValueError, match="decoded"):
        import_decoded(data)
