"""Boundary checks for the recovered Flexible Color normalization ABI."""

import importlib.util
import struct
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
for name in ("inspect_np3", "np3_parameters"):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
normalize = sys.modules["np3_parameters"].normalize_flexible


def neutral() -> bytearray:
    data = bytearray(88)
    data[:4] = b"0100"
    data[12:24] = bytes((128, 1)) * 6
    data[24:48] = bytes([128] * 24)
    data[48:51] = bytes([1] * 3)
    for offset in (52, 56, 60):
        struct.pack_into("<HBB", data, offset, 32768, 128, 128)
    data[64:67] = bytes([1] * 3)
    data[68:72] = bytes((178, 1, 128, 1))
    return data


@pytest.mark.parametrize("version", [b"0100", b"0101"])
def test_neutral_state_and_return_byte(version: bytes) -> None:
    data = neutral()
    data[:4] = version
    assert normalize(data) == (128, [0] * 39 + [50, 0])


@pytest.mark.parametrize("value", [0, 1, 255])
def test_sentinels_default_before_division(value: int) -> None:
    data = neutral()
    data[12:24] = bytes((value, 0)) * 6
    data[24:48] = bytes([value] * 24)
    data[48:51] = bytes(3)
    for offset in (52, 56, 60):
        struct.pack_into("<HBB", data, offset, 65535, value, value)
    data[64:67] = bytes(3)
    data[68:72] = bytes((value, 0)) * 2
    assert normalize(data) == (value, [0] * 39 + [50, 0])


def test_nonunit_scales_signed_hues_and_truncation() -> None:
    data = neutral()
    data[12:16] = bytes((119, 4, 137, 4))
    data[24:30] = bytes((119, 137, 129, 137, 119, 127))
    data[48:51] = bytes((4, 4, 3))
    struct.pack_into("<HBB", data, 52, 32759, 137, 119)
    data[64:67] = bytes((4, 4, 4))
    returned, values = normalize(data)
    assert returned == 119
    assert values[:6] == [-2, 2, 0, 0, 0, 0]
    assert values[6:12] == [-2, 2, 0, 2, -2, 0]
    assert values[30:33] == [-2, 2, -2]


@pytest.mark.parametrize("offset", [13, 48, 49, 50, 64, 65, 66, 69, 71])
def test_ordinary_values_with_zero_scale_fail_closed(offset: int) -> None:
    data = neutral()
    data[offset] = 0
    with pytest.raises(ValueError, match="Zero parameter scale"):
        normalize(data)


@pytest.mark.parametrize("data", [b"", b"0100" + bytes(83), b"0200" + bytes(84)])
def test_unsupported_block_fails_closed(data: bytes) -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        normalize(data)
