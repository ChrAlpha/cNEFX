"""Bound stream recurrence and source key selection without photo fixtures."""

import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import colorbalance  # noqa: E402

from nx_ctudio import _tiff  # noqa: E402


@pytest.fixture
def table(monkeypatch):
    data = bytes(range(256)) * 2
    monkeypatch.setattr(
        colorbalance, "DECRYPT_TABLE_HASH", hashlib.sha256(data).hexdigest()
    )
    return data


def source():
    maker = _tiff.Directory({})
    maker.set(0x97, 7, b"0803" + bytes(162))
    maker.set(0x1D, 2, b"257\0")
    maker.numbers(0xA7, 4, 0x01020304)
    exif = _tiff.Directory({37500: _tiff.Tag(37500, 7, 0, b"", [maker])})
    root = _tiff.Directory({34665: _tiff.Tag(34665, 4, 1, bytes(4), [exif])})
    root.set(272, 2, b"NIKON Z 8\0")
    return root, maker


def test_stream_recurrence_wraps_without_touching_version(table):
    data = b"0803" + bytes(162)
    decoded = colorbalance.decrypt(data, 1, 0, table)
    assert decoded[:12] == b"0803" + bytes.fromhex("60c12386ea4fb51c")
    assert colorbalance.decrypt(decoded, 1, 0, table) == data
    assert data == b"0803" + bytes(162)


def test_source_uses_low_serial_byte_and_xor_of_all_shutter_bytes(table):
    root, _ = source()
    decoded = colorbalance.decrypt_source(root, table)
    assert decoded == colorbalance.decrypt(b"0803" + bytes(162), 1, 4, table)
    assert decoded[4] == 100


@pytest.mark.parametrize("key", [-1, 2**32, True, 1.0, "1", None])
def test_invalid_keys_rejected(table, key):
    for serial, count in ((key, 0), (0, key)):
        with pytest.raises(ValueError, match="32-bit"):
            colorbalance.decrypt(b"0803" + bytes(162), serial, count, table)


@pytest.mark.parametrize(
    "data",
    [b"", b"0703" + bytes(162), b"0803" + bytes(161), bytearray(b"0803" + bytes(162))],
)
def test_invalid_payload_rejected(table, data):
    with pytest.raises(ValueError, match="0803"):
        colorbalance.decrypt(data, 0, 0, table)


@pytest.mark.parametrize("data", [b"", bytes(512), bytearray(range(256)) * 2])
def test_unpinned_lookup_rejected(data):
    with pytest.raises(ValueError, match="table"):
        colorbalance.decrypt(b"0803" + bytes(162), 0, 0, data)


@pytest.mark.parametrize(
    "kind,payload",
    [
        (2, b"\0"),
        (2, b"+1\0"),
        (2, b" 1\0"),
        (2, b"1\0\0"),
        (2, b"4294967296\0"),
        (2, b"1"),
        (7, b"1\0"),
    ],
)
def test_malformed_serial_rejected(table, kind, payload):
    root, maker = source()
    maker.set(0x1D, kind, payload)
    with pytest.raises(ValueError):
        colorbalance.decrypt_source(root, table)


@pytest.mark.parametrize("tag", [0x97, 0x1D, 0xA7])
def test_missing_source_key_rejected(table, tag):
    root, maker = source()
    del maker.tags[tag]
    with pytest.raises(ValueError, match="Missing"):
        colorbalance.decrypt_source(root, table)


@pytest.mark.parametrize("tag", [0x97, 0x1D, 0xA7])
def test_wrong_tag_count_rejected(table, tag):
    root, maker = source()
    maker.tags[tag].count += 1
    with pytest.raises(ValueError):
        colorbalance.decrypt_source(root, table)
