"""Independent container and integer packing contracts for native NEFX output."""

import struct
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from nx_ctudio._tiff import Directory, Tag, read, serialize
from nx_ctudio.nefx import pack14, preview, write_nefx


def test_classic_tiff_relocates_embedded_preview_and_preserves_payload() -> None:
    thumbnail = Directory({})
    thumbnail.numbers(513, 4, 0)
    thumbnail.numbers(514, 4, 4)
    thumbnail.images[513] = [b"jpeg"]
    maker = Directory({17: Tag(17, 4, 1, bytes(4), [thumbnail])})
    maker.set(0x56, 7, b"0100" + bytes(12))
    embedded = Tag(37500, 7, 0, b"", [maker], b"Nikon\0\x02\x11\0\0")
    exif = Directory({37500: embedded})
    root = Directory({34665: Tag(34665, 4, 1, bytes(4), [exif])})
    root.set(271, 2, b"NIKON CORPORATION\0")
    encoded = serialize(root)
    restored = read(encoded)
    restored_maker = restored.tags[34665].children[0].tags[37500].children[0]
    assert restored_maker.tags[17].children[0].images[513] == [b"jpeg"]
    assert restored.tags[271].payload == b"NIKON CORPORATION\0"
    assert restored_maker.tags[0x56].payload == b"0100" + bytes(12)
    assert serialize(restored) == encoded


def test_parser_rejects_truncated_and_cyclic_directory() -> None:
    with pytest.raises(ValueError, match="bounds"):
        read(b"II*\0\x08\0\0\0")
    cycle = b"II*\0\x08\0\0\0" + struct.pack("<H", 1)
    cycle += struct.pack("<HHIII", 34665, 4, 1, 8, 0)
    with pytest.raises(ValueError, match="Repeated"):
        read(cycle)


def test_parser_rejects_nonzero_next_link() -> None:
    with pytest.raises(ValueError, match="next-directory"):
        read(b"II*\0\x08\0\0\0" + struct.pack("<HI", 0, 8))


def test_parser_rejects_repeated_payload_amplification() -> None:
    count, block_size = 512, 1024 * 1024
    offset = 8 + 2 + 12 * count + 4
    content = b"II*\0\x08\0\0\0" + struct.pack("<H", count)
    content += b"".join(
        struct.pack("<HHII", 1000 + i, 7, block_size, offset) for i in range(count)
    )
    content += bytes(4 + block_size)
    with pytest.raises(ValueError, match="copy budget"):
        read(content)


def test_failed_write_never_leaves_a_final_file(tmp_path: Path) -> None:
    output = tmp_path / "output.nefx"
    with patch("nx_ctudio.nefx.encode", return_value=b"content"):
        with patch("nx_ctudio._atomic.os.fsync", side_effect=OSError("Disk full")):
            with pytest.raises(OSError, match="Disk full"):
                write_nefx(output, b"", np.zeros((1, 4, 3), np.uint16), 4)
    assert list(tmp_path.iterdir()) == []


def test_atomic_writer_does_not_replace_an_existing_destination(tmp_path: Path) -> None:
    output = tmp_path / "output.nefx"
    output.write_bytes(b"existing")
    with patch("nx_ctudio.nefx.encode") as encode:
        with pytest.raises(FileExistsError):
            write_nefx(output, b"", np.zeros((1, 4, 3), np.uint16), 4)
        encode.assert_not_called()
    assert output.read_bytes() == b"existing"


def test_packed14_uses_little_endian_bits_and_zero_padding() -> None:
    values = np.array([0, 1, 16382, 16383, 1, 2, 3, 4, 5, 6, 7, 8], np.uint16)
    encoded = pack14(values.reshape(1, 4, 3))
    assert len(encoded) == 32
    decoded = []
    for start in (0, 7, 14):
        value = int.from_bytes(encoded[start : start + 7], "little")
        decoded.extend((value >> shift) & 16383 for shift in (0, 14, 28, 42))
    assert decoded == values.tolist()
    assert encoded[21:] == bytes(11)


def test_preview_omits_last_source_row_and_column() -> None:
    samples = np.full((424, 640, 3), 100, np.uint16)
    samples[-1] = 999
    samples[:, -1] = 888
    result = preview(samples)
    assert np.all(result[:-1, :-1] == 100)
    assert not np.any(result[-1])
    assert not np.any(result[:, -1])
