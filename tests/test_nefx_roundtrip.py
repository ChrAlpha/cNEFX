"""Small synthetic NEFX fixtures decoded by real LibRaw, without vendor assets."""

import binascii
import struct
import sys
from pathlib import Path

import numpy as np
import pytest
import rawpy

from nx_ctudio._atomic import publication
from nx_ctudio._tiff import Directory, Tag, read, serialize
from nx_ctudio.nefx import write_nefx


def synthetic_source() -> bytes:
    """Construct only the metadata consumed by the writer; not an authentic NEF."""
    maker = Directory({})
    maker.numbers(0x45, 3, 0, 0, 64, 48)
    maker.set(0x51, 7, b"0102" + bytes(20))
    maker.set(0x54, 7, b"0100" + struct.pack("<4i", 32, 0, 24, 0))
    maker.set(0x5E, 7, b"0100" + bytes(24))
    info = Directory({})
    for code in (8, 11, 13):
        payload = b"0100" + struct.pack("<4i", 32, 0, 24, 0)
        info.set(code, 7, payload + struct.pack("<H", binascii.crc_hqx(payload, 65535)))
    raw = Directory({})
    for code, value in ((256, 64), (257, 48), (262, 32803), (273, 0), (279, 4)):
        raw.numbers(code, 4, value)
    raw.images[273] = [b"raw!"]
    header = b"Nikon\0\x02\x11\0\0"
    raw.tags[0xC7D5] = Tag(0xC7D5, 7, 0, b"", [info], header)
    exif = Directory({37500: Tag(37500, 7, 0, b"", [maker], header)})
    root = Directory({34665: Tag(34665, 4, 1, bytes(4), [exif])})
    root.set(271, 2, b"NIKON CORPORATION\0")
    root.set(272, 2, b"NIKON Z 8\0")
    root.set(700, 1, b"synthetic XMP preservation sentinel")
    images = [Directory({}) for _ in range(6)]
    images[1] = raw
    images[0].numbers(513, 4, 0)
    images[0].numbers(514, 4, 4)
    images[0].images[513] = [b"jpeg"]
    root.tags[330] = Tag(330, 4, 6, bytes(24), images)
    return serialize(root)


@pytest.mark.parametrize("count", [4, 8, 16, 32])
def test_real_libraw_roundtrip_and_metadata(tmp_path: Path, count: int) -> None:
    source = synthetic_source()
    scale = 2 if count >= 16 else 1
    samples = np.random.default_rng(count).integers(
        0, 16384, (48 * scale, 64 * scale, 3), dtype=np.uint16
    )
    samples[0, 0] = [0, 1, 16383]
    destination = tmp_path / "roundtrip.nefx"
    write_nefx(destination, source, samples, count)
    with rawpy.imread(str(destination)) as decoded:
        np.testing.assert_array_equal(decoded.raw_image[..., :3], samples)
        assert np.all(decoded.raw_image[..., 3:] == 0)
    result = read(destination.read_bytes())
    maker = result.tags[34665].children[0].tags[37500].children[0]
    raw = result.tags[330].children[1]
    info = raw.tags[0xC7D5].children[0]
    assert maker.tags[0x45].integers() == (0, 0, 64 * scale, 48 * scale)
    assert maker.tags[0xE2D].payload[7] == (4, 8, 16, 32).index(count) + 1
    assert len(maker.tags[0xE2E].payload) == 16 + 640 * 424 * 3 * 2
    assert result.tags[700].payload == b"synthetic XMP preservation sentinel"
    assert result.tags[330].children[0].images[513] == [b"jpeg"]
    for code in (8, 11, 13):
        payload = info.tags[code].payload
        assert int.from_bytes(payload[-2:], "little") == binascii.crc_hqx(
            payload[:-2], 65535
        )
    assert list(tmp_path.iterdir()) == [destination]


def test_directory_platform_contract(tmp_path: Path) -> None:
    destination = tmp_path / "output-set"
    if sys.platform == "linux":
        with publication(destination, directory=True) as pending:
            (pending.temporary / "complete").write_bytes(b"complete")
            pending.publish()
        assert (destination / "complete").read_bytes() == b"complete"
    else:
        with pytest.raises(NotImplementedError, match="currently require Linux"):
            with publication(destination, directory=True):
                pytest.fail("Unsupported publication must fail before staging.")
        assert list(tmp_path.iterdir()) == []
