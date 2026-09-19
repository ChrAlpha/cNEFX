"""Streaming output layout and failure boundaries independent of NEFX pixels."""

import hashlib
from io import BytesIO
from pathlib import Path

import pytest

from nx_ctudio._tiff import Directory, ImagePayload, Tag, read, serialize, serialize_to


def graph() -> Directory:
    thumbnail = Directory({})
    thumbnail.numbers(513, 4, 0)
    thumbnail.numbers(514, 4, 5)
    thumbnail.images[513] = [b"jpeg!"]
    maker = Directory({17: Tag(17, 4, 1, bytes(4), [thumbnail])})
    maker.set(0x56, 7, b"0100" + bytes(12))
    exif = Directory({37500: Tag(37500, 7, 0, b"", [maker], b"Nikon\0\x02\x11\0\0")})
    raw = Directory({})
    raw.numbers(273, 4, 0, 0)
    raw.numbers(279, 4, 6, 0)
    raw.images[273] = [b"abcdef", b""]
    root = Directory(
        {
            34665: Tag(34665, 4, 1, bytes(4), [exif]),
            330: Tag(330, 4, 1, bytes(4), [raw]),
        }
    )
    root.set(271, 2, b"NIKON CORPORATION\0")
    root.numbers(274, 3, 1)
    return root


def test_streamed_layout_matches_pre_streaming_bytes(tmp_path: Path) -> None:
    root = graph()
    raw = root.tags[330].children[0]
    raw.images[273][0] = ImagePayload(6, lambda: iter((b"ab", b"", b"cdef")))
    destination = tmp_path / "output.tiff"
    with destination.open("xb") as stream:
        serialize_to(root, stream)
    data = destination.read_bytes()
    assert hashlib.sha256(data).hexdigest() == (
        "4cfac852f5e7f0ef6f9ee70d6ea017b89810bccf43a8c15666172e53585c017a"
    )
    assert serialize(root) == data
    restored = read(data)
    assert restored.tags[330].children[0].images[273] == [b"abcdef", b""]
    maker = restored.tags[34665].children[0].tags[37500].children[0]
    assert maker.tags[17].children[0].images[513] == [b"jpeg!"]
    assert serialize(restored) == data


@pytest.mark.parametrize("chunks", [(b"abc",), (b"abcdefg",)])
def test_deferred_payload_length_mismatch_fails(chunks: tuple[bytes, ...]) -> None:
    root = graph()
    root.tags[330].children[0].images[273][0] = ImagePayload(6, lambda: iter(chunks))
    with pytest.raises(ValueError):
        serialize_to(root, BytesIO())


@pytest.mark.parametrize("length", [-1, 1 << 32])
def test_invalid_payload_size_is_rejected_before_iteration(length: int) -> None:
    def chunks():
        pytest.fail("Invalid reservation must fail before consuming chunks.")
        yield b"unused"

    root = graph()
    root.tags[330].children[0].images[273][0] = ImagePayload(length, chunks)
    with pytest.raises(ValueError):
        serialize_to(root, BytesIO())


def test_stream_rejects_existing_data_without_overwriting() -> None:
    output = BytesIO(b"existing")
    with pytest.raises(ValueError):
        serialize_to(graph(), output)
    assert output.getvalue() == b"existing"


def test_stream_rejects_short_writes() -> None:
    class ShortWriter(BytesIO):
        def write(self, data: bytes) -> int:
            return super().write(data[:-1])

    with pytest.raises(OSError):
        serialize_to(graph(), ShortWriter())


def test_payload_is_consumed_in_bounded_writes() -> None:
    class BoundedWriter(BytesIO):
        def write(self, data: bytes) -> int:
            assert len(data) <= 65536
            return super().write(data)

    payload = b"x" * 65536
    root = graph()
    root.tags[330].children[0].images[273][0] = ImagePayload(
        32 * len(payload), lambda: iter([payload] * 32)
    )
    output = BoundedWriter()
    serialize_to(root, output)
    restored = read(output.getvalue())
    assert restored.tags[330].children[0].images[273][0] == payload * 32
