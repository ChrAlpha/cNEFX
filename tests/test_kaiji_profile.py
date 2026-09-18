"""Bound legacy ICC serialization and verify plane byte order."""

import hashlib
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import kaiji_profile  # noqa: E402


def template(monkeypatch):
    data = bytearray(221888)
    data[36:40] = b"acsp"
    struct.pack_into(">I", data, 128, 5)
    offset = 192
    for index, (signature, length) in enumerate(
        zip(kaiji_profile.TAG_ORDER, (100, 30, 20, 0x36046, 250), strict=True)
    ):
        struct.pack_into(">4sII", data, 132 + index * 12, signature, offset, length)
        offset += length
    result = bytes(data)
    monkeypatch.setattr(
        kaiji_profile, "TEMPLATE_HASH", hashlib.sha256(result).hexdigest()
    )
    return result


@pytest.mark.parametrize("name,owner", [("NX Ctudio", "Test"), ("X" * 254, "")])
@pytest.mark.parametrize("grid", [33, 65])
def test_tag_alignment_and_plane_interleaving(monkeypatch, name, owner, grid):
    base = template(monkeypatch)
    planes = [struct.pack("<H", v) * grid**3 for v in (1, 257, 65280)]
    result = kaiji_profile.build_profile(
        base, planes, description=name, copyright=owner, grid=grid
    )
    assert struct.unpack_from(">I", result)[0] == len(result)
    assert result[16:24] == b"RGB Lab "
    entries = list(struct.iter_unpack(">4sII", result[132:192]))
    assert all(offset % 4 == 0 for _, offset, _ in entries)
    tags = kaiji_profile._tags(result)
    assert tags[b"A2B0"][10] == grid
    assert struct.unpack_from(">3H", tags[b"A2B0"], 0x6034) == (1, 257, 65280)
    assert struct.unpack_from(">3H", tags[b"A2B0"], 0x6034 + (grid**3 - 1) * 6) == (
        1,
        257,
        65280,
    )
    assert tags[b"wtpt"] == kaiji_profile._tags(base)[b"wtpt"]
    assert tags[b"B2A0"] == kaiji_profile._tags(base)[b"B2A0"]
    assert len(result) - entries[-1][1] - entries[-1][2] == 4


def test_unpinned_template_is_rejected():
    with pytest.raises(ValueError, match="pinned"):
        kaiji_profile.build_profile(
            bytes(221888), [], description="NX Ctudio", copyright=""
        )


@pytest.mark.parametrize("text", ["X" * 255, "bad\0name", "line\nbreak", "\x7f"])
def test_profile_text_rejects_truncation_and_control_characters(monkeypatch, text):
    with pytest.raises(ValueError, match="ASCII"):
        kaiji_profile.build_profile(
            template(monkeypatch),
            [bytes(33**3 * 2)] * 3,
            description=text,
            copyright="",
        )


def test_planes_require_complete_little_endian_bytes(monkeypatch):
    with pytest.raises(ValueError, match="planes"):
        kaiji_profile.build_profile(
            template(monkeypatch),
            [bytes(33**3 * 2 - 1)] * 3,
            description="NX Ctudio",
            copyright="",
        )


@pytest.mark.parametrize("grid", [0, 32, 64, 129, 65.0, True])
def test_profile_rejects_unsupported_grid(monkeypatch, grid):
    with pytest.raises(ValueError, match="grid"):
        kaiji_profile.build_profile(
            template(monkeypatch), [], description="NX Ctudio", copyright="", grid=grid
        )
