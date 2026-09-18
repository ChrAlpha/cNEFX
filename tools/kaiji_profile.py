"""Reconstruct the bounded Nikon RGB-to-Lab ICC container from native planes."""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Sequence

import numpy as np

TEMPLATE_HASH = "044663073ae0ab74be677766cd1f402a1640b0c3f796cca48115498a20afceb2"
TAG_ORDER = (b"desc", b"cprt", b"wtpt", b"A2B0", b"B2A0")


def _tags(data: bytes) -> dict[bytes, bytes]:
    if len(data) < 192 or struct.unpack_from(">I", data, 128)[0] != 5:
        raise ValueError("Expected the five-tag profile layout.")
    result = {}
    for signature, offset, size in struct.iter_unpack(">4sII", data[132:192]):
        if offset < 192 or offset + size > len(data) or signature in result:
            raise ValueError("Invalid ICC tag bounds or duplicate tag.")
        result[signature] = data[offset : offset + size]
    if tuple(result) != TAG_ORDER:
        raise ValueError("Unexpected ICC tag order.")
    return result


def _text(value: str) -> bytes:
    if not isinstance(value, str) or len(value) > 254:
        raise ValueError("Expected at most 254 printable ASCII characters.")
    if any(not 32 <= ord(c) <= 126 for c in value):
        raise ValueError("ICC text must be printable ASCII.")
    return value.encode("ascii")


def build_profile(
    template: bytes,
    planes: Sequence[bytes],
    *,
    description: str,
    copyright: str,
    grid: int = 33,
) -> bytes:
    """Reproduce Rome0x28d510 with grid33/65 and zeroed output padding.

    The pinned template supplies header, identity curves/matrix, white point and
    reverse tag. Supplied planes contain little-endian uint16 ICC-Lab codes.
    """
    if (
        type(template) is not bytes
        or len(template) != 221888
        or hashlib.sha256(template).hexdigest() != TEMPLATE_HASH
    ):
        raise ValueError("Only the pinned Nikon NEF profile template is supported.")
    if type(grid) is not int or grid not in (33, 65):
        raise ValueError("Only grid33 and grid65 profile serialization is supported.")
    if len(planes) != 3 or any(
        type(p) is not bytes or len(p) != grid**3 * 2 for p in planes
    ):
        raise ValueError("Expected three complete little-endian grid planes.")
    tags = _tags(template)
    name, owner = _text(description), _text(copyright)
    encoded = name + b"\0"
    # Nikon uses a variable-sized legacy ScriptCode string in its desc payload.
    desc = b"desc" + bytes(4) + struct.pack(">I", len(encoded)) + encoded
    desc += bytes(10) + bytes([len(encoded)]) + encoded
    copyright_tag = b"text" + bytes(4) + owner + b"\0"
    clut = np.stack([np.frombuffer(p, dtype="<u2") for p in planes], axis=-1)
    forward = bytearray(tags[b"A2B0"][:0x6034])
    forward[10] = grid
    forward += clut.astype(">u2").tobytes()
    forward += tags[b"A2B0"][0x36034:0x36046]
    payloads = [desc, copyright_tag, tags[b"wtpt"], bytes(forward), tags[b"B2A0"]]
    output = bytearray(template[:192])
    output[16:24] = b"RGB Lab "
    for index, (signature, payload) in enumerate(zip(TAG_ORDER, payloads, strict=True)):
        output.extend(bytes((-len(output)) % 4))
        struct.pack_into(
            ">4sII", output, 132 + index * 12, signature, len(output), len(payload)
        )
        output.extend(payload)
    # The recovered allocation formula includes four bytes beyond the last tag.
    output.extend(bytes(4))
    struct.pack_into(">I", output, 0, len(output))
    return bytes(output)
