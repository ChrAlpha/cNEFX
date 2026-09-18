# /// script
# requires-python = ">=3.11"
# dependencies = ["capstone==5.0.6", "pefile==2024.8.26", "tifffile==2026.9.15"]
# ///
"""Predict pinned Nikon Adobe defaults from explicit local inputs, without execution.

Usage: uv run tools/adobe_defaults.py --dll PixelShiftCombine.dll --prm prm.bin \
--source input.NEF --count 8
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import capstone as cs
import pefile
import tifffile
from capstone.x86 import X86_OP_IMM, X86_OP_MEM, X86_REG_RBP, X86_REG_RSP
from inspect_prm import decode, records

DLL_HASH = "c82782c51be65d9528f6751bfca9d2bdc7b728baa3ee8eca2e84ba713ed9031c"
PRM_HASH = "8d6f1bf219561f3c7c11c61b9b9e3317a369c9b5be5089ddffd82db96ca7f707"
RESOURCE_HASH = "d44561f16bf877a264ecfbb947f5b6635d6b0e9151fa70ca601d9b131804822f"
CRD = "{http://ns.adobe.com/camera-raw-defaults/1.0/}"
FIELDS = (
    "LuminanceSmoothing",
    "LuminanceNoiseReductionDetail",
    "LuminanceNoiseReductionContrast",
    "ColorNoiseReduction",
    "ColorNoiseReductionDetail",
    "ColorNoiseReductionSmoothness",
    "Sharpness",
    "SharpenRadius",
    "SharpenDetail",
    "SharpenEdgeMasking",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def signed(value: int, bits: int = 8) -> int:
    return (value + (1 << (bits - 1))) % (1 << bits) - (1 << (bits - 1))


def trunc_div(value: int, divisor: int) -> int:
    return (1 if value >= 0 else -1) * (abs(value) // divisor)


def u16(data: bytes, offset: int) -> int:
    require(0 <= offset <= len(data) - 2, "uint16 resource index out of bounds.")
    return struct.unpack_from("<H", data, offset)[0]


def byte(data: bytes, offset: int) -> int:
    require(0 <= offset < len(data), "Byte resource index out of bounds.")
    return data[offset]


def interpolate(table: bytes | list[int], coordinate: int) -> int:
    require(0 <= coordinate <= 108 and len(table) >= 19, "Invalid table coordinate.")
    quotient, remainder = divmod(coordinate, 6)
    if remainder == 0:
        return table[quotient]
    return (
        (6 - remainder) * table[quotient] + remainder * table[quotient + 1] + 3
    ) // 6


def sharp_values(value: int) -> tuple[int, int, int, int]:
    amount = 0 if value <= 0 else value if value <= 8 else min(150, 2 * value - 8)
    detail = max(13, min(25, 25 - max(0, (value - 29) // 4)))
    return amount, 20, detail, 0


def nr_knots(setting: int) -> list[list[int]]:
    require(setting in (0, 2, 4, 6), "Unsupported HighISONoiseReduction value.")
    base = {0: 0, 2: 30, 4: 45, 6: 60}[setting]
    return [[min(100, base + 3 * max(k - 2, 0)) for k in range(19)]] + [
        [value] * 19 for value in (75, 0, 7 if setting == 0 else 25, 50, 50)
    ]


def stack_literals(pe: pefile.PE, rva: int, size: int, rbp_bias: int) -> dict[int, int]:
    decoder = cs.Cs(cs.CS_ARCH_X86, cs.CS_MODE_64)
    decoder.detail = True
    stack: dict[int, int] = {}
    for instruction in decoder.disasm(pe.get_data(rva, size), rva):
        if instruction.mnemonic != "mov" or len(instruction.operands) != 2:
            continue
        target, source = instruction.operands
        if target.type != X86_OP_MEM or source.type != X86_OP_IMM:
            continue
        require(target.mem.base in (X86_REG_RBP, X86_REG_RSP), "Unexpected LUT store.")
        offset = target.mem.disp + (rbp_bias if target.mem.base == X86_REG_RBP else 0)
        for index in range(target.size):
            stack[offset + index] = (source.imm >> (8 * index)) & 255
    return stack


def verify_tables(pe: pefile.PE) -> dict[str, int]:
    expected = bytes(
        value for index in range(-128, 128) for value in sharp_values(index)
    )
    require(
        pe.get_data(0x5EDF60, 1024) == expected, "Sharpening LUT equation mismatch."
    )
    stack = stack_literals(pe, 0x34F0C5, 0x417, 0x100)
    for group, setting in enumerate((0, 2, 4, 6)):
        expected = [value for row in nr_knots(setting) for value in row]
        actual = [stack.get(0x13 + group * 0x90 + index) for index in range(114)]
        require(actual == expected, "Noise-reduction LUT equation mismatch.")
    return {"sharpening_table_bytes_verified": 1024, "noise_table_bytes_verified": 456}


def read_ifd(image: tifffile.TiffFile, data: bytes, offset: int) -> dict[int, object]:
    require(0 <= offset <= len(data) - 2, "IFD offset out of bounds.")
    count = u16(data, offset)
    require(offset + 2 + 12 * count + 4 <= len(data), "Truncated IFD.")
    result = {}
    for index in range(count):
        tag = tifffile.TiffTag.fromfile(image, offset=offset + 2 + 12 * index)
        require(tag.code not in result, "Duplicate metadata tag.")
        result[tag.code] = tag.value
    return result


def read_source(data: bytes) -> tuple[dict[int, object], bytes | None]:
    with tifffile.TiffFile(io.BytesIO(data)) as image:
        require(
            image.byteorder == "<" and not image.is_bigtiff,
            "Expected little-endian TIFF.",
        )
        page = image.pages[0]
        require(
            page.tags[272].value == "NIKON Z 8",
            "Only the pinned Z8 resource is supported.",
        )
        exif = page.tags[34665].value
        maker = exif["MakerNote"]
        xmp = page.tags[700].value if 700 in page.tags else None
    require(
        isinstance(maker, bytes) and maker[:7] == b"Nikon\0\x02",
        "Unknown MakerNote header.",
    )
    body = maker[10:]
    require(body[:4] == b"II*\0" and len(body) >= 8, "Unknown MakerNote byte order.")
    with tifffile.TiffFile(io.BytesIO(body)) as image:
        tags = read_ifd(image, body, struct.unpack_from("<I", body, 4)[0])
    require(xmp is None or isinstance(xmp, bytes), "Expected byte-valued XMP tag.")
    return tags, xmp


def block(
    tags: dict[int, object], key: int, versions: tuple[bytes, ...], size: int
) -> bytes:
    value = tags.get(key)
    require(isinstance(value, bytes) and len(value) >= size, f"Truncated tag {key:#x}.")
    require(value[:4] in versions, f"Unsupported version for tag {key:#x}.")
    return value


def color_balance(tags: dict[int, object], pe: pefile.PE) -> tuple[list[int], int]:
    value = bytearray(block(tags, 0x97, (b"0803",), 162))
    serial, count = tags.get(0x1D), tags.get(0xA7)
    require(
        isinstance(serial, str) and 1 <= len(serial) <= 8,
        "Unsupported serial-key shape.",
    )
    require(
        isinstance(count, int) and 0 <= count <= 0xFFFFFFFF,
        "Invalid shutter-count key.",
    )
    key = 0
    for character in serial.rstrip("\0").encode("ascii"):
        key = (
            key * 10 + (character - 48 if 48 <= character <= 57 else character % 10)
        ) & 0xFFFFFFFF
    ci = pe.get_data(0x5EAA60, 256)[key & 255]
    cj = pe.get_data(0x5EAB60, 256)[
        (count ^ (count >> 8) ^ (count >> 16) ^ (count >> 24)) & 255
    ]
    for index in range(4, len(value)):
        cj = (cj + ci * ((0x60 + index - 4) & 255)) & 255
        value[index] ^= cj
    return [value[index] for index in (0x8C, 0x3C, 0x3E, 0x3D, 0x40, 0x44, 0x43)], u16(
        value, 0x8A
    )


def picture_control(tags: dict[int, object]) -> tuple[int, int, int, dict[int, int]]:
    value = block(tags, 0x23, (b"0310", b"0311"), 108)
    identifier = u16(value, 48)
    indices = {
        0x8001: (0, 0, 0),
        0x8002: (1, 1, 1),
        0x2CF: (2, 2, 2),
        1: (3, 4, 4),
        0x3C2: (3, 5, 5),
        0xC3: (3, 6, 6),
        0x486: (3, 7, 7),
        0x4C7: (3, 8, 8),
        0x28E: (3, 9, 9),
        0x64D: (4, 10, 10),
        0x493: (3, 11, 11),
        0x654: (4, 12, 12),
        0x655: (11, 13, 13),
        0x20: (12, 14, 14),
        0x1001: (1, 35, 1),
        0x100D: (10, 36, 1),
        0x100E: (1, 37, 1),
    }
    if 0x801 <= identifier <= 0x814:
        family, profile, group = (
            5 + (identifier - 0x801) // 4,
            15 + identifier - 0x801,
            15,
        )
    else:
        require(identifier in indices, "Unknown Picture Control identifier.")
        family, profile, group = indices[identifier]
    offsets = (53, 54, 55, 56, 57, 58, 61, 62, 63, 64, 65, 66, 67, 68)
    fields = {index + 16: value[offset] for index, offset in enumerate(offsets)}
    fields.update(dict.fromkeys(range(30, 34), 0))
    if profile == 2 and value[61] == value[63] == value[67] == 255:
        for field, offset in zip(
            (20, 21, 22, 23, 24, 25, 28, 29),
            (90, 91, 94, 95, 96, 97, 98, 99),
            strict=True,
        ):
            fields[field] = value[offset]
    elif profile == 14 and value[63] == value[67] == 255:
        extra = block(tags, 0x5C, (b"0100", b"0101"), 88)
        fields.update({30: extra[12], 31: extra[13], 32: extra[22], 33: extra[23]})
    return family, profile, group, fields


def contrast(
    resource: bytes, a: list[int], b: dict[int, int], family: int, profile: int
) -> int:
    if (b[30] if family == 12 else b[24]) == 1:
        return 0
    if family == 2:
        index = (resource[0x8AC] * ((b[24] + 128) & 255) + a[4]) & 255
        return signed(u16(resource, 0x1E0 + 2 * index), 16)
    if b[24] == 0:
        result = signed(trunc_div(64 * a[1] - 768 + (-1 if a[1] < 12 else 1), 3), 16)
        lower, upper = (
            signed(u16(resource, offset + 2 * profile), 16) for offset in (0x3E0, 0x42C)
        )
        return min(max(result, lower), upper)
    field, divisor, rounding = (b[30], 25, 12) if family == 12 else (b[24], 3, 1)
    if field == 255:
        return 0
    numerator = 64 * (field - 128) + (-rounding if field < 128 else rounding)
    return signed(trunc_div(numerator, divisor), 16)


def component(
    resource: bytes, a: list[int], b: dict[int, int], group: int, clarity: bool
) -> int:
    base = 0x4C4 + 51 * group + (34 if clarity else 0)
    field = b[22 if clarity else 20]
    multiplier, cap = (0x8B1, 40) if clarity else (0x8AF, 48)
    if group == 2:
        result = (
            resource[multiplier] * ((field + 128) & 255) + byte(resource, base + a[6])
        ) & 255
    elif b[16] == 0:
        result = min(
            byte(resource, (0x81C if clarity else 0x7F4) + a[0]),
            byte(resource, base + a[2]),
        )
    else:
        result = (field - (108 if clarity else 116)) & 255
    if group == 15:
        numerator = 256 * (b[18] - 128)
        if clarity:
            numerator += -50 if b[18] < 128 else 50
        alpha = 256 if b[18] == 255 else trunc_div(numerator, 100) & 65535
        result = (
            trunc_div(
                resource[0x8B4 if clarity else 0x8B2] * (256 - alpha) + result * alpha,
                256,
            )
            & 255
        )
    return min(cap, result)


def predict(
    tags: dict[int, object], resource: bytes, pe: pefile.PE, count: int
) -> dict[str, object]:
    a, iso = color_balance(tags, pe)
    family, profile, group, b = picture_control(tags)
    wb = tags.get(0xC)
    require(
        isinstance(wb, tuple)
        and len(wb) >= 4
        and all(isinstance(v, int) and v > 0 for v in wb[:4]),
        "Invalid white-balance rationals.",
    )
    scaled_wb = (512 * (wb[0] / wb[1]), 512 * (wb[2] / wb[3]))
    require(all(v < 2**31 for v in scaled_wb), "WB conversion exceeds int32 range.")
    maximum_wb = max(int(value) & 65535 for value in scaled_wb)
    bucket = next((j for j in range(5) if maximum_wb < u16(resource, 0xA82 + j * 2)), 5)
    contrast_value = contrast(resource, a, b, family, profile)
    row = int(contrast_value * 3 / 64 + 12.5) & 0xFFFFFFFF
    brightness = 6 if b[26] in (1, 255) else (b[26] - 122) & 255
    selector = byte(resource, 0xA9F + 13 * a[0] + brightness)
    contrast_term = byte(resource, 0xBA3 + 8 * row + selector)
    clarity = component(resource, a, b, group, True)
    if family == 2:
        saturation = max(
            0, min(24, signed(resource[0x8AE] * ((b[28] + 128) & 255) + a[5]))
        )
    elif family == 12:
        saturation = max(
            0,
            min(
                24,
                signed(
                    trunc_div(6 * (b[32] - 128) + (-12 if b[32] < 128 else 12), 25) + 12
                ),
            ),
        )
    elif b[28] == 0:
        saturation = min(
            max(a[3], byte(resource, 0x478 + profile)), byte(resource, 0x49E + profile)
        )
    else:
        saturation = 12 if b[28] == 255 else (b[28] - 116) & 255
    iso_term = resource[0x830 + min(108, ((iso - 72) >> 2) & 255)]
    limit = resource[0x830 + min(108, ((u16(resource, 0x8A0) - 72) >> 2) & 255)]
    mode_table = (
        bytes(16)
        + pe.get_data(0x5EE3C0, 16)
        + pe.get_data(0x5EE360, 16)
        + pe.get_data(0x5EE3B0, 16)
    )
    stores = stack_literals(pe, 0x34EFD7, 0x35, 0)
    mode_table += bytes(stores[index] for index in range(64, 76))
    offset = (1 if count == 8 else 3) * 19
    knots = [signed(value) for value in mode_table[offset : offset + 19]]
    q, r = divmod(iso_term, 6)
    require(0 <= iso_term <= 108, "Invalid resource ISO coordinate.")
    weighted = (6 - r) * knots[q] + (r * knots[q + 1] if r else 0)
    adjustment = (
        knots[q]
        if r == 0
        else trunc_div(weighted if weighted + 3 < 0 else weighted + 6, 6)
    )
    coordinate = (
        iso_term
        + resource[0xA8C + bucket]
        + contrast_term
        + byte(resource, 0xC6B + clarity)
        + byte(resource, 0xC94 + saturation)
        + adjustment
    ) & 255
    setting = tags.get(0xB1)
    require(isinstance(setting, int), "Missing HighISONoiseReduction.")
    values = [interpolate(table, min(coordinate, limit)) for table in nr_knots(setting)]
    sharp = component(resource, a, b, group, False)
    flag_data = tags.get(0x62, b"")
    require(
        isinstance(flag_data, bytes) and (0x62 not in tags or len(flag_data) >= 4),
        "Truncated tag 0x62.",
    )
    require(flag_data[:4] != b"0100" or len(flag_data) >= 5, "Truncated tag 0x62.")
    flag = flag_data[:4] == b"0100" and flag_data[4:5] == b"\x02"
    table = (
        (0x64205 if flag else 0x64147)
        if limit < coordinate
        else (0x641CC if flag else 0x6410E)
    )
    scale = interpolate(resource[table : table + 19], min(coordinate, limit))
    index = signed(resource[0x6428A + sharp])
    if sharp >= 12:
        index = signed(trunc_div(index * scale, 128))
    amount, radius, detail, masking = sharp_values(index)
    values += [amount, radius / 10, detail, masking]
    return {
        "defaults": dict(zip(FIELDS, values, strict=True)),
        "intermediates": {
            "family": family,
            "profile": profile,
            "resource_group": group,
            "iso_code": iso,
            "iso_term": iso_term,
            "contrast_fixed_point": contrast_value,
            "clarity_index": clarity,
            "saturation_index": saturation,
            "mode_correction": adjustment,
            "noise_coordinate": coordinate,
            "noise_limit": limit,
            "sharpness_input": sharp,
            "sharpness_signed_index": index,
        },
        "validation_scope": "Static prediction; no vendor comparison for this input",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dll", "prm", "source"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--count", type=int, choices=(4, 8, 16, 32), default=8)
    args = parser.parse_args()
    try:
        dll, prm, source = (
            path.read_bytes() for path in (args.dll, args.prm, args.source)
        )
        require(digest(dll) == DLL_HASH, "Unsupported DLL SHA-256.")
        require(digest(prm) == PRM_HASH, "Unsupported prm.bin SHA-256.")
        resource = decode(
            prm,
            next(
                item
                for item in records(prm)
                if item["type"] == 0x1002 and item["id"] == 1071
            ),
        )
        require(
            digest(resource) == RESOURCE_HASH,
            "Unsupported decoded Z8 resource SHA-256.",
        )
        with pefile.PE(data=dll, fast_load=True) as pe:
            checks = verify_tables(pe)
            tags, xmp = read_source(source)
            report = {
                "schema_version": 1,
                "evidence_kind": "static_default_prediction",
                "input_count": args.count,
                "dll_sha256": DLL_HASH,
                "prm_sha256": PRM_HASH,
                "resource_sha256": RESOURCE_HASH,
                "source_sha256": digest(source),
                "table_checks": checks,
            }
            if args.count in (4, 16) or xmp is None:
                report.update(
                    update_applies=False,
                    reason="Count bypasses update or XMP tag is absent.",
                )
            else:
                require(bool(xmp), "Vendor update rejects an empty XMP tag.")
                root = ET.fromstring(xmp.rstrip(b"\0"))
                existing = {
                    key[len(CRD) :]: value
                    for element in root.iter()
                    for key, value in element.attrib.items()
                    if key.startswith(CRD)
                }
                existing.update(
                    {
                        element.tag[len(CRD) :]: element.text
                        for element in root.iter()
                        if element.tag.startswith(CRD)
                    }
                )
                report.update(
                    update_applies=True,
                    source_defaults=existing,
                    **predict(tags, resource, pe, args.count),
                )
    except (
        OSError,
        ValueError,
        LookupError,
        TypeError,
        struct.error,
        pefile.PEFormatError,
        ET.ParseError,
    ) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=True))
        return 1
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
