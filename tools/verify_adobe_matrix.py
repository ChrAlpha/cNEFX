# /// script
# requires-python = ">=3.12"
# dependencies = ["capstone==5.0.6", "pefile==2024.8.26", "tifffile==2026.9.15"]
# ///
"""Compare predicted Adobe defaults with a pinned DLL on synthetic metadata.

The source manifest must contain synthetic eight-frame packed-14 fixtures.
All modified inputs and vendor outputs are written to a new external directory.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import struct
import subprocess
import time
import xml.etree.ElementTree as ET
from decimal import Decimal, InvalidOperation
from pathlib import Path

import adobe_defaults as defaults
import pefile
import tifffile
from inspect_prm import decode, records


def windows(path: Path) -> str:
    return "Z:" + str(path.resolve()).replace("/", "\\")


def checkpoint(directory: Path, report: dict[str, object]) -> None:
    temporary = directory / "report.json.tmp"
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(directory / "report.json")


def variants(extended: bool) -> list[dict[str, object]]:
    manual = {53: 1, 54: 2, 55: 255, 57: 140, 61: 128, 63: 128, 65: 128, 67: 128}

    def case(name: str, identifier: int, **options: object) -> dict[str, object]:
        return {"name": name, "profile_id": identifier, "pc": manual, **options}

    result = [
        case("standard-manual", 1),
        case("standard-default", 1, pc={**manual, 53: 0, 63: 0, 67: 0}),
        case("profile-03c2-manual", 0x3C2),
        case("creative-0801-half", 0x801, pc={**manual, 55: 178, 61: 132}),
        case(
            "flexible-0020-neutral",
            0x20,
            pc={**manual, 63: 255, 67: 255},
            flexible={12: 128, 22: 128},
        ),
        case(
            "flexible-0020-adjusted",
            0x20,
            pc={**manual, 63: 255, 67: 255},
            flexible={12: 153, 22: 153},
        ),
    ]
    if extended:
        result += [
            case(
                f"profile-{identifier:04x}-default",
                identifier,
                pc={**manual, 53: 0, 63: 0, 67: 0},
            )
            for identifier in (
                0x8001,
                0x8002,
                0xC3,
                0x486,
                0x4C7,
                0x28E,
                0x64D,
                0x493,
                0x654,
                0x655,
                0x1001,
                0x100D,
                0x100E,
            )
        ]
        result += [
            case(
                f"creative-{identifier:04x}-{intensity}",
                identifier,
                pc={**manual, 55: intensity, 61: 136},
                nr=nr,
            )
            for identifier, intensity, nr in (
                (0x801, 128, 2),
                (0x805, 228, 4),
                (0x809, 255, 6),
                (0x80D, 178, 0),
                (0x814, 178, 2),
            )
        ]
        result += [
            case(
                "standard-0311-sharpen-flag2",
                1,
                pc_version="0311",
                flag=2,
                nr=6,
                pc={**manual, 57: 156, 61: 140, 63: 132, 67: 124},
            ),
            case(
                "flexible-0101-negative",
                0x20,
                pc={**manual, 63: 255, 67: 255},
                flexible={12: 103, 22: 103},
                flexible_version="0101",
                nr=4,
            ),
        ]
    return result


def maker_layout(data: bytes) -> dict[int, tuple[int, int]]:
    with tifffile.TiffFile(io.BytesIO(data)) as image:
        offset = image.pages[0].tags[34665].valueoffset
        count = struct.unpack_from("<H", data, offset)[0]
        exif = [
            tifffile.TiffTag.fromfile(image, offset=offset + 2 + index * 12)
            for index in range(count)
        ]
        maker = next(tag for tag in exif if tag.code == 0x927C)
        defaults.require(maker.value[:7] == b"Nikon\0\x02", "Unknown MakerNote header.")
        base = maker.valueoffset + 10
        with tifffile.TiffFile(io.BytesIO(maker.value[10:])) as notes:
            return {
                code: (base + tag.valueoffset, tag.count)
                for code in (0x23, 0x5C, 0x62, 0xB1)
                if (tag := notes.pages[0].tags.get(code)) is not None
            }


def modify(
    data: bytes, variant: dict[str, object]
) -> tuple[bytes, list[dict[str, object]]]:
    layout = maker_layout(data)
    result = bytearray(data)
    changes = []

    def patch(tag: int, relative: int, content: bytes, label: str) -> None:
        offset, count = layout[tag]
        size = count * 2 if tag == 0xB1 else count
        defaults.require(
            0 <= relative <= size - len(content), "Metadata patch out of bounds."
        )
        absolute = offset + relative
        before = bytes(result[absolute : absolute + len(content)])
        result[absolute : absolute + len(content)] = content
        changes.append(
            {
                "tag": hex(tag),
                "offset": relative,
                "absolute_offset": absolute,
                "before_hex": before.hex(),
                "after_hex": content.hex(),
                "field": label,
            }
        )

    patch(0x23, 48, struct.pack("<H", variant["profile_id"]), "Picture Control ID")
    for offset, value in variant["pc"].items():
        patch(0x23, offset, bytes([value]), "Picture Control parameter")
    if "pc_version" in variant:
        patch(0x23, 0, variant["pc_version"].encode("ascii"), "Picture Control version")
    for offset, value in variant.get("flexible", {}).items():
        patch(0x5C, offset, bytes([value]), "Flexible parameter")
    if "flexible_version" in variant:
        patch(0x5C, 0, variant["flexible_version"].encode("ascii"), "Flexible version")
    if "nr" in variant:
        patch(0xB1, 0, struct.pack("<H", variant["nr"]), "High ISO NR")
    if "flag" in variant:
        patch(0x62, 4, bytes([variant["flag"]]), "Sharpening selector")
    return bytes(result), changes


def read_defaults(path: Path) -> tuple[dict[str, str], dict[str, int]]:
    with tifffile.TiffFile(path) as image:
        xmp = image.pages[0].tags[700].value
    root = ET.fromstring(xmp.rstrip(b"\0"))
    found: dict[str, list[str]] = {}
    forms = {"attributes": 0, "elements": 0}
    for element in root.iter():
        for name, value in element.attrib.items():
            if name.startswith(defaults.CRD):
                found.setdefault(name[len(defaults.CRD) :], []).append(value)
                forms["attributes"] += 1
        if element.tag.startswith(defaults.CRD):
            found.setdefault(element.tag[len(defaults.CRD) :], []).append(
                element.text or ""
            )
            forms["elements"] += 1
    values = {}
    for name in defaults.FIELDS:
        defaults.require(name in found, f"Missing output property {name}.")
        defaults.require(
            len(set(found[name])) == 1, f"Conflicting output property {name}."
        )
        values[name] = found[name][0]
    return values, forms


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "dll", "prm", "probe", "wine-prefix", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--extended", action="store_true")
    parser.add_argument("--wine", default="/usr/lib/wine/wine64")
    args = parser.parse_args()
    logging.getLogger("tifffile").setLevel(logging.ERROR)
    repository = Path(__file__).resolve().parents[1]
    if args.output.resolve().is_relative_to(repository):
        parser.error("Runtime artifacts must stay outside the repository.")
    dll, prm = args.dll.read_bytes(), args.prm.read_bytes()
    defaults.require(defaults.digest(dll) == defaults.DLL_HASH, "Unsupported DLL hash.")
    defaults.require(
        defaults.digest(prm) == defaults.PRM_HASH, "Unsupported resource hash."
    )
    defaults.require(
        defaults.digest((args.probe.parent / "prm.bin").read_bytes())
        == defaults.PRM_HASH,
        "Pinned prm.bin must already exist beside the probe.",
    )
    manifest = json.loads(args.manifest.read_text())
    defaults.require(
        manifest["evidence_kind"] == "synthetic_packed14_input_fixture",
        "Expected synthetic packed-14 fixture manifest.",
    )
    frames = sorted(
        (item for item in manifest["files"] if item["input_count"] == 8),
        key=lambda item: item["frame_index"],
    )
    defaults.require(len(frames) == 8, "Expected eight source fixtures.")
    originals = [Path(item["path"]).read_bytes() for item in frames]
    defaults.require(
        all(
            defaults.digest(data) == item["sha256"]
            for data, item in zip(originals, frames, strict=True)
        ),
        "Fixture hash mismatch.",
    )
    resource = decode(
        prm,
        next(
            item
            for item in records(prm)
            if item["type"] == 0x1002 and item["id"] == 1071
        ),
    )
    defaults.require(
        defaults.digest(resource) == defaults.RESOURCE_HASH, "Unexpected Z8 resource."
    )
    args.output.mkdir(parents=True, exist_ok=False)
    planned = variants(args.extended)
    report = {
        "schema_version": 1,
        "evidence_kind": "live_vendor_dll_synthetic_adobe_default_matrix",
        "dll_sha256": defaults.DLL_HASH,
        "prm_sha256": defaults.PRM_HASH,
        "probe_sha256": defaults.digest(args.probe.read_bytes()),
        "predictor_sha256": defaults.digest(Path(defaults.__file__).read_bytes()),
        "runner_sha256": defaults.digest(Path(__file__).read_bytes()),
        "manifest_sha256": defaults.digest(args.manifest.read_bytes()),
        "input_count": 8,
        "output_count": 1,
        "radial_correction": False,
        "source_kind": "Synthetic packed-14 planes and edited inherited Nikon metadata",
        "planned_case_count": len(planned),
        "completed": False,
        "passed": False,
        "cases": [],
    }
    with pefile.PE(data=dll, fast_load=True) as pe:
        report["static_table_checks"] = defaults.verify_tables(pe)
        checkpoint(args.output, report)
        for variant in planned:
            started = time.monotonic()
            case = {
                "name": variant["name"],
                "variant": variant,
                "ok": False,
                "inputs": [],
            }
            directory = args.output / variant["name"]
            directory.mkdir()
            output = directory / "merged.nefx"
            try:
                files = []
                for index, source in enumerate(originals):
                    modified, patches = modify(source, variant)
                    path = directory / f"frame-{index + 1:02d}.NEF"
                    with path.open("xb") as stream:
                        stream.write(modified)
                    files.append(path)
                    case["inputs"].append(
                        {
                            "path": str(path),
                            "sha256": defaults.digest(modified),
                            "source_sha256": frames[index]["sha256"],
                            "patches": patches,
                        }
                    )
                tags, _ = defaults.read_source(files[0].read_bytes())
                prediction = defaults.predict(tags, resource, pe, 8)
                case.update(
                    predicted=prediction["defaults"],
                    intermediates=prediction["intermediates"],
                )
                result = subprocess.run(
                    [
                        args.wine,
                        str(args.probe),
                        windows(args.dll),
                        "normal",
                        "1",
                        windows(output),
                        *map(windows, files),
                    ],
                    env=dict(
                        os.environ,
                        WINEPREFIX=str(args.wine_prefix.resolve()),
                        WINEDEBUG="-all",
                    ),
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=False,
                )
                case.update(
                    return_code=result.returncode,
                    stdout=result.stdout,
                    stderr=result.stderr,
                )
                defaults.require(result.returncode == 0, "Vendor process failed.")
                actual, forms = read_defaults(output)
                differences = {
                    name: {"actual": actual[name], "predicted": value}
                    for name, value in prediction["defaults"].items()
                    if Decimal(actual[name]) != Decimal(str(value))
                }
                case.update(
                    actual=actual,
                    xml_forms=forms,
                    differences=differences,
                    compared_properties=len(defaults.FIELDS),
                    ok=not differences,
                )
            except subprocess.TimeoutExpired as exc:
                case.update(
                    error="Vendor process exceeded 60 seconds.",
                    stdout=(exc.stdout or b"").decode("utf-8", errors="replace"),
                    stderr=(exc.stderr or b"").decode("utf-8", errors="replace"),
                )
            except (
                OSError,
                ValueError,
                LookupError,
                TypeError,
                InvalidOperation,
                ET.ParseError,
                struct.error,
            ) as exc:
                case["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                if output.exists():
                    case.update(
                        output_sha256=defaults.digest(output.read_bytes()),
                        output_size_bytes=output.stat().st_size,
                    )
                case["elapsed_seconds"] = round(time.monotonic() - started, 3)
                report["cases"].append(case)
                checkpoint(args.output, report)
                print(
                    json.dumps(
                        {
                            "name": case["name"],
                            "ok": case["ok"],
                            "error": case.get("error"),
                            "differences": case.get("differences"),
                        }
                    )
                )
    report["original_inputs_unchanged"] = all(
        defaults.digest(Path(item["path"]).read_bytes()) == item["sha256"]
        for item in frames
    )
    report["completed"] = len(report["cases"]) == report["planned_case_count"]
    report["passed"] = (
        report["completed"]
        and report["original_inputs_unchanged"]
        and all(item["ok"] for item in report["cases"])
    )
    report["properties_compared"] = sum(
        item.get("compared_properties", 0) for item in report["cases"]
    )
    checkpoint(args.output, report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
