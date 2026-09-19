"""Replay saved CMS0 evidence with freshly generated native source parameters."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
from pathlib import Path

import numpy as np
from renderer_cms import convert, source_profile, source_table
from renderer_lut import normalized_grid, transform
from renderer_profile import parameters
from verify_image_render import SRGB_HASH, validate_render_events
from verify_image_sdk import PINNED, picture_controls, provider_paths
from verify_image_sdk import validate_events as validate_sdk_events
from verify_np3_header import digest
from verify_renderer_lab_rgb import capture_events, comparison, parameter_comparison

TRACE_EVENTS = {"stage_taps", "stage_pixels", "inner_pixels"}
PAIRS = ((477, 691, None), (2537, 2543, 2539), (2538, 2544, 2540))
RECTANGLES = ((1040, 0, 1248, 208), (0, 0, 208, 172), (208, 0, 258, 172))


def stage_events(events: list) -> dict[int, dict]:
    """Validate every recorded trace, including stages excluded from replay."""
    if type(events) is not list or any(type(e) is not dict for e in events):
        raise ValueError("Expected structured capture events.")
    if json.dumps(
        [e for e in events if e.get("event") == "stage_taps"], sort_keys=True
    ) != json.dumps([{"event": "stage_taps", "installed": True}], sort_keys=True):
        raise ValueError("Missing or duplicate stage-tap installation.")
    capture_events(events)
    stages, seen = {}, set()
    for event in events:
        if event.get("event") not in ("stage_pixels", "inner_pixels"):
            continue
        index = event.get("index")
        if type(index) is not int or index < 1 or index in seen:
            raise ValueError("Duplicate or invalid trace identity.")
        seen.add(index)
        if event["event"] == "inner_pixels":
            continue
        rect = event.get("rect")
        if (
            type(rect) is not list
            or len(rect) != 4
            or any(type(v) is not int for v in rect)
            or min(rect) < 0
            or not 1 < rect[2] - rect[0] <= 4096
            or not 1 < rect[3] - rect[1] <= 4096
            or any(
                type(event.get(k)) is not int for k in ("object", "format", "stride")
            )
            or event["object"] < 1
            or event["format"] not in (0x10060201, 0x10020001, 0x10060203, 0x10060205)
            or event.get("stage")
            not in {
                "nr1",
                "highlight",
                "vignette",
                "lut",
                "nlg",
                "cms",
                "nr2",
                "edge",
                "distortion",
                "cms4",
            }
            or event.get("saved") is not True
            or event.get("error") is not False
        ):
            raise ValueError("Failed or unsupported stage capture.")
        row = (rect[2] - rect[0]) * ((event["format"] >> 16) & 255)
        if (
            not row <= event["stride"] <= 64 * 1024 * 1024
            or row * (rect[3] - rect[1]) > 16 * 1024 * 1024
        ):
            raise ValueError("Unbounded stage capture.")
        stages[index] = event
    if not 1 <= len(stages) <= 512:
        raise ValueError("Missing or excessive stage captures.")
    return stages


def matrix_contract(state: bytes, *, inverse: bool) -> None:
    if len(state) != 0x890 or state[0x230] != int(inverse) or state[0x888] != 0:
        raise ValueError("Unsupported captured matrix state.")
    sampled, auxiliary = (0x3F0, 0x558) if inverse else (0x260, 0x6C0)
    if any(
        struct.unpack_from("<I", state, base + channel * 0x78)[0] != kind
        for base, kind in ((sampled, 5), (auxiliary, 6))
        for channel in range(3)
    ):
        raise ValueError("Unsupported captured matrix curve types.")


def cms_contract(state: bytes, lut: bytes, matrix: bytes, inner_object: int) -> None:
    """Check meaningful mode6 fields, ignoring padding and relocated pointers."""
    if (
        len(state) != 0x748
        or state[0x10:0x13] != bytes(3)
        or struct.unpack_from("<Q", state, 0x600)[0] != 0
        or struct.unpack_from("<I", state, 0x718)[0] != 6
        or struct.unpack_from("<Q", state, 0x720)[0] != inner_object
        or struct.unpack_from("<Q", state, 0x730)[0] == 0
    ):
        raise ValueError("Unsupported CMS0 selector, flags or inner identity.")
    matrix_contract(matrix, inverse=True)
    if len(lut) != 0x8E0:
        raise ValueError("Unexpected LUT state length.")
    image_base = struct.unpack_from("<Q", matrix)[0] - 0xB2A868
    if (
        image_base <= 0
        or image_base % 65536
        or struct.unpack_from("<Q", lut)[0] - 0xB2A478 != image_base
        or struct.unpack_from("<I", lut, 0x22C)[0] != 16
        or struct.unpack_from("<I", lut, 0x578)[0] != 0x60204
        or lut[0x3D0:0x3D3] != b"AAA"
        or struct.unpack_from("<3I", lut, 0x3D4) != (0x41800080,) * 3
        or struct.unpack_from("<3Q", lut, 0x3E0) != (12675, 195, 3)
        or any(
            struct.unpack_from("<I", lut, offset)[0] != 6
            for offset in (0x230, 0x2B0, 0x330, 0x3F8, 0x478, 0x4F8)
        )
    ):
        raise ValueError("Unsupported CMS0 LUT type, grid, curves or format.")


def source_case(source: dict, capture: dict) -> dict:
    flags = (
        "passed",
        "sources_unchanged",
        "native_decryption",
        "picture_import",
        "originals_only",
        "kaiji_tables",
        "prepared_grid",
        "renderer_profile",
    )
    if source.get("evidence_kind") != "native_source_controls_vs_live_sdk_top" or any(
        source.get(name) is not True for name in flags
    ):
        raise ValueError("Unsuccessful or unsupported SDK source report.")
    cases = source.get("cases")
    if type(cases) is not list or not 1 <= len(cases) <= 32:
        raise ValueError("Missing or excessive SDK source cases.")
    for index, case in enumerate(cases):
        if (
            type(case) is not dict
            or type(case.get("index")) is not int
            or case["index"] != index
            or type(case.get("returncode")) is not int
            or case["returncode"] != 0
            or case.get("passed") is not True
            or case.get("variant") != "original"
        ):
            raise ValueError("Failed or invalid SDK source case.")
    selected = cases[0]
    for record in (selected, capture):
        for key in ("source_sha256", "np3_sha256"):
            value = record.get(key)
            if type(value) is not str or not re.fullmatch("[0-9a-f]{64}", value):
                raise ValueError("Invalid source photograph or Picture Control hash.")
    if any(selected.get(k) != capture.get(k) for k in ("source_sha256", "np3_sha256")):
        raise ValueError("Source photograph or Picture Control differs between runs.")
    return selected


def verify(args: argparse.Namespace, report: dict) -> None:
    root = Path(__file__).resolve().parents[1]
    hashes = report["input_hashes"]

    def pinned(path: Path, expected: str | None = None) -> bytes:
        if path.is_symlink() or any(
            p == ".env" or p.startswith(".env.") for p in path.parts
        ):
            raise ValueError("Linked or prohibited evidence path.")
        raw = path.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        key = str(path.resolve())
        if (expected is not None and actual != expected) or (
            key in hashes and hashes[key] != actual
        ):
            raise ValueError("Recorded evidence hash drift.")
        hashes[key] = actual
        return raw

    def recorded_inputs(record: dict) -> None:
        values = record.get("input_hashes")
        if type(values) is not dict or not values:
            raise ValueError("Missing recorded input hashes.")
        for path, expected in values.items():
            if (
                type(path) is not str
                or not Path(path).is_absolute()
                or type(expected) is not str
                or not re.fullmatch("[0-9a-f]{64}", expected)
            ):
                raise ValueError("Invalid recorded input identity.")
            pinned(Path(path), expected)

    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename and Path(filename).resolve().is_relative_to(root):
            pinned(Path(filename))
    capture, source = (
        json.loads(pinned(p)) for p in (args.capture, args.source_report)
    )
    for record in (capture, source):
        recorded_inputs(record)
    for name, expected in PINNED.items():
        pinned(args.vendor / name, expected)
    if (
        capture.get("evidence_kind") != "live_sdk_bounded_rgb16_render"
        or type(capture.get("cases")) is not list
        or len(capture["cases"]) != 1
        or capture.get("bits") != 16
        or capture.get("scale") != 0.03125
    ):
        raise ValueError("Expected one bounded saved render capture.")
    case = capture["cases"][0]
    if (
        type(case.get("returncode")) is not int
        or case["returncode"] != 0
        or type(case.get("index")) is not int
        or case.get("index") != 0
    ):
        raise ValueError("Unsuccessful captured render process.")
    if capture.get("passed") is not True and capture.get("error") != {
        "type": "ValueError",
        "message": "Incomplete or reordered render lifecycle.",
    }:
        raise ValueError("Unrecognized original capture failure.")
    selected = source_case(source, case)
    nef = pinned(Path(case["source"]), case["source_sha256"])
    np3 = pinned(Path(case["np3"]), case["np3_sha256"])
    profile_path = Path(capture["output_profile"])
    destination = pinned(profile_path, SRGB_HASH)
    events = case["events"]
    stages = stage_events(events)
    core = [event for event in events if event.get("event") not in TRACE_EVENTS]
    report["capture_render"] = validate_render_events(
        core, picture_controls(np3), profile_path
    )
    report["capture_providers"] = provider_paths(
        core, capture["input_hashes"], args.vendor.resolve(), args.wine_prefix
    )
    report["original_capture_passed"] = capture["passed"]
    for sdk in source["cases"]:
        matches = [
            p for p, sha in source["input_hashes"].items() if sha == sdk["np3_sha256"]
        ]
        if len(matches) != 1:
            raise ValueError("Missing or ambiguous SDK Picture Control input.")
        validate_sdk_events(
            sdk["events"],
            picture_controls(pinned(Path(matches[0]))),
            kaiji=True,
            prepare_grid=True,
            renderer_profile=True,
        )
        provider_paths(
            sdk["events"],
            source["input_hashes"],
            args.vendor.resolve(),
            args.wine_prefix,
        )
    artifacts = sorted(args.capture.parent.glob("case0.*.bin"))
    if not 1 <= len(artifacts) <= 2048:
        raise ValueError("Missing or excessive capture artifacts.")
    for artifact in artifacts:
        pinned(artifact)

    def stage_bytes(index: int, suffix: str) -> bytes:
        event = stages[index]
        return pinned(
            args.capture.parent / f"case0.stage-{index}-{event['stage']}-{suffix}.bin"
        )

    def inner_bytes(index: int, suffix: str) -> bytes:
        return pinned(args.capture.parent / f"case0.inner-{index}-{suffix}.bin")

    for index, event in stages.items():
        rect = event["rect"]
        size = (
            (rect[2] - rect[0]) * (rect[3] - rect[1]) * ((event["format"] >> 16) & 255)
        )
        if len(stage_bytes(index, "output")) != size:
            raise ValueError("Unexpected saved stage pixel byte count.")
    inner = {e["index"]: e for e in events if e.get("event") == "inner_pixels"}
    for index, event in inner.items():
        inverse = event["kernel_rva"] == 0x53A2F0
        state = inner_bytes(index, "state")
        matrix_contract(state, inverse=inverse)
        matrix, curves = parameters(destination, inverse=inverse)
        captured_matrix = np.frombuffer(
            state, dtype="<f4", count=9, offset=0x3C8 if inverse else 0x238
        ).reshape(3, 3)
        captured_curves = np.array(
            [
                np.frombuffer(
                    inner_bytes(
                        index, f"table-{int(inverse)}-{channel}-{int(not inverse)}"
                    ),
                    dtype="<f4",
                )
                for channel in range(3)
            ]
        )
        check = {
            "index": index,
            "matrix": parameter_comparison(matrix, captured_matrix),
            "curves": parameter_comparison(curves, captured_curves),
        }
        report["parameters"].append(check)
        if any(check[k]["different_words"] for k in ("matrix", "curves")):
            raise ValueError("Native ICC parameters differ from captured parameters.")
        for suffix in ("input", "output"):
            if len(inner_bytes(index, suffix)) != event["width"] * event["height"] * 6:
                raise ValueError("Unexpected saved inner pixel byte count.")
    text = selected["profile"]
    table = source_table(
        nef,
        np3,
        **{
            name: pinned(getattr(args, name))
            for name in ("coefficients", "modes", "blend_calibration", "decoding_table")
        },
    )
    profile = source_profile(
        table,
        pinned(args.template),
        description=text["description"],
        copyright=text["copyright"],
    )
    for name, native, expected in (
        ("table", table, selected["table_sha256"]),
        ("profile", profile, text["sha256"]),
    ):
        target = pinned(args.source_report.parent / f"case0.{name}.bin", expected)
        record = {
            "compared_bytes": len(target),
            "native_sha256": hashlib.sha256(native).hexdigest(),
            "target_sha256": expected,
            "equivalent": native == target,
        }
        report["generation"][name] = record
        if native != target:
            raise ValueError("Fresh native source parameters differ from SDK targets.")
        path = args.output / f"native.{name}.bin"
        with path.open("xb") as stream:
            stream.write(native)
        report["artifacts"][str(path.resolve())] = digest(path)
    report["profile_text"] = {k: text[k] for k in ("description", "copyright")}
    grid = normalized_grid(profile)
    for (nlg, leaf, cms), rect in zip(PAIRS, RECTANGLES, strict=True):
        event, call = stages[nlg], inner[leaf]
        width, height = rect[2] - rect[0], rect[3] - rect[1]
        if (
            event["stage"] != "nlg"
            or event["format"] != 0x10060201
            or tuple(event["rect"]) != rect
            or (call["width"], call["height"], call["kernel_rva"])
            != (width, height, 0x53A2F0)
        ):
            raise ValueError(
                "Incorrect same-run Nlg/inner block identity or dimensions."
            )
        pixels = np.frombuffer(stage_bytes(nlg, "output"), dtype="<u2").reshape(-1, 3)
        lab = np.frombuffer(inner_bytes(leaf, "input"), dtype="<u2").reshape(-1, 3)
        target = np.frombuffer(inner_bytes(leaf, "output"), dtype="<u2").reshape(-1, 3)
        record = {
            "nlg": nlg,
            "inner": leaf,
            "cms": cms,
            "width": width,
            "height": height,
            "rect": list(rect),
            "lab": comparison(transform(pixels, grid, width), lab),
            "rgb": comparison(convert(profile, destination, pixels, width), target),
        }
        report["replays"].append(record)
        reference = cms if cms is not None else 672
        cms_contract(
            stage_bytes(reference, "state"),
            stage_bytes(reference, "inner1"),
            stage_bytes(reference, "inner0"),
            call["object"],
        )
        if cms is not None:
            outer = stages[cms]
            if (
                outer["stage"] != "cms"
                or outer["format"] != 0x10060201
                or tuple(outer["rect"]) != rect
            ):
                raise ValueError("Incorrect CMS0 output rectangle or format.")
            record["outer_rgb"] = comparison(
                target,
                np.frombuffer(stage_bytes(cms, "output"), dtype="<u2").reshape(-1, 3),
            )
        if any(
            record[name]["different_words"]
            for name in ("lab", "rgb", "outer_rgb")
            if name in record
        ):
            raise ValueError("Fresh native CMS0 pixels differ from the saved capture.")
    report["excluded_inner_calls"] = [
        event for index, event in inner.items() if index not in {p[1] for p in PAIRS}
    ]
    report["excluded_nlg_calls"] = [
        event
        for index, event in stages.items()
        if event["stage"] == "nlg" and index not in {p[0] for p in PAIRS}
    ]
    report["sources_unchanged"] = all(
        digest(Path(path)) == sha for path, sha in hashes.items()
    )
    report["artifacts_unchanged"] = all(
        digest(Path(path)) == sha for path, sha in report["artifacts"].items()
    )
    report["passed"] = report["sources_unchanged"] and report["artifacts_unchanged"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "capture",
        "source-report",
        "coefficients",
        "modes",
        "blend-calibration",
        "decoding-table",
        "template",
        "output",
        "vendor",
        "wine-prefix",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if (
        args.output.exists()
        or args.output.is_symlink()
        or args.output.resolve().is_relative_to(root)
    ):
        parser.error("Output must be a new directory outside the repository.")
    args.output.mkdir(parents=True)
    report = {
        "evidence_kind": "saved_cms0_capture_vs_fresh_native_source_parameters",
        "input_hashes": {},
        "artifacts": {},
        "generation": {},
        "parameters": [],
        "replays": [],
        "passed": False,
        "numpy_version": np.__version__,
        "limits": [
            "One ordinary Z8 source and one Flexible Color control; saved evidence.",
            "Nlg input pixels remain captured vendor output; no native source render.",
            "Source tables, grid65, ICC and matrix curves are freshly generated.",
            "Profile text and template remain external serialization inputs.",
            "Intermediate Nlg477/inner691 lacks a captured matching outer CMS tile.",
            "Excluded inner calls validate status and parameters only.",
            "No fresh Wine execution, full-image, NEFX or Lightroom acceptance claim.",
        ],
    }
    try:
        verify(args, report)
    except KeyboardInterrupt:
        report.update(
            passed=False, cancelled=True, error="CMS0 verification cancelled."
        )
    except Exception as exc:
        report.update(
            passed=False, error={"type": type(exc).__name__, "message": str(exc)}
        )
    finally:
        with (args.output / "report.json").open("x") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
    print(
        json.dumps(
            {"passed": report["passed"], "report": str(args.output / "report.json")}
        )
    )
    return 130 if report.get("cancelled") else (0 if report["passed"] else 1)


if __name__ == "__main__":
    raise SystemExit(main())
