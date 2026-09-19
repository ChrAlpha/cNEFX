"""Replay native Nlg spatial-plane recursion from a pinned RGB pyramid capture."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np
from nlg_pyramid import spatial_planes
from verify_image_render import SRGB_HASH, validate_render_events
from verify_image_sdk import PINNED, picture_controls, provider_paths
from verify_np3_header import digest
from verify_renderer_cms import stage_events
from verify_renderer_lab_rgb import comparison
from verify_renderer_nlg import grid_events, grid_values, original_events, row_events
from verify_renderer_vignette import auxiliary_events

CAPTURE_HASH = "5f806d02d83a4e93bb4872f2debeebf7f47076c8bdd99160e31b534658496ad4"
EXTRA_EVENTS = {"nlg_supplier", "nlg_generator", "nlg_pyramid", "nlg_pyramid_plane"}


def level_parameters(raw: bytes, first: bool) -> tuple[int, int, int]:
    if len(raw) != 0xB6 or raw[0xAF] != 1 or raw[0xB3:] != bytes((not first, 1, 0)):
        raise ValueError("Unsupported spatial-plane level parameters.")
    return raw[0xB0], raw[0xB2], raw[0xB1]


def pyramid_events(events: list[dict]) -> dict:
    calls = [e for e in events if e.get("event") == "nlg_pyramid"]
    expected = [
        {
            "event": "nlg_pyramid",
            "index": i,
            "mode": int((i - 1) % 12 >= 6),
            "result": 1,
            "parameters_saved": True,
            "auxiliary": 150 if (i - 1) % 12 == 0 else 0,
        }
        for i in range(1, 25)
    ]
    if json.dumps(calls, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise ValueError("Unexpected pyramid call sequence or status.")
    planes = {}
    for event in events:
        if event.get("event") != "nlg_pyramid_plane":
            continue
        identity = event["index"], event["phase"], event["slot"]
        if identity in planes or event["descriptor_saved"] is not True:
            raise ValueError("Duplicate or missing pyramid descriptor capture.")
        if event["bounded"] and (
            event["saved"] is not True or event["valid"] is not True
        ):
            raise ValueError("Failed bounded pyramid plane capture.")
        planes[identity] = event
    if len(planes) != 130 or sum(e["saved"] for e in planes.values()) != 120:
        raise ValueError("Incomplete pyramid plane coverage.")
    return planes


def plane_values(event: dict, descriptor: bytes, raw: bytes) -> np.ndarray:
    width, height, channels = event["width"], event["height"], event["channels"]
    if (
        len(descriptor) != 0x88
        or any(type(v) is not int for v in (width, height, channels))
        or channels not in (1, 3)
        or not 1 <= width <= 2048
        or not 1 <= height <= 2048
        or event["saved"] is not True
        or event["bounded"] is not True
        or event["valid"] is not True
        or len(raw) != width * height * channels * 2
        or struct.unpack_from("<2I", descriptor, 0x60) != (width, height)
        or struct.unpack_from("<Q", descriptor, 0x28)[0] != height * channels
        or struct.unpack_from("<I", descriptor, 0x68)[0] < width
        or struct.unpack_from("<I", descriptor, 0x78)[0] >= height
    ):
        raise ValueError("Unexpected pyramid plane geometry or saved size.")
    array = (
        np.frombuffer(raw, dtype="<u2")
        .reshape(height, channels, width)
        .transpose(0, 2, 1)
    )
    return array[:, :, 0] if channels == 1 else array


def verify(args: argparse.Namespace, report: dict) -> None:
    root = Path(__file__).resolve().parents[1]

    def pin(path: Path, expected: str | None = None) -> bytes:
        raw = path.read_bytes()
        value, key = hashlib.sha256(raw).hexdigest(), str(path.resolve())
        if expected is not None and value != expected:
            raise ValueError("Recorded input hash drift.")
        if key in report["input_hashes"] and value != report["input_hashes"][key]:
            raise ValueError("Input changed during verification.")
        report["input_hashes"][key] = value
        return raw

    capture = json.loads(pin(args.capture, CAPTURE_HASH))
    if (
        capture.get("passed") is not True
        or capture.get("sources_unchanged") is not True
        or len(capture["cases"]) != 1
    ):
        raise ValueError("Expected a passing single-case capture.")
    for key in ("input_hashes", "output_hashes"):
        for path, expected in capture[key].items():
            pin(Path(path), expected)
    for name, expected in PINNED.items():
        pin(args.vendor / name, expected)
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename and Path(filename).resolve().is_relative_to(root):
            pin(Path(filename))
    profiles = [Path(p) for p, h in capture["input_hashes"].items() if h == SRGB_HASH]
    if len(profiles) != 1:
        raise ValueError("Expected one pinned Nikon sRGB profile.")
    case = capture["cases"][0]
    events = case["events"]
    if case["returncode"] != 0:
        raise ValueError("Vendor capture failed.")
    stage_events(events)
    auxiliary_events(events)
    row_events(events)
    grid_events(events)
    planes = pyramid_events(events)
    core = original_events(
        [e for e in events if e.get("event") not in EXTRA_EVENTS], grids=True
    )
    report["capture_render"] = validate_render_events(
        core, picture_controls(pin(Path(case["np3"]))), profiles[0]
    )
    report["capture_providers"] = provider_paths(
        core, capture["input_hashes"], args.vendor.resolve(), args.wine_prefix
    )
    if report["capture_providers"] != case["loaded_providers"]:
        raise ValueError("Unexpected captured provider list.")

    def snapshot(name: str) -> bytes:
        path = args.capture.parent / name
        return pin(path, capture["output_hashes"][str(path.resolve())])

    render = snapshot("case0.rgb16.bin")
    if hashlib.sha256(render).hexdigest() != case["output_sha256"]:
        raise ValueError("Captured final render hash differs.")

    def plane(index: int, phase: str, slot: int):
        prefix = f"case0.nlg-pyramid-{index}-{phase}{slot}"
        return plane_values(
            planes[index, phase, slot],
            snapshot(prefix + "-descriptor.bin"),
            snapshot(prefix + "-pixels.bin"),
        )

    grids = [
        grid_values(
            snapshot(f"case0.nlg-grid-{i}-descriptor.bin"),
            snapshot(f"case0.nlg-grid-{i}-plane.bin"),
        )
        for i in (1, 2)
    ]
    for start in (7, 19):
        previous = None
        for index in range(start, start + 6):
            first = index == start
            amounts = level_parameters(
                snapshot(f"case0.nlg-pyramid-{index}-parameters.bin"), first
            )
            rgb = plane(index, "output", 13)
            if not first:
                for i, slot in enumerate((9, 10)):
                    report["inputs"].append(
                        {
                            "index": index,
                            "slot": slot,
                            **comparison(previous[i], plane(index, "input", slot)),
                        }
                    )
            previous = spatial_planes(rgb, amounts, previous)
            for i, slot in enumerate((14, 15)):
                report["levels"].append(
                    {
                        "index": index,
                        "slot": slot,
                        **comparison(previous[i], plane(index, "output", slot)),
                    }
                )
        for i in range(2):
            report["shared_grids"].append(
                {"start": start, "grid": i + 1, **comparison(previous[1 - i], grids[i])}
            )
    report["sources_unchanged"] = all(
        digest(Path(p)) == h for p, h in report["input_hashes"].items()
    )
    for key in ("inputs", "levels", "shared_grids"):
        report[key + "_words"] = sum(e["compared_words"] for e in report[key])
    report["passed"] = (
        report["sources_unchanged"]
        and report["levels_words"] == 3786044
        and report["shared_grids_words"] == 2840064
        and all(
            e["different_words"] == 0
            for key in ("inputs", "levels", "shared_grids")
            for e in report[key]
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("capture", "vendor", "wine-prefix", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if (
        args.output.exists()
        or args.output.is_symlink()
        or args.output.resolve().is_relative_to(root)
    ):
        parser.error("Output must be a new external directory.")
    args.output.mkdir(parents=True)
    report = dict(
        evidence_kind="native_Nlg_spatial_planes_from_saved_RGB_pyramids",
        input_hashes={},
        inputs=[],
        levels=[],
        shared_grids=[],
        passed=False,
        limits=[
            "RGB pyramid and numeric parameters remain captured, not source-derived.",
            "Two executions of one Z8 source/control; no additional photograph.",
            "Observed spatial-plane branch only; no full renderer or Lightroom test.",
        ],
    )
    try:
        verify(args, report)
    except KeyboardInterrupt:
        report.update(cancelled=True, passed=False, error="Pyramid replay cancelled.")
    except Exception as exc:
        report.update(
            passed=False, error=dict(type=type(exc).__name__, message=str(exc))
        )
    with (args.output / "report.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            dict(passed=report["passed"], report=str(args.output / "report.json"))
        )
    )
    return 130 if report.get("cancelled") else 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
