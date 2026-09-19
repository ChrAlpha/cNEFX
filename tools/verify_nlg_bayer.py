"""Replay native initial Bayer reduction from a pinned real-source capture."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
from nlg_bayer import reduce_bayer
from verify_image_render import SRGB_HASH, validate_render_events
from verify_image_sdk import PINNED, picture_controls, provider_paths
from verify_nlg_pyramid import EXTRA_EVENTS, pyramid_events
from verify_np3_header import digest
from verify_renderer_cms import stage_events
from verify_renderer_lab_rgb import comparison
from verify_renderer_nlg import grid_events, original_events, row_events
from verify_renderer_vignette import auxiliary_events

CAPTURE_HASH = "3205d9fc07fb623eea02f90bf4c6981a2e8acb7bd0396050efd54ec20a8115bc"
RENDER_HASH = "794b1706677dc854398c1036bd39e202dea742bd6d4acbc4b93c0a8c08eb2aa8"
BAYER_EVENTS = {"nlg_bayer_before", "nlg_bayer_window", "nlg_bayer_after"}
ORIGINS = tuple((y, x) for y in (0, 17, 688, 1375) for x in (0, 904, 1808))


def bayer_events(events: list[dict]) -> list[dict]:
    """Validate the complete observed call contract, including window identity."""
    selected = [e for e in events if e.get("event") in BAYER_EVENTS]
    if len(selected) != 28:
        raise ValueError("Incomplete Bayer capture events.")
    expected, controls = [], []
    for call in (1, 2):
        before = selected[(call - 1) * 14]
        mxcsr = before.get("mxcsr")
        if type(mxcsr) is not int or mxcsr & ~0x3F != 0x1F80:
            raise ValueError("Unexpected captured MXCSR control state.")
        expected.append(
            dict(
                event="nlg_bayer_before",
                call=call,
                valid=True,
                saved=True,
                pattern=0,
                precision=0,
                monochrome=0,
                low=10,
                high=30,
                shift=1,
                clamp=[0, 0],
                source_shape=[5504, 8256],
                output_shape=[1376, 2064],
                cursors=[0, 0, 0],
                mxcsr=mxcsr,
            )
        )
        controls.append(before)
        for window, origin in enumerate(ORIGINS):
            expected.append(
                dict(
                    event="nlg_bayer_window",
                    call=call,
                    window=window,
                    origin=list(origin),
                    source_shape=[4, 1024],
                    output_shape=[1, 256],
                    valid=True,
                    input_unchanged=True,
                    saved=True,
                )
            )
        expected.append(
            dict(
                event="nlg_bayer_after",
                call=call,
                valid=True,
                saved=True,
                descriptors_unchanged=True,
                mxcsr=[mxcsr, mxcsr],
                capture_bytes=call * 120656,
            )
        )
    if json.dumps(selected, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise ValueError("Unexpected Bayer capture identity, geometry or status.")
    return controls


def descriptor(
    raw: bytes, width: int, height: int, channels: int, halo_x: int = 0, halo_y: int = 0
) -> None:
    if (
        len(raw) != 0x88
        or struct.unpack_from("<Q", raw, 0x28)[0] != (height + 2 * halo_y) * channels
        or struct.unpack_from("<2I", raw, 0x60) != (width, height)
        or not width + 2 * halo_x <= struct.unpack_from("<I", raw, 0x68)[0] <= 32768
        or struct.unpack_from("<I", raw, 0x78)[0] != 0
    ):
        raise ValueError("Unexpected captured Bayer descriptor geometry.")


def replay_windows(
    controls: list[dict], snapshot: Callable[[str], bytes], report: dict
) -> None:
    """Keep mismatches in the report while comparing every captured window."""
    for before in controls:
        call = before["call"]
        base = f"case0.nlg-bayer-{call}-window"
        expected = bytearray(32)
        struct.pack_into("<5I", expected, 0, before["pattern"], 8256, 5504, 2064, 1376)
        expected[20:23] = bytes(
            (before["precision"], before["monochrome"], before["shift"])
        )
        struct.pack_into("<2H", expected, 26, before["low"], before["high"])
        if snapshot(f"{base}-1-controls.bin") != expected:
            raise ValueError("Saved Bayer controls differ from events.")
        for name, dims in (
            ("source", (8256, 5504, 1)),
            ("rgb", (2064, 1376, 3, 2, 1)),
            ("aux", (2064, 1376, 1)),
        ):
            raw = snapshot(f"{base}-1-{name}-descriptor-before.bin")
            descriptor(raw, *dims)
            if raw != snapshot(f"{base}-1-{name}-descriptor-after.bin"):
                raise ValueError("Captured Bayer descriptor changed during execution.")
        for window, origin in enumerate(ORIGINS):
            samples = np.frombuffer(snapshot(f"{base}{window}-input.bin"), dtype="<u2")
            if samples.size != 4096:
                raise ValueError("Unexpected Bayer input window size.")
            rgb, aux = reduce_bayer(
                samples.reshape(4, 1024),
                before["pattern"],
                precision=before["precision"],
                monochrome=bool(before["monochrome"]),
                low=before["low"],
                high=before["high"],
                monotonic_shift=before["shift"],
            )
            item = dict(call=call, window=window, origin=list(origin))
            report["windows"].append(item)
            for name, native in (("rgb", rgb.transpose(0, 2, 1)), ("aux", aux)):
                vendor = np.frombuffer(
                    snapshot(f"{base}{window}-{name}.bin"), dtype=native.dtype
                )
                item[name] = comparison(native.reshape(-1), vendor)


def summarize(report: dict) -> None:
    windows = report["windows"]
    report["rgb_words"] = sum(e["rgb"]["compared_words"] for e in windows)
    report["aux_bytes"] = sum(e["aux"]["compared_words"] for e in windows)
    report["passed"] = (
        report["sources_unchanged"]
        and len(report["capture_providers"]) == 28
        and [(e["call"], e["window"]) for e in windows]
        == [(call, window) for call in (1, 2) for window in range(12)]
        and report["rgb_words"] == 18432
        and report["aux_bytes"] == 6144
        and all(
            e[name]["different_words"] == 0 for e in windows for name in ("rgb", "aux")
        )
    )


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
        raise ValueError("Expected a passing single-case source capture.")
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
    if case["returncode"] != 0:
        raise ValueError("Vendor capture failed.")
    events = case["events"]
    stage_events(events)
    auxiliary_events(events)
    row_events(events)
    grid_events(events)
    pyramid_events(events)
    controls = bayer_events(events)
    core = original_events(
        [e for e in events if e.get("event") not in EXTRA_EVENTS | BAYER_EVENTS],
        grids=True,
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

    if hashlib.sha256(snapshot("case0.rgb16.bin")).hexdigest() != RENDER_HASH:
        raise ValueError("Captured final render differs from the baseline.")
    replay_windows(controls, snapshot, report)
    report["sources_unchanged"] = all(
        digest(Path(p)) == h for p, h in report["input_hashes"].items()
    )
    summarize(report)


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
        evidence_kind="native_initial_Bayer_reduction_from_saved_source_windows",
        input_hashes={},
        windows=[],
        passed=False,
        limits=[
            "One lossless Z8 photograph/control, two executions and 24 windows.",
            "Only initial Bayer RGB/auxiliary reduction; controls remain captured.",
            "No source-derived full Bayer chain, pixel-shift NEFX or Lightroom test.",
            "Vendor paths and Wine prefix identify evidence; no vendor code runs.",
        ],
    )
    try:
        verify(args, report)
    except KeyboardInterrupt:
        report.update(cancelled=True, passed=False, error="Bayer replay cancelled.")
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
