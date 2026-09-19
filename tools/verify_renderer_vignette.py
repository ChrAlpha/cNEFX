"""Replay captured mode5 vignette using native radial table and pixel arithmetic."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np
from renderer_vignette import radial_table, transform
from verify_image_render import SRGB_HASH, validate_render_events
from verify_image_sdk import PINNED, picture_controls, provider_paths
from verify_np3_header import digest
from verify_renderer_cms import TRACE_EVENTS, stage_events
from verify_renderer_highlight import TAP_HASH
from verify_renderer_lab_rgb import comparison
from vignette_inputs import source_parameters

from nx_ctudio import _tiff

CAPTURE_HASH = "95836becb7528456146e7b2810f81ba3d1c6b9841c8e248140f59fb9d48a5b1f"
SOURCE_HASH = "28135bfd6d3f271be488e58a30d0a3ad74671d9cb93c0ac7fd67219c93f2fcbd"
PAIRS = ((2491, 2493, (258, 172)), (2492, 2494, (258, 172)), (86, 161, (2064, 1376)))


def auxiliary_events(events: list) -> None:
    expected = [
        {
            "event": "highlight_buffer",
            "index": index,
            "rect": rectangle,
            "channel": 1,
            "bayer": [1, 0],
            "error": False,
            "saved": [1, 1, 1],
        }
        for index, rectangle in (
            (1, [0, 0, 208, 172]),
            (2, [208, 0, 258, 172]),
            (3, [208, 0, 258, 172]),
        )
    ]
    selected = [e for e in events if e.get("event") == "highlight_buffer"]
    if sorted(json.dumps(e, sort_keys=True) for e in selected) != sorted(
        json.dumps(e, sort_keys=True) for e in expected
    ):
        raise ValueError("Unexpected auxiliary v11 highlight capture.")


def parameters(state: bytes) -> tuple[tuple, tuple]:
    """Extract numeric lens parameters only after enforcing the recovered branch."""
    if (
        len(state) != 0xD30
        or struct.unpack_from("<I", state, 0x260)[0] != 5
        or struct.unpack_from("<I", state, 0x2E8)[0] != 0
        or struct.unpack_from("<2d", state, 0x430) != (0.4, 1.0)
        or struct.unpack_from("<2f", state, 0x6D8) != (1.0, 1.0)
        or struct.unpack_from("<H", state, 0x6E0)[0] != 0
        or state[0xD28:0xD2B] != bytes(3)
        or any(
            struct.unpack_from("<Q", state, offset)[0] != 65536
            for offset in (0x2A0, 0x2E0)
        )
    ):
        raise ValueError("Unsupported mode5 vignette state or optional correction.")
    offset = struct.unpack_from("<2f", state, 0x6A8)
    coefficients = tuple(
        struct.unpack_from("<f", state, position)[0]
        for position in (0x6B0, 0x6B8, 0x6C0, 0x6C8)
    )
    return offset, coefficients


def verify(args: argparse.Namespace, report: dict) -> None:
    root = Path(__file__).resolve().parents[1]

    def pinned(path: Path, expected: str | None = None) -> bytes:
        raw = path.read_bytes()
        key, actual = str(path.resolve()), hashlib.sha256(raw).hexdigest()
        if expected is not None and actual != expected:
            raise ValueError("Recorded input hash drift.")
        if key in report["input_hashes"] and report["input_hashes"][key] != actual:
            raise ValueError("Input changed during verification.")
        report["input_hashes"][key] = actual
        return raw

    captured = json.loads(pinned(args.capture, CAPTURE_HASH))
    for path, expected in captured["input_hashes"].items():
        pinned(Path(path), expected)
    for name, expected in PINNED.items():
        pinned(args.vendor / name, expected)
    for name, expected in (
        ("probe_stage_pixels_v11.c", SOURCE_HASH),
        ("highlight_buffer_tap.h", TAP_HASH),
    ):
        pinned(args.capture.parent.parent / name, expected)
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename and Path(filename).resolve().is_relative_to(root):
            pinned(Path(filename))
    if len(captured["cases"]) != 1 or captured["cases"][0]["returncode"] != 0:
        raise ValueError("Expected one successful saved source render.")
    case = captured["cases"][0]
    for name in ("source", "np3"):
        pinned(Path(case[name]), case[f"{name}_sha256"])
    source_root = _tiff.read(pinned(Path(case["source"])))
    stages = stage_events(case["events"])
    auxiliary_events(case["events"])
    core = [
        e
        for e in case["events"]
        if e.get("event") not in TRACE_EVENTS | {"highlight_buffer"}
    ]
    profile = Path(captured["output_profile"])
    pinned(profile, SRGB_HASH)
    report["capture_render"] = validate_render_events(
        core, picture_controls(pinned(Path(case["np3"]))), profile
    )
    report["capture_providers"] = provider_paths(
        core, captured["input_hashes"], args.vendor.resolve(), args.wine_prefix
    )

    def pixels(index: int, suffix: str, shape: tuple) -> np.ndarray:
        raw = pinned(args.capture.parent / f"case0.stage-{index}-{suffix}.bin")
        if len(raw) != int(np.prod(shape)) * 2:
            raise ValueError("Unexpected captured vignette pixel or table size.")
        return np.frombuffer(raw, dtype="<u2").reshape(shape)

    sizes = {target: size for _, target, size in PAIRS} | {156: (2064, 1376)}
    states = {}
    identity = np.arange(65536, dtype="<u2")
    for index, size in sizes.items():
        state = pinned(args.capture.parent / f"case0.stage-{index}-vignette-filter.bin")
        captured_offset, captured_coefficients = parameters(state)
        offset, coefficients = source_parameters(source_root, size)
        if struct.pack("<6f", *offset, *coefficients) != struct.pack(
            "<6f", *captured_offset, *captured_coefficients
        ):
            raise ValueError("Source-derived vignette parameters differ from capture.")
        table = radial_table(size, offset, coefficients)
        if struct.unpack_from("<Q", state, 0x250)[0] != len(table):
            raise ValueError(
                "Generated radial table count differs from captured state."
            )
        states[index] = (offset, coefficients)
        report["parameters"].append(
            {
                "stage": index,
                "size": size,
                "offset": offset,
                "coefficients": coefficients,
            }
        )
        for name, native in (
            ("radial", table),
            ("table0", identity),
            ("table1", identity),
        ):
            report["tables"].append(
                {
                    "stage": index,
                    "kind": name,
                    **comparison(
                        native, pixels(index, f"vignette-{name}", native.shape)
                    ),
                }
            )
    coverage = np.zeros((172, 258), dtype=np.uint8)
    for source, target, size in PAIRS:
        first, second = stages[source], stages[target]
        if (first["stage"], second["stage"]) != ("highlight", "vignette") or (
            first["rect"] != second["rect"]
            or first["format"] != 0x10060201
            or second["format"] != 0x10060201
        ):
            raise ValueError("Vignette stage identity or rectangle mismatch.")
        left, top, right, bottom = second["rect"]
        shape = (bottom - top, right - left, 3)
        original = pixels(source, "highlight-output", shape)
        expected = pixels(target, "vignette-output", shape)
        offset, coefficients = states[target]
        report["replays"].append(
            {
                "source": source,
                "target": target,
                "rect": second["rect"],
                **comparison(
                    transform(original, size, (left, top), offset, coefficients),
                    expected,
                ),
            }
        )
        if size == (258, 172):
            coverage[top:bottom, left:right] += 1
    report["unique_final_words"] = int(coverage.size * 3)
    report["final_coverage_once"] = bool(np.all(coverage == 1))
    report["sources_unchanged"] = all(
        digest(Path(path)) == expected
        for path, expected in report["input_hashes"].items()
    )
    report["passed"] = (
        report["final_coverage_once"]
        and report["sources_unchanged"]
        and all(c["different_words"] == 0 for c in report["tables"] + report["replays"])
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("capture", "vendor", "wine-prefix", "output"):
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
        "evidence_kind": "source_metadata_to_mode5_vignette_vs_saved_vendor_stages",
        "input_hashes": {},
        "parameters": [],
        "tables": [],
        "replays": [],
        "passed": False,
        "numpy_version": np.__version__,
        "limits": [
            "One source/control, mode5, aspect1, unit normalization/cap, strength0.4.",
            "Identity transfer tables, auxiliary correction disabled, unit exponent.",
            "Lens parameters come from the source NEF; default Normal control only.",
            "Saved upstream pixels; no fresh vendor calls or complete native renderer.",
            "No NEFX authenticity or Lightroom acceptance is established.",
        ],
    }
    try:
        verify(args, report)
    except KeyboardInterrupt:
        report.update(cancelled=True, passed=False, error="Vignette replay cancelled.")
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
