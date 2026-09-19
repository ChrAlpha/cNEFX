"""Replay explicit-halo highlight captures with native recovery and fixed gain."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np
from renderer_highlight import recover, tile_halo, transfer_tables, transform
from verify_image_render import SRGB_HASH, validate_render_events
from verify_image_sdk import PINNED, picture_controls, provider_paths
from verify_np3_header import digest
from verify_renderer_cms import TRACE_EVENTS, stage_events
from verify_renderer_lab_rgb import comparison

CAPTURE_HASH = "cc56f48dc64e464f2546705a937d155e055f5374b4aa38f43131d380cb984bfc"
TAP_HASH = "7af3956e3a04ebe2f0afd92c85c036f22653fb7259fbd99630d9d33561d12415"
RECTANGLES = ((0, 0, 208, 172), (208, 0, 258, 172))
STAGES = ((2454, 2456), (2455, 2457))


def buffer_events(events: list) -> dict[int, dict]:
    """Require the four successful calls captured by the bounded v10 probe."""
    selected = [e for e in events if e.get("event") == "highlight_buffer"]
    expected = [
        {
            "event": "highlight_buffer",
            "index": index,
            "rect": list(RECTANGLES[(index - 1) % 2]),
            "channel": 1,
            "bayer": [1, 0],
            "error": False,
            "saved": [1, 1, 1],
        }
        for index in range(1, 5)
    ]
    if sorted(json.dumps(e, sort_keys=True) for e in selected) != sorted(
        json.dumps(e, sort_keys=True) for e in expected
    ):
        raise ValueError("Missing, duplicate or unsupported highlight buffer capture.")
    return {e["index"]: e for e in selected}


def state_contract(state: bytes) -> None:
    """Check the recovered branch, exposure and gain; ignore relocated pointers."""
    if (
        len(state) != 0x590
        or struct.unpack_from("<2f", state, 0x1FC) != (0.0, 0.0)
        or struct.unpack_from("<f", state, 0x20C)[0] != 0.0
        or struct.unpack_from("<f", state, 0x214)[0] != 4.0
        or struct.unpack_from("<3I", state, 0x224) != (4096, 4096, 4096)
        or state[0x3F2] != 1
        or struct.unpack_from("<I", state, 0x3F4)[0] != 0
        or struct.unpack_from("<2I", state, 0x550) != (1, 0)
        or any(
            struct.unpack_from("<Q", state, offset + 0x38)[0] != 65536
            for offset in (0x250, 0x290, 0x320, 0x360)
        )
    ):
        raise ValueError("Unsupported highlight state, exposure, gain or Bayer phase.")


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
    pinned(args.capture.parent.parent / "highlight_buffer_tap.h", TAP_HASH)
    pinned(args.capture.parent.parent / "probe_stage_pixels_v9.c")
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename and Path(filename).resolve().is_relative_to(root):
            pinned(Path(filename))
    if len(captured["cases"]) != 1 or captured["cases"][0]["returncode"] != 0:
        raise ValueError("Expected one successful saved source render.")
    case = captured["cases"][0]
    for name in ("source", "np3"):
        pinned(Path(case[name]), case[f"{name}_sha256"])
    events = case["events"]
    stages = stage_events(events)
    buffers = buffer_events(events)
    core = [
        e for e in events if e.get("event") not in TRACE_EVENTS | {"highlight_buffer"}
    ]
    profile = Path(captured["output_profile"])
    pinned(profile, SRGB_HASH)
    report["capture_render"] = validate_render_events(
        core, picture_controls(pinned(Path(case["np3"]))), profile
    )
    report["capture_providers"] = provider_paths(
        core, captured["input_hashes"], args.vendor.resolve(), args.wine_prefix
    )
    if (report["capture_render"]["width"], report["capture_render"]["height"]) != (
        258,
        172,
    ):
        raise ValueError("Unexpected captured final image dimensions.")

    def pixels(suffix: str, shape: tuple) -> np.ndarray:
        raw = pinned(args.capture.parent / f"case0.{suffix}.bin")
        if len(raw) != int(np.prod(shape)) * 2:
            raise ValueError("Unexpected captured pixel or table size.")
        return np.frombuffer(raw, dtype="<u2").reshape(shape)

    encode, decode = transfer_tables()
    gain = np.minimum(np.arange(65536, dtype=np.int64) * 4, 65535).astype("<u2")
    coverage = np.zeros((172, 258), dtype=np.uint8)
    upstream = np.zeros((172, 258, 3), dtype="<u2")
    for rectangle, (nr1, _) in zip(RECTANGLES, STAGES):
        left, top, right, bottom = rectangle
        upstream[top:bottom, left:right] = pixels(
            f"stage-{nr1}-nr1-output", (bottom - top, right - left, 3)
        )
        coverage[top:bottom, left:right] += 1
    if not np.all(coverage == 1):
        raise ValueError("Missing or overlapping complete upstream image coverage.")
    coverage.fill(0)
    report["gain_only_different_words"] = 0
    for pair, (nr1, highlight) in enumerate(STAGES):
        rect = list(RECTANGLES[pair])
        left, top, right, bottom = rect
        shape = (bottom - top, right - left, 3)
        padded_shape = (shape[0] + 12, shape[1] + 12, 3)
        for stage, name in ((nr1, "nr1"), (highlight, "highlight")):
            if (
                stages[stage]["stage"] != name
                or stages[stage]["rect"] != rect
                or stages[stage]["format"] != 0x10060201
            ):
                raise ValueError("Captured highlight stage identity mismatch.")
        state_contract(
            pinned(
                args.capture.parent / f"case0.stage-{highlight}-highlight-filter.bin"
            )
        )
        original = pixels(f"stage-{nr1}-nr1-output", shape)
        target = pixels(f"stage-{highlight}-highlight-output", shape)
        for index, table in enumerate((gain, gain, decode, encode)):
            report["tables"].append(
                {
                    "stage": highlight,
                    "table": index,
                    **comparison(
                        table,
                        pixels(f"stage-{highlight}-highlight-table{index}", (65536,)),
                    ),
                }
            )
        for index in (pair + 1, pair + 3):
            event = buffers[index]
            if event["rect"] != rect:
                raise ValueError("Captured highlight buffer rectangle mismatch.")
            prefix = f"highlight-buffer-{index}"
            source = pixels(f"{prefix}-input", padded_shape)
            recovered = pixels(f"{prefix}-output", padded_shape)
            generated_halo = tile_halo(upstream, tuple(rect))
            for name, table in (("forward", encode), ("inverse", decode)):
                report["tables"].append(
                    {
                        "buffer": index,
                        "direction": name,
                        **comparison(table, pixels(f"{prefix}-{name}", (65536,))),
                    }
                )
            for name, native, vendor in (
                ("image-to-halo", generated_halo, source),
                ("source-interior", source[6:-6, 6:-6], original),
                ("recovery", recover(source), recovered[6:-6, 6:-6]),
                ("final-stage", transform(source), target),
                ("image-to-final", transform(generated_halo), target),
            ):
                report["replays"].append(
                    {"buffer": index, "kind": name, **comparison(native, vendor)}
                )
        coverage[top:bottom, left:right] += 1
        report["gain_only_different_words"] += comparison(gain[original], target)[
            "different_words"
        ]
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
        "evidence_kind": "saved_explicit_halo_highlight_buffers_vs_native",
        "input_hashes": {},
        "tables": [],
        "replays": [],
        "passed": False,
        "numpy_version": np.__version__,
        "limits": [
            "Pinned v10 capture, one source/control, green reference and Bayer 1/0.",
            "Four calls cover two rectangles; repeated calls are not new images.",
            "Even origin/dimensions, six-pixel halo, Q10 gains 4096.",
            "Saved stage replay only; no fresh vendor calls or native source decoding.",
            "Border parity covers this two-tile image; full rendering is unverified.",
            "No Lightroom acceptance is established.",
        ],
    }
    try:
        verify(args, report)
    except KeyboardInterrupt:
        report.update(cancelled=True, passed=False, error="Highlight replay cancelled.")
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
