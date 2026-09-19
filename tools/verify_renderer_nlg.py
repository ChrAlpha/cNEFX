"""Replay bounded Nlg final rows against hash-pinned saved vendor snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np
from renderer_nlg import decode_state, transform_row
from verify_image_render import SRGB_HASH, validate_render_events
from verify_image_sdk import PINNED, picture_controls, provider_paths
from verify_np3_header import digest
from verify_renderer_cms import TRACE_EVENTS, stage_events
from verify_renderer_lab_rgb import comparison
from verify_renderer_vignette import auxiliary_events

CAPTURE_HASH = "1c5702a58c7b925529237bdbb4c699c37300f88dc469c3dc5a9fdbe5fb2831a7"
RECTANGLES = ((0, 0, 208, 172), (208, 0, 258, 172))
ROWS = (0, 1, 7, 8, 85, 171)
INSTRUMENTATION = TRACE_EVENTS | {"highlight_buffer", "nlg_row"}
RINGS = (0, 0x88, 0x110, 0x198, 0x220)


def row_events(events: list) -> list[dict]:
    """Require complete, unique, successful AVX captures with exact field types."""
    if type(events) is not list or any(type(e) is not dict for e in events):
        raise ValueError("Expected structured Nlg capture events.")
    rows = [e for e in events if e.get("event") == "nlg_row"]
    expected = {(rect, row) for rect in RECTANGLES for row in ROWS}
    seen, indices = set(), set()
    for event in rows:
        rect = event.get("rect")
        if (
            type(rect) is not list
            or len(rect) != 4
            or any(type(v) is not int for v in rect)
            or any(type(event.get(k)) is not int for k in ("index", "row"))
        ):
            raise ValueError("Invalid Nlg row identity or geometry.")
        identity = (tuple(rect), event["row"])
        index = event["index"]
        reference = {
            "event": "nlg_row",
            "index": index,
            "kind": 0,
            "row": event["row"],
            "rect": rect,
            "pitch": 1248,
            "advanced": True,
            "saved": True,
        }
        if (
            identity not in expected
            or identity in seen
            or not 1 <= index <= 12
            or index in indices
            or json.dumps(event, sort_keys=True)
            != json.dumps(reference, sort_keys=True)
        ):
            raise ValueError("Unexpected, duplicate, or failed Nlg row capture.")
        seen.add(identity)
        indices.add(index)
    if seen != expected or indices != set(range(1, 13)):
        raise ValueError("Incomplete Nlg row capture.")
    return sorted(rows, key=lambda event: event["index"])


def original_events(events: list) -> list[dict]:
    """Exclude only named instrumentation; unknown lifecycle events stay visible."""
    return [e for e in events if e.get("event") not in INSTRUMENTATION]


def job_geometry(before: bytes, after: bytes, event: dict) -> tuple[int, int]:
    """Validate numeric geometry and pointer differences without following pointers."""
    if len(before) != 0x3A0 or len(after) != 0x3A0:
        raise ValueError("Expected complete Nlg job snapshots.")
    left, top, right, bottom = event["rect"]
    width = right - left
    coarse = ((left & 7) + width + 7) // 8
    if (
        struct.unpack_from("<4i", before, 0x35C) != tuple(event["rect"])
        or struct.unpack_from("<i", before, 0x358)[0] != event["row"]
        or struct.unpack_from("<I", before, 0x390)[0] != event["pitch"]
        or struct.unpack_from("<2Q", before, 0x338) != (width * (bottom - top),) * 2
        or before[:0x380] != after[:0x380]
        or before[0x388:] != after[0x388:]
        or struct.unpack_from("<Q", after, 0x380)[0]
        - struct.unpack_from("<Q", before, 0x380)[0]
        != event["pitch"]
    ):
        raise ValueError(
            "Nlg job geometry, state stability, or pointer advance mismatch."
        )
    return width, coarse


def ring_values(job: bytes, raw: bytes, offset: int, width: int, coarse: int):
    """Decode saved allocation-order rows, then apply the numeric ring rotation."""
    if len(job) != 0x3A0 or offset not in RINGS:
        raise ValueError("Unsupported Nlg ring descriptor.")
    spatial = offset in (0x110, 0x198)
    columns, count = (coarse, 3) if spatial else (width, 3 if offset == 0 else 1)
    retained = columns + 2 if spatial else columns
    pitch = ((retained + 31) // 32) * 32
    rotation, padding, start = struct.unpack_from("<3i", job, offset + 0x78)
    if (
        struct.unpack_from("<Q", job, offset + 0x28)[0] != count
        or struct.unpack_from("<3I", job, offset + 0x60) != (columns, 1, pitch)
        or not 0 <= rotation < count
        or not 0 <= start < count
        or padding != (2 if spatial else 0)
        or len(raw) != count * pitch * 2
    ):
        raise ValueError("Nlg ring allocation, geometry, or snapshot size mismatch.")
    values = np.frombuffer(raw, dtype="<u2").reshape(count, pitch)
    if spatial:
        values = values[(rotation + start + np.arange(3)) % 3]
    elif rotation != 0 or start != 0:
        raise ValueError("Unsupported Nlg planar ring rotation.")
    return values[:, :retained].copy()


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
    if (
        captured.get("passed") is not True
        or captured.get("sources_unchanged") is not True
        or len(captured["cases"]) != 1
        or captured["cases"][0]["returncode"] != 0
    ):
        raise ValueError("Expected one successful saved Nlg source render.")
    for mapping in (captured["input_hashes"], captured["output_hashes"]):
        for path, expected in mapping.items():
            pinned(Path(path), expected)
    for name, expected in PINNED.items():
        pinned(args.vendor / name, expected)
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename and Path(filename).resolve().is_relative_to(root):
            pinned(Path(filename))
    profiles = [Path(p) for p, h in captured["input_hashes"].items() if h == SRGB_HASH]
    if len(profiles) != 1:
        raise ValueError("Expected one pinned sRGB output profile.")
    profile = profiles[0]
    case = captured["cases"][0]
    for field in ("source", "np3"):
        pinned(Path(case[field]), captured["input_hashes"][case[field]])
    rows = row_events(case["events"])
    stage_events(case["events"])
    auxiliary_events(case["events"])
    core = original_events(case["events"])
    report["capture_render"] = validate_render_events(
        core, picture_controls(pinned(Path(case["np3"]))), profile
    )
    report["capture_providers"] = provider_paths(
        core, captured["input_hashes"], args.vendor.resolve(), args.wine_prefix
    )
    if report["capture_providers"] != case["loaded_providers"]:
        raise ValueError("Captured provider list differs from validated events.")

    def snapshot(filename: str, size: int | None = None) -> bytes:
        path = args.capture.parent / filename
        raw = pinned(path, captured["output_hashes"][str(path.resolve())])
        if size is not None and len(raw) != size:
            raise ValueError("Unexpected Nlg numeric snapshot size.")
        return raw

    render = snapshot("case0.rgb16.bin", report["capture_render"]["bytes"])
    if hashlib.sha256(render).hexdigest() != case["output_sha256"]:
        raise ValueError("Saved final render differs from capture output hash.")
    for event in rows:

        def part(name: str, size: int | None = None) -> bytes:
            return snapshot(f"case0.nlg-row-{event['index']}-{name}.bin", size)

        before, after = part("job-before"), part("job-after")
        width, coarse = job_geometry(before, after, event)
        parameters = decode_state(part("state", 0x10F9C0))
        rings = {
            offset: ring_values(before, part(f"ring{offset:x}"), offset, width, coarse)
            for offset in RINGS
        }
        part("aux-before", width * 2)
        expected = np.frombuffer(part("output", width * 6), dtype="<u2").reshape(
            width, 3
        )
        expected_aux = np.frombuffer(part("aux-after", width * 2), dtype="<u2")
        native, auxiliary = transform_row(
            rings[0].T,
            rings[0x110],
            rings[0x198],
            (event["rect"][0], event["rect"][1] + event["row"]),
            parameters,
        )
        report["replays"].append(
            {
                "index": event["index"],
                "row": event["row"],
                "rect": event["rect"],
                "rgb": comparison(native, expected),
                "auxiliary": comparison(auxiliary, expected_aux),
            }
        )
    report["rgb_words"] = sum(r["rgb"]["compared_words"] for r in report["replays"])
    report["auxiliary_words"] = sum(
        r["auxiliary"]["compared_words"] for r in report["replays"]
    )
    report["sources_unchanged"] = all(
        digest(Path(path)) == expected
        for path, expected in report["input_hashes"].items()
    )
    report["passed"] = (
        report["sources_unchanged"]
        and (report["rgb_words"], report["auxiliary_words"]) == (4644, 1548)
        and all(
            r[k]["different_words"] == 0
            for r in report["replays"]
            for k in ("rgb", "auxiliary")
        )
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
        "evidence_kind": "native_nlg_final_rows_vs_saved_vendor_snapshots",
        "input_hashes": {},
        "replays": [],
        "passed": False,
        "numpy_version": np.__version__,
        "limits": [
            "Twelve AVX rows; one lossless Z8 source and vanilla Flexible Color.",
            "Captured state, RGB and coarse grids; no source reconstruction.",
            "Nonnegative matrix, bounded controls, optional corrections disabled.",
            "No fresh vendor call, complete native renderer, or full-frame Nlg parity.",
            "No NEFX authenticity or Lightroom acceptance is established.",
        ],
    }
    try:
        verify(args, report)
    except KeyboardInterrupt:
        report.update(cancelled=True, passed=False, error="Nlg replay cancelled.")
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
