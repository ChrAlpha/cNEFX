"""Replay captured final CMS states using the pinned native LittleCMS 2.6 build."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path, PureWindowsPath

import numpy as np
from renderer_lcms import LIBRARY_HASH, PROFILE_HASH, transform
from verify_image_render import validate_render_events
from verify_image_sdk import picture_controls, provider_paths
from verify_np3_header import digest
from verify_renderer_cms import TRACE_EVENTS, stage_events
from verify_renderer_lab_rgb import comparison


def state_contract(state: bytes, profile: Path, wine_prefix: Path) -> dict:
    if len(state) != 0x2E8:
        raise ValueError("Expected a complete final CMS state.")
    paths = []
    for offset in (0xA8, 0x1AC):
        raw, delimiter, _ = state[offset : offset + 260].partition(b"\0")
        if not delimiter:
            raise ValueError("Unterminated final CMS profile path.")
        paths.append(PureWindowsPath(raw.decode("ascii")))
    drive_c = wine_prefix.resolve() / "drive_c"
    expected = [
        PureWindowsPath("C:/") / profile.resolve().relative_to(drive_c).as_posix(),
        PureWindowsPath("Z:/") / profile.resolve().as_posix().lstrip("/"),
    ]
    values = [
        struct.unpack_from("<I", state, o)[0]
        for o in (0x2B0, 0x2B4, 0x2B8, 0x2C0, 0x2C4)
    ]
    if (
        paths != expected
        or values != [16, 0, 1, 0x10060201, 0x10060201]
        or state[0x2BC] != 0
        or struct.unpack_from("<Q", state, 0x2D8)[0] == 0
    ):
        raise ValueError("Unsupported final CMS profile, mode or format.")
    return {
        "profiles": [str(p) for p in paths],
        "intent": 0,
        "flags": 0,
        "input_format": 0x4001A,
        "output_format": 0x4001A,
    }


def verify(args: argparse.Namespace, report: dict) -> None:
    root = Path(__file__).resolve().parents[1]

    def pinned(path: Path, expected: str | None = None) -> bytes:
        raw = path.read_bytes()
        actual, key = hashlib.sha256(raw).hexdigest(), str(path.resolve())
        if (expected is not None and actual != expected) or (
            key in report["input_hashes"] and report["input_hashes"][key] != actual
        ):
            raise ValueError("Recorded final CMS input hash drift.")
        report["input_hashes"][key] = actual
        return raw

    pinned(args.library, LIBRARY_HASH)
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename and Path(filename).resolve().is_relative_to(root):
            pinned(Path(filename))
    captured = json.loads(pinned(args.capture))
    for path, expected in captured["input_hashes"].items():
        pinned(Path(path), expected)
    if len(captured["cases"]) != 1 or captured["cases"][0]["returncode"] != 0:
        raise ValueError("Expected one successful saved render.")
    case = captured["cases"][0]
    profile_path = Path(captured["output_profile"])
    profile = pinned(profile_path, PROFILE_HASH)
    events = case["events"]
    stages = stage_events(events)
    core = [e for e in events if e.get("event") not in TRACE_EVENTS]
    final = validate_render_events(
        core, picture_controls(pinned(Path(case["np3"]))), profile_path
    )
    report["capture_render"] = final
    report["providers"] = provider_paths(
        core, captured["input_hashes"], args.vendor.resolve(), args.wine_prefix
    )
    for path in sorted(args.capture.parent.glob("case0.*.bin")):
        pinned(path)
    raw = pinned(args.capture.parent / "case0.rgb16.bin")
    if len(raw) != final["width"] * final["height"] * 6:
        raise ValueError("Unexpected final image byte count.")
    image = np.frombuffer(raw, dtype="<u2").reshape(final["height"], final["width"], 3)
    coverage = np.zeros(image.shape[:2], dtype=np.uint8)
    finals = [e for e in stages.values() if e["stage"] == "cms4"]
    if len(finals) != 2:
        raise ValueError("Expected two captured final output tiles.")
    for event in finals:
        left, top, right, bottom = event["rect"]
        width, height = right - left, bottom - top
        if (
            event["format"] != 0x10060201
            or right > final["width"]
            or bottom > final["height"]
        ):
            raise ValueError("Final tile format or bounds mismatch.")
        prefix = args.capture.parent / f"case0.stage-{event['index']}-cms4"
        config = state_contract(
            pinned(Path(f"{prefix}-state.bin")), profile_path, args.wine_prefix
        )
        matches = []
        for candidate in stages.values():
            if candidate["stage"] == "cms" and candidate["rect"] == event["rect"]:
                state = pinned(
                    args.capture.parent
                    / f"case0.stage-{candidate['index']}-cms-state.bin"
                )
                if (
                    len(state) == 0x748
                    and struct.unpack_from("<I", state, 0x718)[0] == 4
                ):
                    matches.append(candidate)
        if len(matches) != 1 or matches[0]["format"] != 0x10060201:
            raise ValueError("Missing or ambiguous preceding RGB CMS tile.")
        source = pinned(
            args.capture.parent / f"case0.stage-{matches[0]['index']}-cms-output.bin"
        )
        target = pinned(Path(f"{prefix}-output.bin"))
        if len(source) != width * height * 6 or len(target) != len(source):
            raise ValueError("Unexpected final CMS tile byte count.")
        pixels = np.frombuffer(source, dtype="<u2").reshape(-1, 3)
        expected = np.frombuffer(target, dtype="<u2").reshape(-1, 3)
        actual = transform(args.library, profile, pixels, width)
        record = {
            "input_stage": matches[0]["index"],
            "output_stage": event["index"],
            "rect": event["rect"],
            "config": config,
            "native": comparison(actual, expected),
            "final_image": comparison(
                expected, image[top:bottom, left:right].reshape(-1, 3)
            ),
        }
        report["cases"].append(record)
        coverage[top:bottom, left:right] += 1
        if (
            record["native"]["different_words"]
            or record["final_image"]["different_words"]
        ):
            raise ValueError("Final CMS pixels differ from the native replay or image.")
    if np.any(coverage != 1):
        raise ValueError("Incomplete or overlapping final CMS output coverage.")
    report["sources_unchanged"] = all(
        digest(Path(p)) == h for p, h in report["input_hashes"].items()
    )
    report["passed"] = report["sources_unchanged"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("capture", "library", "vendor", "wine-prefix", "output"):
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
        "evidence_kind": "captured_CMS4_vs_native_pinned_LittleCMS_2_6",
        "input_hashes": {},
        "cases": [],
        "passed": False,
        "limits": [
            "One source/control and the pinned NKsRGB-to-NKsRGB transform.",
            "Preceding pixels remain captured; this is not full native rendering.",
            "Native library build is hash-pinned; other builds require validation.",
        ],
    }
    try:
        verify(args, report)
    except KeyboardInterrupt:
        report.update(
            passed=False, cancelled=True, error="Final CMS verification cancelled."
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
