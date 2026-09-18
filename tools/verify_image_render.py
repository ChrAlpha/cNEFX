"""Record guarded small RGB16 renders from the pinned Nikon SDK."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from kaiji_oracle import dependency_hashes
from verify_image_sdk import PINNED, picture_controls, provider_paths
from verify_np3_header import digest, windows

SRGB_HASH = "49caea94c9d36322910350ee37f1fa09629bed70e01cf615a6863ad3f8d1475e"


def record_process(case: dict, result: subprocess.CompletedProcess) -> list[dict]:
    """Keep diagnostics and complete events even after a truncated JSON record."""
    events = []
    case.update(
        stdout=result.stdout,
        stderr=result.stderr,
        returncode=result.returncode,
        events=events,
    )
    for line in result.stdout.splitlines():
        if line.startswith("{"):
            events.append(json.loads(line))
    return events


def validate_render_events(events: list, picture: dict, profile: Path) -> dict:
    if any(type(e) is not dict for e in events):
        raise ValueError("Expected structured SDK events.")
    for event in events:
        if event.get("event") == "sdk_log" and (
            type(event.get("code")) is not int or event["code"] != 0
        ):
            raise ValueError("SDK reported an invalid or nonzero diagnostic.")
    steps = [e for e in events if e.get("event") not in ("provider", "sdk_log")]
    names = [
        "initialized",
        "created",
        "services",
        "metadata_service",
        "metadata_start",
        "metadata_complete",
        "picture_imported",
        "render_service",
        "composer_service",
        "render_composed",
        "render_setup",
        "render_profile",
        "render_profile_path",
        "render_scale",
        "render_bits",
        "render_size",
        "render_target",
        "render_dependencies_released",
        "render_started",
        "render_waited",
        "render_released",
        "render_complete",
        "closed",
    ]
    if [e.get("event") for e in steps] != names:
        raise ValueError("Incomplete or reordered render lifecycle.")
    for event in steps:
        name = event["event"]
        if name in ("render_complete", "render_profile_path"):
            continue
        expected = {"event": name}
        if name not in ("closed", "render_released", "render_dependencies_released"):
            expected["result"] = 0
        if name in ("created", "services", "metadata_service"):
            expected["present"] = True
        if name == "metadata_complete":
            count = event.get("steps")
            if type(count) is not int or not 0 <= count <= 10000:
                raise ValueError("Invalid metadata step count.")
            expected.update(complete=True, steps=count)
        if name == "picture_imported":
            values = event.get("values")
            if type(values) is not list or any(type(v) is not int for v in values):
                raise ValueError("Invalid normalized Picture Control values.")
            expected.update(
                guards_intact=True,
                input_unchanged=True,
                values=picture["normalized_values"],
            )
        if json.dumps(event, sort_keys=True) != json.dumps(expected, sort_keys=True):
            raise ValueError("SDK render lifecycle result or guard mismatch.")
    path_event = steps[12]
    if set(path_event) != {"event", "path_hex"}:
        raise ValueError("Unexpected render profile event.")
    loaded = bytes.fromhex(path_event["path_hex"]).decode("ascii").replace("\\", "/")
    if loaded.casefold() != windows(profile).replace("\\", "/").casefold():
        raise ValueError("Unexpected render output profile.")
    result = steps[-2]
    integer_keys = (
        "width",
        "height",
        "format",
        "bits",
        "bytes",
        "calls",
        "covered",
        "missing",
        "overlaps",
        "rejected",
        "target_released",
        "target_destroyed",
    )
    if any(type(result.get(k)) is not int for k in integer_keys):
        raise ValueError("Render dimensions and counters must be integers.")
    width, height = result["width"], result["height"]
    if not 1 <= width <= 1024 or not 1 <= height <= 1024:
        raise ValueError("Render dimensions exceed the bounded target.")
    pixels = width * height
    if not 1 <= result["calls"] <= pixels:
        raise ValueError("Invalid render callback count.")
    expected = {
        "event": "render_complete",
        "width": width,
        "height": height,
        "format": 1,
        "bits": 16,
        "scale": 0.03125,
        "bytes": pixels * 6,
        "calls": result["calls"],
        "covered": pixels,
        "missing": 0,
        "overlaps": 0,
        "rejected": 0,
        "target_released": 1,
        "target_destroyed": 0,
        "guards_intact": True,
        "metadata_intact": True,
        "saved": True,
    }
    if json.dumps(result, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise ValueError(
            "Render coverage, target lifetime, guards or byte count mismatch."
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("vendor", "probe", "wine-prefix", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--picture-control", type=Path, action="append", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    vendor = args.vendor.resolve()
    if args.output.exists() or args.output.resolve().is_relative_to(root):
        parser.error("Output must be a new directory outside the repository.")
    if any(digest(vendor / name) != value for name, value in PINNED.items()):
        parser.error("Unexpected SDK provider hash.")
    profiles = (
        args.wine_prefix.resolve() / "drive_c/Program Files/Common Files/Nikon/Profiles"
    )
    profile = profiles / "NKsRGB.icm"
    if digest(profile) != SRGB_HASH:
        parser.error("Unexpected Nikon sRGB output profile.")
    pictures = {p: picture_controls(p.read_bytes()) for p in args.picture_control}
    paths = [
        args.probe,
        Path(__file__),
        root / "tools/probe_image_sdk.c",
        root / "tools/image_render.h",
        root / "tools/image_kaiji.h",
        *args.source,
        *args.picture_control,
        *(p for p in vendor.glob("*.dll") if p.name != "zlib1.dll"),
        args.wine_prefix / "drive_c/windows/system32/zlib1.dll",
        *vendor.glob("Services/*.n5m"),
        *(p for p in profiles.iterdir() if p.is_file()),
    ]
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename and Path(filename).resolve().is_relative_to(root):
            paths.append(Path(filename))
    hashes = {str(p.resolve()): digest(p) for p in paths}
    hashes.update(dependency_hashes(vendor / "Rome2.dll", args.probe))
    report = {
        "evidence_kind": "live_sdk_bounded_rgb16_render",
        "input_hashes": hashes,
        "output_profile": str(profile),
        "scale": 0.03125,
        "bits": 16,
        "cases": [],
        "passed": False,
        "limits": [
            "Unmodified vendor rendering is a research oracle, not a native runtime.",
            "Scale 1/32, RGB16 format1, pinned NKsRGB, dimensions at most 1024 each.",
            "Guarded pixel coverage does not establish native or Lightroom parity.",
            "Source paths are supplied inputs; no pixel-shift sequence claim.",
        ],
    }
    args.output.mkdir(parents=True)
    configs = list(itertools.product(args.source, args.picture_control))
    try:
        for index, (source, np3) in enumerate(configs):
            prefix = args.output / f"case{index}"
            case = {
                "index": index,
                "source": str(source.resolve()),
                "source_sha256": hashes[str(source.resolve())],
                "np3": str(np3.resolve()),
                "np3_sha256": hashes[str(np3.resolve())],
                "passed": False,
            }
            report["cases"].append(case)
            try:
                result = subprocess.run(
                    [
                        "/usr/lib/wine/wine64",
                        str(args.probe.resolve()),
                        windows(vendor / "Rome2.dll"),
                        windows(vendor / "Services"),
                        windows(source),
                        windows(prefix),
                        windows(np3),
                        "render",
                    ],
                    cwd=vendor,
                    env=dict(
                        os.environ,
                        WINEPREFIX=str(args.wine_prefix.resolve()),
                        WINEDEBUG="-all",
                        NXC_RENDER_PROFILE=windows(profile),
                    ),
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                case["timeout_seconds"] = 120
                case["stdout"] = (exc.stdout or b"").decode(errors="replace")
                case["stderr"] = (exc.stderr or b"").decode(errors="replace")
                raise ValueError("Render process exceeded its time budget.") from exc
            events = record_process(case, result)
            if result.returncode:
                raise ValueError("Render probe failed.")
            case["render"] = validate_render_events(events, pictures[np3], profile)
            case["loaded_providers"] = provider_paths(
                events, hashes, vendor, args.wine_prefix
            )
            data = prefix.with_suffix(".rgb16.bin").read_bytes()
            if len(data) != case["render"]["bytes"]:
                raise ValueError("Saved RGB16 output has the wrong size.")
            rgb = np.frombuffer(data, dtype="<u2").reshape(-1, 3)
            case.update(
                output_sha256=hashlib.sha256(data).hexdigest(),
                output_bytes=len(data),
                channel_min=rgb.min(axis=0).tolist(),
                channel_max=rgb.max(axis=0).tolist(),
                passed=True,
            )
        report["sources_unchanged"] = all(
            digest(Path(p)) == h for p, h in hashes.items()
        )
        report["passed"] = (
            report["sources_unchanged"]
            and len(report["cases"]) == len(configs)
            and all(c["passed"] for c in report["cases"])
        )
    except (Exception, KeyboardInterrupt) as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    with (args.output / "report.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {"report": str(args.output / "report.json"), "passed": report["passed"]}
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
