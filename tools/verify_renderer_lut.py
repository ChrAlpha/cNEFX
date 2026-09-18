"""Compare the pinned renderer row kernel with the native grid65 RGB/Lab stage."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import re
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
from renderer_lut import normalized_grid, transform
from verify_cml import metrics
from verify_np3_header import digest, windows

RENDER_HASH = "b9572f5b176861f69fe1407e7e1e18734a9983cd73421411efd999f4409bccd8"
WIDTHS = (*range(1, 10), 50, 207, 208)
KERNEL_RVA = 0x528460


def pixel_inputs(width: int) -> np.ndarray:
    """Include cube corners, grid boundaries, fraction ties and seeded samples."""
    if type(width) is not int or width not in WIDTHS:
        raise ValueError("Unexpected oracle row width.")
    corners = list(
        itertools.product((0, 1, 1023, 1024, 32767, 32768, 65534, 65535), repeat=3)
    )
    grid = np.rint(np.arange(65) * 65535 / 64).astype(np.int64)
    edge = np.unique(np.clip((grid[:, None] + (-1, 0, 1)).ravel(), 0, 65535))
    ramps = np.vstack(
        (
            np.column_stack((edge, edge, edge)),
            np.column_stack((edge, 65535 - edge, edge)),
        )
    )
    random = np.random.default_rng(20260921).integers(0, 65536, (4096, 3))
    points = np.vstack((corners, ramps, random)).astype("<u2")
    padding = -len(points) % width
    return np.vstack((points, points[:padding])) if padding else points


def validate_events(events: list, dll: Path, width: int, height: int) -> None:
    if (
        type(events) is not list
        or len(events) != 3
        or any(type(e) is not dict for e in events)
    ):
        raise ValueError("Expected exactly three structured row-kernel events.")
    provider, ready, complete = events
    if set(provider) != {"event", "path_hex"} or provider["event"] != "provider":
        raise ValueError("Missing renderer provider event.")
    try:
        loaded = bytes.fromhex(provider["path_hex"]).decode("ascii").replace("\\", "/")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError("Invalid renderer provider path.") from exc
    if loaded.casefold() != windows(dll).replace("\\", "/").casefold():
        raise ValueError("Unexpected loaded renderer provider.")
    mxcsr = complete.get("mxcsr")
    if (
        type(mxcsr) is not list
        or len(mxcsr) != 2
        or any(type(x) is not int or x not in (0x1F80, 0x1FA0) for x in mxcsr)
    ):
        raise ValueError("Unexpected MXCSR controls or exception flags.")
    expected = [
        {
            "event": "ready",
            "kernel_rva": KERNEL_RVA,
            "nodes": 65,
            "width": width,
            "height": height,
            "read_only_inputs": True,
            "mxcsr": mxcsr[0],
        },
        {
            "event": "complete",
            "rows": height,
            "bytes": width * height * 6,
            "mxcsr": mxcsr,
            "inputs_and_guards_intact": True,
            "saved": True,
            "released": True,
        },
    ]
    if json.dumps([ready, complete], sort_keys=True) != json.dumps(
        expected, sort_keys=True
    ):
        raise ValueError(
            "Row-kernel lifecycle, guards, dimensions or cleanup mismatch."
        )


def validate_capture_report(report: dict) -> list[tuple[int, int, dict]]:
    if (
        type(report) is not dict
        or report.get("evidence_kind") != "instrumented_small_render_pilot"
    ):
        raise ValueError("Unexpected renderer capture report.")
    artifacts, cases = report.get("artifacts"), report.get("cases")
    if type(artifacts) is not dict or not 1 <= len(artifacts) <= 512:
        raise ValueError("Missing or excessive capture artifacts.")
    for name, sha in artifacts.items():
        if (
            type(name) is not str
            or not re.fullmatch(r"case[0-9]+\.[a-z0-9-]+\.bin", name)
            or type(sha) is not str
            or not re.fullmatch(r"[0-9a-f]{64}", sha)
        ):
            raise ValueError("Unsafe capture artifact name or invalid hash.")
    if type(cases) is not list or not 1 <= len(cases) <= 3:
        raise ValueError("Expected one to three captured render cases.")
    selected = []
    for index, case in enumerate(cases):
        if (
            type(case) is not dict
            or type(case.get("index")) is not int
            or case["index"] != index
            or type(case.get("returncode")) is not int
            or case["returncode"] != 0
            or type(case.get("events")) is not list
            or any(type(e) is not dict for e in case["events"])
        ):
            raise ValueError("Invalid capture case or unsuccessful process.")
        profiles = [
            e
            for e in case["events"]
            if e.get("event") == "cml_profile"
            and type(e.get("size")) is int
            and e["size"] == 1672998
        ]
        for profile in profiles:
            if (
                type(profile.get("index")) is not int
                or profile["index"] < 1
                or type(profile.get("saved")) is not int
                or profile["saved"] != 1
                or type(profile.get("result")) is not int
                or profile["result"] != 0
            ):
                raise ValueError("Invalid captured RGB/Lab profile event.")
        blocks = [e for e in case["events"] if e.get("event") == "rgb_lab"]
        if not 1 <= len(blocks) <= 64 or len({e.get("index") for e in profiles}) != len(
            profiles
        ):
            raise ValueError("Missing or ambiguous RGB/Lab capture events.")
        seen = set()
        saved_blocks = 0
        for event in blocks:
            if any(
                type(event.get(k)) is not int
                for k in ("index", "width", "height", "format")
            ):
                raise ValueError("Invalid RGB/Lab capture dimensions.")
            event_id, width, height = event["index"], event["width"], event["height"]
            strides = event.get("strides")
            if (
                event_id in seen
                or event_id < 1
                or not 1 <= width <= 1024
                or not 1 <= height <= 1024
                or event["format"] != 0x10060201
                or event.get("error") is not False
                or json.dumps(event.get("saved")) not in ("[1, 1]", "[0, 0]")
                or type(strides) is not list
                or len(strides) != 6
                or any(type(v) is not int for v in strides)
                or strides[1:3] != [6, 2]
                or strides[4:] != [6, 2]
            ):
                raise ValueError("Unsupported or duplicate RGB/Lab capture block.")
            seen.add(event_id)
            if event["saved"] == [0, 0]:
                if any(
                    e.get("event") == "rgb_kernel" and e.get("index") == event_id
                    for e in case["events"]
                ):
                    raise ValueError("Selected RGB/Lab capture buffers are missing.")
                setup = (width, height) == (1, 1) and strides == [1, 6, 2, 1, 6, 2]
                if not setup and (strides[0] < width * 6 or strides[3] < width * 6):
                    raise ValueError("Unsupported unsaved RGB/Lab row stride.")
                continue
            if strides[0] < width * 6 or strides[3] < width * 6:
                raise ValueError("Unsupported RGB/Lab capture row stride.")
            saved_blocks += 1
            kernels = [
                e
                for e in case["events"]
                if e.get("event") == "rgb_kernel" and e.get("index") == event_id
            ]
            expected = [{"event": "rgb_kernel", "index": event_id, "rva": 0x4F2690}]
            eligible = [p for p in profiles if p["index"] < event_id]
            if (
                json.dumps(kernels, sort_keys=True)
                != json.dumps(expected, sort_keys=True)
                or not eligible
            ):
                raise ValueError("Missing captured RGB/Lab kernel or source profile.")
            selected.append(
                (index, max(eligible, key=lambda p: p["index"])["index"], event)
            )
        if not 1 <= saved_blocks <= 4:
            raise ValueError("Missing or excessive saved RGB/Lab capture blocks.")
    return selected


def load_captures(
    path: Path, sha256: str
) -> tuple[list[dict], dict[str, str], list[dict], list[dict]]:
    if digest(path) != sha256:
        raise ValueError("Changed capture report hash.")
    report = json.loads(path.read_bytes())
    blocks = validate_capture_report(report)
    excluded = [
        {"case": case["index"], "event": event}
        for case in report["cases"]
        for event in case["events"]
        if event.get("event") == "rgb_lab" and event["saved"] == [0, 0]
    ]
    hashes = {str(path.resolve()): sha256}
    for name, expected in report["artifacts"].items():
        artifact = path.parent / name
        if artifact.is_symlink() or digest(artifact) != expected:
            raise ValueError("Changed or linked capture artifact.")
        hashes[str(artifact.resolve())] = expected
    profiles, replay = {}, []
    for index, profile_index, event in blocks:
        profile_path = path.parent / f"case{index}.tap-{profile_index}-profile.bin"
        prefix = path.parent / f"case{index}.rgb-{event['index']}"
        required = [
            profile_path,
            *(
                Path(f"{prefix}-{suffix}.bin")
                for suffix in ("grid", "state", "input", "output")
            ),
        ]
        if any(str(p.resolve()) not in hashes for p in required):
            raise ValueError("Unpinned RGB/Lab capture input.")
        grid = normalized_grid(profile_path.read_bytes())
        captured = np.frombuffer(required[1].read_bytes(), dtype="<u2")
        if captured.size != 65**3 * 3:
            raise ValueError("Unexpected captured grid byte count.")
        grid_result = metrics(grid, captured.reshape(grid.shape))
        if not grid_result["equivalent"]:
            raise ValueError("Native normalized grid differs from renderer capture.")
        state = required[2].read_bytes()
        if (
            len(state) != 0x8D0
            or struct.unpack_from("<I", state, 0x22C)[0] != 16
            or state[0x3D0:0x3D3] != b"AAA"
            or state[0x3D4:0x3E0] != struct.pack("<3I", *([0x41800080] * 3))
            or struct.unpack_from("<3Q", state, 0x3E0) != (12675, 195, 3)
            or any(
                struct.unpack_from("<I", state, o)[0] != 6
                for o in (0x230, 0x2B0, 0x330, 0x3F8, 0x478, 0x4F8)
            )
        ):
            raise ValueError("Unexpected captured LUTRGBLab state.")
        width, height = event["width"], event["height"]
        arrays = [np.frombuffer(p.read_bytes(), dtype="<u2") for p in required[3:]]
        if any(a.size != width * height * 3 for a in arrays):
            raise ValueError("Unexpected captured RGB/Lab block byte count.")
        result = metrics(
            transform(arrays[0].reshape(-1, 3), grid, width), arrays[1].reshape(-1, 3)
        )
        if not result["equivalent"]:
            raise ValueError("Native RGB/Lab block differs from renderer capture.")
        grid_hash = hashlib.sha256(grid.tobytes()).hexdigest()
        profiles.setdefault(
            grid_hash, {"path": profile_path, "grid": grid, "grid_sha256": grid_hash}
        )
        replay.append(
            {
                "case": index,
                "block": event["index"],
                "grid": grid_result,
                "pixels": result,
            }
        )
    if not 1 <= len(profiles) <= 3:
        raise ValueError("Expected at most three distinct captured RGB/Lab grids.")
    return list(profiles.values()), hashes, replay, excluded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dll", "probe", "wine-prefix", "capture-report", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--capture-report-sha256", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.output.exists() or args.output.resolve().is_relative_to(root):
        parser.error("Output must be a new directory outside the repository.")
    if digest(args.dll) != RENDER_HASH:
        parser.error("Unexpected image_render provider hash.")
    profiles, hashes, replay, excluded = load_captures(
        args.capture_report, args.capture_report_sha256
    )
    paths = [args.dll, args.probe, root / "tools/probe_renderer_lut.c"]
    paths.extend(
        Path(m.__file__)
        for m in tuple(sys.modules.values())
        if getattr(m, "__file__", None)
        and Path(m.__file__).resolve().is_relative_to(root)
    )
    hashes.update({str(p.resolve()): digest(p) for p in paths})
    report = {
        "evidence_kind": "live_renderer_rgb_lab_leaf_vs_native",
        "input_hashes": hashes,
        "capture_replay": replay,
        "capture_complete": False,
        "capture_excluded_calls": excluded,
        "widths": list(WIDTHS),
        "cases": [],
        "passed": False,
        "numpy_version": np.__version__,
        "limits": [
            "Unmodified vendor row entry is a research oracle, never a native runtime.",
            "Grid65 identity-curve RGB16/Lab16 stage; reconstructed row descriptor.",
            "Captured source is an instrumented pilot; full rendered images may vary.",
            "Pilot captures are partial: setup calls and later blocks are omitted.",
            "Excluded recorded calls must succeed; their pixels are not compared.",
            "No white-balance, Lab/Lch, complete-render or Lightroom parity claim.",
        ],
    }
    args.output.mkdir(parents=True)
    try:
        for index, profile in enumerate(profiles):
            grid_path = args.output / f"profile{index}.grid.bin"
            with grid_path.open("xb") as stream:
                stream.write(profile["grid"].tobytes())
            for width in WIDTHS:
                pixels = pixel_inputs(width)
                prefix = args.output / f"profile{index}-width{width}"
                source, output = (
                    Path(f"{prefix}.input.bin"),
                    Path(f"{prefix}.output.bin"),
                )
                with source.open("xb") as stream:
                    stream.write(pixels.tobytes())
                case = {
                    "profile": str(profile["path"]),
                    "grid_sha256": profile["grid_sha256"],
                    "input_sha256": digest(source),
                    "width": width,
                    "height": len(pixels) // width,
                    "passed": False,
                }
                report["cases"].append(case)
                result = subprocess.run(
                    [
                        "/usr/lib/wine/wine64",
                        str(args.probe.resolve()),
                        windows(args.dll),
                        windows(grid_path),
                        windows(source),
                        str(width),
                        windows(output),
                    ],
                    cwd=args.dll.resolve().parent.parent,
                    env=dict(
                        os.environ,
                        WINEPREFIX=str(args.wine_prefix.resolve()),
                        WINEDEBUG="-all",
                    ),
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                case.update(
                    returncode=result.returncode,
                    stdout=result.stdout,
                    stderr=result.stderr,
                )
                events = [
                    json.loads(line)
                    for line in result.stdout.splitlines()
                    if line.strip()
                ]
                case["events"] = events
                if result.returncode:
                    raise ValueError("Vendor row-kernel process failed.")
                validate_events(events, args.dll, width, len(pixels) // width)
                if (
                    digest(source) != case["input_sha256"]
                    or digest(grid_path) != profile["grid_sha256"]
                ):
                    raise ValueError("Row-kernel input changed during execution.")
                vendor = np.frombuffer(output.read_bytes(), dtype="<u2")
                if vendor.size != pixels.size:
                    raise ValueError("Unexpected row-kernel output byte count.")
                case["metrics"] = metrics(
                    transform(pixels, profile["grid"], width), vendor.reshape(-1, 3)
                )
                case["output_sha256"] = digest(output)
                if not case["metrics"]["equivalent"]:
                    raise ValueError("Native RGB/Lab row kernel differs from vendor.")
                case["passed"] = True
        if any(digest(Path(p)) != sha for p, sha in hashes.items()):
            raise ValueError("Pinned oracle input changed during verification.")
        report["passed"] = True
    except subprocess.TimeoutExpired as exc:
        report["error"] = "Vendor process exceeded the 30-second time budget."
        report["timeout_stdout"] = (exc.stdout or b"").decode(errors="replace")
        report["timeout_stderr"] = (exc.stderr or b"").decode(errors="replace")
    except (OSError, ValueError, TypeError) as exc:
        report["error"] = str(exc)
    finally:
        with (args.output / "report.json").open("x") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "cases": len(report["cases"]),
                "report": str(args.output / "report.json"),
            }
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
