"""Compare native Lab/Lch16 with fresh CML export calls and saved stage tiles."""

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
from renderer_lch import lab_to_lch, lch_to_lab
from verify_cml import CML_HASH, validate_events
from verify_image_render import SRGB_HASH, validate_render_events
from verify_image_sdk import picture_controls, provider_paths
from verify_np3_header import digest, windows
from verify_renderer_cms import TRACE_EVENTS, stage_events
from verify_renderer_lab_rgb import comparison

LAB_HASH = "511edeab8fec994834c1853b491ab5bfd1730a3296076e64f7b6433199740f84"
LCH_HASH = "3e06ce09b1735ed0234a27e0e749f15f9e0cf1f482434abc53ae07db166771b4"


def bridge_contract(event: dict, call: dict, state: bytes, rect: list, selector: int):
    width, height = rect[2] - rect[0], rect[3] - rect[1]
    if (
        selector not in (3, 4)
        or event["stage"] != "cms"
        or event["rect"] != rect
        or event["format"] != (0x10060203 if selector == 3 else 0x10060201)
        or len(state) != 0x748
        or int.from_bytes(state[0x718:0x71C], "little") != selector
        or int.from_bytes(state[0x720:0x728], "little") != call["object"]
        or call["kernel_rva"] != (0x53B640 if selector == 3 else 0x53A2F0)
        or (call["width"], call["height"]) != (width, height)
    ):
        raise ValueError("Saved Lab/Lch stage identity mismatch.")


def samples(inverse: bool) -> np.ndarray:
    if type(inverse) is not bool:
        raise ValueError("Expected a boolean direction.")
    corners = np.array(
        list(itertools.product((0, 1, 32767, 32768, 32769, 65534, 65535), repeat=3)),
        dtype="<u2",
    )
    random = np.random.default_rng(20260919).integers(0, 65536, (8192, 3), dtype="<u2")
    if inverse:
        hue = np.tile(np.arange(65536), 5)
        chroma = np.repeat((0, 1, 32767, 46341, 65535), 65536)
        boundary = np.column_stack((hue, chroma, hue)).astype("<u2")
    else:
        ratios = np.arange(8193)
        boundary = np.concatenate(
            [
                np.column_stack((ratios, 32768 + first * a, 32768 + second * b)).astype(
                    "<u2"
                )
                for first, second in itertools.product((-1, 1), repeat=2)
                for a, b in (
                    (ratios, np.full(8193, 8192)),
                    (np.full(8193, 8192), ratios),
                )
            ]
        )
    return np.concatenate((corners, boundary, random))


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

    pinned(args.dll, CML_HASH)
    pinned(args.lab_profile, LAB_HASH)
    pinned(args.lch_profile, LCH_HASH)
    pinned(args.probe)
    for name in ("probe_cml.c", "cml_snapshot.h"):
        pinned(root / "tools" / name)
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename and Path(filename).resolve().is_relative_to(root):
            pinned(Path(filename))
    captured = json.loads(pinned(args.capture))
    for path, expected in captured["input_hashes"].items():
        pinned(Path(path), expected)
    if len(captured["cases"]) != 1 or captured["cases"][0]["returncode"] != 0:
        raise ValueError("Expected one successful saved source render.")
    case = captured["cases"][0]
    events = case["events"]
    stages = stage_events(events)
    core = [e for e in events if e.get("event") not in TRACE_EVENTS]
    output_profile = Path(captured["output_profile"])
    pinned(output_profile, SRGB_HASH)
    report["capture_render"] = validate_render_events(
        core, picture_controls(pinned(Path(case["np3"]))), output_profile
    )
    report["capture_providers"] = provider_paths(
        core, captured["input_hashes"], args.dll.parent.resolve(), args.wine_prefix
    )
    inner = {e["index"]: e for e in events if e.get("event") == "inner_pixels"}
    for stage, rgb_lab, lab_rgb, next_stage in (
        (2553, 2557, 2563, 2559),
        (2555, 2558, 2564, 2560),
    ):
        rect = stages[stage]["rect"]
        width, height = rect[2] - rect[0], rect[3] - rect[1]
        for index, selector, leaf in ((stage, 3, rgb_lab), (next_stage, 4, lab_rgb)):
            event = stages[index]
            state = pinned(args.capture.parent / f"case0.stage-{index}-cms-state.bin")
            bridge_contract(event, inner[leaf], state, rect, selector)
        paths = [
            args.capture.parent / f"case0.inner-{rgb_lab}-output.bin",
            args.capture.parent / f"case0.stage-{stage}-cms-output.bin",
            args.capture.parent / f"case0.inner-{lab_rgb}-input.bin",
        ]
        arrays = []
        for path in paths:
            raw = pinned(path)
            if len(raw) != width * height * 6:
                raise ValueError("Unexpected saved Lab/Lch pixel size.")
            arrays.append(np.frombuffer(raw, dtype="<u2").reshape(-1, 3))
        for direction, function, source, target in (
            ("lab-lch", lab_to_lch, arrays[0], arrays[1]),
            ("lch-lab", lch_to_lab, arrays[1], arrays[2]),
        ):
            report["replays"].append(
                {
                    "stage": stage,
                    "direction": direction,
                    "width": width,
                    "height": height,
                    **comparison(function(source), target),
                }
            )
    for inverse, function in ((False, lab_to_lch), (True, lch_to_lab)):
        direction = "lch-lab" if inverse else "lab-lch"
        source_profile, destination = (
            (args.lch_profile, args.lab_profile)
            if inverse
            else (args.lab_profile, args.lch_profile)
        )
        pixels = samples(inverse)
        for count in (*range(1, 10), len(pixels)):
            selected = pixels if count == len(pixels) else pixels[:count]
            prefix = args.output / f"{direction}-{count}"
            input_path, output_path = (
                Path(f"{prefix}.input.bin"),
                Path(f"{prefix}.vendor.bin"),
            )
            with input_path.open("xb") as stream:
                stream.write(selected.tobytes())
            pinned(input_path)
            record = {"direction": direction, "pixels": count}
            report["cases"].append(record)
            process = subprocess.run(
                [
                    "/usr/lib/wine/wine64",
                    str(args.probe.resolve()),
                    windows(args.dll),
                    windows(source_profile),
                    windows(destination),
                    windows(input_path),
                    windows(output_path),
                ],
                cwd=args.dll.parent.resolve(),
                env=dict(
                    os.environ,
                    WINEPREFIX=str(args.wine_prefix.resolve()),
                    WINEDEBUG="-all",
                ),
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
            record.update(
                returncode=process.returncode,
                stdout=process.stdout,
                stderr=process.stderr,
            )
            if process.returncode:
                raise ValueError("CML Lab/Lch probe failed.")
            validate_events(
                [json.loads(line) for line in process.stdout.splitlines() if line],
                args.dll,
                count,
            )
            raw = pinned(output_path)
            if len(raw) != selected.nbytes:
                raise ValueError("Unexpected CML output byte count.")
            target = np.frombuffer(raw, dtype="<u2").reshape(-1, 3)
            record.update(comparison(function(selected), target))
    report["excluded_inner_calls"] = [
        event for index, event in inner.items() if index not in (2557, 2558, 2563, 2564)
    ]
    report["sources_unchanged"] = all(
        digest(Path(path)) == value for path, value in report["input_hashes"].items()
    )
    report["passed"] = report["sources_unchanged"] and all(
        case["different_words"] == 0 for case in report["cases"] + report["replays"]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "dll",
        "probe",
        "wine-prefix",
        "lab-profile",
        "lch-profile",
        "capture",
        "output",
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
        "evidence_kind": "live_CML_Lab_Lch_exports_and_saved_stages_vs_native",
        "input_hashes": {},
        "cases": [],
        "replays": [],
        "passed": False,
        "numpy_version": np.__version__,
        "limits": [
            "Pinned Lab/Lch identity profiles only; forward SIMD branch.",
            "Tables generated using host NumPy transcendental functions.",
            "Fresh calls use single rows, guarded buffers and checked lifecycle.",
            "No native source preprocessing or Lightroom acceptance is established.",
        ],
    }
    try:
        verify(args, report)
    except KeyboardInterrupt:
        report.update(
            cancelled=True, passed=False, error="Lab/Lch verification cancelled."
        )
    except subprocess.TimeoutExpired:
        report["error"] = "Lab/Lch probe timed out after 30 seconds."
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
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
