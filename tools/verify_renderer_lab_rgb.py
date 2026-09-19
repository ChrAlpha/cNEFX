"""Compare captured and fresh Lab/RGB kernel outputs with the native stage."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import platform
import struct
import subprocess
from pathlib import Path

import numpy as np
from renderer_lab_rgb import transform
from verify_image_render import validate_render_events
from verify_image_sdk import picture_controls, provider_paths
from verify_np3_header import digest, windows
from verify_renderer_lut import RENDER_HASH, WIDTHS

KERNEL_RVA = 0x53A2F0


def validate_events(events: list, dll: Path, width: int, height: int) -> None:
    if type(events) is not list or len(events) != 3:
        raise ValueError("Expected three Lab/RGB kernel events.")
    if any(type(event) is not dict for event in events):
        raise ValueError("Expected structured Lab/RGB kernel events.")
    provider, ready, complete = events
    try:
        loaded = bytes.fromhex(provider["path_hex"]).decode("ascii").replace("\\", "/")
    except (KeyError, TypeError, ValueError, UnicodeError) as exc:
        raise ValueError("Invalid kernel provider evidence.") from exc
    if loaded.casefold() != windows(dll).replace("\\", "/").casefold():
        raise ValueError("Unexpected kernel provider.")
    mxcsr = complete.get("mxcsr")
    if (
        type(mxcsr) is not list
        or len(mxcsr) != 2
        or any(
            type(value) is not int or value not in (0x1F80, 0x1FA0) for value in mxcsr
        )
    ):
        raise ValueError("Unexpected kernel MXCSR controls or exception flags.")
    expected = [
        {"event": "provider", "path_hex": provider["path_hex"]},
        {
            "event": "ready",
            "kernel_rva": KERNEL_RVA,
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
    if json.dumps(events, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise ValueError("Kernel lifecycle, output or guard mismatch.")


def capture_events(events: list) -> list[dict]:
    """Retain only the observed bounded branch; validate excluded calls too."""
    selected = []
    indices = set()
    for event in events:
        if type(event) is not dict:
            raise ValueError("Expected structured capture events.")
        if event.get("event") != "inner_pixels":
            continue
        for key in ("index", "object", "format", "width", "height", "kernel_rva"):
            if type(event.get(key)) is not int:
                raise ValueError("Invalid capture dimensions or identity.")
        if event["index"] in indices or event["index"] < 1 or event["object"] < 1:
            raise ValueError("Duplicate or invalid capture identity.")
        indices.add(event["index"])
        if (
            event.get("error") is not False
            or json.dumps(event.get("saved")) != "[1, 1]"
            or not 1 < event["width"] <= 4096
            or not 1 < event["height"] <= 4096
            or event["width"] * event["height"] * 6 > 16 * 1024 * 1024
        ):
            raise ValueError("Failed, incomplete or unbounded inner capture.")
        branch = (event["kernel_rva"], event["format"])
        if branch == (KERNEL_RVA, 0x10060204):
            selected.append(event)
        elif branch != (0x53B640, 0x10060201):
            raise ValueError("Unexpected inner renderer branch.")
    if not selected:
        raise ValueError("No captured Lab/RGB calls.")
    return selected


def pixels(width: int) -> np.ndarray:
    points = list(
        itertools.product((0, 1, 5242, 5243, 32767, 32768, 65534, 65535), repeat=3)
    )
    ramp = np.arange(0, 65536, 257)
    neutral = np.column_stack((ramp, np.full(256, 32768), np.full(256, 32768)))
    random = np.random.default_rng(20260922).integers(0, 65536, (4096, 3))
    result = np.vstack((points, neutral, random)).astype("<u2")
    padding = -len(result) % width
    return np.vstack((result, result[:padding])) if padding else result


def comparison(native: np.ndarray, vendor: np.ndarray) -> dict:
    if native.shape != vendor.shape:
        raise ValueError("Unexpected kernel output dimensions.")
    delta = native.astype(np.int32) - vendor.astype(np.int32)
    return {
        "compared_words": int(native.size),
        "different_words": int(np.count_nonzero(delta)),
        "maximum_error": int(np.max(np.abs(delta))),
        "native_sha256": hashlib.sha256(native.tobytes()).hexdigest(),
        "vendor_sha256": hashlib.sha256(vendor.tobytes()).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dll", "probe", "wine-prefix", "capture", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.output.exists() or args.output.resolve().is_relative_to(root):
        parser.error("Output must be a new directory outside the repository.")
    if digest(args.dll) != RENDER_HASH:
        parser.error("Unexpected renderer provider hash.")
    args.output.mkdir(parents=True)
    report = {
        "evidence_kind": "captured_and_live_LabRGB_kernel_vs_native",
        "input_hashes": {},
        "replays": [],
        "cases": [],
        "passed": False,
        "numpy_version": np.__version__,
        "libc": list(platform.libc_ver()),
        "limits": [
            "Kernel53a2f0,Lab16,identity auxiliary curves,sampled output curves.",
            "Captured matrix/curves are parameters; no native profile generation.",
            "Fresh calls reconstruct guarded pointer state; no complete rendering.",
            "Partial source sampling; no native source or Lightroom acceptance.",
        ],
    }
    hashes = report["input_hashes"]

    def pinned(path):
        raw = path.read_bytes()
        hashes[str(path.resolve())] = hashlib.sha256(raw).hexdigest()
        return raw

    try:
        for path in (
            args.dll,
            args.probe,
            Path(__file__),
            root / "tools/probe_renderer_lab_rgb.c",
            Path(transform.__code__.co_filename),
        ):
            pinned(path)
        source = json.loads(pinned(args.capture))
        for path, expected in source["input_hashes"].items():
            if hashlib.sha256(pinned(Path(path))).hexdigest() != expected:
                raise ValueError("Recorded source-capture input hash drift.")
        if len(source["cases"]) != 1 or source["cases"][0]["returncode"] != 0:
            raise ValueError("Expected one successful guarded source render.")
        case = source["cases"][0]
        events = case["events"]
        selected = capture_events(events)
        extras = {"stage_taps", "stage_pixels", "inner_pixels"}
        if [e for e in events if e.get("event") == "stage_taps"] != [
            {"event": "stage_taps", "installed": True}
        ]:
            raise ValueError("Missing stage-tap installation evidence.")
        if any(
            e.get("error") is not False
            for e in events
            if e.get("event") == "stage_pixels"
        ):
            raise ValueError("Failed stage capture.")
        core = [e for e in events if e.get("event") not in extras]
        profile = Path(source["output_profile"])
        report["capture_render"] = validate_render_events(
            core, picture_controls(pinned(Path(case["np3"]))), profile
        )
        report["capture_providers"] = provider_paths(
            core,
            source["input_hashes"],
            args.dll.resolve().parent.parent,
            args.wine_prefix,
        )
        configs = {}
        for event in selected:
            index, width, height = event["index"], event["width"], event["height"]
            prefix = args.capture.parent / f"case0.inner-{index}"
            state = pinned(Path(f"{prefix}-state.bin"))
            if len(state) != 0x890 or state[0x230] != 1 or state[0x888] != 0:
                raise ValueError("Unsupported captured Lab/RGB state.")
            if any(
                struct.unpack_from("<I", state, 0x558 + i * 0x78)[0] != 6
                or struct.unpack_from("<I", state, 0x3F0 + i * 0x78)[0] != 5
                for i in range(3)
            ):
                raise ValueError("Unsupported captured curve types.")
            matrix = (
                np.frombuffer(state, dtype="<f4", count=9, offset=0x3C8)
                .reshape(3, 3)
                .copy()
            )
            curves = np.array(
                [
                    np.frombuffer(
                        pinned(Path(f"{prefix}-table-1-{i}-0.bin")), dtype="<f4"
                    )
                    for i in range(3)
                ]
            )
            raw = pinned(Path(f"{prefix}-input.bin"))
            target = pinned(Path(f"{prefix}-output.bin"))
            if len(raw) != width * height * 6 or len(target) != len(raw):
                raise ValueError("Unexpected captured pixel size.")
            sample = np.frombuffer(raw, dtype="<u2").reshape(-1, 3)
            vendor = np.frombuffer(target, dtype="<u2").reshape(-1, 3)
            report["replays"].append(
                {
                    "index": index,
                    **comparison(transform(sample, matrix, curves, width), vendor),
                }
            )
            key = hashlib.sha256(matrix.tobytes() + curves.tobytes()).hexdigest()
            configs[key] = (matrix, curves)
        report["excluded_inner_calls"] = [
            e for e in events if e.get("event") == "inner_pixels" and e not in selected
        ]
        ramp = np.tile(np.arange(65536, dtype="<f4"), (3, 1))
        configs["synthetic_signed_matrix"] = (
            np.array([[1, -1, 0], [0, 2, 0], [0, 0, 1]], dtype="<f4"),
            ramp,
        )
        half = np.repeat(
            np.array([0.5, 1.5, 65534.5], dtype="<f4")[:, None], 65536, axis=1
        )
        configs["synthetic_half_even"] = (np.zeros((3, 3), dtype="<f4"), half)
        for config_index, (key, (matrix, curves)) in enumerate(configs.items()):
            matrix_path = args.output / f"config{config_index}.matrix.bin"
            curve_path = args.output / f"config{config_index}.curves.bin"
            matrix_path.write_bytes(matrix.tobytes())
            curve_path.write_bytes(curves.tobytes())
            pinned(matrix_path)
            pinned(curve_path)
            for width in WIDTHS:
                record = {"config": key, "width": width}
                report["cases"].append(record)
                sample = pixels(width)
                input_path = (
                    args.output / f"config{config_index}.width{width}.input.bin"
                )
                output_path = (
                    args.output / f"config{config_index}.width{width}.vendor.bin"
                )
                input_path.write_bytes(sample.tobytes())
                pinned(input_path)
                result = subprocess.run(
                    [
                        "/usr/lib/wine/wine64",
                        str(args.probe.resolve()),
                        windows(args.dll),
                        windows(matrix_path),
                        windows(curve_path),
                        windows(input_path),
                        str(width),
                        windows(output_path),
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
                record.update(
                    returncode=result.returncode,
                    stdout=result.stdout,
                    stderr=result.stderr,
                )
                if result.returncode:
                    raise ValueError("Lab/RGB kernel probe failed.")
                validate_events(
                    [
                        json.loads(line)
                        for line in result.stdout.splitlines()
                        if line.strip()
                    ],
                    args.dll,
                    width,
                    len(sample) // width,
                )
                raw = output_path.read_bytes()
                if len(raw) != sample.nbytes:
                    raise ValueError("Unexpected saved kernel output size.")
                vendor = np.frombuffer(raw, dtype="<u2").reshape(-1, 3)
                record.update(
                    comparison(transform(sample, matrix, curves, width), vendor)
                )
        report["sources_unchanged"] = all(
            digest(Path(path)) == value for path, value in hashes.items()
        )
        report["passed"] = report["sources_unchanged"] and all(
            c["different_words"] == 0 for c in report["cases"] + report["replays"]
        )
    except subprocess.TimeoutExpired as exc:
        report.update(
            error="Kernel probe timed out after30seconds.",
            stdout=(exc.stdout or b"").decode(errors="replace"),
            stderr=(exc.stderr or b"").decode(errors="replace"),
        )
    except (Exception, KeyboardInterrupt) as exc:
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
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
