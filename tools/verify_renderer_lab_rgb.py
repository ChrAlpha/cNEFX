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
from renderer_profile import parameters
from renderer_rgb_lab import transform as transform_rgb
from verify_image_render import validate_render_events
from verify_image_sdk import picture_controls, provider_paths
from verify_np3_header import digest, windows
from verify_renderer_lut import RENDER_HASH, WIDTHS

KERNEL_RVA = 0x53A2F0


def validate_events(
    events: list, dll: Path, width: int, height: int, *, rgb_lab: bool = False
) -> None:
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
    allowed_flags = (0x1F80, 0x1FA0, 0x1F81, 0x1FA1) if rgb_lab else (0x1F80, 0x1FA0)
    if (
        type(mxcsr) is not list
        or len(mxcsr) != 2
        or any(type(value) is not int or value not in allowed_flags for value in mxcsr)
        or mxcsr[0] not in (0x1F80, 0x1FA0)
    ):
        raise ValueError("Unexpected kernel MXCSR controls or exception flags.")
    expected = [
        {"event": "provider", "path_hex": provider["path_hex"]},
        {
            "event": "ready",
            "kernel_rva": 0x53B640 if rgb_lab else KERNEL_RVA,
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


def capture_events(events: list, *, rgb_lab: bool = False) -> list[dict]:
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
        if branch not in ((KERNEL_RVA, 0x10060204), (0x53B640, 0x10060201)):
            raise ValueError("Unexpected inner renderer branch.")
        if (branch[0] == 0x53B640) == rgb_lab:
            selected.append(event)
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


def parameter_comparison(native: np.ndarray, vendor: np.ndarray) -> dict:
    if (
        native.dtype != np.dtype("<f4")
        or vendor.dtype != native.dtype
        or native.shape != vendor.shape
    ):
        raise ValueError("Expected matching float32 parameter arrays.")
    return {
        "compared_words": int(native.size),
        "different_words": int(
            np.count_nonzero(native.view("<u4") != vendor.view("<u4"))
        ),
        "native_sha256": hashlib.sha256(native.tobytes()).hexdigest(),
        "vendor_sha256": hashlib.sha256(vendor.tobytes()).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dll", "probe", "wine-prefix", "capture", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--rgb-lab", action="store_true")
    parser.add_argument("--icc-profile", type=Path)
    args = parser.parse_args()
    native_transform = transform_rgb if args.rgb_lab else transform
    root = Path(__file__).resolve().parents[1]
    if args.output.exists() or args.output.resolve().is_relative_to(root):
        parser.error("Output must be a new directory outside the repository.")
    if digest(args.dll) != RENDER_HASH:
        parser.error("Unexpected renderer provider hash.")
    args.output.mkdir(parents=True)
    report = {
        "evidence_kind": "captured_and_live_matrix_kernel_vs_native",
        "direction": "rgb-lab" if args.rgb_lab else "lab-rgb",
        "parameter_source": "native_icc" if args.icc_profile else "captured_state",
        "parameter_cases": [],
        "input_hashes": {},
        "replays": [],
        "cases": [],
        "passed": False,
        "numpy_version": np.__version__,
        "libc": list(platform.libc_ver()),
        "limits": [
            "Selected matrix branch only; identity auxiliary and sampled RGB curves.",
            "The matrix/TRC ICC, when supplied, remains an external profile input.",
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
            Path(native_transform.__code__.co_filename),
        ):
            pinned(path)
        native_parameters = None
        if args.icc_profile:
            pinned(Path(parameters.__code__.co_filename))
            pinned(root / "tools/cml_luts.py")
            native_parameters = parameters(
                pinned(args.icc_profile), inverse=not args.rgb_lab
            )
        source = json.loads(pinned(args.capture))
        for path, expected in source["input_hashes"].items():
            if hashlib.sha256(pinned(Path(path))).hexdigest() != expected:
                raise ValueError("Recorded source-capture input hash drift.")
        if len(source["cases"]) != 1 or source["cases"][0]["returncode"] != 0:
            raise ValueError("Expected one successful guarded source render.")
        case = source["cases"][0]
        events = case["events"]
        selected = capture_events(events, rgb_lab=args.rgb_lab)
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
            if (
                len(state) != 0x890
                or state[0x230] != int(not args.rgb_lab)
                or state[0x888] != 0
            ):
                raise ValueError("Unsupported captured Lab/RGB state.")
            auxiliary = 0x6C0 if args.rgb_lab else 0x558
            curve_base = 0x260 if args.rgb_lab else 0x3F0
            if any(
                struct.unpack_from("<I", state, auxiliary + i * 0x78)[0] != 6
                or struct.unpack_from("<I", state, curve_base + i * 0x78)[0] != 5
                for i in range(3)
            ):
                raise ValueError("Unsupported captured curve types.")
            matrix = (
                np.frombuffer(
                    state, dtype="<f4", count=9, offset=0x238 if args.rgb_lab else 0x3C8
                )
                .reshape(3, 3)
                .copy()
            )
            curves = np.array(
                [
                    np.frombuffer(
                        pinned(
                            Path(
                                f"{prefix}-table-{0 if args.rgb_lab else 1}-{i}"
                                f"-{1 if args.rgb_lab else 0}.bin"
                            )
                        ),
                        dtype="<f4",
                    )
                    for i in range(3)
                ]
            )
            if native_parameters is not None:
                expected_matrix, expected_curves = matrix, curves
                matrix, curves = native_parameters
                checked = {
                    "index": index,
                    "matrix": parameter_comparison(matrix, expected_matrix),
                    "curves": parameter_comparison(curves, expected_curves),
                }
                report["parameter_cases"].append(checked)
                if (
                    checked["matrix"]["different_words"]
                    or checked["curves"]["different_words"]
                ):
                    raise ValueError(
                        "Native ICC parameters differ from the captured renderer."
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
                    **comparison(
                        native_transform(sample, matrix, curves, width), vendor
                    ),
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
        if args.rgb_lab:
            del configs["synthetic_signed_matrix"], configs["synthetic_half_even"]
            configs["synthetic_normalized_identity"] = (
                np.eye(3, dtype="<f4"),
                ramp / np.float32(65535),
            )
            configs["synthetic_saturated_xyz"] = (
                np.full((3, 3), 16, dtype="<f4"),
                np.ones_like(ramp),
            )
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
                        *(["rgb-lab"] if args.rgb_lab else []),
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
                    rgb_lab=args.rgb_lab,
                )
                raw = output_path.read_bytes()
                if len(raw) != sample.nbytes:
                    raise ValueError("Unexpected saved kernel output size.")
                vendor = np.frombuffer(raw, dtype="<u2").reshape(-1, 3)
                record.update(
                    comparison(native_transform(sample, matrix, curves, width), vendor)
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
    except KeyboardInterrupt:
        report.update(
            passed=False, cancelled=True, error="Matrix verification cancelled."
        )
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
