"""Measure Nikon CML and native Little CMS on identical ICC/pixel inputs."""

from __future__ import annotations

import argparse
import ctypes as c
import hashlib
import itertools
import json
import os
import subprocess
from pathlib import Path

import numpy as np
from cml_interpolation import transform
from verify_np3_header import digest, windows

CML_HASH = "42d71afbe2d6e95d39a71ef7a6a4dfb2819e8d923c7b7e8d8b86d1166687df6f"
SRGB_HASH = "49caea94c9d36322910350ee37f1fa09629bed70e01cf615a6863ad3f8d1475e"


def pixel_inputs() -> np.ndarray:
    corners = list(
        itertools.product((0, 1, 128, 256, 1024, 32768, 65280, 65535), repeat=3)
    )
    ramp = np.repeat(np.arange(0, 65536, 17).reshape(-1, 1), 3, axis=1)
    random = np.random.default_rng(20260920).integers(0, 65536, (4096, 3))
    return np.vstack((corners, ramp, random)).astype("<u2")


def validate_events(
    events: list, dll: Path, pixels: int, snapshot: bool = False
) -> None:
    if not events or events[0].get("event") != "provider":
        raise ValueError("Missing CML provider event.")
    loaded = bytes.fromhex(events[0]["path_hex"]).decode("ascii").replace("\\", "/")
    if loaded.casefold() != windows(dll).replace("\\", "/").casefold():
        raise ValueError("Unexpected loaded CML provider.")
    expected = [
        {"event": name, "result": 0}
        for name in (
            "xform_created",
            "input_profile",
            "output_profile",
            "intent",
            "proof",
            "transformer_created",
            "transform_set",
            "input_bits",
            "output_bits",
            "processed",
        )
    ]
    if snapshot:
        expected.extend(
            [
                {
                    "event": "cml_state",
                    "dimensions": [32, 32, 32],
                    "channels": [3, 3],
                    "mode": 1,
                    "entry_rva": 0x1F320,
                    "inner_rva": 0x245C0,
                    "flags": [1, 0, 0, 0],
                    "identity_masks": [7, 0],
                    "mxcsr_rounding": [0, 0],
                },
                {"event": "snapshots", "result": 0},
            ]
        )
    expected.append(
        {
            "event": "complete",
            "pixels": pixels,
            "saved": True,
            "inputs_and_guards_intact": True,
            "released": True,
        }
    )
    if len(events) != len(expected) + 1 or any(
        json.dumps(event, sort_keys=True) != json.dumps(reference, sort_keys=True)
        for event, reference in zip(events[1:], expected, strict=True)
    ):
        raise ValueError("CML lifecycle, guards, output or cleanup mismatch.")


def replay_snapshots(output: Path, pixels: np.ndarray, vendor: np.ndarray) -> dict:
    arrays, hashes = [], {}
    for name, shape in (
        ("grid", (32, 32, 32, 4)),
        ("input-lut", (3, 65536)),
        ("output-lut", (3, 65536)),
    ):
        path = Path(f"{output}.{name}.bin")
        hashes[name] = digest(path)
        arrays.append(np.frombuffer(path.read_bytes(), dtype="<u2").reshape(shape))
    if np.any(arrays[0][..., 3]) or not np.array_equal(
        arrays[1], np.tile(np.arange(65536, dtype="<u2"), (3, 1))
    ):
        raise ValueError("Expected zero grid padding and exact identity input LUTs.")
    native = transform(pixels, *arrays)
    result = metrics(native, vendor)
    result["snapshot_sha256"] = hashes
    result["native_sha256"] = hashlib.sha256(native.tobytes()).hexdigest()
    return result


def little_cms(
    library: Path, source: bytes, target: bytes, pixels: np.ndarray, flags: int
) -> tuple[np.ndarray, int]:
    lcms = c.CDLL(str(library.resolve()))
    lcms.cmsGetEncodedCMMversion.restype = c.c_uint32
    lcms.cmsOpenProfileFromMem.argtypes = [c.c_void_p, c.c_uint32]
    lcms.cmsOpenProfileFromMem.restype = c.c_void_p
    lcms.cmsCreateTransform.argtypes = [
        c.c_void_p,
        c.c_uint32,
        c.c_void_p,
        c.c_uint32,
        c.c_uint32,
        c.c_uint32,
    ]
    lcms.cmsCreateTransform.restype = c.c_void_p
    lcms.cmsDoTransform.argtypes = [c.c_void_p, c.c_void_p, c.c_void_p, c.c_uint32]
    lcms.cmsDeleteTransform.argtypes = [c.c_void_p]
    lcms.cmsCloseProfile.argtypes = [c.c_void_p]
    profiles, transform = [], None
    try:
        for data in (source, target):
            profile = lcms.cmsOpenProfileFromMem(data, len(data))
            if not profile:
                raise ValueError("Little CMS rejected an ICC profile.")
            profiles.append(profile)
        transform = lcms.cmsCreateTransform(
            profiles[0], 0x4001A, profiles[1], 0x4001A, 0, flags
        )
        if not transform:
            raise ValueError("Little CMS rejected the RGB16 transform.")
        output = np.empty_like(pixels)
        lcms.cmsDoTransform(
            transform, pixels.ctypes.data, output.ctypes.data, len(pixels)
        )
        return output, lcms.cmsGetEncodedCMMversion()
    finally:
        if transform:
            lcms.cmsDeleteTransform(transform)
        for profile in profiles:
            lcms.cmsCloseProfile(profile)


def metrics(native: np.ndarray, vendor: np.ndarray) -> dict:
    if (
        native.dtype != np.dtype("<u2")
        or vendor.dtype != native.dtype
        or native.shape != vendor.shape
    ):
        raise ValueError("Expected matching uint16 pixel arrays.")
    delta = native.astype(np.int32) - vendor.astype(np.int32)
    differences = int(np.count_nonzero(delta))
    return {
        "different_words": differences,
        "compared_words": int(delta.size),
        "max_absolute_error": int(np.max(np.abs(delta))),
        "mean_absolute_error": float(np.mean(np.abs(delta))),
        "equivalent": differences == 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "dll",
        "probe",
        "wine-prefix",
        "lcms",
        "output-profile",
        "profile-report",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--profile-report-sha256", required=True)
    parser.add_argument("--snapshot", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.output.exists() or args.output.resolve().is_relative_to(root):
        parser.error("Output must be a new directory outside the repository.")
    if (
        digest(args.dll) != CML_HASH
        or digest(args.output_profile) != SRGB_HASH
        or digest(args.profile_report) != args.profile_report_sha256
    ):
        parser.error("Unexpected CML, output profile or source report hash.")
    oracle = json.loads(args.profile_report.read_bytes())
    if oracle.get("passed") is not True or not oracle.get("cases"):
        parser.error("Expected successful ICC serialization evidence.")
    paths = [
        args.dll,
        args.probe,
        args.lcms,
        args.output_profile,
        args.profile_report,
        Path(__file__),
        root / "tools/probe_cml.c",
        root / "tools/cml_snapshot.h",
        root / "tools/cml_interpolation.py",
        root / "tools/verify_np3_header.py",
    ]
    profiles = []
    for case in oracle["cases"]:
        path = args.profile_report.parent / f"case{case['index']}.profile.bin"
        if case.get("passed") is not True or digest(path) != case.get(
            "profile", {}
        ).get("sha256"):
            parser.error("Changed or unsuccessful source ICC profile.")
        paths.append(path)
        profiles.append(path)
    hashes = {str(p.resolve()): digest(p) for p in paths}
    args.output.mkdir(parents=True)
    pixels = pixel_inputs()
    payload = pixels.tobytes()
    pixel_path = args.output / "input.bin"
    with pixel_path.open("xb") as stream:
        stream.write(payload)
    report = {
        "evidence_kind": "live_cml_vs_native_lcms_shared_icc_pixels",
        "input_hashes": hashes,
        "pixels_sha256": hashlib.sha256(payload).hexdigest(),
        "pixels_per_case": len(pixels),
        "cases": [],
        "passed": False,
        "equivalent": False,
        "numpy_version": np.__version__,
        "limits": [
            "Shared ICC/pixel inputs, not decoded-image rendering.",
            "Perceptual intent0, proof0, interleaved RGB16; no added LUTs.",
            "passed means complete evidence, not numeric equivalence.",
            "No claim that Little CMS is a pixel-exact Nikon substitute.",
            "Interpolation replay uses captured tables, not native ICC composition.",
        ],
    }
    try:
        for index, profile in enumerate(profiles):
            output = args.output / f"case{index}.vendor.bin"
            result = subprocess.run(
                [
                    "/usr/lib/wine/wine64",
                    str(args.probe.resolve()),
                    windows(args.dll),
                    windows(profile),
                    windows(args.output_profile),
                    windows(pixel_path),
                    windows(output),
                    *(["snapshot"] if args.snapshot else []),
                ],
                cwd=args.dll.resolve().parent,
                env=dict(
                    os.environ,
                    WINEPREFIX=str(args.wine_prefix.resolve()),
                    WINEDEBUG="-all",
                ),
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            events = [
                json.loads(line)
                for line in result.stdout.splitlines()
                if line.startswith("{")
            ]
            case = {
                "index": index,
                "events": events,
                "returncode": result.returncode,
                "stderr": result.stderr,
                "comparisons": [],
                "passed": False,
            }
            report["cases"].append(case)
            if result.returncode:
                raise ValueError("CML process failed.")
            validate_events(events, args.dll, len(pixels), args.snapshot)
            data = output.read_bytes()
            if len(data) != len(payload):
                raise ValueError("CML output size mismatch.")
            case["vendor_sha256"] = hashlib.sha256(data).hexdigest()
            vendor = np.frombuffer(data, dtype="<u2").reshape(pixels.shape)
            if args.snapshot:
                case["interpolation"] = replay_snapshots(output, pixels, vendor)
                if not case["interpolation"]["equivalent"]:
                    raise ValueError(
                        "Native CML interpolation differs from vendor output."
                    )
            for flags in (0, 0x100):
                native, version = little_cms(
                    args.lcms,
                    profile.read_bytes(),
                    args.output_profile.read_bytes(),
                    pixels,
                    flags,
                )
                target = args.output / f"case{index}.lcms-{flags}.bin"
                with target.open("xb") as stream:
                    stream.write(native.tobytes())
                case["comparisons"].append(
                    dict(
                        metrics(native, vendor),
                        flags=flags,
                        sha256=digest(target),
                        version=version,
                    )
                )
            case["passed"] = True
        report["sources_unchanged"] = (
            all(digest(Path(p)) == h for p, h in hashes.items())
            and pixel_path.read_bytes() == payload
        )
        report["passed"] = report["sources_unchanged"] and all(
            x["passed"] for x in report["cases"]
        )
        report["equivalent"] = report["passed"] and all(
            c["equivalent"] for x in report["cases"] for c in x["comparisons"]
        )
    except (Exception, KeyboardInterrupt) as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    with (args.output / "report.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "report": str(args.output / "report.json"),
                "passed": report["passed"],
                "equivalent": report["equivalent"],
            }
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
