# /// script
# requires-python = ">=3.12"
# dependencies = ["pefile==2024.8.26"]
# ///
"""Compare native Z8 blend selection against the unmodified vendor leaf."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import struct
import subprocess
from pathlib import Path

import pefile
from kaiji_mix import CALIBRATION_HASH, _matrix_product, blend_mix
from verify_np3_header import DLL_HASH, digest, windows


def cases(calibration: bytes, originals: list[list[int]]) -> list[list[int]]:
    base = [16384, 0, 0, 0, 16384, 0, 0, 0, 16384, 512, 512, 17, 0]
    result = originals + [base]
    for mode in range(22):
        for value in (-(2**31), -1281, -1280, -1279, -3, 0, 3, 3840, 3841, 2**31 - 1):
            result.append(base[:11] + [mode, value])
    for threshold, _ in struct.iter_unpack("<2h", calibration[18:342]):
        for delta in (-4, -1, 0, 1, 4):
            for mode in (0, 1, 17, 20, 21):
                result.append(base[:11] + [mode, threshold * 4 + delta])
    levels = (-32768, -32767, -1024, -1, 0, 1, 2, 511, 512, 513, 1024, 32767)
    for red in levels:
        for blue in levels:
            for mode in (1, 17, 21):
                result.append(base[:9] + [red, blue, mode, 0])
    for index in range(9):
        for value in (-32768, -16384, -1, 0, 1, 16384, 32767):
            row = base.copy()
            row[index] = value
            result.append(row)
    result.append([0] * 9 + base[9:])
    rng = random.Random(20260920)
    for _ in range(2048):
        result.append(
            [rng.randrange(-32768, 32768) for _ in range(11)]
            + [rng.randrange(22), rng.randrange(-(2**31), 2**31)]
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dll", "probe", "image-report", "wine-prefix", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--image-report-sha256", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.output.exists() or args.output.resolve().is_relative_to(root):
        parser.error("Output must be a new directory outside the repository.")
    if (
        digest(args.dll) != DLL_HASH
        or digest(args.image_report) != args.image_report_sha256
    ):
        parser.error("Unexpected DLL or image report hash.")
    with pefile.PE(data=args.dll.read_bytes(), fast_load=True) as pe:
        calibration = b"".join(
            pe.get_data(rva, size)
            for rva, size in (
                (0x4F3C78, 18),
                (0x71B840, 324),
                (0x6FA770, 2816),
                (0x7061E8, 4),
            )
        )
    if hashlib.sha256(calibration).hexdigest() != CALIBRATION_HASH:
        parser.error("Unexpected Z8 blend calibration.")
    oracle = json.loads(args.image_report.read_bytes())
    if (
        oracle.get("passed") is not True
        or oracle.get("kaiji_tables") is not True
        or not oracle.get("cases")
    ):
        parser.error("Expected successful real image factory evidence.")
    paths = [
        args.dll,
        args.probe,
        args.image_report,
        Path(__file__),
        root / "tools/kaiji_mix.py",
        root / "tools/probe_kaiji_mix.c",
        root / "tools/verify_np3_header.py",
    ]
    originals, original_results = [], []
    for case in oracle["cases"]:
        path = args.image_report.parent / f"case{case['index']}.kaiji.bin"
        if case.get("passed") is not True or digest(path) != case.get("kaiji_sha256"):
            parser.error("Unsuccessful or changed live factory snapshot.")
        paths.append(path)
        data = path.read_bytes()
        if len(data) != 0x17F8:
            parser.error("Incorrect Kaiji snapshot size.")
        row = list(struct.unpack_from("<9h", data, 0xB4))
        row += list(struct.unpack_from("<2h", data, 0x152E))
        row += [
            struct.unpack_from("<I", data, 0x66C)[0],
            struct.unpack_from("<i", data, 0x674)[0],
        ]
        expected = struct.unpack_from("<H", data, 0x1528)[0]
        if row not in originals:
            originals.append(row)
            original_results.append(expected)
    hashes = {str(p.resolve()): digest(p) for p in paths}
    configs = cases(calibration, originals)
    payload = b"NXCM001\0" + struct.pack("<I", len(configs))
    payload += b"".join(struct.pack("<13i", *row) for row in configs)
    args.output.mkdir(parents=True)
    input_path = args.output / "cases.bin"
    with input_path.open("xb") as stream:
        stream.write(payload)
    with (args.output / "blend-calibration.bin").open("xb") as stream:
        stream.write(calibration)
    report = {
        "evidence_kind": "native_blend_selection_vs_live_vendor_leaf",
        "input_hashes": hashes,
        "calibration_sha256": CALIBRATION_HASH,
        "cases_sha256": hashlib.sha256(payload).hexdigest(),
        "pe_reader": {"name": "pefile", "version": pefile.__version__},
        "coverage": {
            "total": len(configs),
            "real_image_inputs": len(originals),
            "random": 2048,
            "seed": 20260920,
        },
        "cases": [],
        "passed": False,
        "limits": [
            "Camera1071 and family12 only; other camera calibration paths excluded.",
            "Inputs include explicit signed16 matrices/WB levels and WB mode/value.",
            "Metadata derivation and rendered-image parity are not established here.",
        ],
    }
    try:
        result = subprocess.run(
            [
                "/usr/lib/wine/wine64",
                str(args.probe.resolve()),
                windows(args.dll),
                windows(input_path),
            ],
            cwd=args.dll.resolve().parent,
            env=dict(
                os.environ, WINEPREFIX=str(args.wine_prefix.resolve()), WINEDEBUG="-all"
            ),
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        events = [
            json.loads(line)
            for line in result.stdout.splitlines()
            if line.startswith("{")
        ]
        report.update(returncode=result.returncode, stderr=result.stderr, events=events)
        completion = {
            "event": "complete",
            "cases": len(configs),
            "state_and_guards_unchanged": True,
        }
        if (
            result.returncode
            or len(events) != len(configs) + 2
            or events[-1] != completion
        ):
            raise ValueError("Oracle status, guards or coverage mismatch.")
        provider = events[0]
        loaded = bytes.fromhex(provider.get("path_hex", "")).decode("ascii")
        if (
            provider.get("event") != "provider"
            or loaded.replace("\\", "/").casefold()
            != windows(args.dll).replace("\\", "/").casefold()
        ):
            raise ValueError("Unexpected loaded Rome provider.")
        for i, (config, event) in enumerate(zip(configs, events[1:-1], strict=True)):
            if (
                event.get("event") != "case"
                or type(event.get("index")) is not int
                or event["index"] != i
                or type(event.get("result")) is not int
                or not 0 <= event["result"] <= 10
            ):
                raise ValueError("Malformed vendor case result.")
            native = blend_mix(calibration, config[:9], *config[9:])
            matrix = _matrix_product(struct.unpack_from("<9h", calibration), config[:9])
            if (
                not isinstance(event.get("matrix"), list)
                or len(event["matrix"]) != 9
                or any(type(v) is not int for v in event["matrix"])
            ):
                raise ValueError("Malformed vendor matrix result.")
            report["cases"].append(
                {
                    "index": i,
                    "inputs": config,
                    "native": native,
                    "vendor": event["result"],
                    "matrix_native": matrix,
                    "matrix_vendor": event["matrix"],
                    "passed": native == event["result"] and matrix == event["matrix"],
                }
            )
        report["real_factory_matches"] = all(
            report["cases"][i]["vendor"] == expected
            for i, expected in enumerate(original_results)
        )
        report["sources_unchanged"] = (
            all(digest(Path(p)) == value for p, value in hashes.items())
            and input_path.read_bytes() == payload
        )
        report["passed"] = (
            report["real_factory_matches"]
            and report["sources_unchanged"]
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
