# /// script
# requires-python = ">=3.12"
# dependencies = ["pefile==2024.8.26"]
# ///
"""Compare the native camera-index leaf with pinned, unmodified Nikon code."""

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
from kaiji_grading import COEFFICIENT_HASH
from kaiji_index import MODE_TABLE_HASH, camera_index
from verify_np3_header import DLL_HASH, digest, windows


def cases(coefficients: bytes) -> list[list[int]]:
    base = [0, 0, 14, 6, 12, 0, 20, 0, 0, 1]
    result = [base.copy()]
    for threshold in struct.unpack_from("<5H", coefficients, 0xA82):
        for delta in (-1, 0, 1):
            for side in (0, 1):
                row = base.copy()
                row[side] = threshold + delta
                result.append(row)
    for profile in range(20):
        for brightness in range(13):
            row = base.copy()
            row[2:4] = profile, brightness
            result.append(row)
    for profile in (0, 1, 5, 9, 15):
        for contrast in range(-256, 257):
            row = base.copy()
            row[2], row[5] = profile, contrast
            result.append(row)
    for field, size in ((4, 25), (6, 41)):
        for value in range(size):
            row = base.copy()
            row[field] = value
            result.append(row)
    for mode in range(1, 5):
        for offset in range(109):
            row = base.copy()
            row[7:] = offset, 2, mode
            result.append(row)
    rng = random.Random(20260919)
    random_cases = []
    for _ in range(512):
        random_cases.append(
            [
                rng.randrange(65536),
                rng.randrange(65536),
                rng.randrange(20),
                rng.randrange(13),
                rng.randrange(25),
                rng.randrange(-256, 257),
                rng.randrange(41),
                rng.randrange(109),
                rng.choice((0, 2)),
                rng.randrange(1, 5),
            ]
        )
    result.extend(random_cases)
    for offset in range(109):
        row = base.copy()
        row[7:] = offset, 0, 0
        result.append(row)
    result.extend(row[:8] + [0, 0] for row in random_cases)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dll", "probe", "coefficients", "wine-prefix", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.output.exists() or args.output.resolve().is_relative_to(root):
        parser.error("Output must be a new directory outside the repository.")
    dll, coefficients = args.dll.read_bytes(), args.coefficients.read_bytes()
    if (
        hashlib.sha256(dll).hexdigest() != DLL_HASH
        or hashlib.sha256(coefficients).hexdigest() != COEFFICIENT_HASH
    ):
        parser.error("Unexpected vendor DLL or coefficient hash.")
    with pefile.PE(data=dll, fast_load=True) as pe:
        mode_table = (
            bytes(16)
            + b"".join(pe.get_data(rva, 16) for rva in (0x7CAE50, 0x7B8C60, 0x7CAE40))
            + b"".join(pe.get_data(rva, 4) for rva in (0x27E19F, 0x27E1AF, 0x27E1B9))
        )
    if hashlib.sha256(mode_table).hexdigest() != MODE_TABLE_HASH:
        parser.error("Unexpected mode correction table.")
    sources = [
        args.probe,
        Path(__file__),
        Path(camera_index.__code__.co_filename),
        root / "tools/probe_kaiji_index.c",
        Path(digest.__code__.co_filename),
    ]
    hashes = {str(path.resolve()): digest(path) for path in sources}
    hashes[str(args.dll.resolve())], hashes[str(args.coefficients.resolve())] = (
        DLL_HASH,
        COEFFICIENT_HASH,
    )
    configs = cases(coefficients)
    payload = (
        b"NXCI001\0"
        + struct.pack("<I", len(configs))
        + b"".join(struct.pack("<10i", *row) for row in configs)
    )
    args.output.mkdir(parents=True)
    input_path = args.output / "cases.bin"
    with input_path.open("xb") as stream:
        stream.write(payload)
    with (args.output / "mode-corrections.bin").open("xb") as stream:
        stream.write(mode_table)
    report = {
        "evidence_kind": "native_camera_selector_vs_live_vendor_leaf",
        "input_hashes": hashes,
        "mode_table_sha256": MODE_TABLE_HASH,
        "cases_sha256": hashlib.sha256(payload).hexdigest(),
        "pe_reader": {"name": "pefile", "version": pefile.__version__},
        "coverage": {
            "total": len(configs),
            "processing_zero_mode_zero": sum(row[8:] == [0, 0] for row in configs),
            "processing_two": sum(row[8] == 2 for row in configs),
        },
        "cases": [],
        "passed": False,
        "limits": [
            "Raw input fields, not full metadata/application-state recovery.",
            "Pinned Z8 coefficients; other camera resources are untested.",
        ],
    }
    try:
        result = subprocess.run(
            [
                "/usr/lib/wine/wine64",
                str(args.probe.resolve()),
                windows(args.dll),
                windows(args.coefficients),
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
        report.update(
            returncode=result.returncode,
            stderr=result.stderr,
            completion=events[-1] if events else None,
        )
        if (
            result.returncode
            or len(events) != 4492
            or len(configs) != 4491
            or events[-1]
            != {"event": "complete", "cases": 4491, "input_and_guards_unchanged": True}
        ):
            raise ValueError("Unexpected oracle status or coverage.")
        for i, (config, event) in enumerate(zip(configs, events[:-1], strict=True)):
            if (
                event.get("event") != "case"
                or type(event.get("index")) is not int
                or event["index"] != i
                or type(event.get("result")) is not int
                or not 0 <= event["result"] <= 108
            ):
                raise ValueError("Unexpected oracle case.")
            native = camera_index(coefficients, mode_table, config)
            report["cases"].append(
                {
                    "index": i,
                    "inputs": config,
                    "native": native,
                    "vendor": event["result"],
                    "passed": native == event["result"],
                }
            )
        report["sources_unchanged"] = (
            all(digest(Path(p)) == h for p, h in hashes.items())
            and input_path.read_bytes() == payload
        )
        report["passed"] = report["sources_unchanged"] and all(
            c["passed"] for c in report["cases"]
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
