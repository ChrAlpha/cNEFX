"""Compare native Grading curves with bounded live vendor table generation."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import random
import struct
import subprocess
from pathlib import Path

from kaiji_grading import COEFFICIENT_HASH, grading_tables
from verify_np3_header import DLL_HASH, digest, windows


def cases() -> list[dict]:
    result = [
        {"name": "neutral", "groups": [[0, 0, 0]] * 3, "blending": 0, "balance": 0}
    ]
    settings = [
        (0, 100, 0),
        (120, 100, 0),
        (240, 100, 0),
        (360, 100, 0),
        (0, 0, -100),
        (0, 0, 100),
    ]
    for group, (hue, chroma, brightness) in itertools.product(range(3), settings):
        groups = [[0, 0, 0] for _ in range(3)]
        groups[group] = [hue, chroma, brightness]
        result.append(
            {
                "name": f"group{group}-h{hue}-c{chroma}-b{brightness}",
                "groups": groups,
                "blending": 0,
                "balance": 0,
            }
        )
    for blending, balance in itertools.product((0, 50, 100), (-100, 0, 100)):
        result.append(
            {
                "name": f"mixed-blend{blending}-balance{balance}",
                "groups": [[15, 75, -30], [145, 50, 50], [270, 100, -60]],
                "blending": blending,
                "balance": balance,
            }
        )
    rng = random.Random(20260919)
    for i in range(12):
        result.append(
            {
                "name": f"random{i}",
                "groups": [
                    [rng.randrange(361), rng.randrange(101), rng.randrange(-100, 101)]
                    for _ in range(3)
                ],
                "blending": rng.randrange(101),
                "balance": rng.randrange(-100, 101),
            }
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dll", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--coefficients", type=Path, required=True)
    parser.add_argument("--wine-prefix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.output.exists() or args.output.resolve().is_relative_to(root):
        parser.error("Output must be a new directory outside the repository.")
    if digest(args.dll) != DLL_HASH:
        parser.error("Unexpected Rome2.dll hash.")
    coefficients = args.coefficients.read_bytes()
    if hashlib.sha256(coefficients).hexdigest() != COEFFICIENT_HASH:
        parser.error("Unexpected coefficient resource hash.")
    paths = [
        args.dll,
        args.probe,
        args.coefficients,
        Path(__file__),
        root / "tools/kaiji_grading.py",
        root / "tools/flexible_math.py",
        root / "tools/probe_kaiji_state.c",
    ]
    hashes = {str(path.resolve()): digest(path) for path in paths}
    report = {
        "evidence_kind": "native_grading_vs_live_vendor_table_generation",
        "input_hashes": hashes,
        "camera_id": 1071,
        "profile_family": 12,
        "grading_indices": [0, 1, 2],
        "seed": 20260919,
        "cases": [],
        "passed": False,
        "limits": [
            "Only three 65-node source and three 4096-node expanded Grading LUTs.",
            "Parameter-state group order; UI group names remain unverified.",
            "Synthetic context omits scalar controls, custom curves and Blender.",
            "The vendor generator probe checks suffix guards only.",
        ],
    }
    args.output.mkdir(parents=True)
    try:
        for case in cases():
            sources, expanded = grading_tables(
                coefficients, case["groups"], case["blending"], case["balance"]
            )
            values = [0] * 30 + list(itertools.chain.from_iterable(case["groups"]))
            values.extend((case["blending"], case["balance"]))
            state = struct.pack("<41i", *values)
            state_path = args.output / f"{case['name']}.state.bin"
            output_path = args.output / f"{case['name']}.bin"
            with state_path.open("xb") as stream:
                stream.write(state)
            result = subprocess.run(
                [
                    "/usr/lib/wine/wine64",
                    str(args.probe.resolve()),
                    windows(args.dll),
                    windows(state_path),
                    windows(output_path),
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
            case.update(
                returncode=result.returncode,
                events=events,
                stderr=result.stderr,
                state_sha256=hashlib.sha256(state).hexdigest(),
                passed=False,
            )
            report["cases"].append(case)
            if result.returncode or len(events) != 1:
                raise ValueError("Vendor generation failed.")
            event = events[0]
            if (
                event.get("event") != "kaiji_state"
                or event.get("success") is not True
                or event.get("guards_intact") is not True
                or event.get("output_bytes") != 0x34370
                or state_path.read_bytes() != state
            ):
                raise ValueError("Vendor generation status or input check failed.")
            data = output_path.read_bytes()
            if len(data) != 0x34370:
                raise ValueError("Unexpected output table size.")
            native = sources + expanded
            vendor = tuple(
                struct.unpack_from("<65H", data, offset)
                for offset in (0x4E, 0xD0, 0x152)
            ) + tuple(
                struct.unpack_from("<4096H", data, offset)
                for offset in (0x1D4, 0x21D4, 0x41D4)
            )
            differences = [
                sum(a != b for a, b in zip(n, v, strict=True))
                for n, v in zip(native, vendor, strict=True)
            ]
            case.update(
                output_sha256=hashlib.sha256(data).hexdigest(),
                differences=differences,
                passed=not any(differences),
            )
        report["sources_unchanged"] = all(
            digest(Path(p)) == h for p, h in hashes.items()
        )
        report["compared_words"] = len(report["cases"]) * 3 * (65 + 4096)
        report["passed"] = (
            len(report["cases"]) == 40
            and report["sources_unchanged"]
            and all(case["passed"] for case in report["cases"])
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
                "cases": len(report["cases"]),
            }
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
