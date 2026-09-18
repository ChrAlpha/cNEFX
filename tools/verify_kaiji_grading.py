"""Compare native Kaiji table stages with bounded live vendor generation."""

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

from kaiji_blender import blender_tables
from kaiji_grading import COEFFICIENT_HASH, grading_tables
from kaiji_input import input_matrix
from kaiji_tone import tone_tables
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


def blender_cases() -> list[dict]:
    result = [{"name": "neutral", "regions": [[0, 0, 0]] * 8}]
    for region, channel, value in itertools.product(range(8), range(3), (-100, 100)):
        regions = [[0, 0, 0] for _ in range(8)]
        regions[region][channel] = value
        result.append(
            {
                "name": f"region{region}-channel{channel}-value{value}",
                "regions": regions,
            }
        )
    rng = random.Random(20260919)
    for i in range(12):
        result.append(
            {
                "name": f"random{i}",
                "regions": [
                    [rng.randrange(-100, 101) for _ in range(3)] for _ in range(8)
                ],
            }
        )
    return result


def tone_cases() -> list[dict]:
    combinations = itertools.product(
        (-120, 0, 120), (-120, 0, 120), (-100, 0, 100), (-100, 0, 100)
    )
    result = [
        {"name": f"corner{i}", "tone": list(values)}
        for i, values in enumerate(combinations)
    ]
    rng = random.Random(20260919)
    result.extend(
        {
            "name": f"random{i}",
            "tone": [
                rng.randrange(-120, 121),
                rng.randrange(-120, 121),
                rng.randrange(-100, 101),
                rng.randrange(-100, 101),
            ],
        }
        for i in range(16)
    )
    return result


def matrix_cases() -> list[dict]:
    return [
        {
            "name": f"s{saturation}-a{strength}-m{selector}",
            "saturation": saturation,
            "strength": strength,
            "matrix": selector,
        }
        for saturation, strength, selector in itertools.product(
            (-50, -25, 0, 25, 50), (0, 128, 256), (0, 1, 2)
        )
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dll", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--coefficients", type=Path, required=True)
    parser.add_argument("--wine-prefix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("grading", "blender", "tone", "input"), default="grading"
    )
    parser.add_argument("--model-index", type=int, choices=(0, 14), default=0)
    parser.add_argument("--strength", type=int, default=0)
    args = parser.parse_args()
    if not 0 <= args.strength <= 256:
        parser.error("Strength must be in [0, 256].")
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
        root / "tools/kaiji_blender.py",
        root / "tools/kaiji_tone.py",
        root / "tools/kaiji_input.py",
        root / "tools/flexible_math.py",
        root / "tools/probe_kaiji_state.c",
    ]
    hashes = {str(path.resolve()): digest(path) for path in paths}
    report = {
        "evidence_kind": f"native_{args.stage}_vs_live_vendor_table_generation",
        "stage": args.stage,
        "input_hashes": hashes,
        "camera_id": 1071,
        "profile_family": 12,
        "model_index": args.model_index,
        "grading_indices": [0, 1, 2],
        "seed": 20260919,
        "cases": [],
        "passed": False,
        "limits": [
            "Grading: 65/4096-node LUTs. Blender: six sectors. Tone: 257/4096 nodes.",
            "Parameter-state group order; UI group names remain unverified.",
            "Selected model context; tone uses engine units, not NP3 input units.",
            "Custom curves, complete NP3 import and rendered pixels are not compared.",
            "The vendor generator probe checks suffix guards only.",
        ],
    }
    args.output.mkdir(parents=True)
    try:
        selected = {
            "grading": cases,
            "blender": blender_cases,
            "tone": tone_cases,
            "input": matrix_cases,
        }[args.stage]()
        for case in selected:
            config = {
                "model": args.model_index,
                "strength": case.get("strength", args.strength),
                "saturation": case.get("saturation", 0),
                "matrix": case.get("matrix", 0),
            }
            extra = [str(value) for value in case.get("tone", [0, 0, 0, 0])]
            extra.extend(str(value) for value in config.values())
            if args.stage == "grading":
                sources, expanded = grading_tables(
                    coefficients, case["groups"], case["blending"], case["balance"]
                )
                native = sources + expanded
                values = [0] * 30 + list(itertools.chain.from_iterable(case["groups"]))
                values.extend((case["blending"], case["balance"]))
            elif args.stage == "blender":
                native = blender_tables(
                    coefficients, case["regions"], model_index=args.model_index
                )
                values = (
                    [0] * 6
                    + list(itertools.chain.from_iterable(case["regions"]))
                    + [0] * 11
                )
            elif args.stage == "tone":
                native = tone_tables(coefficients, *case["tone"])
                values = [0] * 41
            else:
                native = (
                    input_matrix(
                        coefficients,
                        config["saturation"],
                        config["strength"],
                        config["matrix"],
                    ),
                )
                values = [0] * 41
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
                    *extra,
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
                configuration=config,
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
            if args.stage == "tone" and event.get("tone") != case["tone"]:
                raise ValueError("Probe did not apply the requested tone state.")
            if any(event.get(key) != value for key, value in config.items()):
                raise ValueError("Probe did not apply the requested profile state.")
            data = output_path.read_bytes()
            if len(data) != 0x34370:
                raise ValueError("Unexpected output table size.")
            if args.stage == "grading":
                vendor = tuple(
                    struct.unpack_from("<65H", data, offset)
                    for offset in (0x4E, 0xD0, 0x152)
                ) + tuple(
                    struct.unpack_from("<4096H", data, offset)
                    for offset in (0x1D4, 0x21D4, 0x41D4)
                )
            elif args.stage == "blender":
                vendor = tuple(
                    struct.unpack_from("<32h", data, 0x6228 + channel * 0x80)
                    for channel in range(6)
                )
            elif args.stage == "tone":
                vendor = (
                    struct.unpack_from("<257H", data, 0x2E044),
                    struct.unpack_from("<4096H", data, 0x2E246),
                )
            else:
                vendor = (struct.unpack_from("<9h", data, 0x61D4),)
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
        count = {"grading": 40, "blender": 61, "tone": 97, "input": 45}[args.stage]
        words = {
            "grading": 3 * (65 + 4096),
            "blender": 6 * 32,
            "tone": 257 + 4096,
            "input": 9,
        }[args.stage]
        report["compared_words"] = len(report["cases"]) * words
        report["passed"] = (
            len(report["cases"]) == count
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
