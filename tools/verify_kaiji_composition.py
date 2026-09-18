"""Compare combined native Kaiji controls and locate unexplained table changes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import struct
import subprocess
from pathlib import Path

from flexible_math import round_divide32
from kaiji_blender import blender_tables
from kaiji_contrast import contrast_tables
from kaiji_grading import COEFFICIENT_HASH, grading_tables
from kaiji_input import input_matrix
from kaiji_tone import tone_tables
from verify_np3_header import DLL_HASH, digest, windows


def controls() -> list[list[int]]:
    result = [[0] * 39 + [50, 0]]
    rng = random.Random(20260919)
    for _ in range(16):
        values = [rng.randrange(-100, 101) for _ in range(5)] + [rng.randrange(-50, 51)]
        values.extend(rng.randrange(-100, 101) for _ in range(24))
        for _ in range(3):
            values.extend(
                (rng.randrange(361), rng.randrange(101), rng.randrange(-100, 101))
            )
        values.extend((rng.randrange(101), rng.randrange(-100, 101)))
        result.append(values)
    return result


def expected_ranges(coefficients: bytes, values: list[int]) -> list[tuple[int, bytes]]:
    """Known generator ranges for model14, strength256, selector0, brightness6."""
    result = []

    def words(offset: int, data, code: str = "H") -> None:
        result.append((offset, struct.pack(f"<{len(data)}{code}", *data)))

    groups = [values[30 + i * 3 : 33 + i * 3] for i in range(3)]
    sources, expanded = grading_tables(coefficients, groups, values[39], values[40])
    for offset, row in zip(
        (0x4E, 0xD0, 0x152, 0x1D4, 0x21D4, 0x41D4), sources + expanded, strict=True
    ):
        words(offset, row)
    words(0x61D4, input_matrix(coefficients, values[5], 256, 0), "h")
    regions = [values[6 + i * 3 : 9 + i * 3] for i in range(8)]
    for channel, row in enumerate(
        blender_tables(coefficients, regions, model_index=14)
    ):
        words(0x6228 + channel * 128, row, "h")
        words(0x6268 + channel * 128, row, "h")
        words(0x213C6 + channel * 64, [max(-512, min(511, v)) for v in row], "h")
    contrast = round_divide32(values[0] * 64, 25)
    sources, expanded = contrast_tables(coefficients, contrast, 6, model_index=14)
    words(0x25836, [256 - abs(contrast), abs(contrast)])
    for offset, row in zip((0x2583A, 0x25A3C, 0x25C3E, 0x25E40), sources, strict=True):
        words(offset, row)
    for offset, row in zip((0x26044, 0x2A044), expanded, strict=True):
        words(offset, row, "i")
    source, expanded_tone = tone_tables(
        coefficients,
        round_divide32(values[1] * 6, 5),
        round_divide32(values[2] * 6, 5),
        values[3],
        values[4],
    )
    words(0x2E044, source)
    words(0x2E246, expanded_tone)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dll", "probe", "coefficients", "wine-prefix", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
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
        *(
            root / "tools" / name
            for name in (
                "kaiji_blender.py",
                "kaiji_contrast.py",
                "kaiji_grading.py",
                "kaiji_input.py",
                "kaiji_tone.py",
                "flexible_math.py",
                "probe_kaiji_state.c",
            )
        ),
    ]
    hashes = {str(p.resolve()): digest(p) for p in paths}
    report = {
        "evidence_kind": "combined_native_controls_vs_vendor_table_ranges",
        "input_hashes": hashes,
        "cases": [],
        "passed": False,
        "seed": 20260919,
        "limits": [
            "Manual field mapping, model14/strength256/selector0/brightness6.",
            "Unchanged bytes outside recovered ranges remain vendor-derived.",
            "Complete native table generation is not established by delta coverage.",
            "No complete application import or rendered-image comparison.",
        ],
    }
    args.output.mkdir(parents=True)
    baseline = None
    mask = bytearray(0x34370)
    try:
        for index, values in enumerate(controls()):
            ranges = expected_ranges(coefficients, values)
            if index == 0:
                for offset, data in ranges:
                    if any(mask[offset : offset + len(data)]):
                        raise ValueError("Native table ranges overlap.")
                    mask[offset : offset + len(data)] = bytes([1]) * len(data)
                report["native_range_bytes"] = sum(mask)
            state = struct.pack("<41i", *values)
            state_path = args.output / f"case{index}.state.bin"
            table_path = args.output / f"case{index}.bin"
            with state_path.open("xb") as stream:
                stream.write(state)
            config = [
                round_divide32(values[1] * 6, 5),
                round_divide32(values[2] * 6, 5),
                values[3],
                values[4],
                14,
                256,
                values[5],
                0,
                round_divide32(values[0] * 64, 25),
                6,
            ]
            result = subprocess.run(
                [
                    "/usr/lib/wine/wine64",
                    str(args.probe.resolve()),
                    windows(args.dll),
                    windows(state_path),
                    windows(table_path),
                    *(str(v) for v in config),
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
                "normalized_values": values,
                "state_sha256": hashlib.sha256(state).hexdigest(),
                "returncode": result.returncode,
                "events": events,
                "stderr": result.stderr,
                "passed": False,
            }
            report["cases"].append(case)
            expected_event = {
                "event": "kaiji_state",
                "tone": config[:4],
                "model": 14,
                "strength": 256,
                "saturation": values[5],
                "matrix": 0,
                "contrast": config[8],
                "brightness": 6,
                "guards_intact": True,
                "success": True,
                "output_bytes": 0x34370,
            }
            if (
                result.returncode
                or len(events) != 1
                or any(events[0].get(k) != v for k, v in expected_event.items())
            ):
                raise ValueError("Unexpected oracle status or configuration.")
            actual = table_path.read_bytes()
            if len(actual) != len(mask) or state_path.read_bytes() != state:
                raise ValueError("Output size or input stability failure.")
            if baseline is None:
                baseline = actual
            predicted = bytearray(baseline)
            for offset, data in ranges:
                predicted[offset : offset + len(data)] = data
            mismatches = [
                i
                for i, (a, b) in enumerate(zip(actual, predicted, strict=True))
                if a != b
            ]
            unknown = [i for i in mismatches if not mask[i]]
            case.update(
                sha256=hashlib.sha256(actual).hexdigest(),
                known_range_different_bytes=len(mismatches) - len(unknown),
                unexplained_change_bytes=len(unknown),
                first_unexplained_offsets=[hex(i) for i in unknown[:32]],
                passed=not mismatches,
            )
        report["sources_unchanged"] = all(
            digest(Path(p)) == h for p, h in hashes.items()
        )
        report["passed"] = (
            len(report["cases"]) == 17
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
