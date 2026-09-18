"""Compare combined native Kaiji controls and locate unexplained table changes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import struct
import subprocess
import sys
from pathlib import Path

from flexible_math import round_divide32
from kaiji_grading import COEFFICIENT_HASH
from kaiji_oracle import dependency_hashes, loaded_provider_matches
from kaiji_table import generate_table
from kaiji_table import parameter_ranges as expected_ranges
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


def configurations(camera_sweep: bool) -> list[tuple[list[int], int, int]]:
    values = controls()
    if not camera_sweep:
        return [(row, 0, 0) for row in values]
    return (
        [(values[0], 0, index) for index in range(109)]
        + [(values[0], mix, 54) for mix in range(1, 11)]
        + [(row, (i * 7) % 11, (i * 29) % 109) for i, row in enumerate(values[1:], 1)]
    )


def exposure_configurations() -> list[tuple[list[int], int, int, int]]:
    values = controls()
    return [
        (values[0], 0, index, adjustment)
        for index in (0, 54, 108)
        for adjustment in range(-36, 31)
    ] + [
        (row, i * 7 % 11, i * 29 % 109, -(i * 7 % 36 + 1))
        for i, row in enumerate(values[1:], 1)
    ]


def range_mask(ranges: list[tuple[int, bytes]], *, complete: bool) -> bytearray:
    mask = bytearray(0x34370)
    if complete and (
        len(ranges) != 1 or ranges[0][0] != 0 or len(ranges[0][1]) != len(mask)
    ):
        raise ValueError("Expected one complete native table snapshot.")
    for offset, data in ranges:
        if (
            type(offset) is not int
            or type(data) is not bytes
            or offset < 0
            or offset + len(data) > len(mask)
            or any(mask[offset : offset + len(data)])
        ):
            raise ValueError("Native table ranges overlap or exceed the layout.")
        mask[offset : offset + len(data)] = bytes([1]) * len(data)
    if sum(mask) != (len(mask) if complete else 69670):
        raise ValueError("Incomplete native table coverage.")
    return mask


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dll", "probe", "coefficients", "wine-prefix", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--native-complete", action="store_true")
    parser.add_argument("--camera-sweep", action="store_true")
    parser.add_argument("--exposure-sweep", action="store_true")
    parser.add_argument("--adjustment-tables", type=Path)
    args = parser.parse_args()
    if args.camera_sweep and not args.native_complete:
        parser.error("Camera sweep requires complete native table comparison.")
    if args.exposure_sweep and (
        not args.native_complete or args.camera_sweep or args.adjustment_tables is None
    ):
        parser.error(
            "Exposure sweep requires complete mode and adjustment tables only."
        )
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
        Path(__file__),
        *(
            Path(sys.modules[name].__file__)
            for name in (
                "kaiji_blender",
                "kaiji_camera",
                "kaiji_contrast",
                "kaiji_grading",
                "kaiji_input",
                "kaiji_table",
                "kaiji_tone",
                "flexible_math",
            )
        ),
        root / "tools/probe_kaiji_state.c",
    ]
    hashes = {str(p.resolve()): digest(p) for p in paths}
    hashes[str(args.coefficients.resolve())] = hashlib.sha256(coefficients).hexdigest()
    hashes[str((root / "tools/kaiji_oracle.py").resolve())] = digest(
        root / "tools/kaiji_oracle.py"
    )
    hashes.update(dependency_hashes(args.dll, args.probe))
    adjustment_tables = None
    if args.adjustment_tables is not None:
        from kaiji_camera import ADJUSTMENT_BYTES, ADJUSTMENT_HASH

        adjustment_tables = args.adjustment_tables.read_bytes()
        if (
            len(adjustment_tables) != ADJUSTMENT_BYTES
            or hashlib.sha256(adjustment_tables).hexdigest() != ADJUSTMENT_HASH
        ):
            parser.error("Unexpected adjustment table snapshot.")
        hashes[str(args.adjustment_tables.resolve())] = ADJUSTMENT_HASH
    configs = (
        exposure_configurations()
        if args.exposure_sweep
        else [(v, mix, index, 0) for v, mix, index in configurations(args.camera_sweep)]
    )
    report = {
        "evidence_kind": (
            "complete_native_synthetic_table_vs_live_vendor"
            if args.native_complete
            else "combined_native_controls_vs_vendor_table_ranges"
        ),
        "input_hashes": hashes,
        "cases": [],
        "passed": False,
        "seed": 20260919,
        "camera_sweep": args.camera_sweep,
        "exposure_sweep": args.exposure_sweep,
        "limits": (
            [
                "Model14/family12, strength256/selector0/brightness6/hue128.",
                "Camera mix/index are explicit probe inputs, not application-derived.",
                "State+152c/+153e stays zero; exposure adjustment is explicit.",
                "External coefficients; no application import or custom curve.",
                "Leading output-space matrix is zero, suitable for scalar RGB only.",
            ]
            if args.native_complete
            else [
                "Manual field mapping, model14/strength256/selector0/brightness6.",
                "Unchanged bytes outside recovered ranges remain vendor-derived.",
                "Delta coverage does not establish complete native tables.",
                "No complete application import or rendered-image comparison.",
            ]
        ),
    }
    args.output.mkdir(parents=True)
    baseline = None
    try:
        for index, (values, blend_mix, camera_index, exposure_adjustment) in enumerate(
            configs
        ):
            ranges = (
                [
                    (
                        0,
                        generate_table(
                            coefficients,
                            values,
                            blend_mix=blend_mix,
                            camera_index=camera_index,
                            exposure_adjustment=exposure_adjustment,
                            adjustment_tables=adjustment_tables,
                        ),
                    )
                ]
                if args.native_complete
                else expected_ranges(coefficients, values)
            )
            mask = range_mask(ranges, complete=args.native_complete)
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
            if args.native_complete:
                config.extend((round_divide32(values[5] * 6, 25) + 12, 16, 128))
            if args.camera_sweep or args.exposure_sweep:
                config.extend((blend_mix, camera_index))
            if args.exposure_sweep:
                config.append(exposure_adjustment)
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
                "blend_mix": blend_mix,
                "camera_index": camera_index,
                "exposure_adjustment": exposure_adjustment,
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
            if args.native_complete:
                expected_event.update(raw1522=config[10], raw1526=16, raw1538=128)
            if args.camera_sweep or args.exposure_sweep:
                expected_event.update(raw1528=blend_mix, raw152a=camera_index)
            if args.exposure_sweep:
                expected_event.update(raw1533=exposure_adjustment)
            if (
                result.returncode
                or len(events) != 1
                or any(events[0].get(k) != v for k, v in expected_event.items())
                or not loaded_provider_matches(events[0], args.dll)
            ):
                raise ValueError("Unexpected oracle status or configuration.")
            actual = table_path.read_bytes()
            if len(actual) != len(mask) or state_path.read_bytes() != state:
                raise ValueError("Output size or input stability failure.")
            if baseline is None:
                baseline = actual
            if args.native_complete:
                predicted = ranges[0][1]
            else:
                predicted = bytearray(baseline)
                for offset, data in ranges:
                    predicted[offset : offset + len(data)] = data
            if args.native_complete:
                with (args.output / f"case{index}.native.bin").open("xb") as stream:
                    stream.write(predicted)
            mismatches = [
                i
                for i, (a, b) in enumerate(zip(actual, predicted, strict=True))
                if a != b
            ]
            unknown = [i for i in mismatches if not mask[i]]
            case.update(
                sha256=hashlib.sha256(actual).hexdigest(),
                native_sha256=hashlib.sha256(predicted).hexdigest(),
                known_range_different_bytes=len(mismatches) - len(unknown),
                unexplained_change_bytes=len(unknown),
                first_unexplained_offsets=[hex(i) for i in unknown[:32]],
                passed=not mismatches,
            )
        report["sources_unchanged"] = all(
            digest(Path(p)) == h for p, h in hashes.items()
        )
        report["passed"] = (
            len(report["cases"])
            == (217 if args.exposure_sweep else (135 if args.camera_sweep else 17))
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
