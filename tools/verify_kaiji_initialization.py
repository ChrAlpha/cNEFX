"""Isolate Kaiji initialization fields with pinned vendor-generated tables."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import struct
import subprocess
from pathlib import Path

from flexible_math import round_divide32
from kaiji_oracle import dependency_hashes, loaded_provider_matches
from kaiji_scalar import ScalarTable
from verify_kaiji_composition import controls
from verify_np3_header import DLL_HASH, digest, windows


def cases() -> list[tuple[str, list[int], tuple[int, int, int]]]:
    neutral = [0] * 39 + [50, 0]
    result = [
        (f"bias-{a}-{b}-{c}", neutral, (a, b, c))
        for a, b, c in itertools.product((0, 12), (0, 16), (0, 128))
    ]
    result.append(("hue-sentinel", neutral, (12, 16, 255)))
    for index, values in enumerate(controls()[1:4]):
        result.append(
            (
                f"mixed-{index}",
                values,
                (round_divide32(values[5] * 6, 25) + 12, 16, 128),
            )
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dll", "probe", "wine-prefix", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.output.exists() or args.output.resolve().is_relative_to(root):
        parser.error("Output must be a new directory outside the repository.")
    if digest(args.dll) != DLL_HASH:
        parser.error("Unexpected Rome2.dll hash.")
    sources = [
        args.dll,
        args.probe,
        Path(__file__),
        root / "tools/probe_kaiji_state.c",
        Path(ScalarTable.evaluate.__code__.co_filename),
        Path(round_divide32.__code__.co_filename),
        Path(controls.__code__.co_filename),
        Path(dependency_hashes.__code__.co_filename),
        Path(digest.__code__.co_filename),
    ]
    hashes = {str(p.resolve()): digest(p) for p in sources}
    hashes.update(dependency_hashes(args.dll, args.probe))
    report = {
        "evidence_kind": "live_vendor_initialization_field_factorial",
        "input_hashes": hashes,
        "cases": [],
        "passed": False,
        "limits": [
            "Manual model14/family12 state; this is not full application import.",
            "Primary RGB values use the native evaluator on vendor tables.",
            "Independent vendor scalar captures are required to confirm RGB parity.",
            "State/output suffix guards do not establish complete memory safety.",
        ],
    }
    args.output.mkdir(parents=True)
    baseline = None
    tables = {}
    try:
        for name, values, biases in cases():
            state = struct.pack("<41i", *values)
            state_path = args.output / f"{name}.state.bin"
            table_path = args.output / f"{name}.bin"
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
                *biases,
            ]
            case = {
                "name": name,
                "normalized_values": values,
                "config": config,
                "state_sha256": hashlib.sha256(state).hexdigest(),
                "passed": False,
            }
            report["cases"].append(case)
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
            case.update(
                returncode=result.returncode, events=events, stderr=result.stderr
            )
            expected = dict(
                zip(
                    (
                        "model",
                        "strength",
                        "saturation",
                        "matrix",
                        "contrast",
                        "brightness",
                        "raw1522",
                        "raw1526",
                        "raw1538",
                    ),
                    config[4:],
                    strict=True,
                )
            )
            expected.update(
                event="kaiji_state",
                tone=config[:4],
                guards_intact=True,
                success=True,
                output_bytes=0x34370,
            )
            if (
                result.returncode
                or len(events) != 1
                or any(events[0].get(k) != v for k, v in expected.items())
                or not loaded_provider_matches(events[0], args.dll)
            ):
                raise ValueError("Unexpected oracle status or configuration.")
            data = table_path.read_bytes()
            if len(data) != 0x34370 or state_path.read_bytes() != state:
                raise ValueError("Output size or input stability failure.")
            if baseline is None:
                baseline = data
            table = ScalarTable(data)
            samples = [
                table.evaluate(point)
                for point in ((32, 0, 0), (0, 32, 0), (0, 0, 32), (16, 16, 16))
            ]
            matrix = list(struct.unpack_from("<9h", data, 0x2375C))
            case.update(
                sha256=hashlib.sha256(data).hexdigest(),
                changed_bytes=sum(a != b for a, b in zip(baseline, data, strict=True)),
                chroma_matrix=matrix,
                native_primary_and_gray=samples,
                native_primary_chromatic=any(len(set(p)) > 1 for p in samples[:3]),
                passed=True,
            )
            tables[name] = data
        report["checks"] = {
            "neutral_hue_matrix_is_identity": all(
                c["chroma_matrix"] == [1024, 0, 0, 0, 1024, 0, 0, 0, 1024]
                for c in report["cases"]
                if c["config"][-1] in (128, 255)
            ),
            "native_chroma_requires_initialized_hue": all(
                c["native_primary_chromatic"] == (c["config"][-1] != 0)
                for c in report["cases"][:9]
            ),
            "hue_sentinel_matches_neutral": (
                tables["hue-sentinel"] == tables["bias-12-16-128"]
            ),
            "sources_unchanged": all(digest(Path(p)) == h for p, h in hashes.items()),
        }
        report["passed"] = len(report["cases"]) == 12 and all(report["checks"].values())
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
