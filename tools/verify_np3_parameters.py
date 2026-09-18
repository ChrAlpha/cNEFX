"""Verify decoded NP3 parameter buffers against unmodified Nikon code."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from inspect_np3 import Record, parse, serialize
from np3_parameters import BASE_PAIRS, BASE_SINGLE, parameter_blocks
from verify_np3_header import DLL_HASH, digest, windows


def variants(records: tuple[Record, ...]) -> list[tuple[str, bytes]]:
    output = []
    identifiers = sorted(
        set(BASE_PAIRS) | set(BASE_SINGLE) | set(range(0x1900, 0x1F00, 256))
    )
    fields = {record.identifier: record.payload for record in records}
    for identifier in identifiers + [0x500, 0x1F00, 0x2000, 0x10100]:
        original = fields.get(identifier)
        if identifier == 0x10100:
            replacement = b"nxc-oracle\0\0"
        elif identifier == 0x1F00:
            replacement = bytes(range(118, 142)) + original[24:]
        elif identifier == 0x2000:
            replacement = (
                b"".join(
                    (0x8000 + 300 + 20 * i).to_bytes(2, "big")
                    + bytes((136 + i, 124 - i))
                    for i in range(3)
                )
                + original[12:16]
                + bytes((188, 1, 120, 1))
            )
        else:
            replacement = bytes((137, original[1]))
        changed = tuple(
            Record(
                r.identifier, replacement if r.identifier == identifier else r.payload
            )
            for r in records
        )
        if original is None:
            changed += (Record(identifier, replacement),)
        output.append((f"record-{identifier:05x}", serialize(changed)))
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--dll", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--wine-prefix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if digest(args.dll) != DLL_HASH:
        parser.error("Unexpected Nikon DLL hash.")
    args.output.mkdir(exist_ok=False)
    inputs = [
        (path.name, path.read_bytes(), str(path))
        for path in sorted(args.samples.glob("*.NP3"))
    ]
    inputs += [
        (name, data, "synthetic single-record mutation of vanilla.NP3")
        for name, data in variants(parse((args.samples / "vanilla.NP3").read_bytes()))
    ]
    report = {
        "evidence_kind": "live_vendor_np3_parameter_parser_comparison",
        "passed": False,
        "expected_cases": len(inputs),
        "cases": [],
        "dll_sha256": DLL_HASH,
        "probe_sha256": digest(args.probe),
        "runner_sha256": digest(Path(__file__)),
        "model_sha256": digest(Path(__file__).with_name("np3_parameters.py")),
        "limits": (
            "Decoded parameter buffers only; no image rendering or camera import."
        ),
    }
    try:
        if not inputs:
            raise ValueError("No reference inputs.")
        for index, (name, data, provenance) in enumerate(inputs):
            path = args.output / f"input-{index:03d}.NP3"
            with path.open("xb") as stream:
                stream.write(data)
            expected = parameter_blocks(parse(data))
            result = subprocess.run(
                [
                    "/usr/lib/wine/wine64",
                    str(args.probe.resolve()),
                    windows(args.dll),
                    windows(path),
                    "parameters",
                ],
                cwd=args.dll.resolve().parent,
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
            events = [
                json.loads(line)
                for line in result.stdout.splitlines()
                if line.startswith("{")
            ]
            parameters = [
                event for event in events if event.get("event") == "parameters"
            ]
            case = {
                "name": name,
                "provenance": provenance,
                "input_sha256": digest(path),
                "returncode": result.returncode,
                "stderr": result.stderr,
                "passed": False,
                "events": events,
            }
            report["cases"].append(case)
            if result.returncode != 0 or len(parameters) != 1:
                raise ValueError("Vendor parameter parser failed.")
            actual = parameters[0]
            differences = {}
            for key, predicted in expected.items():
                decoded = bytes.fromhex(actual[key + "_hex"])
                if len(decoded) != len(predicted):
                    raise ValueError("Unexpected parameter buffer size.")
                differences[key] = sum(
                    a != b for a, b in zip(decoded, predicted, strict=True)
                )
            case.update(
                differences=differences,
                bytes_compared=sum(map(len, expected.values())),
                passed=actual["success"]
                and actual["guards_intact"]
                and not any(differences.values()),
            )
        report["passed"] = len(report["cases"]) == len(inputs) and all(
            case["passed"] for case in report["cases"]
        )
    except (Exception, KeyboardInterrupt) as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    with (args.output / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "passed": report["passed"]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
