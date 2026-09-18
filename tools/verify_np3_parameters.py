"""Verify decoded NP3 parameter buffers against unmodified Nikon code."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from inspect_np3 import Record, parse, serialize
from np3_parameters import BASE_PAIRS, BASE_SINGLE, normalize_flexible, parameter_blocks
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


def normalization_variants(records: tuple[Record, ...]) -> list[tuple[str, bytes]]:
    output = []
    values = (0, 1, 255, 2, 254, 127, 129, 128)
    hues = (0, 1, 65535, 32767, 32769, 65534, 32768, 36000)
    for scale in (1, 3, 4, 255):
        for value, hue in zip(values, hues, strict=True):
            replacements = {
                identifier: bytes((value, scale))
                for identifier in range(0x1900, 0x1F00, 0x100)
            }
            replacements[0x1F00] = bytes([value] * 24 + [scale] * 3 + [0])
            replacements[0x2000] = (
                hue.to_bytes(2, "big") + bytes((value, value))
            ) * 3 + bytes([scale] * 3 + [0] + [value, scale] * 2)
            changed = tuple(
                Record(r.identifier, replacements.get(r.identifier, r.payload))
                for r in records
            )
            output.append(
                (f"normalized-{value:03d}-scale-{scale:03d}", serialize(changed))
            )
    replacements = {
        identifier: bytes((107 + i * 9, 2 + i * 3))
        for i, identifier in enumerate(range(0x1900, 0x1F00, 0x100))
    }
    replacements[0x1F00] = bytes(17 + 9 * i for i in range(24)) + bytes((3, 5, 7, 0))
    replacements[0x2000] = b"".join(
        (32531 + i * 221).to_bytes(2, "big") + bytes((153 - i * 17, 107 + i * 25))
        for i in range(3)
    ) + bytes((11, 13, 17, 0, 201, 19, 47, 23))
    output.append(
        (
            "normalized-asymmetric",
            serialize(
                tuple(
                    Record(r.identifier, replacements.get(r.identifier, r.payload))
                    for r in records
                )
            ),
        )
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--dll", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--wine-prefix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", help="Run only the exact named case.")
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
    inputs += [
        (name, data, "synthetic scale/sentinel mutation of vanilla.NP3")
        for name, data in normalization_variants(
            parse((args.samples / "vanilla.NP3").read_bytes())
        )
    ]
    if args.case:
        inputs = [item for item in inputs if item[0] == args.case]
        if not inputs:
            parser.error("Unknown case name.")
    report = {
        "evidence_kind": "live_vendor_np3_parameter_parser_comparison",
        "passed": False,
        "expected_cases": len(inputs),
        "cases": [],
        "dll_sha256": DLL_HASH,
        "probe_sha256": digest(args.probe),
        "runner_sha256": digest(Path(__file__)),
        "model_sha256": digest(Path(__file__).with_name("np3_parameters.py")),
        "normalization_rva": "0x131b90",
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
            returned, values = normalize_flexible(expected["flexible"])
            normalized = (
                actual["normalized_return"] == returned
                and actual["normalized_values"] == values
                and actual["normalized_guard_intact"]
            )
            case.update(
                normalized_values_equal=normalized,
                normalized_values_compared=len(values),
                differences=differences,
                bytes_compared=sum(map(len, expected.values())),
                passed=actual["success"]
                and actual["guards_intact"]
                and normalized
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
