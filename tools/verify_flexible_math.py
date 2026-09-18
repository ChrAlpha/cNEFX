"""Compare synthetic integer cases with pinned, unmodified Rome2 leaf routines."""

from __future__ import annotations

import argparse
import json
import os
import random
import struct
import subprocess
from collections import Counter
from pathlib import Path
from typing import NamedTuple

from flexible_math import INT32_MAX, INT32_MIN, clamp32, interpolate, round_divide32
from verify_np3_header import DLL_HASH, digest, windows

SEED = 20260918
LEAVES = {
    0: {"name": "round_divide32", "rva": "0x29d8e0"},
    1: {"name": "clamp32", "rva": "0x29d8d0"},
    2: {"name": "unsigned_shift6", "rva": "0x2ae600", "shift": 6, "signed": False},
    3: {"name": "unsigned_shift7", "rva": "0x2ae640", "shift": 7, "signed": False},
    4: {"name": "signed_shift6", "rva": "0x2ae680", "shift": 6, "signed": True},
    5: {"name": "signed_shift4", "rva": "0x2ae6c0", "shift": 4, "signed": True},
    6: {"name": "unsigned_shift4", "rva": "0x2ae700", "shift": 4, "signed": False},
}


class Case(NamedTuple):
    name: str
    operation: int
    a: int
    b: int = 0
    c: int = 0
    samples: tuple[int, ...] = ()


def expected(case: Case) -> int | None:
    try:
        if case.operation == 0:
            return round_divide32(case.a, case.b)
        if case.operation == 1:
            return clamp32(case.a, case.b, case.c)
        leaf = LEAVES[case.operation]
        return interpolate(
            case.samples, case.a, interval_shift=leaf["shift"], signed=leaf["signed"]
        )
    except (ValueError, KeyError):
        return None


def cases() -> list[Case]:
    rng = random.Random(SEED)
    output: list[Case] = []
    divisors = (1, 2, 3, 4, 5, 63, 64, 65, 100, 127, 128, 255, 256, INT32_MAX)
    for divisor in divisors:
        numerators = {INT32_MIN, INT32_MIN + 1, -1, 0, 1, INT32_MAX - 1, INT32_MAX}
        for quotient in (0, 1, 2, 1000):
            for side in (-1, 0, 1):
                value = quotient * divisor + divisor // 2 + side
                numerators.update(
                    n for n in (value, -value) if INT32_MIN <= n <= INT32_MAX
                )
        for numerator in sorted(numerators):
            output.append(Case(f"divide-{numerator}-{divisor}", 0, numerator, divisor))
    for index in range(1500):
        output.append(
            Case(
                f"divide-random-{index}",
                0,
                rng.randint(INT32_MIN, INT32_MAX),
                rng.randint(1, INT32_MAX),
            )
        )
    for low, high in (
        (INT32_MIN, INT32_MAX),
        (-4, 9),
        (0, 0),
        (0, 4095),
        (-32768, 32767),
    ):
        for value in sorted(
            {
                INT32_MIN,
                INT32_MAX,
                low,
                high,
                -1,
                0,
                1,
                max(INT32_MIN, low - 1),
                min(INT32_MAX, high + 1),
            }
        ):
            output.append(Case(f"clamp-{low}-{high}-{value}", 1, value, low, high))
    for index in range(500):
        low, high = sorted(
            (rng.randint(INT32_MIN, INT32_MAX), rng.randint(INT32_MIN, INT32_MAX))
        )
        output.append(
            Case(
                f"clamp-random-{index}", 1, rng.randint(INT32_MIN, INT32_MAX), low, high
            )
        )
    for operation in range(2, 7):
        leaf = LEAVES[operation]
        low, high = (-32768, 32767) if leaf["signed"] else (0, 65535)
        arrays = [
            tuple([low] * 9),
            tuple([high] * 9),
            tuple(low if i % 2 else high for i in range(9)),
            tuple((high - low) * i // 8 + low for i in range(9)),
            tuple((high - low) * (8 - i) // 8 + low for i in range(9)),
            (0, 1, 0, 3, 2, 1, 3, 0, 0),
            *(tuple(rng.randint(low, high) for _ in range(9)) for _ in range(3)),
        ]
        for number, samples in enumerate(arrays):
            for index in range((len(samples) - 1) << leaf["shift"]):
                output.append(
                    Case(
                        f"interpolate-{operation}-{number}-{index}",
                        operation,
                        index,
                        samples=samples,
                    )
                )
        for index in (-1, 8 << leaf["shift"], INT32_MAX):
            output.append(
                Case(
                    f"rejected-index-{operation}-{index}",
                    operation,
                    index,
                    samples=arrays[0],
                )
            )
        output.append(Case(f"rejected-empty-{operation}", operation, 0))
        output.append(
            Case(f"rejected-singleton-{operation}", operation, 0, samples=(0,))
        )
    output.extend(
        (
            Case("rejected-zero-divisor", 0, 1, 0),
            Case("rejected-negative-divisor", 0, 1, -1),
            Case("rejected-reversed-clamp", 1, 0, 1, -1),
            Case("rejected-unknown-operation", 7, 0),
        )
    )
    return output


def encode(inputs: list[Case]) -> bytes:
    records = [b"NXFM001\0", struct.pack("<I", len(inputs))]
    for case in inputs:
        records.append(
            struct.pack(
                "<IiiiI", case.operation, case.a, case.b, case.c, len(case.samples)
            )
        )
        records.append(
            struct.pack(
                f"<{len(case.samples)}H", *(value & 65535 for value in case.samples)
            )
        )
    return b"".join(records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dll", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--wine-prefix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if digest(args.dll) != DLL_HASH:
        parser.error("Unexpected Nikon DLL hash.")
    args.output.mkdir(exist_ok=False)
    inputs = cases()
    path = args.output / "cases.bin"
    with path.open("xb") as stream:
        stream.write(encode(inputs))
    predictions = [expected(case) for case in inputs]
    input_hash = digest(path)
    report = {
        "evidence_kind": "live_vendor_integer_leaf_comparison_synthetic_inputs",
        "passed": False,
        "seed": SEED,
        "expected_cases": len(inputs),
        "vendor_calls_expected": sum(value is not None for value in predictions),
        "rejected_cases_expected": sum(value is None for value in predictions),
        "cases_by_operation": dict(Counter(case.operation for case in inputs)),
        "dll_sha256": DLL_HASH,
        "probe_sha256": digest(args.probe),
        "runner_sha256": digest(Path(__file__)),
        "model_sha256": digest(Path(__file__).with_name("flexible_math.py")),
        "probe_source_sha256": digest(
            Path(__file__).with_name("probe_flexible_math.c")
        ),
        "input_sha256": input_hash,
        "leaves": LEAVES,
        "limits": (
            "Integer arithmetic only; no image rendering, color parity "
            "or camera acceptance."
        ),
    }
    stdout, stderr = "", ""
    try:
        result = subprocess.run(
            [
                "/usr/lib/wine/wine64",
                str(args.probe.resolve()),
                windows(args.dll),
                windows(path),
            ],
            cwd=args.dll.resolve().parent,
            env=dict(
                os.environ, WINEPREFIX=str(args.wine_prefix.resolve()), WINEDEBUG="-all"
            ),
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        stdout, stderr = result.stdout, result.stderr
        report["returncode"] = result.returncode
        events = [
            json.loads(line) for line in stdout.splitlines() if line.startswith("{")
        ]
        actual = [event for event in events if event.get("event") == "case"]
        completions = [event for event in events if event.get("event") == "complete"]
        if (
            result.returncode
            or len(actual) != len(inputs)
            or completions != [{"event": "complete", "cases": len(inputs)}]
        ):
            raise ValueError("Vendor oracle did not complete the case stream.")
        differences = []
        guards_unchanged = True
        for index, (case, predicted, event) in enumerate(
            zip(inputs, predictions, actual, strict=True)
        ):
            guards_unchanged &= event.get("input_and_guards_unchanged") is True
            if (
                event.get("index") != index
                or event.get("accepted") is not (predicted is not None)
                or (predicted is not None and event.get("result") != predicted)
            ):
                differences.append(
                    {
                        "index": index,
                        "case": case._asdict(),
                        "expected": predicted,
                        "actual": event,
                    }
                )
        report.update(
            cases_compared=len(actual),
            different_cases=len(differences),
            first_differences=differences[:20],
            input_and_guards_unchanged=guards_unchanged,
            input_file_unchanged=digest(path) == input_hash,
            dll_unchanged=digest(args.dll) == DLL_HASH,
        )
        report["passed"] = (
            not differences
            and guards_unchanged
            and report["input_file_unchanged"]
            and report["dll_unchanged"]
        )
    except (Exception, KeyboardInterrupt) as exc:
        if isinstance(exc, subprocess.TimeoutExpired):
            stdout = exc.stdout or b""
            stderr = exc.stderr or b""
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    for filename, content in (("stdout.jsonl", stdout), ("stderr.txt", stderr)):
        with (args.output / filename).open("xb") as stream:
            stream.write(content.encode() if isinstance(content, str) else content)
    with (args.output / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "passed": report["passed"]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
