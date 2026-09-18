"""Compare native scalar RGB evaluation with a pinned complete vendor lattice."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from kaiji_scalar import ScalarTable


def validate_oracle(oracle: object) -> list[dict]:
    if not isinstance(oracle, dict):
        raise ValueError("Expected a scalar lattice oracle object.")
    kinds = {
        "live_vendor_scalar_rgb_capture": "rgb",
        "live_vendor_scalar_matrix_initialization_controls": "rgb_and_lab",
    }
    kind = oracle.get("evidence_kind")
    scope = kinds.get(kind) if isinstance(kind, str) else None
    if scope is None or oracle.get("output_scope", scope) != scope:
        raise ValueError("Unsupported scalar oracle evidence kind or output scope.")
    if (
        oracle.get("passed") is not True
        or oracle.get("error") is not None
        or type(oracle.get("grid")) is not int
        or oracle["grid"] != 33
        or type(oracle.get("selector")) is not int
        or oracle["selector"] != 0
    ):
        raise ValueError("Unsuccessful scalar oracle or unsupported grid/selector.")
    required_checks = {"processing_modes_equal", "sources_unchanged"}
    for processing in (0, 2):
        required_checks.update(
            f"p{processing}_{check}"
            for check in ("rgb_matrix_invariant", "rgb_in_range")
        )
        if scope == "rgb_and_lab":
            required_checks.update(
                f"p{processing}_{check}"
                for check in ("zero_matrix_lab", "initialized_lab_varies")
            )
    checks = oracle.get("checks")
    if (
        not isinstance(checks, dict)
        or not required_checks <= checks.keys()
        or any(value is not True for value in checks.values())
    ):
        raise ValueError("Missing or unsuccessful scalar oracle checks.")
    flags = (1,) if scope == "rgb" else (0, 1)
    expected_modes = set(itertools.product((0, 2), flags, (0, 1)))
    cases = oracle.get("cases")
    if not isinstance(cases, list) or len(cases) != len(expected_modes):
        raise ValueError("Incomplete scalar lattice oracle.")
    selected, seen = [], set()
    for case in cases:
        if not isinstance(case, dict) or any(
            type(case.get(key)) is not int
            for key in ("processing", "flags", "matrix", "returncode")
        ):
            raise ValueError("Malformed scalar oracle case.")
        mode = case["processing"], case["flags"], case["matrix"]
        if mode not in expected_modes or mode in seen:
            raise ValueError("Repeated or unsupported scalar oracle mode.")
        seen.add(mode)
        events = case.get("events")
        if (
            case.get("passed") is not True
            or case["returncode"] != 0
            or not isinstance(events, list)
            or len(events) != 1
            or not isinstance(events[0], dict)
        ):
            raise ValueError("Unsuccessful scalar oracle case.")
        expected = {
            "event": "kaiji_scalar_lattice",
            "processing": mode[0],
            "flags": mode[1],
            "matrix": mode[2],
            "selector": 0,
            "grid": 33,
            "calls": 33**3,
            "guards_intact": True,
            "inputs_unchanged": True,
            "success": True,
        }
        if any(
            type(events[0].get(key)) is not type(value) or events[0][key] != value
            for key, value in expected.items()
        ):
            raise ValueError("Scalar oracle status, guards or coverage mismatch.")
        if mode[1] == 1:
            selected.append(case)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--oracle-sha256", required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.output.exists() or args.output.resolve().is_relative_to(root):
        parser.error("Output must be a new path outside the repository.")
    snapshot = args.oracle.read_bytes()
    if hashlib.sha256(snapshot).hexdigest() != args.oracle_sha256:
        parser.error("Unexpected oracle report hash.")
    try:
        oracle = json.loads(snapshot)
        selected = validate_oracle(oracle)
    except ValueError as exc:
        parser.error(str(exc))
    table = args.table.read_bytes()
    table_hash = hashlib.sha256(table).hexdigest()
    if (
        not isinstance(oracle.get("input_hashes"), dict)
        or oracle["input_hashes"].get(str(args.table.resolve())) != table_hash
    ):
        parser.error("Unexpected table bytes.")
    implementation = Path(ScalarTable.evaluate.__code__.co_filename)
    before = hashlib.sha256(implementation.read_bytes()).hexdigest()
    evaluator = ScalarTable(table)
    actual = np.array(
        [evaluator.evaluate(point) for point in itertools.product(range(33), repeat=3)],
        dtype="<u2",
    ).T
    differences, maximum, plane_hashes = 0, 0, {}
    for case in selected:
        key = case["processing"], case["matrix"]
        name = f"p{key[0]}-f1-m{key[1]}.bin"
        data = (args.oracle.parent / name).read_bytes()
        if len(data) != actual.nbytes or hashlib.sha256(data).hexdigest() != case.get(
            "sha256"
        ):
            parser.error("Unexpected RGB plane bytes.")
        expected = np.frombuffer(data, dtype="<u2").reshape(actual.shape)
        if np.any(expected > 0xFF00):
            parser.error("Unwritten or out-of-range RGB samples.")
        delta = actual.astype(np.int32) - expected.astype(np.int32)
        differences += int(np.count_nonzero(delta))
        maximum = max(maximum, int(np.max(np.abs(delta))))
        plane_hashes[name] = case["sha256"]
    stable = hashlib.sha256(implementation.read_bytes()).hexdigest() == before
    report = {
        "evidence_kind": "native_scalar_rgb_vs_pinned_vendor_lattice",
        "oracle_sha256": args.oracle_sha256,
        "table_sha256": table_hash,
        "plane_hashes": plane_hashes,
        "implementation_sha256": before,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "implementation_unchanged": stable,
        "points": 33**3,
        "unique_channel_words": int(actual.size),
        "comparisons": len(selected),
        "different_channel_words": differences,
        "max_absolute_error": maximum,
        "passed": stable and differences == 0,
        "limits": [
            "Vendor-generated table input, not a complete native generator.",
            "Synthetic lattice RGB, not image decoding or full NP3 import.",
            "Lab output and other grid sizes are outside this comparison.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
