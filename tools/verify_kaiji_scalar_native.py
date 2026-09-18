"""Compare native scalar RGB evaluation with a pinned complete vendor lattice."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from kaiji_scalar import ScalarTable


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
    oracle = json.loads(snapshot)
    count = 4 if oracle.get("output_scope") == "rgb" else 8
    if (
        oracle.get("passed") is not True
        or oracle.get("grid") != 33
        or len(oracle.get("cases", [])) != count
    ):
        parser.error("Incomplete scalar lattice oracle.")
    table = args.table.read_bytes()
    table_hash = hashlib.sha256(table).hexdigest()
    if oracle["input_hashes"].get(str(args.table.resolve())) != table_hash:
        parser.error("Unexpected table bytes.")
    implementation = Path(ScalarTable.evaluate.__code__.co_filename)
    before = hashlib.sha256(implementation.read_bytes()).hexdigest()
    evaluator = ScalarTable(table)
    actual = np.array(
        [evaluator.evaluate(point) for point in itertools.product(range(33), repeat=3)],
        dtype="<u2",
    ).T
    differences, maximum, plane_hashes = 0, 0, {}
    selected = []
    for case in oracle["cases"]:
        if case["flags"] != 1:
            continue
        key = case["processing"], case["matrix"]
        if key in selected or case.get("passed") is not True:
            parser.error("Repeated or unsuccessful RGB case.")
        selected.append(key)
        name = f"p{key[0]}-f1-m{key[1]}.bin"
        data = (args.oracle.parent / name).read_bytes()
        if (
            len(data) != actual.nbytes
            or hashlib.sha256(data).hexdigest() != case["sha256"]
        ):
            parser.error("Unexpected RGB plane bytes.")
        expected = np.frombuffer(data, dtype="<u2").reshape(actual.shape)
        delta = actual.astype(np.int32) - expected.astype(np.int32)
        differences += int(np.count_nonzero(delta))
        maximum = max(maximum, int(np.max(np.abs(delta))))
        plane_hashes[name] = case["sha256"]
    if set(selected) != set(itertools.product((0, 2), (0, 1))):
        parser.error("Incomplete RGB mode/matrix coverage.")
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
