"""Replay saved NP3 vendor oracle reports against the native curve generator."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from inspect_np3 import parse

from nx_ctudio.picture_curve import generate_curve

ORACLE_SHA256 = "8908203ebeb6e221e3c54df58e08a198599718f9b8c9d759f0066b5406d3bd9d"
MATRIX_SHA256 = "315c68b51ff1d247c5f32ccda57de2d7327ac7d46972cc792351e182ea5f0165"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def vendor_table(source: dict, path: Path) -> tuple[int, ...]:
    if source.get("returncode") != 0 or source.get("event", {}).get("status") != 0:
        raise ValueError(f"Vendor case {source.get('name')} lacks a successful status.")
    if source["event"].get("guards_intact") is not True:
        raise ValueError(f"Vendor case {source.get('name')} has damaged guards.")
    if source.get("source_sha256") != digest(path):
        raise ValueError(f"Vendor case {source.get('name')} input hash mismatch.")
    table = source["event"].get("table")
    if not isinstance(table, list) or len(table) != 257:
        raise ValueError(f"Vendor case {source.get('name')} has no 257-entry table.")
    return tuple(table)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(
        Path(__file__).resolve().parents[1]
    ):
        parser.error("Output must be a new path outside the repository.")
    oracle = json.loads(args.oracle.read_text())
    matrix = json.loads(args.matrix.read_text())
    if digest(args.oracle) != ORACLE_SHA256 or digest(args.matrix) != MATRIX_SHA256:
        parser.error("Unexpected pinned curve oracle report hash.")
    cases = []
    for source in oracle["cases"]:
        path = args.samples / source["name"]
        payload = next(
            record.payload
            for record in parse(path.read_bytes())
            if record.identifier == 2
        )
        expected = vendor_table(source, path)
        cases.append(
            {
                "name": source["name"],
                "source_sha256": digest(path),
                "differences": sum(
                    a != b
                    for a, b in zip(generate_curve(payload), expected, strict=True)
                ),
            }
        )
    matrix_cases = []
    for source in matrix["cases"]:
        path = args.matrix_dir / f"{source['name']}.NP3"
        payload = next(
            record.payload
            for record in parse(path.read_bytes())
            if record.identifier == 2
        )
        expected = vendor_table(source, path)
        matrix_cases.append(
            {
                "name": source["name"],
                "source_sha256": digest(path),
                "differences": sum(
                    a != b
                    for a, b in zip(generate_curve(payload), expected, strict=True)
                ),
            }
        )
    report = {
        "evidence_kind": "native_np3_curve_replay",
        "oracle_sha256": digest(args.oracle),
        "matrix_sha256": digest(args.matrix),
        "cases": cases + matrix_cases,
        "passed": all(item["differences"] == 0 for item in cases + matrix_cases),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "passed": report["passed"],
                "cases": len(report["cases"]),
            }
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
