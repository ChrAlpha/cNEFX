"""Replay saved NP3 vendor oracle reports against the native curve generator."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nx_ctudio.picture_curve import generate_table

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inspect_np3 import parse


def digest(path: Path) -> str:
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


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
    cases = []
    for source in oracle["cases"]:
        path = args.samples / source["name"]
        payload = next(
            record.payload
            for record in parse(path.read_bytes())
            if record.identifier == 2
        )
        cases.append(
            {
                "name": source["name"],
                "source_sha256": digest(path),
                "differences": sum(
                    a != b
                    for a, b in zip(
                        generate_table(payload), source["event"]["table"], strict=True
                    )
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
        matrix_cases.append(
            {
                "name": source["name"],
                "source_sha256": digest(path),
                "differences": sum(
                    a != b
                    for a, b in zip(
                        generate_table(payload), source["event"]["table"], strict=True
                    )
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
    args.output.write_text(json.dumps(report, indent=2) + "\n")
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
