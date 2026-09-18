"""Replay saved NP3 vendor oracle reports against the native curve generator."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from inspect_np3 import parse

from nx_ctudio.picture_curve import generate_curve

ORACLE_SHA256 = "8908203ebeb6e221e3c54df58e08a198599718f9b8c9d759f0066b5406d3bd9d"
MATRIX_SHA256 = "315c68b51ff1d247c5f32ccda57de2d7327ac7d46972cc792351e182ea5f0165"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_report(path: Path, expected_hash: str, count: int) -> tuple[list[dict], str]:
    data = path.read_bytes()
    actual_hash = digest(data)
    if actual_hash != expected_hash:
        raise ValueError("Unexpected pinned curve oracle report hash.")
    report = json.loads(data)
    cases = report.get("cases") if isinstance(report, dict) else None
    if not isinstance(cases, list) or len(cases) != count:
        raise ValueError(f"Expected exactly {count} cases in {path.name}.")
    names = [source.get("name") for source in cases if isinstance(source, dict)]
    if len(names) != count or any(not isinstance(name, str) for name in names):
        raise ValueError(f"Malformed case names in {path.name}.")
    if len(set(names)) != count:
        raise ValueError(f"Duplicate case names in {path.name}.")
    return cases, actual_hash


def vendor_table(source: dict, source_hash: str) -> tuple[int, ...]:
    if source.get("returncode") != 0 or source.get("event", {}).get("status") != 0:
        raise ValueError(f"Vendor case {source.get('name')} lacks a successful status.")
    if source["event"].get("guards_intact") is not True:
        raise ValueError(f"Vendor case {source.get('name')} has damaged guards.")
    if source.get("source_sha256") != source_hash:
        raise ValueError(f"Vendor case {source.get('name')} input hash mismatch.")
    table = source["event"].get("table")
    if (
        not isinstance(table, list)
        or len(table) != 257
        or any(type(value) is not int or not 0 <= value <= 32767 for value in table)
    ):
        raise ValueError(f"Vendor case {source.get('name')} has no 257-entry table.")
    return tuple(table)


def replay_cases(sources: list[dict], directory: Path, suffix: str = "") -> list[dict]:
    cases = []
    for source in sources:
        data = (directory / f"{source['name']}{suffix}").read_bytes()
        source_hash = digest(data)
        expected = vendor_table(source, source_hash)
        payloads = [record.payload for record in parse(data) if record.identifier == 2]
        if len(payloads) != 1:
            raise ValueError(f"Vendor case {source['name']} needs exactly one curve.")
        cases.append(
            {
                "name": source["name"],
                "source_sha256": source_hash,
                "differences": sum(
                    a != b
                    for a, b in zip(generate_curve(payloads[0]), expected, strict=True)
                ),
            }
        )
    return cases


def provenance() -> dict:
    root = Path(__file__).resolve().parents[1]
    result = {}
    for name, path in (
        ("runner", Path(__file__)),
        ("generator", Path(generate_curve.__code__.co_filename)),
        ("parser", Path(parse.__code__.co_filename)),
    ):
        result[f"{name}_sha256"] = digest(path.read_bytes())
    result["git_revision"] = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    result["git_worktree_dirty"] = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    return result


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
    try:
        source_provenance = provenance()
        oracle, oracle_hash = read_report(args.oracle, ORACLE_SHA256, 5)
        matrix, matrix_hash = read_report(args.matrix, MATRIX_SHA256, 18)
        cases = replay_cases(oracle, args.samples)
        cases.extend(replay_cases(matrix, args.matrix_dir, ".NP3"))
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    report = {
        "evidence_kind": "native_np3_curve_replay",
        **source_provenance,
        "oracle_sha256": oracle_hash,
        "matrix_sha256": matrix_hash,
        "cases": cases,
        "samples_compared": len(cases) * 257,
        "passed": all(item["differences"] == 0 for item in cases),
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
