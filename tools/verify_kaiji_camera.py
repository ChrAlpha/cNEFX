"""Replay native camera sections against pinned initialization and matrix oracles."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from flexible_math import interpolate, round_divide32
from kaiji_camera import camera_ranges
from kaiji_grading import COEFFICIENT_HASH
from kaiji_oracle import MOON_HASH, RESOURCE_HASH
from verify_np3_header import DLL_HASH


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def snapshots(path: Path, expected_hash: str, kind: str, count: int):
    data = path.read_bytes()
    if digest(data) != expected_hash:
        raise ValueError("Unexpected oracle report hash.")
    oracle = json.loads(data)
    checks = oracle.get("checks", {})
    stable = checks.get("sources_unchanged", oracle.get("sources_unchanged"))
    cases = oracle.get("cases", [])
    if (
        oracle.get("passed") is not True
        or oracle.get("error") is not None
        or oracle.get("evidence_kind") != kind
        or stable is not True
        or any(value is not True for value in checks.values())
        or len(cases) != count
    ):
        raise ValueError("Incomplete or unsuccessful oracle report.")
    hashes = oracle["input_hashes"]
    for basename, expected in (
        ("Rome2.dll", DLL_HASH),
        ("Moon2.dll", MOON_HASH),
        ("prm.bin", RESOURCE_HASH),
    ):
        matches = [v for p, v in hashes.items() if Path(p).name == basename]
        if matches != [expected]:
            raise ValueError("Oracle has missing or unexpected vendor dependencies.")
    names = set()
    for case in cases:
        name = case.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9-]+", name):
            raise ValueError("Unsupported oracle case name.")
        if name in names:
            raise ValueError("Repeated oracle case.")
        names.add(name)
        events = case.get("events", [])
        if (
            case.get("passed") is not True
            or type(case.get("returncode")) is not int
            or case["returncode"] != 0
            or len(events) != 1
        ):
            raise ValueError("Unsuccessful oracle case.")
        event = events[0]
        expected = {
            "event": "kaiji_state",
            "output_bytes": 0x34370,
            "model": 14,
            "guards_intact": True,
            "success": True,
        }
        if any(
            type(event.get(k)) is not type(v) or event[k] != v
            for k, v in expected.items()
        ):
            raise ValueError("Unexpected oracle status, guards, or model.")
        moon_path = event.get("moon_path")
        if not isinstance(moon_path, str) or not moon_path.lower().startswith("z:/"):
            raise ValueError("Missing loaded coefficient provider path.")
        if hashes.get(moon_path[2:]) != MOON_HASH:
            raise ValueError("Loaded coefficient provider does not match its pin.")
        state = (path.parent / f"{name}.state.bin").read_bytes()
        table = (path.parent / f"{name}.bin").read_bytes()
        if (
            len(state) != 164
            or digest(state) != case.get("state_sha256")
            or len(table) != 0x34370
            or digest(table) != case.get("sha256", case.get("output_sha256"))
        ):
            raise ValueError("Unexpected state or table snapshot.")
        yield name, event, table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("initialization", "matrix", "coefficients", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("initialization", "matrix"):
        parser.add_argument(f"--{name}-sha256", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.output.exists() or args.output.resolve().is_relative_to(root):
        parser.error("Output must be a new path outside the repository.")
    coefficients = args.coefficients.read_bytes()
    if digest(coefficients) != COEFFICIENT_HASH:
        parser.error("Unexpected coefficient resource hash.")
    sources = [
        Path(__file__),
        Path(camera_ranges.__code__.co_filename),
        Path(interpolate.__code__.co_filename),
        Path(round_divide32.__code__.co_filename),
    ]
    hashes = {str(p.resolve()): digest(p.read_bytes()) for p in sources}
    report = {
        "evidence_kind": "native_camera_sections_vs_saved_vendor_tables",
        "input_reports": {},
        "implementation_hashes": hashes,
        "coefficient_sha256": digest(coefficients),
        "cases": [],
        "passed": False,
        "limits": [
            "Saved-output replay, not fresh vendor execution.",
            "Model14/family12 only, with state+152a/+152c zero.",
            "Partial tables; complete generation and image rendering remain untested.",
        ],
    }
    try:
        for path, report_hash, kind, count in (
            (
                args.initialization,
                args.initialization_sha256,
                "live_vendor_initialization_field_factorial",
                12,
            ),
            (
                args.matrix,
                args.matrix_sha256,
                "native_input_vs_live_vendor_table_generation",
                45,
            ),
        ):
            report["input_reports"][str(path.resolve())] = report_hash
            for name, event, table in snapshots(path, report_hash, kind, count):
                ranges = camera_ranges(
                    coefficients,
                    model_index=event["model"],
                    selector=event["matrix"],
                    saturation_bias=event.get("raw1522", 0),
                )
                mismatches = [
                    sum(
                        a != b
                        for a, b in zip(
                            data, table[offset : offset + len(data)], strict=True
                        )
                    )
                    for offset, data in ranges
                ]
                report["cases"].append(
                    {
                        "name": name,
                        "oracle": str(path.resolve()),
                        "table_sha256": digest(table),
                        "selector": event["matrix"],
                        "saturation_bias": event.get("raw1522", 0),
                        "compared_bytes": sum(len(data) for _, data in ranges),
                        "different_bytes": sum(mismatches),
                        "range_differences": {
                            hex(offset): diff
                            for (offset, _), diff in zip(
                                ranges, mismatches, strict=True
                            )
                        },
                    }
                )
        report["sources_unchanged"] = all(
            digest(Path(p).read_bytes()) == h for p, h in hashes.items()
        )
        report["passed"] = (
            len(report["cases"]) == 57
            and report["sources_unchanged"]
            and all(c["different_bytes"] == 0 for c in report["cases"])
        )
    except (Exception, KeyboardInterrupt) as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": str(args.output), "passed": report["passed"]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
