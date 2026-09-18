"""Compare public NP3 headers with the unmodified Nikon metadata parser."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from inspect_np3 import describe, parse

DLL_HASH = "45c8b91ddd61a103711d755a57c45db4ce8431b9ec9ce7a1f4fac28fe25582e2"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def windows(path: Path) -> str:
    return "Z:" + str(path.resolve()).replace("/", "\\")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", type=Path, nargs="+")
    parser.add_argument("--dll", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--wine-prefix", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if digest(args.dll) != DLL_HASH:
        parser.error("DLL differs from the analyzed version.")
    if args.report.exists():
        parser.error("Report already exists.")
    report = {
        "evidence_kind": "live_vendor_np3_header_oracle_public_samples",
        "dll_sha256": DLL_HASH,
        "probe_sha256": digest(args.probe),
        "runner_sha256": digest(Path(__file__)),
        "cases": [],
        "passed": False,
        "limits": (
            "Header parsing only; no full import, camera acceptance or rendering."
        ),
    }
    try:
        for path in args.files:
            before = digest(path)
            data = path.read_bytes()
            records = parse(data)
            native = describe(records)
            result = subprocess.run(
                [
                    "/usr/lib/wine/wine64",
                    str(args.probe.resolve()),
                    windows(args.dll),
                    windows(path),
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
            expected_name = next(
                r.payload.hex() for r in records if r.identifier == 0x200
            )
            checks = {
                "returncode": result.returncode == 0,
                "one_event": len(events) == 1,
                "source_unchanged": digest(path) == before,
            }
            if len(events) == 1:
                checks.update(
                    status=events[0].get("status") == 0,
                    version_mask=events[0].get("version_mask") == 64,
                    profile_id=events[0].get("profile_id") == native["profile_id"],
                    name=events[0].get("name_hex") == expected_name,
                )
            report["cases"].append(
                {
                    "path": str(path.resolve()),
                    "sha256": before,
                    "checks": checks,
                    "events": events,
                    "stderr": result.stderr,
                    "passed": all(checks.values()),
                }
            )
        report["passed"] = len(report["cases"]) == len(args.files) and all(
            item["passed"] for item in report["cases"]
        )
    except (Exception, KeyboardInterrupt) as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    with args.report.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"report": str(args.report), "passed": report["passed"]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
