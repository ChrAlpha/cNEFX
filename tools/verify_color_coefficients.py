"""Compare the Nikon color engine coefficient API with native resource decoding."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from inspect_prm import decode, records
from verify_np3_header import digest, windows

MOON_HASH = "f02f7499564c14f18933ee2b0bda961c25889f0240a30bfe042fa165ba2e8da0"
PRM_HASH = "8d6f1bf219561f3c7c11c61b9b9e3317a369c9b5be5089ddffd82db96ca7f707"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dll", required=True, type=Path)
    parser.add_argument("--prm", required=True, type=Path)
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--wine-prefix", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if digest(args.dll) != MOON_HASH or digest(args.prm) != PRM_HASH:
        parser.error("Unexpected pinned module or resource hash.")
    if args.output.resolve().is_relative_to(Path(__file__).resolve().parents[1]):
        parser.error("Research artifacts must remain outside the repository.")
    args.output.mkdir(exist_ok=False)
    report = {
        "evidence_kind": "live_vendor_color_coefficient_api_comparison",
        "passed": False,
        "dll_sha256": MOON_HASH,
        "prm_sha256": PRM_HASH,
        "probe_sha256": digest(args.probe),
        "runner_sha256": digest(Path(__file__)),
        "decoder_sha256": digest(Path(__file__).with_name("inspect_prm.py")),
        "engine": 0x1002,
        "camera": 1071,
        "limit": "Coefficient loading only; no rendering or generated color tables.",
    }
    try:
        container = args.prm.read_bytes()
        matches = [
            r for r in records(container) if r["type"] == 0x1002 and r["id"] == 1071
        ]
        if len(matches) != 1:
            raise ValueError("Expected exactly one Z8 coefficient resource.")
        expected = decode(container, matches[0])
        probe = args.output / "probe.exe"
        shutil.copyfile(args.probe, probe)
        shutil.copyfile(args.prm, args.output / "prm.bin")
        output = args.output / "coefficients.bin"
        result = subprocess.run(
            [
                "/usr/lib/wine/wine64",
                str(probe.resolve()),
                windows(args.dll),
                windows(output),
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
        events = [
            json.loads(line)
            for line in result.stdout.splitlines()
            if line.startswith("{")
        ]
        report.update(returncode=result.returncode, events=events, stderr=result.stderr)
        if result.returncode or len(events) != 1 or not events[0].get("success"):
            raise ValueError("Nikon coefficient API failed.")
        actual = output.read_bytes()
        differences = sum(a != b for a, b in zip(expected, actual, strict=True))
        report.update(
            bytes_compared=len(expected),
            differences=differences,
            output_sha256=digest(output),
            passed=differences == 0,
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
