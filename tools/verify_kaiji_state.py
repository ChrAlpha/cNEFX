"""Drive the vendor Kaiji generator from a normalized public NP3 state."""

from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
from pathlib import Path

from inspect_np3 import parse
from np3_parameters import normalize_flexible, parameter_blocks
from verify_np3_header import digest, windows

ROME_HASH = "45c8b91ddd61a103711d755a57c45db4ce8431b9ec9ce7a1f4fac28fe25582e2"
OUTPUT_BYTES = 0x34370


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--np3", type=Path, required=True)
    parser.add_argument("--dll", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--wine-prefix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if digest(args.dll) != ROME_HASH:
        parser.error("Unexpected Rome2.dll hash.")
    if args.output.exists() or args.output.resolve().is_relative_to(
        Path(__file__).resolve().parents[1]
    ):
        parser.error("Output must be a new path outside the repository.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = parse(args.np3.read_bytes())
    blocks = parameter_blocks(records)
    returned, normalized = normalize_flexible(blocks["flexible"])
    state_path = args.output.with_suffix(".state.bin")
    state_path.write_bytes(struct.pack("<41i", *normalized))
    result = subprocess.run(
        [
            "/usr/lib/wine/wine64",
            str(args.probe.resolve()),
            windows(args.dll),
            windows(state_path),
            windows(args.output),
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
        json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")
    ]
    report = {
        "evidence_kind": "live_unmodified_vendor_np3_state_to_kaiji_tables",
        "input_sha256": digest(args.np3),
        "rome2_sha256": ROME_HASH,
        "probe_sha256": digest(args.probe),
        "runner_sha256": digest(Path(__file__)),
        "normalized_return": returned,
        "normalized_values": normalized,
        "returncode": result.returncode,
        "events": events,
        "stderr": result.stderr,
        "passed": False,
        "limits": [
            "State mapping uses the recovered 41-value order and a synthetic "
            "camera/profile context.",
            "Output tables remain opaque external artifacts; no rendered pixels are "
            "compared.",
        ],
    }
    if result.returncode == 0 and len(events) == 1 and args.output.exists():
        report.update(
            output_bytes=args.output.stat().st_size,
            output_sha256=digest(args.output),
        )
        report["passed"] = (
            events[0].get("success") is True
            and events[0].get("guards_intact") is True
            and report["output_bytes"] == OUTPUT_BYTES
        )
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
