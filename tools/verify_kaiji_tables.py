"""Run the bounded Rome2 Flexible Color table-generator oracle."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from verify_np3_header import digest, windows

ROME_HASH = "45c8b91ddd61a103711d755a57c45db4ce8431b9ec9ce7a1f4fac28fe25582e2"
OUTPUT_BYTES = 0x34370
OUTPUT_HASH = "b6baa6389b252bda4aad71876b426be2383cde4326e681d10a984861b2060248"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dll", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--wine-prefix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", default="neutral")
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    if digest(args.dll) != ROME_HASH:
        parser.error("Unexpected Rome2.dll hash.")
    if args.output.exists() or args.output.resolve().is_relative_to(
        Path(__file__).resolve().parents[1]
    ):
        parser.error("Output must be a new path outside the repository.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "/usr/lib/wine/wine64",
            str(args.probe.resolve()),
            windows(args.dll),
            windows(args.output),
            args.mode,
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
        "evidence_kind": "live_unmodified_vendor_kaiji_table_generator",
        "rome2_sha256": ROME_HASH,
        "probe_sha256": digest(args.probe),
        "runner_sha256": digest(Path(__file__)),
        "camera_id": 1071,
        "profile_family": 12,
        "mode": args.mode,
        "output_bytes_expected": OUTPUT_BYTES,
        "output_path": str(args.output.resolve()),
        "returncode": result.returncode,
        "events": events,
        "stderr": result.stderr,
        "passed": False,
        "limits": [
            "Neutral synthetic parameter object only; no user NP3 state import.",
            "Generated table bytes are kept outside the repository.",
            "This is table generation evidence, not rendered image parity.",
        ],
    }
    if result.returncode == 0 and len(events) == 1 and args.output.exists():
        data = args.output.read_bytes()
        report.update(
            output_size_bytes=len(data),
            output_sha256=digest(args.output),
            nonzero_bytes=sum(bool(value) for value in data),
        )
        report["passed"] = (
            events[0].get("success") is True
            and events[0].get("guards_intact") is True
            and len(data) == OUTPUT_BYTES
            and (
                args.expected_sha256 is None
                or report["output_sha256"] == args.expected_sha256
            )
        )
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
