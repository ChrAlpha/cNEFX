"""Compare recovered Kaiji matrix initialization over complete 33-cube lattices."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import subprocess
from pathlib import Path

import numpy as np
from verify_np3_header import DLL_HASH, digest, windows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dll", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--table-sha256", required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--wine-prefix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rgb-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.output.exists() or args.output.resolve().is_relative_to(root):
        parser.error("Output must be a new directory outside the repository.")
    if digest(args.dll) != DLL_HASH:
        parser.error("Unexpected Rome2.dll hash.")
    table = args.table.read_bytes()
    if len(table) != 0x34370 or hashlib.sha256(table).hexdigest() != args.table_sha256:
        parser.error("Unexpected table size or hash.")
    sources = [
        args.dll,
        args.table,
        args.probe,
        Path(__file__),
        root / "tools/probe_kaiji_scalar_mode.c",
    ]
    before = {str(path.resolve()): digest(path) for path in sources}
    args.output.mkdir(parents=True)
    report = {
        "evidence_kind": (
            "live_vendor_scalar_rgb_capture"
            if args.rgb_only
            else "live_vendor_scalar_matrix_initialization_controls"
        ),
        "output_scope": "rgb" if args.rgb_only else "rgb_and_lab",
        "input_hashes": before,
        "grid": 33,
        "selector": 0,
        "plane_format": "Three planar little-endian uint16 channels; (r*33+g)*33+b",
        "cases": [],
        "checks": {},
        "passed": False,
        "limits": [
            "Recovered scalar context only; the application factory is bypassed.",
            "Matrix and mode controls do not establish native rendering parity.",
            "The supplied table may omit NP3 scalar settings and custom tone curves.",
        ],
    }
    planes = {}
    try:
        flags_to_test = (1,) if args.rgb_only else (0, 1)
        for processing, flags, matrix in itertools.product(
            (0, 2), flags_to_test, (0, 1)
        ):
            output = args.output / f"p{processing}-f{flags}-m{matrix}.bin"
            result = subprocess.run(
                [
                    "/usr/lib/wine/wine64",
                    str(args.probe.resolve()),
                    windows(args.dll),
                    windows(args.table),
                    windows(output),
                    str(flags),
                    str(matrix),
                    str(processing),
                ],
                cwd=args.dll.resolve().parent,
                env=dict(
                    os.environ,
                    WINEPREFIX=str(args.wine_prefix.resolve()),
                    WINEDEBUG="-all",
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
            case = {
                "processing": processing,
                "flags": flags,
                "matrix": matrix,
                "returncode": result.returncode,
                "events": events,
                "stderr": result.stderr,
                "passed": False,
            }
            report["cases"].append(case)
            if result.returncode or len(events) != 1:
                raise ValueError("Scalar probe failed.")
            event = events[0]
            if not all(
                event.get(key) is True
                for key in ("guards_intact", "inputs_unchanged", "success")
            ):
                raise ValueError("Probe reported a failed guard or output check.")
            expected = {
                "event": "kaiji_scalar_lattice",
                "processing": processing,
                "flags": flags,
                "matrix": matrix,
                "selector": 0,
                "grid": 33,
                "calls": 33**3,
            }
            if any(event.get(key) != value for key, value in expected.items()):
                raise ValueError("Unexpected scalar probe configuration or coverage.")
            data = output.read_bytes()
            if len(data) != 3 * 33**3 * 2:
                raise ValueError("Unexpected output plane size.")
            values = np.frombuffer(data, dtype="<u2").reshape(3, 33**3)
            planes[processing, flags, matrix] = values
            case.update(
                sha256=hashlib.sha256(data).hexdigest(),
                minimum=values.min(axis=1).tolist(),
                maximum=values.max(axis=1).tolist(),
                center=values[:, (16 * 33 + 16) * 33 + 16].tolist(),
                passed=True,
            )
        checks = report["checks"]
        for processing in (0, 2):
            checks[f"p{processing}_rgb_matrix_invariant"] = np.array_equal(
                planes[processing, 1, 0], planes[processing, 1, 1]
            )
            checks[f"p{processing}_rgb_in_range"] = bool(
                np.all(planes[processing, 1, 1] <= 0xFF00)
            )
            if not args.rgb_only:
                checks[f"p{processing}_zero_matrix_lab"] = bool(
                    np.all(
                        planes[processing, 0, 0] == np.array([0, 32768, 32768])[:, None]
                    )
                )
                checks[f"p{processing}_initialized_lab_varies"] = bool(
                    np.all(np.ptp(planes[processing, 0, 1], axis=1) > 0)
                )
        checks["processing_modes_equal"] = all(
            np.array_equal(planes[0, flags, matrix], planes[2, flags, matrix])
            for flags, matrix in itertools.product(flags_to_test, (0, 1))
        )
        checks["sources_unchanged"] = all(
            digest(Path(path)) == value for path, value in before.items()
        )
        report["passed"] = len(report["cases"]) == 4 * len(flags_to_test) and all(
            checks.values()
        )
    except (Exception, KeyboardInterrupt) as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    with (args.output / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {"report": str(args.output / "report.json"), "passed": report["passed"]}
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
