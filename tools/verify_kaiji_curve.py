"""Compare native 12-bit NP3 curves with the vendor Kaiji replacement branch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
from pathlib import Path

from flexible_math import interpolate
from inspect_np3 import parse
from kaiji_oracle import dependency_hashes, loaded_provider_matches
from verify_np3_header import DLL_HASH, digest, windows
from verify_picture_curve import MATRIX_SHA256, ORACLE_SHA256, read_report

from nx_ctudio.picture_curve import generate_curve


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "dll",
        "probe",
        "wine-prefix",
        "samples",
        "oracle",
        "matrix",
        "matrix-dir",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.output.exists() or args.output.resolve().is_relative_to(root):
        parser.error("Output must be a new directory outside the repository.")
    if digest(args.dll) != DLL_HASH:
        parser.error("Unexpected Rome2.dll hash.")
    public, _ = read_report(args.oracle, ORACLE_SHA256, 5)
    synthetic, _ = read_report(args.matrix, MATRIX_SHA256, 18)
    inputs = [(case, args.samples / case["name"]) for case in public]
    inputs.extend((case, args.matrix_dir / f"{case['name']}.NP3") for case in synthetic)
    paths = [
        args.dll,
        args.probe,
        Path(__file__),
        root / "tools/probe_kaiji_curve.c",
        Path(generate_curve.__code__.co_filename),
        root / "tools/flexible_math.py",
        root / "tools/inspect_np3.py",
        root / "tools/kaiji_oracle.py",
    ]
    hashes = {str(path.resolve()): digest(path) for path in paths}
    hashes.update(dependency_hashes(args.dll, args.probe))
    report = {
        "evidence_kind": "native_custom_curve_vs_vendor_kaiji_replacement",
        "input_hashes": hashes,
        "oracle_sha256": ORACLE_SHA256,
        "matrix_sha256": MATRIX_SHA256,
        "cases": [],
        "passed": False,
        "limits": [
            "Public NP3 curve import and recovered internal curve-list view.",
            "Complete application state and rendered images are not compared.",
            "Extreme-control comparison covers four tone controls only.",
        ],
    }
    args.output.mkdir(parents=True)
    try:
        for source, path in inputs:
            snapshot = path.read_bytes()
            source_hash = hashlib.sha256(snapshot).hexdigest()
            if source_hash != source["source_sha256"]:
                raise ValueError("NP3 source differs from the pinned input.")
            payload = next(
                record.payload for record in parse(snapshot) if record.identifier == 2
            )
            nodes = generate_curve(payload, sample_max=4095)
            expanded = tuple(
                interpolate(nodes, i, interval_shift=4) for i in range(4096)
            )
            baseline = None
            for mode in ("neutral", "extreme"):
                output = args.output / f"{source['name']}-{mode}.bin"
                result = subprocess.run(
                    [
                        "/usr/lib/wine/wine64",
                        str(args.probe.resolve()),
                        windows(args.dll),
                        windows(path),
                        windows(output),
                        mode,
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
                    "name": source["name"],
                    "mode": mode,
                    "source_sha256": source_hash,
                    "returncode": result.returncode,
                    "events": events,
                    "stderr": result.stderr,
                    "passed": False,
                }
                report["cases"].append(case)
                if result.returncode or len(events) != 1:
                    raise ValueError("Custom-curve oracle failed.")
                event = events[0]
                if (
                    event.get("event") != "kaiji_custom_curve"
                    or event.get("mode") != mode
                    or event.get("status") != 0
                    or event.get("guards_intact") is not True
                    or event.get("success") is not True
                    or event.get("output_bytes") != 0x34370
                    or not loaded_provider_matches(event, args.dll)
                ):
                    raise ValueError("Custom-curve oracle status or guard failure.")
                data = output.read_bytes()
                if len(data) != 0x34370 or path.read_bytes() != snapshot:
                    raise ValueError("Output size or source stability failure.")
                actual = (
                    struct.unpack_from("<257H", data, 0x2E044),
                    struct.unpack_from("<4096H", data, 0x2E246),
                )
                differences = [
                    sum(a != b for a, b in zip(n, v, strict=True))
                    for n, v in zip((nodes, expanded), actual, strict=True)
                ]
                if mode == "neutral":
                    baseline = actual
                case.update(
                    output_sha256=hashlib.sha256(data).hexdigest(),
                    differences=differences,
                    tone_controls_bypassed=actual == baseline,
                    passed=not any(differences) and actual == baseline,
                )
        report["sources_unchanged"] = all(
            digest(Path(p)) == h for p, h in hashes.items()
        )
        report["compared_words"] = len(report["cases"]) * (257 + 4096)
        report["passed"] = (
            len(report["cases"]) == 46
            and report["sources_unchanged"]
            and all(case["passed"] for case in report["cases"])
        )
    except (Exception, KeyboardInterrupt) as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    with (args.output / "report.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {"report": str(args.output / "report.json"), "passed": report["passed"]}
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
