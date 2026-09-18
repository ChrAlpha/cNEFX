"""Compare native 12-bit NP3 curves with the vendor Kaiji replacement branch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

from flexible_math import interpolate
from inspect_np3 import Record, parse, serialize
from kaiji_grading import COEFFICIENT_HASH
from kaiji_oracle import dependency_hashes, loaded_provider_matches
from kaiji_table import generate_table
from np3_parameters import normalize_flexible, parameter_blocks
from verify_kaiji_composition import controls
from verify_np3_header import DLL_HASH, digest, windows
from verify_picture_curve import MATRIX_SHA256, ORACLE_SHA256, read_report

from nx_ctudio.picture_curve import generate_curve


def mixed_snapshot(snapshot: bytes, values: list[int]) -> bytes:
    replacements = {
        0x1900 + index * 256: bytes((128 + values[index], 1)) for index in range(6)
    }
    replacements[0x1F00] = bytes(128 + v for v in values[6:30]) + bytes((1, 1, 1, 0))
    replacements[0x2000] = b"".join(
        (32768 + values[30 + i * 3]).to_bytes(2, "big")
        + bytes((128 + values[31 + i * 3], 128 + values[32 + i * 3]))
        for i in range(3)
    ) + bytes((1, 1, 1, 0, 128 + values[39], 1, 128 + values[40], 1))
    changed = serialize(
        tuple(
            Record(r.identifier, replacements.get(r.identifier, r.payload))
            for r in parse(snapshot)
        )
    )
    _, actual = normalize_flexible(parameter_blocks(parse(changed))["flexible"])
    if actual != values:
        raise ValueError("Mixed NP3 mutation does not preserve requested controls.")
    return changed


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
    parser.add_argument("--complete-native", action="store_true")
    parser.add_argument("--coefficients", type=Path)
    args = parser.parse_args()
    if args.complete_native and args.coefficients is None:
        parser.error("Complete native comparison requires external coefficients.")
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
        Path(read_report.__code__.co_filename),
        root / "tools/flexible_math.py",
        root / "tools/inspect_np3.py",
        root / "tools/kaiji_oracle.py",
    ]
    hashes = {str(path.resolve()): digest(path) for path in paths}
    hashes.update(dependency_hashes(args.dll, args.probe))
    coefficients = None
    if args.complete_native:
        coefficients = args.coefficients.read_bytes()
        coefficient_hash = hashlib.sha256(coefficients).hexdigest()
        if coefficient_hash != COEFFICIENT_HASH:
            parser.error("Unexpected coefficient resource hash.")
        hashes[str(args.coefficients.resolve())] = coefficient_hash
        for name in (
            "kaiji_table",
            "kaiji_camera",
            "kaiji_blender",
            "kaiji_contrast",
            "kaiji_grading",
            "kaiji_input",
            "kaiji_tone",
            "np3_parameters",
            "verify_kaiji_composition",
        ):
            path = Path(sys.modules[name].__file__).resolve()
            hashes[str(path)] = digest(path)
    report = {
        "evidence_kind": (
            "native_np3_parameters_and_curve_vs_vendor_tables"
            if args.complete_native
            else "native_custom_curve_vs_vendor_kaiji_replacement"
        ),
        "input_hashes": hashes,
        "oracle_sha256": ORACLE_SHA256,
        "matrix_sha256": MATRIX_SHA256,
        "cases": [],
        "passed": False,
        "limits": [
            "Public NP3 curve import and recovered internal curve-list view.",
            "Complete application state and rendered images are not compared.",
            "Extreme-control comparison covers four tone controls only.",
            "Normalized parameter assignment bypasses full application initialization.",
            "Camera calibration fields stay zero; no source-image rendering.",
        ],
    }
    args.output.mkdir(parents=True)
    try:
        if args.complete_native:
            source, path = inputs[0]
            snapshot = path.read_bytes()
            if hashlib.sha256(snapshot).hexdigest() != source["source_sha256"]:
                raise ValueError("Unexpected base NP3 snapshot.")
            for index, values in enumerate(controls()[1:9]):
                changed = mixed_snapshot(snapshot, values)
                target = args.output / f"mixed-{index}.NP3"
                with target.open("xb") as stream:
                    stream.write(changed)
                inputs.append(
                    (
                        {
                            "name": f"mixed-{index}",
                            "source_sha256": hashlib.sha256(changed).hexdigest(),
                            "base_sha256": source["source_sha256"],
                            "requested_values": values,
                        },
                        target,
                    )
                )
        for source, path in inputs:
            snapshot = path.read_bytes()
            source_hash = hashlib.sha256(snapshot).hexdigest()
            if source_hash != source["source_sha256"]:
                raise ValueError("NP3 source differs from the pinned input.")
            records = parse(snapshot)
            payloads = [record.payload for record in records if record.identifier == 2]
            if len(payloads) != 1:
                raise ValueError("Expected exactly one NP3 curve record.")
            payload = payloads[0]
            normalized_return, normalized_values = normalize_flexible(
                parameter_blocks(records)["flexible"]
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
                        *(["normalized"] if args.complete_native else []),
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
                    "base_sha256": source.get("base_sha256"),
                    "requested_values": source.get("requested_values"),
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
                if args.complete_native and (
                    event.get("normalized") is not True
                    or event.get("normalization_return") != normalized_return
                    or event.get("normalized_values") != normalized_values
                ):
                    raise ValueError("Unexpected vendor NP3 normalization.")
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
                if args.complete_native:
                    values = normalized_values.copy()
                    if mode == "extreme":
                        values[1:5] = [100, -100, 100, -100]
                    native = generate_table(coefficients, values, curve_payload=payload)
                    if type(native) is not bytes or len(native) != 0x34370:
                        raise ValueError("Incomplete native NP3 table.")
                    different_bytes = sum(
                        a != b for a, b in zip(native, data, strict=True)
                    )
                    with (args.output / f"{source['name']}-{mode}.native.bin").open(
                        "xb"
                    ) as stream:
                        stream.write(native)
                    case.update(
                        native_sha256=hashlib.sha256(native).hexdigest(),
                        different_bytes=different_bytes,
                        compared_bytes=len(native),
                        passed=case["passed"] and different_bytes == 0,
                    )
        report["sources_unchanged"] = all(
            digest(Path(p)) == h for p, h in hashes.items()
        )
        report["compared_words"] = len(report["cases"]) * (257 + 4096)
        report["passed"] = (
            len(report["cases"]) == (62 if args.complete_native else 46)
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
