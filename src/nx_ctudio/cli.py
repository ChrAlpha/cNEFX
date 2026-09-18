"""Command-line entry point for NX Ctudio."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .inspect import inspect_file


def emit(report: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(report, indent=2, ensure_ascii=True))
        return
    for item in report["files"]:
        print(f"{item['status']}: {item['path']}")
        if item["status"] == "error":
            print(f"  {item['error']['code']}: {item['error']['message']}")
            continue
        print(f"  SHA-256: {item['sha256']}")
        for key, value in item["metadata"].items():
            print(f"  {key}: {value}")
    if "error" in report:
        print(f"nxc: {report['error']['message']}", file=sys.stderr)
    print(
        "Metadata inspection only; RAW decoding and sequence validation not performed."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="nxc", description="NX Ctudio: independent Nikon RAW workflow tools."
    )
    parser.add_argument(
        "--version", action="version", version=f"NX Ctudio {__version__}"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser(
        "inspect",
        help="Inspect file metadata and hashes without modifying inputs.",
        description=(
            "Inspect explicit input files in the supplied order. Requires ExifTool. "
            "Numeric metadata values are preserved as reported by ExifTool. "
            "This command does not decode RAW pixels or validate "
            "a pixel-shift sequence."
        ),
    )
    inspect.add_argument(
        "files", nargs="+", type=Path, help="Explicit input file paths."
    )
    inspect.add_argument(
        "--json", action="store_true", help="Write a JSON report to stdout."
    )
    merge = commands.add_parser(
        "merge",
        help="Experimentally synthesize ordered Z8 frames into native NEFX.",
        description=(
            "Native experimental Z8 pixel-shift synthesis. Supply a complete sequence "
            "in capture order. HE/HE*, radial correction, color rendering and "
            "Adobe default updates are not supported. Output is never overwritten."
        ),
    )
    merge.add_argument("files", nargs="+", type=Path)
    merge.add_argument("--output", "-o", type=Path, required=True)
    merge.add_argument("--json", action="store_true")
    picture_curve = commands.add_parser(
        "picture-curve",
        help="Regenerate an NP3 tone curve table.",
        description=(
            "Read a bounded NP3 container and regenerate its 257-entry tone curve "
            "with the native curve implementation."
        ),
    )
    picture_curve.add_argument("file", type=Path)
    picture_curve.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "merge":
        from .merge import merge as synthesize

        try:
            report = synthesize(args.files, args.output)
            exit_code = 0
        except (Exception, KeyboardInterrupt) as exc:
            exit_code = 130 if isinstance(exc, KeyboardInterrupt) else 1
            report = {
                "status": "error",
                "backend": "native",
                "error": {
                    "code": "cancelled" if exit_code == 130 else "merge_failed",
                    "message": str(exc) or "Synthesis cancelled.",
                },
            }
        if args.json:
            print(json.dumps(report, indent=2, ensure_ascii=True))
        elif exit_code:
            print(f"nxc: {report['error']['message']}", file=sys.stderr)
        else:
            print(f"ok: {report['output']}")
            print(f"  SHA-256: {report['sha256']}")
            print(f"  Samples verified: {report['samples_verified']}")
            for limitation in report["limitations"]:
                print(f"  {limitation}")
            for warning in report["warnings"]:
                print(f"nxc: {warning}", file=sys.stderr)
        return exit_code
    if args.command == "picture-curve":
        from .picture_control import extract_curve, read_np3

        report: dict[str, Any] = {"schema_version": 1, "status": "error"}
        try:
            path = args.file.resolve(strict=True)
            report.update(
                status="ok",
                path=str(path),
                picture_control=extract_curve(read_np3(path)),
            )
            exit_code = 0
        except (OSError, ValueError) as exc:
            report["error"] = {"code": "picture_control_error", "message": str(exc)}
            exit_code = 1
        if args.json:
            print(json.dumps(report, indent=2, ensure_ascii=True))
        elif exit_code:
            print(f"nxc: {report['error']['message']}", file=sys.stderr)
        else:
            print(f"ok: {report['path']}")
            print(f"  Picture Control: {report['picture_control']['name']}")
            print(f"  Profile ID: {report['picture_control']['profile_id']}")
            print("  Curve samples: 257")
        return exit_code
    report: dict[str, Any] = {
        "schema_version": 1,
        "tool": {"name": "NX Ctudio", "version": __version__},
        "raw_decode": "not_performed",
        "sequence_validation": "not_performed",
        "files": [],
    }
    executable = shutil.which("exiftool")
    if not executable:
        report["error"] = {
            "code": "dependency_missing",
            "message": "ExifTool is required. Install it and add exiftool to PATH.",
        }
        emit(report, args.json)
        return 2
    try:
        version = subprocess.run(
            [executable, "-config", "", "-ver"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        report["error"] = {"code": "dependency_error", "message": str(exc)}
        emit(report, args.json)
        return 2
    report["tool"]["exiftool_version"] = version
    try:
        for path in args.files:
            report["files"].append(inspect_file(path, executable))
    except KeyboardInterrupt:
        report["error"] = {"code": "cancelled", "message": "Inspection cancelled."}
        emit(report, args.json)
        return 130
    emit(report, args.json)
    return 0 if all(item["status"] == "ok" for item in report["files"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
