"""Compare native Z8 source controls with actual SDK TOPParam initialization."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from image_controls import source_controls, verify_top
from kaiji_grading import COEFFICIENT_HASH
from kaiji_index import MODE_TABLE_HASH
from kaiji_oracle import MOON_HASH
from verify_colorbalance import ASTEROID_HASH, STRATO_HASH
from verify_np3_header import DLL_HASH, digest, windows

from nx_ctudio import _tiff

PINNED = {
    "Rome2.dll": DLL_HASH,
    "Asteroid11.dll": ASTEROID_HASH,
    "Strato10.dll": STRATO_HASH,
    "Moon2.dll": MOON_HASH,
    "CML10.dll": "42d71afbe2d6e95d39a71ef7a6a4dfb2819e8d923c7b7e8d8b86d1166687df6f",
    "Services/metadata.n5m": (
        "3e1299471c257a48fbe595782a171d14a38a7a6263b070c06cb02c9983b16390"
    ),
    "Services/picture_control.n5m": (
        "485a56e5be41dfeaece9426b026e2f843f6e6cab8372aeca72252f141f246484"
    ),
}


def provider_paths(
    events: list, hashes: dict[str, str], vendor: Path, wine_prefix: Path
) -> list[str]:
    loaded = []
    names = {Path(path).name.casefold(): path for path in hashes}
    for event in events:
        if event.get("event") != "provider":
            continue
        value = bytes.fromhex(event["path_hex"]).decode("ascii").replace("\\", "/")
        name = value.rsplit("/", 1)[-1].casefold()
        if name not in names:
            continue
        expected = names[name]
        target = Path(expected)
        drive = wine_prefix.resolve() / "drive_c"
        canonical = (
            "C:/" + target.relative_to(drive).as_posix()
            if target.is_relative_to(drive)
            else windows(target)
        )
        if value.casefold() != canonical.replace("\\", "/").casefold():
            raise ValueError(f"Unexpected loaded provider: {name}")
        loaded.append(expected)
    required = ["Rome2.dll", "Asteroid11.dll", "Strato10.dll", "Moon2.dll", "CML10.dll"]
    required += ["Services/metadata.n5m", "Services/picture_control.n5m"]
    if not {str((vendor / name).resolve()) for name in required} <= set(loaded):
        raise ValueError("Missing required loaded providers.")
    return sorted(set(loaded))


def validate_events(events: list) -> None:
    steps = [e for e in events if e.get("event") not in ("provider", "sdk_log")]
    expected = [
        {"event": "initialized", "result": 0},
        {"event": "created", "result": 0, "present": True},
        {"event": "services", "result": 0, "present": True},
        {"event": "metadata_service", "result": 0, "present": True},
        {"event": "metadata_start", "result": 0},
        {"event": "metadata_complete", "result": 0, "complete": True},
        {"event": "top_initialized", "result": 0},
        {"event": "top_applied", "result": 0},
        {"event": "top_guards", "intact": True, "saved": True},
        {"event": "closed"},
    ]
    if len(steps) != len(expected) or any(
        any(type(event.get(k)) is not type(v) or event[k] != v for k, v in ref.items())
        for event, ref in zip(steps, expected, strict=True)
    ):
        raise ValueError("SDK lifecycle, TOP status or guard mismatch.")
    if any(e.get("event") == "sdk_log" and e.get("code") for e in events):
        raise ValueError("SDK reported a nonzero diagnostic code.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "dll",
        "probe",
        "wine-prefix",
        "coefficients",
        "mode-table",
        "metadata-report",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--metadata-report-sha256", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.output.exists() or args.output.resolve().is_relative_to(root):
        parser.error("Output must be a new directory outside the repository.")
    coefficients, modes = args.coefficients.read_bytes(), args.mode_table.read_bytes()
    if (
        digest(args.dll) != DLL_HASH
        or digest(args.coefficients) != COEFFICIENT_HASH
        or digest(args.mode_table) != MODE_TABLE_HASH
        or digest(args.metadata_report) != args.metadata_report_sha256
    ):
        parser.error("Unexpected pinned input.")
    oracle = json.loads(args.metadata_report.read_bytes())
    if (
        oracle.get("passed") is not True
        or oracle.get("sources_unchanged") is not True
        or oracle.get("source_white_balance_validation") is not True
        or not oracle.get("cases")
    ):
        parser.error("Expected successful source white-balance evidence.")
    vendor = args.dll.resolve().parent
    profiles = (
        args.wine_prefix.resolve() / "drive_c/Program Files/Common Files/Nikon/Profiles"
    )
    if any(digest(vendor / name) != value for name, value in PINNED.items()):
        parser.error("Unexpected SDK provider hash.")
    if digest(profiles / "NKsRGB.icm") != (
        "49caea94c9d36322910350ee37f1fa09629bed70e01cf615a6863ad3f8d1475e"
    ):
        parser.error("Unexpected Nikon sRGB profile.")
    paths = [
        args.dll,
        args.probe,
        args.coefficients,
        args.mode_table,
        args.metadata_report,
        Path(__file__),
        root / "tools/probe_image_sdk.c",
        Path(_tiff.__file__),
        *(
            root / "tools" / name
            for name in (
                "image_controls.py",
                "color_inputs.py",
                "colorbalance.py",
                "kaiji_index.py",
                "kaiji_grading.py",
                "verify_np3_header.py",
                "verify_colorbalance.py",
                "kaiji_oracle.py",
            )
        ),
        *(p for p in vendor.glob("*.dll") if p.name != "zlib1.dll"),
        args.wine_prefix / "drive_c/windows/system32/zlib1.dll",
        *vendor.glob("Services/*.n5m"),
        *profiles.glob("*.icm"),
    ]
    hashes = {str(p.resolve()): digest(p) for p in paths}
    args.output.mkdir(parents=True)
    report = {
        "evidence_kind": "native_source_controls_vs_live_sdk_top",
        "input_hashes": hashes,
        "cases": [],
        "passed": False,
        "limits": [
            "Selected metadata fields and selector only, not complete TOP parity.",
            "Other selector arguments are SDK outputs, not natively recovered.",
            "Original photographs are ordinary Z8 captures; variants modify metadata.",
            "No Flexible Color rendering, image-domain or Lightroom parity claim.",
        ],
    }
    try:
        for source in oracle["cases"]:
            if source.get("passed") is not True:
                raise ValueError("Unsuccessful source-metadata case.")
            matches = [
                Path(p)
                for p, h in oracle["input_hashes"].items()
                if h == source["source_sha256"]
            ]
            if len(matches) != 1 or digest(matches[0]) != source["source_sha256"]:
                raise ValueError("Source photograph hash mismatch or ambiguity.")
            path = matches[0]
            index = source["index"]
            decoded_path = args.metadata_report.parent / f"case{index}.decoded.bin"
            decoded = decoded_path.read_bytes()
            if hashlib.sha256(decoded).hexdigest() != source["decoded_block_sha256"]:
                raise ValueError("Decoded ColorBalance hash mismatch.")
            hashes[str(path.resolve())] = source["source_sha256"]
            hashes[str(decoded_path.resolve())] = digest(decoded_path)
            expected = source_controls(
                _tiff.read(path.read_bytes()), decoded, coefficients
            )
            prefix = args.output / f"case{index}"
            result = subprocess.run(
                [
                    "/usr/lib/wine/wine64",
                    str(args.probe.resolve()),
                    windows(args.dll),
                    windows(vendor / "Services"),
                    windows(path),
                    windows(prefix),
                ],
                cwd=vendor,
                env=dict(
                    os.environ,
                    WINEPREFIX=str(args.wine_prefix.resolve()),
                    WINEDEBUG="-all",
                ),
                capture_output=True,
                text=True,
                timeout=60,
            )
            events = [
                json.loads(line)
                for line in result.stdout.splitlines()
                if line.startswith("{")
            ]
            case = {
                "index": index,
                "source_sha256": source["source_sha256"],
                "variant": source["variant"],
                "events": events,
                "returncode": result.returncode,
                "stderr": result.stderr,
                "passed": False,
            }
            report["cases"].append(case)
            if result.returncode:
                raise ValueError("SDK probe failed.")
            validate_events(events)
            case["loaded_providers"] = provider_paths(
                events, hashes, vendor, args.wine_prefix
            )
            for suffix in ("initial", "active"):
                data = prefix.with_suffix(f".{suffix}.bin").read_bytes()
                case[suffix] = verify_top(
                    data, expected, coefficients, modes, active=suffix == "active"
                )
                case[f"{suffix}_sha256"] = hashlib.sha256(data).hexdigest()
            case["passed"] = True
        report["sources_unchanged"] = all(
            digest(Path(p)) == h for p, h in hashes.items()
        )
        report["passed"] = (
            report["sources_unchanged"]
            and len(report["cases"]) == len(oracle["cases"])
            and all(c["passed"] for c in report["cases"])
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
