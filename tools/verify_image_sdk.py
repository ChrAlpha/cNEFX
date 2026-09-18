"""Compare native Z8 source controls with actual SDK TOPParam initialization."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from color_inputs import source_blend_inputs
from image_controls import source_controls, verify_top
from image_kaiji import verify_image_table, verify_prepared_grid
from inspect_np3 import parse
from kaiji_grading import COEFFICIENT_HASH
from kaiji_index import MODE_TABLE_HASH
from kaiji_mix import CALIBRATION_HASH
from kaiji_oracle import MOON_HASH, dependency_hashes
from kaiji_profile import build_profile
from np3_parameters import normalize_flexible, parameter_blocks
from verify_colorbalance import ASTEROID_HASH, STRATO_HASH
from verify_np3_header import DLL_HASH, digest, windows

from nx_ctudio import _tiff
from nx_ctudio.picture_curve import generate_curve

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
    "Services/image_render.n5m": (
        "b9572f5b176861f69fe1407e7e1e18734a9983cd73421411efd999f4409bccd8"
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
    required = list(PINNED)
    if not {str((vendor / name).resolve()) for name in required} <= set(loaded):
        raise ValueError("Missing required loaded providers.")
    return sorted(set(loaded))


def picture_controls(data: bytes) -> dict:
    records = parse(data)
    blocks = parameter_blocks(records)
    _, values = normalize_flexible(blocks["flexible"])
    curve = next((record.payload for record in records if record.identifier == 2), None)
    if curve is not None:
        generate_curve(curve)
    return {
        "picture_control_id": 32,
        "profile_row": 14,
        "profile_family": 12,
        "normalized_values": values,
        "custom_curve": curve is not None,
    }


def validate_events(
    events: list,
    picture: dict | None = None,
    *,
    kaiji: bool = False,
    prepare_grid: bool = False,
    renderer_profile: bool = False,
) -> None:
    if type(renderer_profile) is not bool or (renderer_profile and not prepare_grid):
        raise ValueError("Renderer profile requires engine preparation.")
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
    if picture is not None:
        expected.insert(
            6,
            {
                "event": "picture_imported",
                "result": 0,
                "guards_intact": True,
                "input_unchanged": True,
                "values": picture["normalized_values"],
            },
        )
    if kaiji:
        if picture is None:
            raise ValueError("Kaiji validation requires an imported NP3.")
        expected[-1:-1] = [
            {"event": "kaiji_cms", "result": 0, "present": True},
            {"event": "kaiji_config", "result": 0},
            {"event": "kaiji_engine", "value": 9},
            {"event": "kaiji_tables", "guards_intact": True, "saved": True},
            {"event": "kaiji_released", "guards_intact": True},
        ]
    if prepare_grid:
        if not kaiji:
            raise ValueError("Engine preparation requires Kaiji table validation.")
        expected.insert(
            -2,
            {
                "event": "kaiji_prepared",
                "error": False,
                "grid": 65 if renderer_profile else 33,
                "selector": int(renderer_profile),
                "output_flags": 0,
                "saved": True,
            },
        )
        expected.insert(
            -2,
            {
                "event": "kaiji_profile",
                "error": False,
                "guards_intact": True,
                "saved": True,
            },
        )
    if len(steps) != len(expected) or any(
        any(type(event.get(k)) is not type(v) or event[k] != v for k, v in ref.items())
        for event, ref in zip(steps, expected, strict=True)
    ):
        raise ValueError("SDK lifecycle, TOP status or guard mismatch.")
    if picture is not None and any(type(v) is not int for v in steps[6]["values"]):
        raise ValueError("Malformed normalized Picture Control values.")
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
    parser.add_argument("--picture-control", type=Path, action="append", default=[])
    parser.add_argument("--originals-only", action="store_true")
    parser.add_argument("--kaiji", action="store_true")
    parser.add_argument("--blend-calibration", type=Path)
    parser.add_argument("--prepare-grid", action="store_true")
    parser.add_argument("--renderer-profile", action="store_true")
    args = parser.parse_args()
    if args.renderer_profile and not args.prepare_grid:
        parser.error("--renderer-profile requires --prepare-grid.")
    if args.prepare_grid and not args.kaiji:
        parser.error("--prepare-grid requires --kaiji.")
    if args.kaiji and (not args.picture_control or args.blend_calibration is None):
        parser.error("--kaiji requires --picture-control and --blend-calibration.")
    blend_calibration = args.blend_calibration.read_bytes() if args.kaiji else None
    if args.kaiji and hashlib.sha256(blend_calibration).hexdigest() != CALIBRATION_HASH:
        parser.error("Unexpected blend calibration hash.")
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
        root / "tools/image_render.h",
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
                "np3_parameters.py",
                "inspect_np3.py",
            )
        ),
        *(p for p in vendor.glob("*.dll") if p.name != "zlib1.dll"),
        args.wine_prefix / "drive_c/windows/system32/zlib1.dll",
        *vendor.glob("Services/*.n5m"),
        *profiles.glob("*.icm"),
        Path(generate_curve.__code__.co_filename),
        root / "src/nx_ctudio/picture_control.py",
        *args.picture_control,
    ]
    hashes = {str(p.resolve()): digest(p) for p in paths}
    if args.kaiji:
        hashes[str(args.blend_calibration.resolve())] = CALIBRATION_HASH
        hashes.update(dependency_hashes(args.dll, args.probe))
        dependencies = [root / "tools/image_kaiji.h"]
        dependencies += list((root / "tools").glob("kaiji_*.py"))
        dependencies += [root / "tools/image_kaiji.py", root / "tools/flexible_math.py"]
        hashes.update({str(p.resolve()): digest(p) for p in dependencies})
    args.output.mkdir(parents=True)
    report = {
        "evidence_kind": "native_source_controls_vs_live_sdk_top",
        "input_hashes": hashes,
        "cases": [],
        "passed": False,
        "picture_import": bool(args.picture_control),
        "originals_only": args.originals_only,
        "kaiji_tables": args.kaiji,
        "prepared_grid": args.prepare_grid,
        "renderer_profile": args.renderer_profile,
        "limits": [
            "Selected metadata fields and selector only, not complete TOP parity.",
            "Without --kaiji, other selector arguments remain SDK outputs "
            "and curve checks cover activation only.",
            "Kaiji comparison covers model14, full strength, matrix0, brightness6, "
            "no application EV adjustment and source ISO bucket6.",
            "Blend inputs require source AUTO1/AUTO0 and decrypted ColorBalance0803; "
            "application WB overrides and native decryption remain unimplemented.",
            "Calibration row0 and auxiliary selector22 are bounded observed defaults; "
            "general metadata derivation of those defaults remains unresolved.",
            "Selected Kaiji fields and complete tables, not every parameter byte.",
            "Without --prepare-grid, comparison precedes engine preparation "
            "and its output matrix.",
            "Prepared Lab lattice parity is not decoded-image or ICC consumer parity.",
            "Original photographs are ordinary Z8 captures; variants modify metadata.",
            "No Flexible Color rendering, image-domain or Lightroom parity claim.",
        ],
    }
    try:
        selected = [
            s
            for s in oracle["cases"]
            if not args.originals_only or s["variant"] == "original"
        ]
        configs = [(s, p) for s in selected for p in (args.picture_control or [None])]
        if not configs:
            raise ValueError("No source cases selected.")
        for run_index, (source, np3) in enumerate(configs):
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
            tree = _tiff.read(path.read_bytes())
            expected = source_controls(tree, decoded, coefficients)
            blend_inputs = source_blend_inputs(tree, decoded) if args.kaiji else None
            picture = picture_controls(np3.read_bytes()) if np3 else None
            prefix = args.output / f"case{run_index}"
            result = subprocess.run(
                [
                    "/usr/lib/wine/wine64",
                    str(args.probe.resolve()),
                    windows(args.dll),
                    windows(vendor / "Services"),
                    windows(path),
                    windows(prefix),
                    *([windows(np3)] if np3 else []),
                    *(["kaiji"] if args.kaiji else []),
                    *(
                        ["prepare-renderer" if args.renderer_profile else "prepare"]
                        if args.prepare_grid
                        else []
                    ),
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
                "index": run_index,
                "source_index": index,
                "source_sha256": source["source_sha256"],
                "variant": source["variant"],
                "events": events,
                "returncode": result.returncode,
                "stderr": result.stderr,
                "passed": False,
                "np3_sha256": digest(np3) if np3 else None,
            }
            report["cases"].append(case)
            if result.returncode:
                raise ValueError("SDK probe failed.")
            validate_events(
                events,
                picture,
                kaiji=args.kaiji,
                prepare_grid=args.prepare_grid,
                renderer_profile=args.renderer_profile,
            )
            case["loaded_providers"] = provider_paths(
                events, hashes, vendor, args.wine_prefix
            )
            for suffix in ("initial", "active"):
                data = prefix.with_suffix(f".{suffix}.bin").read_bytes()
                case[suffix] = verify_top(
                    data, expected, coefficients, modes, active=suffix == "active"
                )
                if picture is not None and suffix == "active":
                    if any(case[suffix][k] != v for k, v in picture.items()):
                        raise ValueError("NP3 import disagrees with active TOPParam.")
                case[f"{suffix}_sha256"] = hashlib.sha256(data).hexdigest()
            if args.kaiji:
                snapshots = {}
                for name, size in (
                    ("request", 0x2F0),
                    ("config", 0xDC8),
                    ("kaiji", 0x17F8),
                    ("table", 0x34370),
                ):
                    snapshots[name] = prefix.with_suffix(f".{name}.bin").read_bytes()
                    if len(snapshots[name]) != size:
                        raise ValueError(f"Incorrect {name} snapshot size.")
                    case[f"{name}_sha256"] = hashlib.sha256(snapshots[name]).hexdigest()
                curve = next(
                    (r.payload for r in parse(np3.read_bytes()) if r.identifier == 2),
                    None,
                )
                case["kaiji"] = verify_image_table(
                    snapshots["kaiji"],
                    snapshots["table"],
                    picture,
                    expected,
                    case["active"],
                    coefficients,
                    modes,
                    curve,
                    blend_inputs,
                    blend_calibration,
                    output_selector=int(args.renderer_profile),
                )
                if args.prepare_grid:
                    prepared = prefix.with_suffix(".prepared-table.bin").read_bytes()
                    planes = [
                        prefix.with_suffix(f".plane{i}.bin").read_bytes()
                        for i in range(3)
                    ]
                    case["prepared_grid"] = verify_prepared_grid(
                        snapshots["table"],
                        prepared,
                        planes,
                        renderer_profile=args.renderer_profile,
                    )
                    case["prepared_table_sha256"] = hashlib.sha256(prepared).hexdigest()
                    case["plane_sha256"] = [
                        hashlib.sha256(p).hexdigest() for p in planes
                    ]
                    profile = prefix.with_suffix(".profile.bin").read_bytes()
                    event = next(e for e in events if e.get("event") == "kaiji_profile")
                    description = bytes.fromhex(event["description_hex"]).decode(
                        "ascii"
                    )
                    copyright = bytes.fromhex(event["copyright_hex"]).decode("ascii")
                    native = build_profile(
                        (profiles / "NKDBase_NEF.icm").read_bytes(),
                        planes,
                        description=description,
                        copyright=copyright,
                        grid=65 if args.renderer_profile else 33,
                    )
                    if (
                        type(event.get("bytes")) is not int
                        or event["bytes"] != len(profile)
                        or profile != native
                    ):
                        raise ValueError(
                            "Native ICC serialization disagrees with vendor."
                        )
                    case["profile"] = {
                        "compared_bytes": len(native),
                        "different_bytes": 0,
                        "sha256": hashlib.sha256(native).hexdigest(),
                        "description": description,
                        "copyright": copyright,
                        "text_source": "Captured profile text; verified native planes",
                    }
            case["passed"] = True
        report["sources_unchanged"] = all(
            digest(Path(p)) == h for p, h in hashes.items()
        )
        report["passed"] = (
            report["sources_unchanged"]
            and len(report["cases"]) == len(configs)
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
