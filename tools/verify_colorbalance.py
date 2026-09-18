"""Compare native ColorBalance import with actual Nikon NEF metadata loading."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
from pathlib import Path

from color_inputs import rational_gains, source_white_balance, white_balance_levels
from colorbalance import import_decoded, iso_code
from verify_np3_header import DLL_HASH, digest, windows

from nx_ctudio import _tiff

ASTEROID_HASH = "00733061041d2c36fc580b770c6d94a7872e1bcef7f93befde07e33ff7204ed6"
STRATO_HASH = "140ca66b7be15a892af07820a608cd9956c71aa42b780de38f36e963e9bbf221"
DECODER = r"""
use strict;
use warnings;
use Image::ExifTool;
use Image::ExifTool::Nikon;
use JSON::PP;
local $/;
my $input = decode_json(<STDIN>);
my $data = pack('H*', $input->{payload});
my $plain = Image::ExifTool::Nikon::Decrypt(
    \$data, 4, undef, $input->{serial}, $input->{count});
binmode STDOUT;
print $plain;
"""


def provider_matches(actual: object, dll: Path, name: str = "Strato10.dll") -> bool:
    expected = windows(dll.resolve().parent / name)
    return (
        isinstance(actual, str)
        and actual.replace("\\", "/").casefold()
        == expected.replace("\\", "/").casefold()
    )


def validate_color_events(events: list, root: _tiff.Directory, rome: Path) -> dict:
    rgb = source_white_balance(root)
    maker = root.tags[34665].children[0].tags[37500].children[0]
    if 0xE2D in maker.tags:
        raise ValueError("Only absent processing metadata is validated.")
    if (
        len(events) != 5
        or not isinstance(events[0], dict)
        or not provider_matches(events[0].get("rome_path"), rome, "Rome2.dll")
    ):
        raise ValueError("Unexpected loaded Rome2.dll provider or event count.")
    expected = [
        {
            "event": "white_balance",
            "rome_path": events[0]["rome_path"],
            "status": 1,
            "guards_intact": True,
            "values": list(rgb),
        }
    ]
    for key, source in ((0x8000C, 12), (0x8003B, 59), (0x80E17, 12)):
        expected.append(
            {
                "event": "color_tag",
                "key": key,
                "success": True,
                "guards_intact": True,
                "hex": struct.pack("<4d", *rational_gains(maker.tags[source])).hex(),
            }
        )
    expected.append(
        {
            "event": "color_tag",
            "key": 0x80E2D,
            "success": False,
            "guards_intact": True,
            "hex": "a5" * 16,
        }
    )
    if (
        events != expected
        or any(
            type(event[key]) is not type(value)
            for event, reference in zip(events, expected, strict=True)
            for key, value in reference.items()
            if key != "values"
        )
        or any(type(value) not in (int, float) for value in events[0]["values"])
    ):
        raise ValueError("White-balance result, typed tag, or guard mismatch.")
    return {
        "native_rgb": rgb,
        "native_top_wb_levels": white_balance_levels(rgb),
        "compared_tag_bytes": 96,
        "compared_rgb_values": 3,
        "top_levels_evidence": "Static initializer mapping; not a live TOP snapshot.",
        "processing_tag_present": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dll", "probe", "wine-prefix", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--rome", type=Path)
    parser.add_argument("--wb-mutations", action="store_true")
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.output.exists() or args.output.resolve().is_relative_to(root):
        parser.error("Output must be a new directory outside the repository.")
    if digest(args.dll) != ASTEROID_HASH:
        parser.error("Unexpected Asteroid11.dll.")
    if digest(args.dll.resolve().parent / "Strato10.dll") != STRATO_HASH:
        parser.error("Unexpected Strato10.dll dependency.")
    if args.rome and digest(args.rome) != DLL_HASH:
        parser.error("Unexpected Rome2.dll.")
    if args.wb_mutations and not args.rome:
        parser.error("White-balance mutations require --rome.")
    perl = json.loads(
        subprocess.run(
            [
                "perl",
                "-MImage::ExifTool",
                "-MImage::ExifTool::Nikon",
                "-MJSON::PP",
                "-e",
                "print encode_json({version=>$Image::ExifTool::VERSION,modules=>"
                '[$INC{"Image/ExifTool.pm"},$INC{"Image/ExifTool/Nikon.pm"}]});',
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    )
    paths = [
        args.dll,
        args.dll.parent / "Strato10.dll",
        args.probe,
        Path(__file__),
        root / "tools/probe_colorbalance.c",
        Path(import_decoded.__code__.co_filename),
        Path(_tiff.__file__),
        Path(digest.__code__.co_filename),
        Path(source_white_balance.__code__.co_filename),
        *(Path(p) for p in perl["modules"]),
    ]
    if args.rome:
        paths.append(args.rome)
    hashes = {str(p.resolve()): digest(p) for p in paths}
    report = {
        "evidence_kind": "native_colorbalance_import_vs_live_nef_metadata_loader",
        "input_hashes": hashes,
        "exiftool_version": perl["version"],
        "source_white_balance_validation": args.rome is not None,
        "white_balance_mutations": args.wb_mutations,
        "cases": [],
        "passed": False,
        "limits": [
            "Decryption uses installed open-source ExifTool, not native Python.",
            "Only 166-byte little-endian ColorBalance0803 is supported.",
            "No full application initialization or rendered-image parity.",
            "WB scope: Z8, unity fine WB, inactive 0103 adjustment, no overrides.",
            "Mutations are controlled metadata experiments, not camera captures.",
        ],
    }
    args.output.mkdir(parents=True)
    try:
        cases = [(path, "original", None) for path in args.files]
        if args.wb_mutations:
            for index, path in enumerate(args.files):
                snapshot = path.read_bytes()
                hashes[str(path.resolve())] = hashlib.sha256(snapshot).hexdigest()
                for label, gains in (
                    ("relocated", None),
                    ("rb", (600, 700, 512, 512)),
                    ("above-two", (1536, 2048, 512, 512)),
                    ("unity", (512, 512, 512, 512)),
                    ("split-green", (977, 787, 768, 1024)),
                    ("fractional", (513, 511, 512, 512)),
                ):
                    tree = _tiff.read(snapshot)
                    maker = tree.tags[34665].children[0].tags[37500].children[0]
                    if gains is not None:
                        denominator = 511 if label == "fractional" else 512
                        maker.set(
                            12,
                            5,
                            b"".join(struct.pack("<II", g, denominator) for g in gains),
                        )
                    target = args.output / f"source{index}-{label}.nef"
                    with target.open("xb") as stream:
                        stream.write(_tiff.serialize(tree))
                    cases.append((target, label, hashes[str(path.resolve())]))
        for index, (path, variant, parent_hash) in enumerate(cases):
            snapshot = path.read_bytes()
            source_hash = hashlib.sha256(snapshot).hexdigest()
            hashes[str(path.resolve())] = source_hash
            tree = _tiff.read(snapshot)
            maker = tree.tags[34665].children[0].tags[37500].children[0]
            raw = maker.tags[0x97].payload
            if len(raw) != 166 or raw[:4] != b"0803":
                raise ValueError("Unsupported source ColorBalance record.")
            decoded = subprocess.run(
                ["perl", "-e", DECODER],
                input=json.dumps(
                    {
                        "payload": raw.hex(),
                        "serial": int(maker.tags[0x1D].payload.rstrip(b"\0")),
                        "count": maker.tags[0xA7].integers()[0],
                    }
                ).encode(),
                capture_output=True,
                check=True,
                timeout=30,
            ).stdout
            native = import_decoded(decoded)
            output = args.output / f"case{index}.vendor.bin"
            result = subprocess.run(
                [
                    "/usr/lib/wine/wine64",
                    str(args.probe.resolve()),
                    windows(args.dll),
                    windows(path),
                    windows(output),
                    *([windows(args.rome)] if args.rome else []),
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
            )
            events = [
                json.loads(line)
                for line in result.stdout.splitlines()
                if line.startswith("{")
            ]
            case = {
                "index": index,
                "source_sha256": source_hash,
                "variant": variant,
                "parent_source_sha256": parent_hash,
                "raw_block_sha256": hashlib.sha256(raw).hexdigest(),
                "decoded_block_sha256": hashlib.sha256(decoded).hexdigest(),
                "returncode": result.returncode,
                "events": events,
                "stderr": result.stderr,
                "passed": False,
            }
            report["cases"].append(case)
            if (
                len(events) != (8 if args.rome else 3)
                or not isinstance(events[1], dict)
                or not provider_matches(events[1].get("strato_path"), args.dll)
            ):
                raise ValueError("Unexpected loaded Strato10.dll provider.")
            expected = [
                {"event": "load_init", "success": True},
                {
                    "event": "read_metadata",
                    "success": True,
                    "strato_path": events[1]["strato_path"],
                },
                {
                    "event": "colorbalance_iso",
                    "status": 1,
                    "value": iso_code(decoded),
                    "guards_intact": True,
                    "success": True,
                },
            ]
            if result.returncode or events[:3] != expected:
                raise ValueError(
                    "Metadata loader failed or returned unexpected fields."
                )
            if args.rome:
                case["color_inputs"] = validate_color_events(
                    events[3:], tree, args.rome
                )
            vendor = output.read_bytes()
            if len(vendor) != 0xD08:
                raise ValueError("Unexpected typed ColorBalance size.")
            for suffix, payload in (("decoded.bin", decoded), ("native.bin", native)):
                with (args.output / f"case{index}.{suffix}").open("xb") as stream:
                    stream.write(payload)
            differences = sum(a != b for a, b in zip(native, vendor, strict=True))
            case.update(
                native_sha256=hashlib.sha256(native).hexdigest(),
                vendor_sha256=hashlib.sha256(vendor).hexdigest(),
                compared_bytes=len(native),
                different_bytes=differences,
                packed_iso_offset="0x8a",
                typed_iso_offset="0xa0",
                native_iso_code=iso_code(decoded),
                vendor_iso_code=struct.unpack_from("<H", vendor, 0xA0)[0],
                passed=differences == 0,
            )
        report["sources_unchanged"] = all(
            digest(Path(p)) == h for p, h in hashes.items()
        )
        report["passed"] = (
            len(report["cases"]) == len(cases)
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
