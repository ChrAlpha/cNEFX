"""Compare native ColorBalance import with actual Nikon NEF metadata loading."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
from pathlib import Path

from colorbalance import import_decoded, iso_code
from verify_np3_header import digest, windows

from nx_ctudio import _tiff

ASTEROID_HASH = "00733061041d2c36fc580b770c6d94a7872e1bcef7f93befde07e33ff7204ed6"
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dll", "probe", "wine-prefix", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.output.exists() or args.output.resolve().is_relative_to(root):
        parser.error("Output must be a new directory outside the repository.")
    if digest(args.dll) != ASTEROID_HASH:
        parser.error("Unexpected Asteroid11.dll.")
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
        *(Path(p) for p in perl["modules"]),
    ]
    hashes = {str(p.resolve()): digest(p) for p in paths}
    report = {
        "evidence_kind": "native_colorbalance_import_vs_live_nef_metadata_loader",
        "input_hashes": hashes,
        "exiftool_version": perl["version"],
        "cases": [],
        "passed": False,
        "limits": [
            "Decryption uses installed open-source ExifTool, not native Python.",
            "Only 166-byte little-endian ColorBalance0803 is supported.",
            "No full application initialization or rendered-image parity.",
        ],
    }
    args.output.mkdir(parents=True)
    try:
        for index, path in enumerate(args.files):
            snapshot = path.read_bytes()
            source_hash = hashlib.sha256(snapshot).hexdigest()
            hashes[str(path.resolve())] = source_hash
            maker = _tiff.read(snapshot).tags[34665].children[0].tags[37500].children[0]
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
                "raw_block_sha256": hashlib.sha256(raw).hexdigest(),
                "decoded_block_sha256": hashlib.sha256(decoded).hexdigest(),
                "returncode": result.returncode,
                "events": events,
                "stderr": result.stderr,
                "passed": False,
            }
            report["cases"].append(case)
            expected = [
                {"event": "load_init", "success": True},
                {"event": "read_metadata", "success": True},
                {
                    "event": "colorbalance_iso",
                    "status": 1,
                    "value": iso_code(decoded),
                    "guards_intact": True,
                    "success": True,
                },
            ]
            if result.returncode or events != expected:
                raise ValueError(
                    "Metadata loader failed or returned unexpected fields."
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
            len(report["cases"]) == len(args.files)
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
