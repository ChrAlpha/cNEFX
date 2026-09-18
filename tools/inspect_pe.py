# /// script
# requires-python = ">=3.11"
# dependencies = ["pefile==2024.8.26"]
# ///
"""Inventory a local PE binary without loading or executing it.

Usage: uv run tools/inspect_pe.py /path/to/PixelShiftCombine.dll
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pefile


def inventory(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    with pefile.PE(data=data, fast_load=True) as pe:
        pe.parse_data_directories(directories=[0, 1, 13])
        imports = [
            {
                "module": entry.dll.decode("ascii"),
                "symbols": [
                    {
                        "name": item.name.decode("ascii") if item.name else None,
                        "ordinal": item.ordinal,
                        "iat_rva": hex(item.address - pe.OPTIONAL_HEADER.ImageBase),
                    }
                    for item in entry.imports
                ],
            }
            for entry in getattr(pe, "DIRECTORY_ENTRY_IMPORT", [])
        ]
        exports = [
            {
                "name": item.name.decode("ascii") if item.name else None,
                "ordinal": item.ordinal,
                "rva": hex(item.address),
                "forwarder": item.forwarder.decode("ascii") if item.forwarder else None,
            }
            for item in getattr(
                getattr(pe, "DIRECTORY_ENTRY_EXPORT", None), "symbols", []
            )
        ]
        return {
            "schema_version": 1,
            "evidence_kind": "static_pe_metadata",
            "reader": {"name": "pefile", "version": pefile.__version__},
            "filename": path.name,
            "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "machine": hex(pe.FILE_HEADER.Machine),
            "image_base": hex(pe.OPTIONAL_HEADER.ImageBase),
            "imports": imports,
            "delay_import_modules": [
                entry.dll.decode("ascii")
                for entry in getattr(pe, "DIRECTORY_ENTRY_DELAY_IMPORT", [])
            ],
            "exports": exports,
            "parser_warnings": pe.get_warnings(),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    args = parser.parse_args()
    try:
        report = inventory(args.binary)
    except (OSError, pefile.PEFormatError, UnicodeError) as exc:
        parser.exit(1, f"inspect_pe: {exc}\n")
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
