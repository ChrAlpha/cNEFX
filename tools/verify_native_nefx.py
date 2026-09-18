"""Verify the native NEFX writer against pinned saved Nikon outputs."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import rawpy
from inspect_nefx_layout import compare, digest
from verify_nefx_metadata import validate
from verify_nefx_preview import verify as verify_preview

from nx_ctudio import __version__
from nx_ctudio.nefx import write_nefx


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.output.resolve().is_relative_to(root):
        parser.error("Research output must be outside the repository.")
    args.output.mkdir(exist_ok=False)
    logging.getLogger("tifffile").setLevel(logging.ERROR)
    specification = json.loads(
        (root / "research/pixel-shift-serialization.json").read_text()
    )
    result = {
        "evidence_kind": "native_nefx_writer_saved_synthetic_vendor_comparison",
        "passed": False,
        "cases": [],
        "code_sha256": {
            str(p.relative_to(root)): digest(p.read_bytes())
            for p in (
                Path(__file__).resolve(),
                root / "src/nx_ctudio/nefx.py",
                root / "src/nx_ctudio/_tiff.py",
                root / "tools/verify_nefx_metadata.py",
            )
        },
    }
    try:
        for index, (reference, count) in enumerate(
            zip(specification["verification"]["artifacts"], (4, 4, 32, 16), strict=True)
        ):
            source, vendor = (Path(reference[key]) for key in ("source", "output"))
            for key, path in (("source", source), ("output", vendor)):
                if digest(path.read_bytes()) != reference[key + "_sha256"]:
                    raise ValueError("Saved source or vendor checksum mismatch.")
            with rawpy.imread(str(vendor)) as image:
                samples = image.raw_image[..., :3].copy()
            output = args.output / f"case-{index}.nefx"
            write_nefx(output, source.read_bytes(), samples, count)
            with rawpy.imread(str(output)) as image:
                differences = int(np.count_nonzero(image.raw_image[..., :3] != samples))
            metadata = validate(
                source, output, count, software=f"NX Ctudio {__version__}"
            )
            layout = compare(source, output)
            preview = verify_preview(output)
            images_preserved = not layout["removed_payloads"] and all(
                p["preserved"]
                for p in layout["payloads"]
                if p["path"] != "IFD0/SubIFD[1]/strips[0]"
            )
            case = {
                "source": str(source),
                "source_sha256": reference["source_sha256"],
                "vendor": str(vendor),
                "vendor_sha256": reference["output_sha256"],
                "output": str(output),
                "output_sha256": digest(output.read_bytes()),
                "input_count": count,
                "samples": int(samples.size),
                "different_samples": differences,
                "metadata_checks": metadata["checks"],
                "preview": preview,
                "source_image_payloads_preserved": images_preserved,
                "passed": differences == 0
                and metadata["passed"]
                and preview["differences"] == 0
                and images_preserved,
            }
            result["cases"].append(case)
        result["passed"] = len(result["cases"]) == 4 and all(
            c["passed"] for c in result["cases"]
        )
    except (Exception, KeyboardInterrupt) as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
    with (args.output / "report.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "passed": result["passed"]}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
