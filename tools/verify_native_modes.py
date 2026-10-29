"""Replay all nine native output modes against saved synthetic Nikon NEFX."""

from __future__ import annotations

import argparse
import io
import json
import logging
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import rawpy
import tifffile
from verify_native_cli import digest, shutter_field


def maker_blocks(path: Path) -> dict[str, str]:
    with tifffile.TiffFile(path) as image:
        maker = image.pages[0].tags[34665].value["MakerNote"]
    with tifffile.TiffFile(io.BytesIO(maker[10:])) as image:
        return {
            str(code): image.pages[0].tags[code].value.hex()
            for code in (0x56, 0xE2D, 0xE2E)
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--executable", type=Path, help="Native cnefx executable.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.output.resolve().is_relative_to(root):
        parser.error("Research output must be outside the repository.")
    args.output.mkdir(exist_ok=False)
    cli_prefix = (
        [str(args.executable.resolve())]
        if args.executable
        else [sys.executable, "-m", "nx_ctudio.cli"]
    )
    logging.getLogger("tifffile").setLevel(logging.ERROR)
    pins: dict[Path, str] = {}

    def pin(path: Path, expected: str | None = None) -> str:
        value = digest(path)
        if value != pins.get(path, value) or (
            expected is not None and value != expected
        ):
            raise ValueError(f"Artifact changed or hash mismatch: {path}")
        pins[path] = value
        return value

    report = {
        "evidence_kind": "native_cli_saved_synthetic_nine_mode_comparison",
        "passed": False,
        "cases": [],
        "runtime": "cnefx" if args.executable else "python_reference",
        "code_sha256": {
            str(p): pin(p)
            for p in [
                *sorted((root / "src/nx_ctudio").glob("*.py")),
                *sorted((root / "src").glob("*.cpp")),
                *sorted((root / "src").glob("*.hpp")),
                *sorted((root / "include/cnefx").glob("*.hpp")),
                root / "CMakeLists.txt",
                Path(__file__),
                Path(__file__).with_name("verify_native_cli.py"),
            ]
        },
        "limitations": (
            "Synthetic metadata and pixels; no authentic sequence or Lightroom test. "
            "Radial correction is disabled in this saved corpus."
        ),
    }
    try:
        if args.executable:
            report["executable_sha256"] = pin(args.executable.resolve())
            report["executable_version"] = subprocess.run(
                [*cli_prefix, "--version"],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            ).stdout.strip()
        pin(args.matrix)
        pin(args.manifest)
        matrix = json.loads(args.matrix.read_text())
        manifest = json.loads(args.manifest.read_text())
        if matrix["evidence_kind"] != "live_vendor_dll_synthetic_distinct_frame_matrix":
            raise ValueError("Expected saved vendor mode matrix.")
        if pin(args.manifest) != matrix["manifest_sha256"]:
            raise ValueError("Matrix and fixture manifest differ.")
        expected_modes = {
            (4, 1),
            (8, 1),
            (8, 2),
            (16, 1),
            (16, 4),
            (32, 1),
            (32, 2),
            (32, 4),
            (32, 8),
        }
        if {
            (c["inputs"], c["outputs"]) for c in matrix["cases"]
        } != expected_modes or len(matrix["cases"]) != 9:
            raise ValueError("Expected all nine unique modes.")
        copies = {}
        for count in (4, 8, 16, 32):
            directory = args.output / f"inputs-{count}"
            directory.mkdir()
            records = sorted(
                (f for f in manifest["files"] if f["input_count"] == count),
                key=lambda f: f["frame_index"],
            )
            if [f["frame_index"] for f in records] != list(range(count)):
                raise ValueError("Incomplete ordered fixture set.")
            copies[count] = []
            first_shutter = None
            for index, record in enumerate(records):
                path = Path(record["path"])
                pin(path, record["sha256"])
                content = bytearray(path.read_bytes())
                offset, shutter = shutter_field(content)
                if first_shutter is None:
                    first_shutter = shutter
                struct.pack_into("<I", content, offset, first_shutter + index)
                target = directory / path.name
                with target.open("xb") as stream:
                    stream.write(content)
                pin(target)
                copies[count].append(target)
        for case in matrix["cases"]:
            count, outputs = case["inputs"], case["outputs"]
            if not case["ok"] or len(case["comparisons"]) != outputs:
                raise ValueError("Saved vendor case is incomplete or failed.")
            directory = args.output / f"merge-{count}-{outputs}"
            command = [
                *cli_prefix,
                "merge",
                *map(str, copies[count]),
                "--output-directory",
                str(directory),
                "--outputs",
                str(outputs),
                "--json",
            ]
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=180
            )
            observed = {
                "inputs": count,
                "outputs": outputs,
                "returncode": result.returncode,
                "comparisons": [],
                "passed": False,
            }
            report["cases"].append(observed)
            if result.returncode:
                observed.update(stdout=result.stdout, stderr=result.stderr)
                raise ValueError("Native CLI failed.")
            actual_report = json.loads(result.stdout)
            if json.loads((directory / "manifest.json").read_text()) != actual_report:
                raise ValueError("Published manifest and CLI report differ.")
            if (
                actual_report["output_count"] != outputs
                or len(actual_report["outputs"]) != outputs
            ):
                raise ValueError("Native output count differs.")
            for index, (item, reference) in enumerate(
                zip(actual_report["outputs"], case["comparisons"], strict=True)
            ):
                native = Path(item["output"])
                pin(native, item["sha256"])
                prefix = args.matrix.parent / f"merge-{count}-{outputs}.nefx"
                vendor = (
                    prefix if outputs == 1 else Path(f"{prefix}.part-{index + 1}.nefx")
                )
                pin(vendor, reference["output_sha256"])
                with rawpy.imread(str(native)) as a, rawpy.imread(str(vendor)) as b:
                    first, second = a.raw_image[..., :3], b.raw_image[..., :3]
                    if first.shape != second.shape:
                        raise ValueError("Native geometry differs from vendor.")
                    different = int(np.count_nonzero(first != second))
                    samples = int(first.size)
                matches = maker_blocks(native) == maker_blocks(vendor)
                observed["comparisons"].append(
                    {
                        "native": str(native),
                        "native_sha256": item["sha256"],
                        "vendor": str(vendor),
                        "vendor_sha256": reference["output_sha256"],
                        "input_indices": item["input_indices"],
                        "samples": samples,
                        "different_samples": different,
                        "capture_marker_preview_equal": matches,
                        "passed": different == 0
                        and matches
                        and item["input_indices"] == reference["input_indices"],
                    }
                )
            rerun = subprocess.run(command, capture_output=True, text=True, timeout=30)
            observed["overwrite_rejected"] = rerun.returncode == 1 and all(
                pin(Path(item["output"]), item["sha256"])
                for item in actual_report["outputs"]
            )
            observed["passed"] = observed["overwrite_rejected"] and all(
                c["passed"] for c in observed["comparisons"]
            )
        for path in pins:
            pin(path)
        report["passed"] = len(report["cases"]) == 9 and all(
            c["passed"] for c in report["cases"]
        )
    except (Exception, KeyboardInterrupt) as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    report["artifact_sha256"] = {str(p): h for p, h in pins.items()}
    with (args.output / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"passed": report["passed"], "output": str(args.output)}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
