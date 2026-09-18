"""Compare the sparse-repair reference with the pinned DLL on synthetic planes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import struct
import subprocess
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

from sparse_reference import Selection, repair

DLL_SHA256 = "c82782c51be65d9528f6751bfca9d2bdc7b728baa3ee8eca2e84ba713ed9031c"
WIDTH, HEIGHT = 16, 12
PLANE_SIZE = WIDTH * HEIGHT
CASE_BYTES = 4 * PLANE_SIZE * 2


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def windows(path: Path) -> str:
    return "Z:" + str(path.resolve()).replace("/", "\\")


def fixtures(count: int) -> list[tuple[str, Selection, list[list[int]]]]:
    rng = random.Random(0x34BA60)
    base = Selection(HEIGHT, WIDTH)
    cases = []
    for value in (0, 1, 299, 300, 301, 499, 500, 501, 1999, 2000, 2001, 16383):
        cases.append(
            (f"constant-{value}", base, [[value] * PLANE_SIZE for _ in range(4)])
        )
    selectors = [
        base,
        replace(base, first_row=2),
        replace(base, first_row=9),
        replace(base, first_row=2, inner_step=1, inner_after=5),
        replace(base, first_row=2, middle_count=3, middle_step=2),
        replace(base, first_row=3, outer_count=2, outer_step=2),
        replace(base, half_grid_start=6),
        replace(base, reference_width=24),
        replace(base, reference_height=16, first_row=6),
        replace(base, middle_count=2),
        replace(base, outer_count=0),
        replace(base, reference_width=12),
    ]
    boundaries = (0, 1, 299, 300, 301, 499, 500, 501, 1999, 2000, 2001, 16383)
    for index in range(count):
        mode = index % 4
        planes = []
        for _ in range(4):
            if mode == 0:
                planes.append([rng.randrange(16384) for _ in range(PLANE_SIZE)])
            elif mode == 1:
                planes.append([rng.choice(boundaries) for _ in range(PLANE_SIZE)])
            elif mode == 2:
                center = rng.choice((500, 2000, 8000))
                planes.append(
                    [center + rng.randrange(-100, 101) for _ in range(PLANE_SIZE)]
                )
            else:
                planes.append([rng.choice((0, 0, 1, 2, 10)) for _ in range(PLANE_SIZE)])
        cases.append(
            (f"random-{index}", selectors[(index // 4) % len(selectors)], planes)
        )
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dll", required=True, type=Path)
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--wine-prefix", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cases", type=int, default=10000)
    parser.add_argument("--wine", default="/usr/lib/wine/wine64")
    args = parser.parse_args()
    if digest(args.dll) != DLL_SHA256:
        parser.error("Unexpected DLL checksum.")
    if not 1 <= args.cases <= 90000:
        parser.error("Case count must be in 1..90000.")
    args.output.mkdir(parents=True, exist_ok=False)
    cases = fixtures(args.cases)
    input_path = args.output / "input.bin"
    output_path = args.output / "output.bin"
    report: dict[str, object] = {
        "schema_version": 1,
        "evidence_kind": "live_internal_vendor_oracle_synthetic_sparse_repair",
        "dll_sha256": DLL_SHA256,
        "probe_sha256": digest(args.probe),
        "reference_sha256": digest(Path(__file__).with_name("sparse_reference.py")),
        "runner_sha256": digest(Path(__file__)),
        "seed": 0x34BA60,
        "width": WIDTH,
        "height": HEIGHT,
        "cases": len(cases),
        "passed": False,
    }
    try:
        with input_path.open("xb") as stream:
            stream.write(struct.pack("<4sI", b"NXCS", len(cases)))
            for _, selection, planes in cases:
                selection.coordinates(WIDTH, HEIGHT)
                stream.write(selection.pack())
                for plane in planes:
                    stream.write(struct.pack(f"<{PLANE_SIZE}H", *plane))
        with input_path.open("rb") as source, output_path.open("xb") as target:
            result = subprocess.run(
                [args.wine, str(args.probe), windows(args.dll)],
                env=dict(
                    os.environ,
                    WINEPREFIX=str(args.wine_prefix.resolve()),
                    WINEDEBUG="-all",
                ),
                stdin=source,
                stdout=target,
                stderr=subprocess.PIPE,
                timeout=60,
                check=False,
            )
        report.update(
            return_code=result.returncode,
            stderr=result.stderr.decode("utf-8", errors="replace"),
        )
        if result.returncode:
            raise ValueError("Vendor probe failed.")
        changed = 0
        differences = 0
        mismatch_cases = 0
        selected_samples = 0
        trace: Counter[str] = Counter()
        failures = []
        with output_path.open("rb") as stream:
            if stream.read(8) != struct.pack("<4sI", b"NXCO", len(cases)):
                raise ValueError("Unexpected output header.")
            for name, selection, planes in cases:
                block = stream.read(CASE_BYTES)
                if len(block) != CASE_BYTES:
                    raise ValueError(f"Truncated output at {name}.")
                actual = struct.unpack(f"<{PLANE_SIZE * 4}H", block)
                expected = repair(planes, WIDTH, HEIGHT, selection, trace)
                selected_samples += 4 * len(selection.coordinates(WIDTH, HEIGHT))
                mismatches = []
                for index, value in enumerate(actual):
                    frame, pixel = divmod(index, PLANE_SIZE)
                    changed += value != planes[frame][pixel]
                    if value != expected[frame][pixel]:
                        differences += 1
                        if len(mismatches) < 8:
                            mismatches.append(
                                {
                                    "frame": frame,
                                    "y": pixel // WIDTH,
                                    "x": pixel % WIDTH,
                                    "actual": value,
                                    "expected": expected[frame][pixel],
                                }
                            )
                if mismatches:
                    mismatch_cases += 1
                    if len(failures) < 20:
                        failures.append(
                            {
                                "name": name,
                                "selection": asdict(selection),
                                "mismatches": mismatches,
                            }
                        )
            if stream.read(1):
                raise ValueError("Unexpected trailing output bytes.")
        report.update(
            samples=len(cases) * PLANE_SIZE * 4,
            selected_samples=selected_samples,
            changed_samples=changed,
            differences=differences,
            mismatch_cases=mismatch_cases,
            failures=failures,
            reference_branch_counts=dict(sorted(trace.items())),
            passed=differences == 0 and changed > 0,
        )
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        report["error"] = str(error)
    finally:
        for name, path in (("input", input_path), ("output", output_path)):
            if path.exists():
                report[f"{name}_sha256"] = digest(path)
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
