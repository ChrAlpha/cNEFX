"""Compare cNEFX lattice arithmetic with the pinned external firmware project."""

from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import json
import struct
import subprocess
from pathlib import Path

import numpy as np

REVISION = "13c3709a35c58f9cf6fc9c1043705712074651eb"


class Frame(ct.Structure):
    _fields_ = [
        ("frame_id", ct.c_uint32),
        ("dx2", ct.c_int32),
        ("dy2", ct.c_int32),
        ("black", ct.c_uint16 * 4),
        ("white", ct.c_uint16 * 4),
        ("cfa", ct.c_uint8 * 4),
    ]


class Plan(ct.Structure):
    _fields_ = [
        ("sensor_width", ct.c_uint32),
        ("sensor_height", ct.c_uint32),
        ("frame_count", ct.c_uint32),
        ("scale", ct.c_uint32),
        ("x0", ct.c_int32),
        ("y0", ct.c_int32),
        ("width", ct.c_uint32),
        ("height", ct.c_uint32),
        ("group_id", ct.c_uint64),
        ("frames", Frame * 32),
    ]


READ = ct.CFUNCTYPE(
    ct.c_int,
    ct.c_void_p,
    ct.c_uint32,
    ct.c_uint32,
    ct.POINTER(ct.c_uint16),
    ct.c_uint32,
)
CANCEL = ct.CFUNCTYPE(ct.c_int, ct.c_void_p)


def run(reference: Path, executable: Path, output: Path) -> dict:
    revision = subprocess.check_output(
        ["git", "-C", str(reference), "rev-parse", "HEAD"], text=True, timeout=10
    ).strip()
    if revision != REVISION:
        raise ValueError(
            "The external reference revision differs from the reviewed pin."
        )
    subprocess.run(
        [
            "git",
            "-C",
            str(reference),
            "diff",
            "--exit-code",
            REVISION,
            "--",
            "z8pixelshift/src",
        ],
        check=True,
        timeout=10,
    )
    output.mkdir(parents=True)
    source = reference / "z8pixelshift/src"
    library = output / "reference.so"
    build = [
        "cc",
        "-O2",
        "-fPIC",
        "-shared",
        "-ffp-contract=off",
        str(source / "psm.c"),
        str(source / "afrow.c"),
        "-lm",
        "-o",
        str(library),
    ]
    subprocess.run(build, check=True, timeout=30)
    oracle = ct.CDLL(str(library))
    oracle.psm_scratch_size.argtypes = [ct.POINTER(Plan)]
    oracle.psm_scratch_size.restype = ct.c_size_t
    oracle.psm_merge_row.argtypes = [
        ct.POINTER(Plan),
        ct.c_uint32,
        READ,
        CANCEL,
        ct.c_void_p,
        ct.c_void_p,
        ct.c_size_t,
        ct.POINTER(ct.c_uint16),
    ]
    cases = []
    phase_x, phase_y = (0, 1, 1, 0), (0, 0, 1, 1)
    for count in (4, 8, 16, 32):
        for seed in range(3):
            width, height = 64, 48
            # The reference validates the whole crop against every displacement,
            # including unsampled half-pixel phases. Its paired 2x mode needs 4.
            crop = (4 + seed % 2, 4 + seed % 2, 56, 40)
            scale = 2 if count >= 16 else 1
            frames = np.random.default_rng(20261029 + count + seed).integers(
                0, 16384, (count, height, width), dtype=np.uint16
            )
            if seed < 2:
                frames.fill(0 if seed == 0 else 16383)
            plan = Plan(
                width,
                height,
                count,
                scale,
                crop[0] * scale,
                crop[1] * scale,
                crop[2] * scale,
                crop[3] * scale,
                1,
            )
            reduced = count // 2 if count in (8, 32) else count
            for index in range(count):
                local = index % reduced
                group, position = divmod(local, 4)
                frame = plan.frames[index]
                frame.frame_id = index
                frame.dx2 = 2 * phase_x[position] + (
                    phase_x[group] if scale == 2 else 0
                )
                frame.dy2 = 2 * phase_y[position] + (
                    phase_y[group] if scale == 2 else 0
                )
                if index >= reduced:
                    frame.dy2 += 4
                frame.white[:] = (16383,) * 4
                frame.cfa[:] = (0, 1, 1, 2)
            read_errors = []

            @READ
            def read_row(_, index, row, target, length):
                if index >= count or row >= height or length != width:
                    read_errors.append((index, row, length))
                    return 1
                ct.memmove(target, frames[index, row].ctypes.data, width * 2)
                return 0

            @CANCEL
            def cancel(_):
                return 0

            scratch_size = oracle.psm_scratch_size(ct.byref(plan))
            scratch = (ct.c_uint64 * ((scratch_size + 7) // 8))()
            expected = np.empty((plan.height, plan.width, 3), dtype=np.uint16)
            for row in range(plan.height):
                # Meet the external callback's 8-byte alignment contract.
                values = np.zeros(plan.width * 3 + 4, dtype=np.uint16)
                assert values.ctypes.data % 8 == 0
                status = oracle.psm_merge_row(
                    ct.byref(plan),
                    row,
                    read_row,
                    cancel,
                    None,
                    scratch,
                    scratch_size,
                    values.ctypes.data_as(ct.POINTER(ct.c_uint16)),
                )
                assert status == 0 and not read_errors, (
                    count,
                    seed,
                    row,
                    status,
                    read_errors,
                )
                expected[row] = values[: plan.width * 3].reshape(plan.width, 3)
            payload = (
                struct.pack("<9I", count, width, height, *crop, 3201, 0)
                + b"0100"
                + bytes(24 + 32)
                + frames.astype("<u2").tobytes()
            )
            input_path, result_path = (
                output / f"{count}-{seed}.input",
                output / f"{count}-{seed}.rgb",
            )
            input_path.write_bytes(payload)
            subprocess.run(
                [str(executable), "--replay", str(input_path), str(result_path)],
                check=True,
                timeout=30,
            )
            actual = np.frombuffer(result_path.read_bytes(), dtype="<u2").reshape(
                expected.shape
            )
            differences = int(np.count_nonzero(expected != actual))
            cases.append(
                {
                    "frames": count,
                    "seed": seed,
                    "samples": int(expected.size),
                    "different_samples": differences,
                    "fixture_sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
            assert differences == 0, cases[-1]
    report = {
        "evidence_kind": "third_party_source_lattice_comparison",
        "passed": True,
        "reference_revision": revision,
        "command": build,
        "scope": "Pair averaging, phase mapping and integer green truncation.",
        "limitations": (
            "AF repair, radial correction, file identity, serialization and "
            "authentic captures are separate evidence."
        ),
        "source_sha256": {
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in ("psm.c", "psm.h", "afrow.c", "afrow.h")
        },
        "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "cases": cases,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.output.resolve().is_relative_to(root):
        parser.error("External reference artifacts must stay outside the repository.")
    report = run(
        args.reference.resolve(), args.executable.resolve(), args.output.resolve()
    )
    print(json.dumps({"passed": report["passed"], "cases": len(report["cases"])}))


if __name__ == "__main__":
    main()
