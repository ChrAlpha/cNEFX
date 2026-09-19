"""Compare native RGB16 standard transfer tables against the pinned DLL export."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path

import numpy as np
from raw_transfer import srgb_lut
from verify_np3_header import digest, windows

UTILITY_HASH = "4b07ab68499500434d5b79e68a835dd0770d8d4350c8276ccaec1b359beed088"


def math_provider() -> Path:
    """Resolve the actual native powf provider through the dynamic loader."""

    class Info(ctypes.Structure):
        _fields_ = [
            ("filename", ctypes.c_char_p),
            ("base", ctypes.c_void_p),
            ("symbol", ctypes.c_char_p),
            ("address", ctypes.c_void_p),
        ]

    name = ctypes.util.find_library("m")
    if name is None:
        raise ValueError("Native libm is unavailable.")
    libm, runtime = ctypes.CDLL(name), ctypes.CDLL(None)
    lookup = runtime.dladdr
    lookup.argtypes = [ctypes.c_void_p, ctypes.POINTER(Info)]
    lookup.restype = ctypes.c_int
    info = Info()
    if not lookup(ctypes.cast(libm.powf, ctypes.c_void_p), ctypes.byref(info)):
        raise ValueError("Cannot resolve the native powf provider.")
    return Path(os.fsdecode(info.filename)).resolve(strict=True)


def validate_event(event: dict, dll: Path) -> None:
    if type(event) is not dict or type(event.get("provider_hex")) is not str:
        raise ValueError("Missing structured transfer provider evidence.")
    provider = bytes.fromhex(event["provider_hex"]).decode("ascii").replace("\\", "/")
    if provider.casefold() != windows(dll).replace("\\", "/").casefold():
        raise ValueError("Unexpected loaded transfer provider.")
    expected = {
        "event": "raw_transfer",
        "provider_hex": event["provider_hex"],
        "bytes": 262144,
        "unchanged": True,
        "saved": True,
        "released": True,
    }
    if json.dumps(event, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise ValueError("Transfer table lifecycle or output evidence mismatch.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dll", "probe", "wine-prefix", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.output.exists() or args.output.resolve().is_relative_to(root):
        parser.error("Output must be a new directory outside the repository.")
    if digest(args.dll) != UTILITY_HASH:
        parser.error("Unexpected utilities.dll hash.")
    paths = [
        args.dll,
        args.probe,
        Path(__file__),
        Path(srgb_lut.__code__.co_filename),
        Path(digest.__code__.co_filename),
        root / "tools/probe_raw_transfer.c",
        math_provider(),
    ]
    hashes = {str(p.resolve()): digest(p) for p in paths}
    report = {
        "evidence_kind": "live_utilities_standard_transfer_vs_native_libm",
        "input_hashes": hashes,
        "libc": list(platform.libc_ver()),
        "numpy_version": np.__version__,
        "cases": [],
        "passed": False,
        "limits": [
            "Fixed GetStdLut(0,gamma,16,16),gamma0/1 only.",
            "Numerical parity is scoped to the recorded native libm.",
            "Table generation does not establish graph activation or full rendering.",
        ],
    }
    args.output.mkdir(parents=True)
    try:
        output = args.output / "vendor.bin"
        result = subprocess.run(
            [
                "/usr/lib/wine/wine64",
                str(args.probe.resolve()),
                windows(args.dll),
                windows(output),
            ],
            cwd=args.dll.resolve().parent,
            env=dict(
                os.environ, WINEPREFIX=str(args.wine_prefix.resolve()), WINEDEBUG="-all"
            ),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        report.update(
            returncode=result.returncode, stdout=result.stdout, stderr=result.stderr
        )
        if result.returncode:
            raise ValueError("Transfer probe failed.")
        lines = [
            json.loads(line) for line in result.stdout.splitlines() if line.strip()
        ]
        if len(lines) != 1:
            raise ValueError("Unexpected transfer event count.")
        validate_event(lines[0], args.dll)
        raw = output.read_bytes()
        if len(raw) != 262144:
            raise ValueError("Unexpected standard transfer table size.")
        tables = np.frombuffer(raw, dtype="<u2").reshape(2, 65536)
        report["output_sha256"] = hashlib.sha256(raw).hexdigest()
        for gamma in (0, 1):
            native = srgb_lut(inverse=bool(gamma))
            delta = native.astype(np.int32) - tables[gamma].astype(np.int32)
            case = {
                "gamma": gamma,
                "compared_words": 65536,
                "different_words": int(np.count_nonzero(delta)),
                "maximum_error": int(np.max(np.abs(delta))),
                "native_sha256": hashlib.sha256(native.tobytes()).hexdigest(),
                "vendor_sha256": hashlib.sha256(tables[gamma].tobytes()).hexdigest(),
            }
            report["cases"].append(case)
        report["sources_unchanged"] = all(
            digest(Path(p)) == h for p, h in hashes.items()
        )
        report["passed"] = report["sources_unchanged"] and all(
            c["different_words"] == 0 for c in report["cases"]
        )
    except subprocess.TimeoutExpired as exc:
        report.update(
            error="Transfer oracle exceeded 30 seconds.",
            stdout=(exc.stdout or b"").decode(errors="replace"),
            stderr=(exc.stderr or b"").decode(errors="replace"),
        )
    except (OSError, ValueError, TypeError) as exc:
        report["error"] = str(exc)
    except KeyboardInterrupt:
        report.update(
            passed=False, cancelled=True, error="Transfer verification cancelled."
        )
    finally:
        with (args.output / "report.json").open("x") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
    print(
        json.dumps(
            {"passed": report["passed"], "report": str(args.output / "report.json")}
        )
    )
    return 130 if report.get("cancelled") else (0 if report["passed"] else 1)


if __name__ == "__main__":
    raise SystemExit(main())
