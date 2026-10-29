"""Bounded Linux end-to-end comparison on hash-pinned synthetic capture copies."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import rawpy

import nx_ctudio


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def measured(command: list[str], directory: Path, timeout: int, memory: int) -> dict:
    directory.mkdir()
    stdout, stderr = directory / "stdout.json", directory / "stderr.txt"
    usage_path = directory / "usage.json"
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    bounded = [
        "timeout",
        "--kill-after=5s",
        f"{timeout}s",
        "prlimit",
        f"--as={memory}",
        "--",
        "/usr/bin/time",
        "-o",
        str(usage_path),
        "-f",
        '{"max_rss_kib":%M,"user_seconds":%U,"system_seconds":%S}',
        *command,
    ]
    start = time.monotonic()
    with stdout.open("x") as out, stderr.open("x") as err:
        process = subprocess.run(bounded, stdout=out, stderr=err, env=env)
    usage = {}
    if usage_path.exists():
        lines = usage_path.read_text().splitlines()
        if lines and lines[-1].startswith("{"):
            usage = json.loads(lines[-1])
    result = {
        "command": bounded,
        "returncode": process.returncode,
        "elapsed_seconds": time.monotonic() - start,
        "user_seconds": usage.get("user_seconds"),
        "system_seconds": usage.get("system_seconds"),
        "max_rss_kib": usage.get("max_rss_kib"),
        "measurement": "GNU time on the target process; parent RSS excluded",
        "stdout": str(stdout),
        "stderr": str(stderr),
    }
    if process.returncode == 0:
        result["cli"] = json.loads(stdout.read_text())
    return result


def compare(first: dict, second: dict) -> dict:
    total = different = 0
    for a, b in zip(first["outputs"], second["outputs"], strict=True):
        assert a["input_indices"] == b["input_indices"]
        with rawpy.imread(a["output"]) as left, rawpy.imread(b["output"]) as right:
            x, y = left.raw_image[..., :3], right.raw_image[..., :3]
            assert x.shape == y.shape
            for row in range(0, x.shape[0], 128):
                different += int(
                    np.count_nonzero(x[row : row + 128] != y[row : row + 128])
                )
            total += x.size
    return {"samples": total, "different_samples": different}


def run(args) -> dict:
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True)
    source = json.loads(args.input_report.read_text())
    if digest(args.input_report) != args.input_report_sha256:
        raise ValueError("Input report identity differs.")
    inputs = source["inputs"]
    if len(inputs) != 32:
        raise ValueError("Expected the recorded complete synthetic 32-frame corpus.")
    for item in inputs:
        if digest(Path(item["path"])) != item["sha256"]:
            raise ValueError("Synthetic input hash mismatch.")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    python_root = Path(nx_ctudio.__file__).parent
    report = {
        "evidence_kind": "full_resolution_synthetic_cpp_python_resource_comparison",
        "passed": False,
        "revision": revision,
        "executable": str(args.executable),
        "executable_sha256": digest(args.executable),
        "input_report": str(args.input_report),
        "input_report_sha256": args.input_report_sha256,
        "inputs": inputs,
        "outputs": args.outputs,
        "radial": args.radial,
        "host": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "rawpy": rawpy.__version__,
        "rawpy_libraw": rawpy.libraw_version,
        "code_sha256": {
            str(p): digest(p)
            for p in [
                *sorted(python_root.glob("*.py")),
                *sorted((root / "src").glob("*.cpp")),
                *sorted((root / "src").glob("*.hpp")),
                *sorted((root / "include/cnefx").glob("*.hpp")),
                Path(__file__),
                root / "CMakeLists.txt",
                root / "cmake/LibRaw.cmake",
            ]
        },
        "budget": {
            "concurrency": 1,
            "timeout_seconds_per_run": args.timeout,
            "address_space_bytes_per_run": args.memory_mib * 1024**2,
        },
        "limitations": (
            "Repeated ordinary RAW with synthetic identity metadata. This measures "
            "resources and decoded agreement, not authentic capture quality or "
            "downstream acceptance. Input hashing warms the file cache."
        ),
        "runs": [],
        "comparisons": [],
    }
    try:
        for iteration in range(args.runs):
            order = ("native", "python") if iteration % 2 == 0 else ("python", "native")
            pair = {}
            for runtime in order:
                destination = args.output / f"{runtime}-{iteration}"
                prefix = (
                    [str(args.executable)]
                    if runtime == "native"
                    else [sys.executable, "-m", "nx_ctudio.cli"]
                )
                command = [
                    *prefix,
                    "merge",
                    *[item["path"] for item in inputs],
                    "--output-directory",
                    str(destination / "outputs"),
                    "--outputs",
                    str(args.outputs),
                    "--json",
                ]
                if args.radial:
                    command.append("--chromatic-aberration-correction")
                observed = measured(
                    command, destination, args.timeout, args.memory_mib * 1024**2
                )
                observed.update(runtime=runtime, iteration=iteration)
                report["runs"].append(observed)
                if observed["returncode"] or observed["cli"]["status"] != "ok":
                    raise RuntimeError(
                        f"{runtime} benchmark failed; inspect retained logs."
                    )
                pair[runtime] = observed["cli"]
                print(
                    json.dumps(
                        {
                            k: observed[k]
                            for k in (
                                "runtime",
                                "iteration",
                                "elapsed_seconds",
                                "max_rss_kib",
                            )
                        }
                    ),
                    flush=True,
                )
            equality = compare(pair["native"], pair["python"])
            report["comparisons"].append(equality)
            if equality["different_samples"]:
                raise ValueError("Decoded full-resolution samples differ.")
        for item in inputs:
            if digest(Path(item["path"])) != item["sha256"]:
                raise ValueError("Input changed during benchmark.")
        if digest(args.executable) != report["executable_sha256"]:
            raise ValueError("Executable changed during benchmark.")
        report["passed"] = True
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--input-report", type=Path, required=True)
    parser.add_argument("--input-report-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--outputs", type=int, choices=(1, 2, 4, 8), default=4)
    parser.add_argument("--radial", action="store_true")
    parser.add_argument("--runs", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--memory-mib", type=int, default=4096)
    args = parser.parse_args()
    if platform.system() != "Linux":
        parser.error("This resource runner uses Linux GNU time and prlimit.")
    if not Path("/usr/bin/time").is_file():
        parser.error("Install GNU time for target-process RSS measurements.")
    root = Path(__file__).resolve().parents[1]
    args.output, args.executable = args.output.resolve(), args.executable.resolve()
    if args.output.is_relative_to(root):
        parser.error("Benchmark outputs must stay outside the repository.")
    report = run(args)
    print(
        json.dumps(
            {"passed": report["passed"], "report": str(args.output / "report.json")}
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
