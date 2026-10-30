"""Native build verification and standalone archive checks; no capture assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

import rawpy

from ci_fixture import GOLDEN_SHA256, check_rgb, merge_expectations, sequence

ROOT = Path(__file__).resolve().parents[1]


def run(*args: str, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(args, check=True, timeout=300, **kwargs)


def version() -> str:
    text = (ROOT / "CMakeLists.txt").read_text(encoding="ascii")
    match = re.search(r"project\(cNEFX VERSION ([0-9.]+) LANGUAGES CXX\)", text)
    if not match:
        raise ValueError("Missing CMake project version.")
    value = match[1]
    if os.environ.get("GITHUB_REF_TYPE") == "tag":
        if os.environ.get("GITHUB_REF_NAME") != f"v{value}":
            raise ValueError("The version tag must match the native project version.")
    return value


def check() -> None:
    version()
    base = os.environ.get("CI_BASE_SHA", "")
    if not base or set(base) == {"0"}:
        base = run(
            "git", "merge-base", "HEAD", "origin/master", capture_output=True, text=True
        ).stdout.strip()
        if (
            base
            == run(
                "git", "rev-parse", "HEAD", capture_output=True, text=True
            ).stdout.strip()
        ):
            base = "HEAD^"
    run("git", "diff", "--check", base, "HEAD")
    paths = (
        run(
            "git",
            "diff",
            "--name-only",
            "--diff-filter=ACMR",
            "-z",
            base,
            "HEAD",
            capture_output=True,
        )
        .stdout.decode()
        .split("\0")
    )
    python = [p for p in paths if p.endswith(".py") and Path(p).is_file()]
    if python:
        run(sys.executable, "-m", "ruff", "check", "--no-cache", "--", *python)
        run(
            sys.executable,
            "-m",
            "ruff",
            "format",
            "--no-cache",
            "--check",
            "--",
            *python,
        )
    for name in paths:
        path = Path(name)
        if path.is_file() and path.suffix in {
            ".py",
            ".cpp",
            ".hpp",
            ".json",
            ".md",
            ".cmake",
            ".yml",
        }:
            path.read_text(encoding="ascii")


def executable(build: Path, config: str, name: str) -> Path:
    name += ".exe" if os.name == "nt" else ""
    for path in (build / name, build / config / name):
        if path.is_file():
            return path.resolve()
    raise ValueError(f"Missing built executable: {name}")


def verify(build: Path, config: str) -> None:
    output = build / "verification"
    if output.exists():
        raise ValueError(
            "Verification directory exists; preserve it or choose a fresh build."
        )
    output.mkdir()
    for script, program in (
        ("core", "cnefx_core_test"),
        ("format", "cnefx_format_test"),
        ("cli", "cnefx"),
    ):
        args = [
            sys.executable,
            str(ROOT / f"tools/verify_cpp_{script}.py"),
            "--executable",
            str(executable(build, config, program)),
            "--work-directory",
            str(output / script),
        ]
        if script == "cli":
            args += ["--injector", str(executable(build, config, "cnefx_merge_test"))]
        run(*args)


def smoke(program: Path, work: Path) -> dict:
    env = {"PATH": ""}
    for key in ("SYSTEMROOT", "WINDIR", "TEMP", "TMP"):
        if key in os.environ:
            env[key] = os.environ[key]

    def invoke(*args: str) -> subprocess.CompletedProcess:
        return run(
            str(program), *args, cwd=work, env=env, capture_output=True, text=True
        )

    assert invoke("--version").stdout.strip() == f"cNEFX {version()}"
    assert "Usage:" in invoke("--help").stdout
    cases = []
    for count, outputs in ((4, 1), (32, 4)):
        folder = work / f"capture-{count}"
        folder.mkdir()
        paths = sequence(folder, count)
        destination = work / ("single.nefx" if outputs == 1 else "set")
        option = "--output" if outputs == 1 else "--output-directory"
        result = invoke(
            "merge",
            *map(str, paths),
            option,
            str(destination),
            "--outputs",
            str(outputs),
            "--json",
        )
        report = json.loads(result.stdout)
        assert report["status"] == "ok"
        items = report["outputs"] if outputs > 1 else [report]
        golden = merge_expectations("NIKON Z 8", count, outputs, pattern="constant")
        assert [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths] == golden[
            "input_sha256"
        ]
        for item, expected in zip(items, golden["outputs"], strict=True):
            assert (
                item.get("input_indices", list(range(count)))
                == expected["input_indices"]
            )
            with rawpy.imread(item["output"]) as image:
                check_rgb(image.raw_image[..., :3], expected)
        cases.append(
            {
                "inputs": count,
                "outputs": outputs,
                "samples_verified": report["samples_verified"],
            }
        )
    return {
        "passed": True,
        "path_empty": True,
        "expectations_sha256": GOLDEN_SHA256,
        "cases": cases,
    }


def package(build: Path, config: str) -> None:
    destination = build / "packages"
    run(
        "cpack",
        "--config",
        str(build / "CPackConfig.cmake"),
        "-C",
        config,
        "-B",
        str(destination),
    )
    archives = sorted(destination.glob("cnefx-*.tar.gz")) + sorted(
        destination.glob("cnefx-*.zip")
    )
    if len(archives) != 1:
        raise ValueError("Expected one native package archive.")
    archive = archives[0]
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert Path(str(archive) + ".sha256").read_text().split()[0] == digest
    with tempfile.TemporaryDirectory(prefix="package-check-", dir=build) as temporary:
        work = Path(temporary)
        if archive.suffix == ".zip":
            with zipfile.ZipFile(archive) as source:
                source.extractall(work)
        else:
            with tarfile.open(archive) as source:
                source.extractall(work, filter="data")
        binaries = list(work.glob("*/bin/cnefx")) + list(work.glob("*/bin/cnefx.exe"))
        assert len(binaries) == 1
        info_path = next(work.glob("*/share/cnefx/build-info.json"))
        info = json.loads(info_path.read_text())
        assert info["version"] == version() and info["libraw_version"] == "0.22.3"
        assert (
            info["libraw_archive_sha256"]
            == "789f3c33466210a1a0907f747a97c715d21bbf4af712f257fcbbd90111b227e4"
        )
        assert (info_path.parent / "libraw" / "COPYRIGHT").is_file()
        report = smoke(binaries[0], work)
        report.update(
            build=info,
            archive_sha256=digest,
            archive_bytes=archive.stat().st_size,
            executable_bytes=binaries[0].stat().st_size,
        )
    (destination / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "verify", "package"))
    parser.add_argument("--build-dir", type=Path, default=Path("build/release"))
    parser.add_argument("--config", default="Release")
    args = parser.parse_args()
    if args.command == "check":
        check()
    elif args.command == "verify":
        verify(args.build_dir.resolve(), args.config)
    else:
        package(args.build_dir.resolve(), args.config)


if __name__ == "__main__":
    main()
