"""Bounded CI checks and standalone packaging; no vendor or capture assets."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args: str, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(args, check=True, timeout=600, **kwargs)


def version() -> str:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="ascii"))
    expected = project["project"]["version"]
    from nx_ctudio import __version__

    if expected != __version__ or expected != importlib.metadata.version("nx-ctudio"):
        raise ValueError("Project, installed package, and CLI versions must match.")
    if os.environ.get("GITHUB_REF_TYPE") == "tag":
        if os.environ.get("GITHUB_REF_NAME") != f"v{expected}":
            raise ValueError("The version tag must match the package version.")
    return expected


def check() -> None:
    version()
    base = os.environ.get("CI_BASE_SHA", "")
    if not base or set(base) == {"0"}:
        # A new branch/tag or manual dispatch checks its complete branch delta.
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
    changed = (
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
    paths = [name for name in changed if name.endswith(".py") and Path(name).is_file()]
    if paths:
        run(sys.executable, "-m", "ruff", "check", "--", *paths)
        run(sys.executable, "-m", "ruff", "format", "--check", "--", *paths)


def smoke(executable: Path) -> None:
    """Exercise the installed/frozen CLI outside the checkout without PATH tools."""
    expected = version()
    with tempfile.TemporaryDirectory(prefix="nxc-smoke-") as directory:
        work = Path(directory)
        env = os.environ.copy()
        for name in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
            env.pop(name, None)
        env["PATH"] = ""

        def invoke(*args: str, code: int = 0) -> subprocess.CompletedProcess:
            result = subprocess.run(
                [str(executable), *args],
                cwd=work,
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode != code:
                raise RuntimeError(f"CLI {args}: {result.returncode}: {result.stderr}")
            return result

        assert invoke("--version").stdout.strip() == f"NX Ctudio {expected}"
        assert "merge" in invoke("--help").stdout
        assert "--chromatic-aberration-correction" in invoke("merge", "--help").stdout
        assert "usage:" in invoke("merge", code=2).stderr
        report = json.loads(invoke("inspect", "missing.NEF", "--json", code=2).stdout)
        assert report["error"]["code"] == "dependency_missing"
        report = json.loads(
            invoke(
                "merge",
                "missing.NEF",
                "-o",
                "out.nefx",
                "--json",
                code=1,
            ).stdout
        )
        assert report["error"]["code"] == "merge_failed"
        source = work / "source.NEF"
        source.write_bytes(b"synthetic duplicate input")
        report = json.loads(
            invoke(
                "merge",
                *([str(source)] * 4),
                "-o",
                "out.nefx",
                "--json",
                code=1,
            ).stdout
        )
        assert "Duplicate" in report["error"]["message"]
        files = [work / f"invalid-{i}.NEF" for i in range(4)]
        for item in files:
            item.write_bytes(b"synthetic invalid TIFF")
        report = json.loads(
            invoke(
                "merge",
                *map(str, files),
                "-o",
                "out.nefx",
                "--json",
                code=1,
            ).stdout
        )
        assert report["error"]["code"] == "merge_failed"
        assert set(work.iterdir()) == {source, *files}
    print(
        json.dumps({"status": "ok", "executable": str(executable), "version": expected})
    )


def build(target: str) -> None:
    expected = version()
    machine = {"aarch64": "arm64", "AMD64": "x86_64"}.get(
        platform.machine(), platform.machine()
    )
    system = {"Linux": "linux", "Darwin": "macos", "Windows": "windows"}[
        platform.system()
    ]
    if target != f"{system}-{machine}":
        raise ValueError("Build target must match the native runner architecture.")
    dist = ROOT / "dist"
    work = ROOT / "build" / target
    work.mkdir(parents=True, exist_ok=True)
    run(
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onedir",
        "--name",
        "nxc",
        "--distpath",
        str(work / "dist"),
        "--workpath",
        str(work / "work"),
        "--specpath",
        str(work),
        "--collect-all",
        "rawpy",
        "--copy-metadata",
        "numpy",
        "--copy-metadata",
        "rawpy",
        "--copy-metadata",
        "tifffile",
        str(ROOT / "src/nx_ctudio/__main__.py"),
        cwd=ROOT,
    )
    bundle = work / "dist/nxc"
    manifest = {
        "schema_version": 1,
        "version": expected,
        "target": target,
        "revision": run(
            "git", "rev-parse", "HEAD", capture_output=True, text=True
        ).stdout.strip(),
        "python": platform.python_version(),
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "rawpy", "tifffile", "pyinstaller")
        },
        "lock_sha256": hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest(),
        "limitations": [
            "Experimental; authentic capture and downstream application "
            "acceptance remain open.",
            "Output-directory publication requires Linux renameat2 "
            "and RENAME_NOREPLACE.",
            "Metadata inspection requires a separately installed ExifTool on PATH.",
            "Unsigned build; platform signing and notarization are not provided.",
        ],
    }
    (bundle / "build-info.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="ascii"
    )
    dist.mkdir(exist_ok=True)
    archive = shutil.make_archive(
        str(dist / f"nxc-{expected}-{target}"),
        "zip" if system == "windows" else "gztar",
        root_dir=bundle.parent,
        base_dir=bundle.name,
    )
    with tempfile.TemporaryDirectory(prefix="nxc-extracted-") as directory:
        shutil.unpack_archive(archive, directory)
        smoke(Path(directory) / "nxc" / ("nxc.exe" if system == "windows" else "nxc"))
    artifacts = sorted(
        path
        for path in dist.iterdir()
        if path.is_file() and path.name.endswith((".whl", ".tar.gz", ".zip"))
    )
    checksums = []
    for path in artifacts:
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        checksums.append(f"{digest}  {path.name}\n")
    (dist / "SHA256SUMS").write_text("".join(checksums), encoding="ascii")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check")
    smoke_parser = commands.add_parser("smoke")
    smoke_parser.add_argument("--executable", type=Path)
    build_parser = commands.add_parser("build")
    build_parser.add_argument("--target", required=True)
    args = parser.parse_args()
    if args.command == "check":
        check()
    elif args.command == "build":
        build(args.target)
    else:
        executable = args.executable or Path(sys.executable).with_name(
            "nxc.exe" if sys.platform == "win32" else "nxc"
        )
        smoke(executable.resolve(strict=True))


if __name__ == "__main__":
    main()
