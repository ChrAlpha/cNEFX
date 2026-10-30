"""Exercise all native CLI modes, JSON reports and failure atomicity."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import rawpy

import ci_fixture
from tiff_fixture import tree_digest
from verify_cpp_format import pixels


def command(argv: list[str], code: int = 0) -> subprocess.CompletedProcess:
    result = subprocess.run(argv, capture_output=True, text=True, timeout=45)
    assert result.returncode == code, (argv, result)
    return result


def run(executable: Path, injector: Path, directory: Path) -> dict:
    directory.mkdir(parents=True)
    ci_fixture.pixels = pixels
    cases = []
    modes = (
        (4, 1),
        (8, 1),
        (8, 2),
        (16, 1),
        (16, 4),
        (32, 1),
        (32, 2),
        (32, 4),
        (32, 8),
    )
    for model in ("NIKON Z 8", "NIKON Z f", "NIKON Z6_3", "NIKON Z5_2"):
        for count, outputs in modes:
            folder = directory / f"{model.replace(' ', '_')}-{count}-{outputs}"
            folder.mkdir()
            sources = ci_fixture.sequence(folder, count, model=model)
            output = folder / "result"
            result = command(
                [
                    str(executable),
                    "merge",
                    *map(str, sources),
                    "--outputs",
                    str(outputs),
                    "--output-directory",
                    str(output),
                    "--json",
                ]
            )
            report = json.loads(result.stdout)
            assert report == json.loads((output / "manifest.json").read_text())
            assert report["status"] == "ok" and report["schema_version"] == 1
            assert report["camera"]["model"] == model
            assert report["output_count"] == outputs
            assert report["warnings"] == []
            assert len(list(output.iterdir())) == outputs + 1
            golden = ci_fixture.merge_expectations(model, count, outputs)
            assert [
                hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
            ] == golden["input_sha256"]
            for item, expected in zip(
                report["outputs"], golden["outputs"], strict=True
            ):
                path = Path(item["output"])
                assert item["input_indices"] == expected["input_indices"]
                assert item["shape"] == expected["shape"]
                assert item["samples_verified"] == expected["samples"]
                assert item["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
                with rawpy.imread(str(path)) as image:
                    ci_fixture.check_rgb(image.raw_image[..., :3], expected)
                assert tree_digest(path.read_bytes()) == expected["tree_sha256"]
            for item, path in zip(report["inputs"], sources, strict=True):
                assert item["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
            cases.append(
                {
                    "model": model,
                    "inputs": count,
                    "outputs": outputs,
                    "samples": report["samples_verified"],
                    "samples_match": True,
                }
            )
    folder = directory / "single"
    folder.mkdir()
    sources = ci_fixture.sequence(folder)
    target = folder / "unicode space-\u6d4b\u8bd5.nefx"
    args = [str(executable), "merge", *map(str, sources), "-o", str(target), "--json"]
    report = json.loads(command(args).stdout)
    assert report["output"] == str(target)
    failure = json.loads(command(args, 1).stdout)
    assert failure["error"]["code"] == "merge_failed"
    quoted_target = folder / 'quoted-".nefx'
    quoted = json.loads(
        command(
            [
                str(executable),
                "merge",
                *map(str, sources),
                "-o",
                str(quoted_target),
                "--json",
            ],
            1 if os.name == "nt" else 0,
        ).stdout
    )
    if os.name == "nt":
        assert quoted["error"]["code"] == "merge_failed"
        assert not quoted_target.exists()
    else:
        assert quoted["output"] == str(quoted_target)
    for extra in (
        [],
        ["--json"],
        ["--outputs", "0"],
        ["--unknown"],
        ["--outputs", "2", "-o", "a.nefx"],
    ):
        result = command([str(executable), "merge", *extra], 2)
        assert not result.stdout and result.stderr
    assert "cNEFX" in command([str(executable), "--version"]).stdout
    assert "Usage:" in command([str(executable), "--help"]).stdout
    injected = []
    for mode in (
        "order",
        "preflight",
        "unread",
        "replacement",
        "cancel_decode",
        "write_failure",
        "after_write",
        "cancel",
        "restored",
        "late_hash",
        "target_race",
    ):
        folder = directory / f"inject-{mode}"
        folder.mkdir()
        sources = ci_fixture.sequence(folder, 8)
        command(
            [
                str(injector),
                "--inject",
                mode,
                str(folder / "result"),
                *map(str, sources),
            ]
        )
        injected.append(mode)
    cancellation = "not_run_on_windows"
    if os.name != "nt":
        folder = directory / "signal"
        folder.mkdir()
        sources = ci_fixture.sequence(folder, 32, shape=(480, 640))
        args = [
            str(executable),
            "merge",
            *map(str, sources),
            "--output",
            str(folder / "cancelled.nefx"),
            "--json",
        ]
        process = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        try:
            deadline = time.monotonic() + 10
            while not list(folder.glob(".cnefx-*")):
                assert process.poll() is None
                assert time.monotonic() < deadline
                time.sleep(0.001)
            process.send_signal(signal.SIGINT)
            stdout, stderr = process.communicate(timeout=10)
            assert process.returncode == 130, (stdout, stderr)
            assert json.loads(stdout)["error"]["code"] == "cancelled"
            assert not (folder / "cancelled.nefx").exists()
            assert not list(folder.glob(".cnefx-*"))
            cancellation = "SIGINT_exit_130_no_output_or_staging"
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    report = {
        "evidence_kind": "authored_native_cli_modes_and_lifecycle",
        "expectations_sha256": ci_fixture.GOLDEN_SHA256,
        "passed": True,
        "cases": cases,
        "injected_failures": injected,
        "cancellation": cancellation,
        "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
    }
    (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--injector", type=Path, required=True)
    parser.add_argument("--work-directory", type=Path, required=True)
    args = parser.parse_args()
    report = run(
        args.executable.resolve(),
        args.injector.resolve(),
        args.work_directory.resolve(),
    )
    print(json.dumps({"cases": len(report["cases"]), "passed": report["passed"]}))


if __name__ == "__main__":
    main()
