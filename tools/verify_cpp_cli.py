"""Exercise all native CLI modes, JSON reports and failure atomicity."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import rawpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import ci_fixture  # noqa: E402
from verify_cpp_format import compare_tree, pixels  # noqa: E402

from nx_ctudio._tiff import read  # noqa: E402
from nx_ctudio.nefx import encode  # noqa: E402
from nx_ctudio.pixel_shift import output_groups, reconstruct  # noqa: E402


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
            selector = b"0100" + bytes([(4, 8, 16, 32).index(count) + 1]) + bytes(23)
            for item, group in zip(
                report["outputs"], output_groups(count, outputs), strict=True
            ):
                expected = reconstruct(
                    [pixels(i) for i in group], (4, 4, 56, 40), selector, 100
                )
                path = Path(item["output"])
                assert item["input_indices"] == list(group)
                assert item["shape"] == list(expected.shape)
                assert item["samples_verified"] == expected.size
                assert item["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
                with rawpy.imread(str(path)) as image:
                    np.testing.assert_array_equal(image.raw_image[..., :3], expected)
                compare_tree(
                    read(path.read_bytes()),
                    read(encode(sources[group[0]].read_bytes(), expected, len(group))),
                )
            for item, path in zip(report["inputs"], sources, strict=True):
                assert item["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
            cases.append(
                {
                    "model": model,
                    "inputs": count,
                    "outputs": outputs,
                    "samples": report["samples_verified"],
                    "different_samples": 0,
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
