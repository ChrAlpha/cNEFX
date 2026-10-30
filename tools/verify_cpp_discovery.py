"""Synthetic discovery: folders, patterns, repeated runs and partial failures."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np

import ci_fixture
from tiff_fixture import read, serialize
from verify_cpp_sequences import (
    check_outputs,
    merge,
    name,
    ordinary,
    samples,
    sequence,
)


def zero_configuration(executable, directory):
    """A folder of three sequences, an ordinary photo and other files."""
    built = sequence(directory / "build", [4] * 3)
    folder = directory / "card"
    folder.mkdir()
    # Lowercase extensions count; hidden files and subdirectories do not. A
    # recursive scan would find a duplicate shot in the nested copy.
    sources = [
        folder / (path.stem + (".nef" if i >= 8 else ".NEF"))
        for i, path in enumerate(built)
    ]
    for path, target in zip(built, sources, strict=True):
        shutil.copy2(path, target)
    ordinary(sources[0], folder / "DSC_0001.NEF")
    # Another camera's big-endian NEF cannot hold a pixel-shift capture here.
    (folder / "OTHER.NEF").write_bytes(b"MM\0*\0\0\0\x08" + bytes(64))
    (folder / "notes.txt").write_text("not a capture\n")
    (folder / "._S1-01.NEF").write_bytes(b"\0\5\26\7 AppleDouble")
    (folder / "nested").mkdir()
    shutil.copy2(built[0], folder / "nested" / built[0].name)
    golden = ci_fixture.merge_expectations("NIKON Z 8", 4, 1, pattern="constant")
    report = merge(executable, folder)
    assert "output_directory" not in report
    assert [(item["path"], item["reason"][:16]) for item in report["skipped"]] == [
        (str(folder / "DSC_0001.NEF"), "No active pixel-"),
        (str(folder / "OTHER.NEF"), "Unsupported NEF:"),
    ]
    check_outputs(report, sources, 4, 1, 3, golden, True)
    assert sorted(p.name for p in folder.iterdir() if p.suffix == ".NEFX") == [
        name(sources[i * 4], 1, 0) for i in range(3)
    ]
    assert not list(folder.glob(".cnefx-*"))
    # A repeated run recognizes the outputs and merges nothing again.
    hashes = {
        p: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in folder.rglob("*")
        if p.is_file()
    }
    report = merge(executable, folder)
    assert report["status"] == "ok" and report["outputs"] == []
    assert [item["existing"] for item in report["sequences"]] == [
        [str(folder / name(sources[i * 4], 1, 0))] for i in range(3)
    ]
    assert hashes == {
        p: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in folder.rglob("*")
        if p.is_file()
    }
    # Explicit files in reverse order: sequences follow first appearance and
    # captures their shot numbers.
    target = directory / "reversed"
    merged = folder / name(sources[0], 1, 0)
    report = merge(executable, merged, *reversed(built), "--output-directory", target)
    assert report["skipped"] == [
        {"path": str(merged), "reason": "Already merged NEFX."}
    ]
    order = [p for number in (2, 1, 0) for p in built[number * 4 : number * 4 + 4]]
    assert [item["path"] for item in report["inputs"]] == [str(p) for p in order]
    for item, number in zip(report["outputs"], (2, 1, 0), strict=True):
        assert Path(item["output"]).name == name(built[number * 4], 1, 0)
        pixels = samples(Path(item["output"])).astype(np.int32) - number * 256
        ci_fixture.check_rgb(pixels, golden["outputs"][0])
    return {"skipped": ["ordinary", "merged"], "outputs": 3, "reversed": True}


def mixed_sizes(executable, directory):
    """Sequences of different lengths and models keep their own offsets and outputs."""
    directory.mkdir()
    sources = sequence(directory, [4, 32], pattern="spatial")
    for path in sources[4:]:
        tree = read(path.read_bytes())
        tree.set(272, 2, b"NIKON Z f\0")
        path.write_bytes(serialize(tree))
    report = merge(executable, *sources, "--output-directory", directory / "result")
    assert len(report["sequences"]) == 2 and report["output_count"] == 2
    models = ("NIKON Z 8", "NIKON Z f")
    for item, (number, count, first) in zip(
        report["outputs"], ((0, 4, 0), (1, 32, 4)), strict=True
    ):
        model = models[number]
        assert report["sequences"][number]["camera"]["model"] == model
        expected = ci_fixture.merge_expectations(model, count, 1)["outputs"][0]
        assert item["sequence_index"] == number
        assert item["input_indices"] == [i + first for i in expected["input_indices"]]
        ci_fixture.check_rgb(samples(Path(item["output"])), expected)
    return {"captures_per_sequence": [4, 32]}


def descriptors(executable, directory):
    """Skipped files close at once and captures open one sequence at a time."""
    if os.name == "nt":
        return "not_run_on_windows"
    import resource

    directory.mkdir()
    few = sequence(directory / "few", [4])
    for index in range(200):
        ordinary(few[0], few[0].parent / f"DSC_{index:04d}.NEF")
    many = sequence(directory / "many", [32] * 3, pattern="spatial")
    for sources in ([few[0].parent], many):

        def limit():
            resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))

        result = subprocess.run(
            [str(executable), "merge", *map(str, sources), "--json"],
            capture_output=True,
            text=True,
            timeout=120,
            preexec_fn=limit,
        )
        assert result.returncode == 0, result
        assert json.loads(result.stdout)["status"] == "ok"
    return {"descriptor_limit": 64, "skipped_files": 200, "captures": 96}


def resume(executable, directory):
    """Problems affect only their own sequence; repeating the run finishes the rest."""
    sources = sequence(directory, [4] * 3)
    folders = sorted({path.parent for path in sources})
    missing, held = sources[5], sources[5].with_suffix(".held")
    missing.rename(held)
    blocker = sources[8].parent / name(sources[8], 1, 0)
    blocker.write_bytes(b"not a NEFX")
    corrupt = folders[0] / "S1-99.NEF"
    corrupt.write_bytes(b"not a TIFF")
    report = merge(executable, *folders, code=1)
    assert report["status"] == "partial"
    assert [item["status"] for item in report["sequences"]] == [
        "merged",
        "failed",
        "failed",
    ]
    assert "3 of 4 captures" in report["sequences"][1]["error"]
    assert "not this sequence's merged NEFX" in report["sequences"][2]["error"]
    # A file that is not even a readable TIFF cannot hold a capture: skipped.
    assert report["errors"] == []
    assert [(item["path"], item["reason"][:16]) for item in report["skipped"]] == [
        (str(corrupt), "Unsupported NEF:")
    ]
    assert [Path(o["output"]).name for o in report["outputs"]] == [
        name(sources[0], 1, 0)
    ]
    held.rename(missing)
    blocker.unlink()
    corrupt.unlink()
    report = merge(executable, *folders)
    assert [item["status"] for item in report["sequences"]] == [
        "already_merged",
        "merged",
        "merged",
    ]
    assert [o["sequence_index"] for o in report["outputs"]] == [1, 2]
    report = merge(executable, *folders)
    assert report["status"] == "ok" and report["outputs"] == []
    # A sequence with one of two outputs left resumes with the missing one only.
    pair = sequence(directory / "pair", [8], pattern="spatial")
    merge(executable, pair[0].parent, "--outputs", 2)
    second = pair[0].parent / name(pair[0], 2, 1)
    second.unlink()
    report = merge(executable, pair[0].parent, "--outputs", 2)
    assert report["sequences"][0]["existing"] == [
        str(pair[0].parent / name(pair[0], 2, 0))
    ]
    assert [Path(o["output"]) for o in report["outputs"]] == [second]
    expected = ci_fixture.merge_expectations("NIKON Z 8", 8, 2)["outputs"][1]
    shots = [
        report["inputs"][i]["shot_number"]
        for i in report["outputs"][0]["input_indices"]
    ]
    assert shots == [i + 1 for i in expected["input_indices"]]
    ci_fixture.check_rgb(samples(second), expected)
    assert not list(directory.rglob(".cnefx-*"))
    return {
        "first": "partial",
        "second": "remaining_two",
        "third": "nothing_to_do",
        "missing_output": "resumed_alone",
    }


def patterns(executable, directory):
    """Unexpanded patterns, overlapping inputs and non-NEF matches."""
    sources = sequence(directory, [4])
    folder = sources[0].parent
    (folder / "S1-01.JPG").write_bytes(b"jpeg")
    # Windows shells and quoted arguments pass the pattern through unexpanded.
    report = merge(
        executable, folder / "s1-*", folder, "--output-directory", directory / "result"
    )
    assert [item["path"] for item in report["inputs"]] == [str(p) for p in sources]
    assert report["skipped"] == [
        {"path": str(folder / "S1-01.JPG"), "reason": "Not a NEF file."}
    ]
    # '?' matches one UTF-8 character.
    ordinary(sources[0], folder / "R\u00e9.NEF")
    for pattern, text in (
        (folder / "R?.NEF", "No pixel-shift captures"),
        (folder / "*.XYZ", "No files match"),
        (directory / "seq*" / "S1-01.NEF", "only in file names"),
    ):
        assert text in merge(executable, pattern, code=1)["error"]["message"]
    return {"pattern": "s1-*", "deduplicated": 4}


def failures(executable, directory):
    cases = []
    partial = {
        "hardlink": "Hard link to another input.",
        "mixed_exposure": "Sequence metadata differs.",
        "repeated_copy": "Repeated captures",
    }
    for mode in (
        "set_incomplete",
        "only_ordinary",
        "single_file",
        "missing_input",
        *partial,
    ):
        folder = directory / mode
        sources = sequence(folder, [4] * 3)
        arguments = list(sources)
        if mode == "missing_input":
            arguments = [sources[0].parent, folder / "sequenc-2"]
        elif mode == "mixed_exposure":
            tree = read(sources[5].read_bytes())
            tree.tags[34665].children[0].set(33434, 5, struct.pack("<II", 1, 100))
            sources[5].write_bytes(serialize(tree))
        elif mode == "repeated_copy":
            shutil.copytree(sources[0].parent, folder / "backup")
            arguments.append(folder / "backup")
        elif mode == "set_incomplete":
            arguments = [*sources[:-1], "--output-directory", folder / "result"]
        elif mode == "only_ordinary":
            ordinary(sources[0], folder / "ordinary.NEF")
            arguments = [folder / "ordinary.NEF"]
        elif mode == "single_file":
            arguments.extend(["--output", folder / "result.nefx"])
        else:
            os.link(sources[0], folder / "link.NEF")
            arguments.append(folder / "link.NEF")
        report = merge(executable, *arguments, code=1)
        assert report["schema_version"] == 2
        produced = list(folder.rglob("*.NEFX")) + list(folder.rglob("result*"))
        if mode in partial:
            assert report["status"] == "partial"
            failed = [
                item for item in report["sequences"] if item["status"] == "failed"
            ]
            messages = [item["error"] for item in failed] + [
                item["reason"] for item in report["errors"]
            ]
            assert len(messages) == 1 and partial[mode] in messages[0], (mode, report)
            assert len(report["outputs"]) == (3 if mode == "hardlink" else 2)
            message = messages[0]
        else:
            assert report["error"]["code"] == "merge_failed", (mode, report)
            assert not produced, mode
            message = report["error"]["message"]
        assert not list(folder.rglob(".cnefx-*")), mode
        cases.append({"case": mode, "message": message})
    return cases


def run(executable: Path, directory: Path) -> dict:
    directory.mkdir(parents=True)
    report = {
        "evidence_kind": "synthetic_discovery_and_batch_lifecycle",
        "passed": True,
        "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "zero_configuration": zero_configuration(executable, directory / "zero"),
        "mixed_sizes": mixed_sizes(executable, directory / "mixed-sizes"),
        "descriptors": descriptors(executable, directory / "descriptors"),
        "resume": resume(executable, directory / "resume"),
        "patterns": patterns(executable, directory / "patterns"),
        "failures": failures(executable, directory / "failures"),
    }
    (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--work-directory", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.executable.resolve(), args.work_directory.resolve())
    print(json.dumps({"passed": report["passed"]}))
