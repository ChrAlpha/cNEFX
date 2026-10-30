"""Synthetic multi-sequence discovery, grouping, isolation and publication."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
from pathlib import Path

import numpy as np
import rawpy

import ci_fixture
from tiff_fixture import read, serialize, tree_digest
from verify_cpp_format import pixels as spatial_pixels

MODES = ((4, 1), (8, 1), (8, 2), (16, 1), (16, 4), (32, 1), (32, 2), (32, 4), (32, 8))


def parts(tree):
    exif = tree.tags[34665].children[0]
    radial = tree.tags[330].children[1].tags[0xC7D5].children[0]
    return exif, exif.tags[37500].children[0], radial


def sequence(
    directory: Path,
    counts: list[int],
    model: str = "NIKON Z 8",
    *,
    calibration: bool = False,
    pattern: str = "constant",
) -> list[Path]:
    """Consecutive sequences like AE brackets: one burst and exposure each."""
    paths = []
    original = ci_fixture.pixels
    try:
        for number, count in enumerate(counts):
            folder = directory / f"sequence-{number + 1}"
            folder.mkdir(parents=True)

            def pixels(index, shape, number=number):
                if pattern == "spatial":
                    return spatial_pixels(index, shape)
                if pattern == "random":
                    generator = np.random.default_rng((number, index))
                    return generator.integers(0, 16384, shape, dtype=np.uint16)
                return np.full(shape, 1400 + index * 4 + number * 256, dtype=np.uint16)

            ci_fixture.pixels = pixels
            for index, path in enumerate(
                ci_fixture.sequence(folder, count, model=model)
            ):
                tree = read(path.read_bytes())
                exif, maker, _ = parts(tree)
                capture = struct.pack("<III", 123 + number, index + 1, 1)
                maker.set(0x56, 7, b"0100" + capture)
                maker.numbers(0xA7, 4, 101 + len(paths))
                selector = bytearray(maker.tags[0x5E].payload)
                selector[5] = index + 1
                maker.set(0x5E, 7, bytes(selector))
                exif.set(33434, 5, struct.pack("<II", 1, 100 << number))
                exif.set(37380, 10, struct.pack("<ii", -number, 1))
                if calibration:
                    exif.numbers(34855, 3, 100 << number)
                    gains = struct.pack("<8I", 2 + number, 1, 1, 1, 1, 1, 1, 1)
                    maker.set(0x0C, 5, gains)
                renamed = folder / f"S{number + 1}-{index + 1:02d}.NEF"
                renamed.write_bytes(serialize(tree))
                path.unlink()
                paths.append(renamed)
    finally:
        ci_fixture.pixels = original
    return paths


def command(args: list, code: int = 0) -> subprocess.CompletedProcess:
    args = [str(arg) for arg in args]
    result = subprocess.run(args, capture_output=True, text=True, timeout=45)
    assert result.returncode == code, (args, result)
    return result


def merge(executable, *arguments, code=0) -> dict:
    result = command([executable, "merge", *arguments, "--json"], code)
    return json.loads(result.stdout)


def name(first: Path, outputs: int, local: int) -> str:
    suffix = f"_{local + 1:02d}" if outputs > 1 else ""
    return f"{first.stem}_merged{suffix}.NEFX"


def samples(path):
    with rawpy.imread(str(path)) as image:
        return image.raw_image[..., :3].copy()


def check_outputs(report, sources, count, outputs, sequences, golden, offset):
    """Outputs follow sequence order; each matches its own sequence's pixels."""
    assert report["status"] == "ok" and report["schema_version"] == 2
    assert report["warnings"] == [] and len(report["sequences"]) == sequences
    assert all(item["status"] == "merged" for item in report["sequences"])
    assert report["output_count"] == sequences * outputs
    assert [item["path"] for item in report["inputs"]] == [str(p) for p in sources]
    assert [item["sequence_index"] for item in report["inputs"]] == [
        i // count for i in range(len(sources))
    ]
    for index, item in enumerate(report["outputs"]):
        number, local = divmod(index, outputs)
        expected = golden["outputs"][local]
        assert item["sequence_index"] == number
        assert item["input_indices"] == [
            i + number * count for i in expected["input_indices"]
        ]
        path = Path(item["output"])
        assert path.name == name(sources[number * count], outputs, local)
        assert item["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        pixels = samples(path).astype(np.int32) - (number * 256 if offset else 0)
        ci_fixture.check_rgb(pixels, expected)
        source = parts(read(sources[number * count].read_bytes()))[0]
        written = parts(read(path.read_bytes()))[0]
        for code in (33434, 33437, 37380, 34855):
            assert written.tags[code].payload == source.tags[code].payload


def check_case(executable, folder, model, count, outputs, jobs, corrected):
    folder.mkdir()
    # Saved constant-scene expectations exist for Z8 4->1 and 32->4 only. Other
    # modes use the saved spatial-pattern matrix, identical in every sequence.
    constant = model == "NIKON Z 8" and (count, outputs) in ((4, 1), (32, 4))
    pattern = "constant" if constant else "spatial"
    sources = sequence(folder, [count] * 3, model, calibration=True, pattern=pattern)
    hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in sources]
    target = folder / "result"
    options = ["--outputs", outputs, "--output-directory", target, "--jobs", jobs]
    if corrected:
        options.append("--chromatic-aberration-correction")
    report = merge(executable, *sources, *options)
    assert report == json.loads((target / "manifest.json").read_text())
    assert report["output_directory"] == str(target)
    assert len(list(target.iterdir())) == 3 * outputs + 1
    golden = ci_fixture.merge_expectations(model, count, outputs, pattern=pattern)
    check_outputs(report, sources, count, outputs, 3, golden, constant)
    assert hashes == [hashlib.sha256(p.read_bytes()).hexdigest() for p in sources]
    return {
        "model": model,
        "captures_per_sequence": count,
        "outputs_per_sequence": outputs,
        "jobs": jobs,
        "corrected": corrected,
        "pattern": pattern,
        "input_sha256": hashes,
        "output_sha256": [o["sha256"] for o in report["outputs"]],
    }


def ordinary(source: Path, target: Path) -> None:
    """A capture whose pixel-shift flag is off, like an ordinary photo."""
    tree = read(source.read_bytes())
    maker = parts(tree)[1]
    capture = bytearray(maker.tags[0x56].payload)
    capture[12] = 0
    maker.set(0x56, 7, bytes(capture))
    target.write_bytes(serialize(tree))


def configure(source, target, count, capture, iso, selection, radial):
    """Write sequence-specific ISO, sparse selection and radial metadata."""
    tree = read(source.read_bytes())
    exif, maker, correction = parts(tree)
    exif.numbers(34855, 3, iso)
    mode = (4, 8, 16, 32).index(count) + 1
    fields = (mode, capture + 1, 48, 64, 2, 6 + selection * 4, 0, 0, 1, 1, 1, 0, 5)
    maker.set(0x5E, 7, b"0100" + struct.pack("<BB6H3BxBxB3x", *fields))
    center = (radial * 400 + capture * 80, 256, -200 - radial * 300, 256)
    maker.set(0x54, 7, b"0100" + struct.pack("<4i", *center))
    coefficients = bytearray(correction.tags[7].payload)
    values = (10000 + radial * 2500, 8000, -1000, -6000, 4000, 2000)
    for offset, value in zip((44, 28, 20, 80, 64, 56), values, strict=True):
        struct.pack_into("<i", coefficients, offset, value)
    correction.set(7, 7, bytes(coefficients))
    target.write_bytes(serialize(tree))


def isolation(executable, directory):
    """One run equals separate runs; each sequence keeps its own settings."""

    def settings(number):
        # Sparse repair applies only at ISO 3200 or below.
        iso = (3200, 3200, 6400)[number]
        return {"iso": iso, "selection": number, "radial": number}

    cases = []
    for count, outputs in ((4, 1), (32, 4)):
        folder = directory / f"isolation-{count}-{outputs}"
        folder.mkdir()
        paths = sequence(folder, [count] * 3, calibration=True, pattern="random")
        for index, path in enumerate(paths):
            number, capture = divmod(index, count)
            configure(path, path, count, capture, **settings(number))

        def run(sources, target):
            options = ["--outputs", outputs, "--output-directory", target]
            report = merge(
                executable, *sources, *options, "--chromatic-aberration-correction"
            )
            return [Path(item["output"]) for item in report["outputs"]]

        separate = [
            run(paths[i * count : (i + 1) * count], folder / f"separate-{i}")
            for i in range(3)
        ]
        for index, path in enumerate(run(paths, folder / "together")):
            expected = separate[index // outputs][index % outputs]
            assert tree_digest(path.read_bytes()) == tree_digest(expected.read_bytes())
        # Negative controls: giving one sequence the first sequence's setting
        # changes samples, so the comparison above detects a shared setting.
        controls = {}
        for field, number in (("selection", 1), ("iso", 2), ("radial", 1)):
            control = folder / f"control-{field}"
            control.mkdir()
            changed = settings(number) | {field: settings(0)[field]}
            copies = []
            for capture in range(count):
                source = paths[number * count + capture]
                copies.append(control / source.name)
                configure(source, copies[-1], count, capture, **changed)
            differences = [
                int(np.count_nonzero(samples(left) != samples(right)))
                for left, right in zip(
                    separate[number], run(copies, control / "result"), strict=True
                )
            ]
            assert all(differences), (count, field, differences)
            controls[field] = differences
        cases.append(
            {
                "captures_per_sequence": count,
                "outputs_per_sequence": outputs,
                "input_sha256": [
                    hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
                ],
                "control_changed_samples": controls,
            }
        )
    return cases


def run(executable: Path, injector: Path, directory: Path) -> dict:
    directory.mkdir(parents=True)
    cases = []
    for model in ("NIKON Z 8", "NIKON Z f"):
        for count, outputs in MODES:
            for jobs, corrected in ((1, False), (2, True)):
                folder = (
                    directory / f"{model.replace(' ', '_')}-{count}-{outputs}-{jobs}"
                )
                cases.append(
                    check_case(
                        executable, folder, model, count, outputs, jobs, corrected
                    )
                )
    separated = isolation(executable, directory)
    injected = []
    for mode, target in (
        *(
            (mode, "result")
            for mode in (
                "incomplete",
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
            )
        ),
        ("write_failure", "-"),
        ("cancel", "-"),
        ("race_last", "-"),
        ("race_second", "-"),
        ("cancel_later", "-"),
        ("preflight", "-"),
    ):
        folder = directory / f"inject-{mode}-{'beside' if target == '-' else 'set'}"
        count, outputs = (8, 2) if mode == "race_second" else (4, 1)
        sources = sequence(folder, [count] * (2 if mode == "race_second" else 3))
        destination = target if target == "-" else folder / target
        command([injector, "--inject", mode, destination, outputs, *sources])
        injected.append({"mode": mode, "beside_sources": target == "-"})
    report = {
        "evidence_kind": "synthetic_multi_sequence_saved_spatial_and_offset_samples",
        "passed": True,
        "expectations_sha256": ci_fixture.GOLDEN_SHA256,
        "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "rawpy": rawpy.__version__,
        "rawpy_libraw": rawpy.libraw_version,
        "cases": cases,
        "isolation": separated,
        "injected_failures": injected,
    }
    (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
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
