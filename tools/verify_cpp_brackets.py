"""Synthetic AE bracket batching: saved pixels, per-exposure isolation, atomicity."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
from pathlib import Path

import numpy as np
import rawpy

import ci_fixture
from tiff_fixture import read, serialize, tree_digest
from verify_cpp_format import pixels as spatial_pixels

MODES = ((4, 1), (8, 1), (8, 2), (16, 1), (16, 4), (32, 1), (32, 2), (32, 4), (32, 8))


def sequence(
    directory: Path,
    count: int,
    brackets: int,
    model: str = "NIKON Z 8",
    *,
    reset: bool = True,
    calibration: bool = False,
    pattern: str = "constant",
) -> list[Path]:
    paths = []
    original = ci_fixture.pixels
    try:
        for bracket in range(brackets):
            folder = directory / f"exposure-{bracket}"
            folder.mkdir()

            def pixels(index, shape, bracket=bracket):
                if pattern == "spatial":
                    return spatial_pixels(index, shape)
                if pattern == "random":
                    generator = np.random.default_rng((bracket, index))
                    return generator.integers(0, 16384, shape, dtype=np.uint16)
                return np.full(shape, 1400 + index * 4 + bracket * 256, dtype=np.uint16)

            ci_fixture.pixels = pixels
            current = ci_fixture.sequence(folder, count, model=model)
            for index, path in enumerate(current):
                tree = read(path.read_bytes())
                exif = tree.tags[34665].children[0]
                maker = exif.tags[37500].children[0]
                shot = index + 1 + (0 if reset else bracket * count)
                maker.set(
                    0x56, 7, b"0100" + struct.pack("<III", 123 + bracket, shot, 1)
                )
                maker.numbers(0xA7, 4, 101 + bracket * count + index)
                selector = bytearray(maker.tags[0x5E].payload)
                selector[5] = index + 1
                maker.set(0x5E, 7, bytes(selector))
                exif.set(33434, 5, struct.pack("<II", 1, 100 << bracket))
                exif.set(37380, 10, struct.pack("<ii", -bracket, 1))
                if calibration:
                    exif.numbers(34855, 3, 100 << bracket)
                    maker.set(
                        0x0C, 5, struct.pack("<8I", 2 + bracket, 1, 1, 1, 1, 1, 1, 1)
                    )
                path.write_bytes(serialize(tree))
            paths.extend(current)
    finally:
        ci_fixture.pixels = original
    return paths


def command(args: list[str], code: int = 0) -> subprocess.CompletedProcess:
    result = subprocess.run(args, capture_output=True, text=True, timeout=45)
    assert result.returncode == code, (args, result)
    return result


def arguments(executable, sources, target, brackets=0, outputs=1):
    return [
        str(executable),
        "merge",
        *map(str, sources),
        *(["--ae-bracketing", str(brackets)] if brackets else []),
        "--outputs",
        str(outputs),
        "--output-directory",
        str(target),
        "--json",
    ]


def check_case(
    executable, folder, model, count, outputs, brackets, jobs, corrected, *, expected=0
):
    folder.mkdir()
    # Saved constant-scene expectations exist for 4->1 and 32->4 only. Other
    # modes use the full saved spatial-pattern matrix without pixel transforms.
    pattern = "constant" if (count, outputs) in ((4, 1), (32, 4)) else "spatial"
    sources = sequence(
        folder,
        count,
        brackets,
        model,
        reset=jobs == 1,
        calibration=True,
        pattern=pattern,
    )
    hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in sources]
    target = folder / "result"
    args = arguments(executable, sources, target, expected, outputs) + [
        "--jobs",
        str(jobs),
    ]
    if corrected:
        args.append("--chromatic-aberration-correction")
    report = json.loads(command(args).stdout)
    assert report == json.loads((target / "manifest.json").read_text())
    assert report["status"] == "ok" and report["schema_version"] == 1
    assert report["sequence_validation"] == "complete_consecutive_exposure_sequences"
    assert report["camera"]["model"] == model
    assert report["output_count"] == brackets * outputs
    assert report["ae_bracketing"]["exposure_count"] == brackets
    assert report["ae_bracketing"]["outputs_per_exposure"] == outputs
    assert report["ae_bracketing"]["grouping"] == (
        "explicit_count_contiguous_sequences"
        if expected
        else "metadata_contiguous_sequences"
    )
    assert len(list(target.iterdir())) == brackets * outputs + 1
    assert [item["sha256"] for item in report["inputs"]] == hashes
    golden = ci_fixture.merge_expectations(
        "NIKON Z 8" if pattern == "constant" else model, count, outputs, pattern=pattern
    )
    for bracket, exposure in enumerate(report["ae_bracketing"]["exposures"]):
        assert exposure == {
            "bracket_index": bracket,
            "iso": 100 << bracket,
            "input_indices": list(range(bracket * count, (bracket + 1) * count)),
            "exposure_time": [1, 100 << bracket],
            "f_number": [8, 1],
            "exposure_bias_ev": [-bracket, 1],
        }
    for index, item in enumerate(report["outputs"]):
        bracket, local = divmod(index, outputs)
        expected_output = golden["outputs"][local]
        assert item["bracket_index"] == bracket
        assert item["input_indices"] == [
            i + bracket * count for i in expected_output["input_indices"]
        ]
        path = Path(item["output"])
        assert path.name == f"bracket-{bracket + 1:02d}-merged-{local + 1:02d}.nefx"
        assert item["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        with rawpy.imread(str(path)) as image:
            # Constant offsets commute with integer reconstruction and zero radial
            # coefficients. Compare to independently pinned pre-C++ sample hashes.
            offset = bracket * 256 if pattern == "constant" else 0
            pixels = image.raw_image[..., :3].astype(np.int32) - offset
            ci_fixture.check_rgb(pixels, expected_output)
        source = read(sources[item["input_indices"][0]].read_bytes())
        written = read(path.read_bytes())
        for code in (33434, 33437, 37380, 34855):
            assert (
                written.tags[34665].children[0].tags[code].payload
                == source.tags[34665].children[0].tags[code].payload
            )
    assert hashes == [hashlib.sha256(path.read_bytes()).hexdigest() for path in sources]
    assert report["samples_verified"] == sum(
        item["samples_verified"] for item in report["outputs"]
    )
    return {
        "model": model,
        "captures_per_exposure": count,
        "exposures": brackets,
        "outputs_per_exposure": outputs,
        "jobs": jobs,
        "corrected": corrected,
        "pattern": pattern,
        "expected_exposures": expected,
        "input_sha256": hashes,
        "output_sha256": [o["sha256"] for o in report["outputs"]],
    }


def configure(source, target, count, capture, iso, selection, radial):
    """Write exposure-specific ISO, sparse selection and radial metadata."""
    tree = read(source.read_bytes())
    exif = tree.tags[34665].children[0]
    maker = exif.tags[37500].children[0]
    correction = tree.tags[330].children[1].tags[0xC7D5].children[0]
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


def samples(path):
    with rawpy.imread(str(path)) as image:
        return image.raw_image[..., :3].copy()


def isolation(executable, directory):
    """Batch outputs equal separate merges; each exposure keeps its own settings."""

    def settings(bracket):
        # Sparse repair applies only at ISO 3200 or below.
        iso = (3200, 3200, 6400)[bracket]
        return {"iso": iso, "selection": bracket, "radial": bracket}

    cases = []
    for count, outputs in ((4, 1), (32, 4)):
        folder = directory / f"isolation-{count}-{outputs}"
        folder.mkdir()
        paths = sequence(folder, count, 3, calibration=True, pattern="random")
        for index, path in enumerate(paths):
            bracket, capture = divmod(index, count)
            configure(path, path, count, capture, **settings(bracket))

        def merge(sources, target):
            args = arguments(executable, sources, target, outputs=outputs)
            result = command(args + ["--chromatic-aberration-correction"])
            return [Path(o["output"]) for o in json.loads(result.stdout)["outputs"]]

        separate = [
            merge(paths[i * count : (i + 1) * count], folder / f"separate-{i}")
            for i in range(3)
        ]
        for index, path in enumerate(merge(paths, folder / "batch")):
            expected = separate[index // outputs][index % outputs]
            assert tree_digest(path.read_bytes()) == tree_digest(expected.read_bytes())
        # Negative controls: giving one exposure the first exposure's setting
        # changes samples, so the comparison above detects a shared setting.
        controls = {}
        for field, bracket in (("selection", 1), ("iso", 2), ("radial", 1)):
            control = folder / f"control-{field}"
            control.mkdir()
            changed = settings(bracket) | {field: settings(0)[field]}
            copies = []
            for capture in range(count):
                source = paths[bracket * count + capture]
                copies.append(control / source.name)
                configure(source, copies[-1], count, capture, **changed)
            differences = [
                int(np.count_nonzero(samples(left) != samples(right)))
                for left, right in zip(
                    separate[bracket], merge(copies, control / "result"), strict=True
                )
            ]
            assert all(differences), (count, field, differences)
            controls[field] = differences
        cases.append(
            {
                "captures_per_exposure": count,
                "outputs_per_exposure": outputs,
                "input_sha256": [
                    hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
                ],
                "control_changed_samples": controls,
            }
        )
    return cases


def rejections(executable, directory):
    cases = []
    for mode in (
        "missing_frame",
        "missing_exposure",
        "reversed_exposures",
        "interleaved",
        "duplicate",
        "hardlink",
        "mixed_exposure",
        "same_exposure",
        "serial",
        "shutter_gap",
        "backwards_time",
        "geometry",
        "calibration",
        "unsupported_model",
        "existing",
        "expected_count",
        "auto_single_destination",
        "mode_mismatch",
    ):
        folder = directory / mode
        folder.mkdir()
        sources = sequence(folder, 4, 3)
        target = folder / (
            "result.nefx" if mode == "auto_single_destination" else "result"
        )
        if mode == "missing_frame":
            sources.pop()
        elif mode == "missing_exposure":
            sources = sources[:8]
        elif mode == "reversed_exposures":
            sources = sources[8:] + sources[4:8] + sources[:4]
        elif mode == "interleaved":
            sources = [sources[i] for j in range(4) for i in (j, j + 4, j + 8)]
        elif mode == "duplicate":
            sources[-1] = sources[0]
        elif mode == "hardlink":
            sources[-1].unlink()
            os.link(sources[0], sources[-1])
        elif mode == "existing":
            target.mkdir()
        else:
            selected = (
                sources[4:8]
                if mode
                in {
                    "same_exposure",
                    "serial",
                    "shutter_gap",
                    "backwards_time",
                    "geometry",
                    "mode_mismatch",
                }
                else sources[-1:]
            )
            if mode == "unsupported_model":
                selected = sources
            for path in selected:
                tree = read(path.read_bytes())
                exif = tree.tags[34665].children[0]
                maker = exif.tags[37500].children[0]
                if mode in {"mixed_exposure", "same_exposure"}:
                    exif.set(33434, 5, struct.pack("<II", 1, 100))
                elif mode == "serial":
                    exif.set(42033, 2, b"different-camera\0")
                elif mode == "shutter_gap":
                    value = maker.tags[0xA7].integers()[0]
                    maker.numbers(0xA7, 4, value + 4)
                elif mode == "backwards_time":
                    exif.set(36867, 2, b"2026:09:19 00:00:00\0")
                elif mode == "geometry":
                    maker.numbers(0x45, 3, 4, 4, 54, 40)
                elif mode == "calibration":
                    maker.set(0x0C, 5, struct.pack("<8I", 3, 1, 1, 1, 1, 1, 1, 1))
                elif mode == "unsupported_model":
                    tree.set(272, 2, b"NIKON Z f\0")
                elif mode == "mode_mismatch":
                    selector = bytearray(maker.tags[0x5E].payload)
                    selector[4] = 2
                    maker.set(0x5E, 7, bytes(selector))
                path.write_bytes(serialize(tree))
        expected = (
            3 if mode == "missing_exposure" else 2 if mode == "expected_count" else 0
        )
        args = arguments(executable, sources, target, expected)
        if mode == "auto_single_destination":
            args[args.index("--output-directory")] = "--output"
        report = json.loads(command(args, 1).stdout)
        assert report["error"]["code"] == "merge_failed", (mode, report)
        if mode == "auto_single_destination":
            assert "Detected multiple exposures" in report["error"]["message"]
        if mode == "existing":
            assert target.is_dir() and not list(target.iterdir())
        else:
            assert not target.exists(), mode
        assert not list(folder.glob(".cnefx-*")), mode
        cases.append({"case": mode, "error": report["error"]["message"]})
    return cases


def run(executable: Path, injector: Path, directory: Path) -> dict:
    directory.mkdir(parents=True)
    cases = []
    for model in ("NIKON Z 8", "NIKON Z6_3"):
        for count, outputs in MODES:
            for jobs, corrected in ((1, False), (2, True)):
                folder = (
                    directory / f"{model.replace(' ', '_')}-{count}-{outputs}-{jobs}"
                )
                cases.append(
                    check_case(
                        executable, folder, model, count, outputs, 3, jobs, corrected
                    )
                )
    for brackets in (2, 9):
        cases.append(
            check_case(
                executable,
                directory / f"bound-{brackets}",
                "NIKON Z 8",
                32,
                8,
                brackets,
                1,
                False,
            )
        )
    # Total input count alone is ambiguous: e.g. 16 captures can be one 16-frame
    # sequence or four 4-frame exposures. The camera's mode determines grouping.
    for count, brackets in ((4, 2), (4, 4), (8, 2), (8, 4), (16, 2)):
        cases.append(
            check_case(
                executable,
                directory / f"metadata-{count}-{brackets}",
                "NIKON Z 8",
                count,
                1,
                brackets,
                1,
                False,
            )
        )
    for count, outputs, brackets in ((4, 1, 3), (32, 8, 9)):
        cases.append(
            check_case(
                executable,
                directory / f"expected-{count}-{brackets}",
                "NIKON Z 8",
                count,
                outputs,
                brackets,
                1,
                False,
                expected=brackets,
            )
        )
    failures = rejections(executable, directory)
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
        sources = sequence(folder, 4, 3)
        command(
            [
                str(injector),
                "--inject-brackets",
                mode,
                str(folder / "result"),
                *map(str, sources),
            ]
        )
        injected.append(mode)
    for options in (
        ["--ae-bracketing"],
        ["--ae-bracketing", "0"],
        ["--ae-bracketing", "1"],
        ["--ae-bracketing", "10"],
        ["--ae-bracketing", "-3"],
        ["--ae-bracketing", "3x"],
        ["--ae-bracketing", "4294967296"],
        ["--ae-bracketing", "3", "--ae-bracketing", "3"],
        ["--ae-bracketing", "3", "--output", "result.nefx"],
    ):
        result = command(
            [str(executable), "merge", "absent.NEF", *options, "--json"], 2
        )
        assert not result.stdout and result.stderr
    report = {
        "evidence_kind": "synthetic_ae_bracket_saved_spatial_and_offset_samples",
        "passed": True,
        "expectations_sha256": ci_fixture.GOLDEN_SHA256,
        "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "rawpy": rawpy.__version__,
        "rawpy_libraw": rawpy.libraw_version,
        "cases": cases,
        "isolation": isolation(executable, directory),
        "rejections": failures,
        "injected_failures": injected,
        "acceptance_gaps": [
            "Authentic AE captures and NX Studio references",
            "Lightroom HDR acceptance",
        ],
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
