# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.4.6", "rawpy==0.27.1", "tifffile==2026.9.15"]
# ///
"""Write and validate a research Linear DNG from a decoded Z8 NEFX.

This preserves raw samples, not Nikon rendering or Adobe profile behavior.
Generated photographs and reports must stay outside the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
import rawpy
import tifffile


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rationals(values: np.ndarray) -> tuple[int, ...]:
    result = []
    for value in values.flat:
        fraction = Fraction(float(value)).limit_denominator(1000000)
        result.extend((fraction.numerator, fraction.denominator))
    return tuple(result)


def read_source(path: Path) -> tuple[np.ndarray, dict]:
    with tifffile.TiffFile(path) as image:
        root = image.pages[0]
        if 50706 in root.tags:
            raise ValueError("Expected a source NEFX, not an existing DNG.")
        make, model = (root.tags[code].value for code in (271, 272))
        if (make, model) != ("NIKON CORPORATION", "NIKON Z 8"):
            raise ValueError("This research probe only supports Nikon Z8 NEFX.")
        orientation = int(root.tags[274].value)
        if orientation not in range(1, 9):
            raise ValueError("Unsupported orientation.")
    with rawpy.imread(str(path)) as image:
        image.unpack()
        if image.raw_type != rawpy.RawType.Stack or image.num_colors != 3:
            raise ValueError("Expected decoded three-channel NEFX, not Bayer NEF.")
        if image.color_desc != b"RGBG" or image.raw_image.shape[2] != 4:
            raise ValueError("Unsupported decoded channel order.")
        sizes = image.sizes
        if (
            sizes.crop_left_margin != 0
            or sizes.crop_top_margin != 0
            or (sizes.crop_height, sizes.crop_width)
            != (sizes.raw_height, sizes.raw_width)
            or sizes.pixel_aspect != 1
        ):
            raise ValueError("Expected an already cropped, square-pixel NEFX.")
        samples = image.raw_image[..., :3].copy()
        black = np.asarray(image.black_level_per_channel[:3], dtype=np.float64)
        if image.black_level_per_channel[1] != image.black_level_per_channel[3]:
            raise ValueError("Different green black levels need a separate contract.")
        if not np.all(black == black[0]):
            raise ValueError("Unequal channel ranges need a separate color contract.")
        white = int(image.white_level)
        gains = np.asarray(image.camera_whitebalance[:3], dtype=np.float64)
        matrix = image.rgb_xyz_matrix[:3].astype(np.float64)
        if white != 16383 or np.any(black < 0) or np.any(black >= white):
            raise ValueError("Unsupported black level or sample code range.")
        if not np.all(np.isfinite(gains)) or np.any(gains <= 0):
            raise ValueError("Missing or invalid camera white balance.")
        if not np.all(np.isfinite(matrix)) or abs(np.linalg.det(matrix)) < 1e-6:
            raise ValueError("Missing or singular color matrix.")
        neutral = gains[1] / gains
        if samples.max() > white:
            raise ValueError("Samples exceed the recovered 14-bit code range.")
        return samples, {
            "make": make,
            "model": model,
            "orientation": orientation,
            "sizes": sizes._asdict(),
            "black_level": black.tolist(),
            "white_level": white,
            "decoder_camera_white_level": image.camera_white_level_per_channel,
            "camera_gains": gains.tolist(),
            "as_shot_neutral": neutral.tolist(),
            "color_matrix_xyz_to_camera": matrix.tolist(),
            "calibration_illuminant": 0,
        }


def write_dng(path: Path, samples: np.ndarray, metadata: dict) -> None:
    height, width, _ = samples.shape
    tags = [
        (271, "s", 0, metadata["make"], False),
        (272, "s", 0, metadata["model"], False),
        (274, "H", 1, metadata["orientation"], False),
        (50706, "B", 4, (1, 4, 0, 0), False),
        (50707, "B", 4, (1, 3, 0, 0), False),
        (50708, "s", 0, metadata["model"], False),
        (50713, "H", 2, (1, 1), False),
        (50714, "2I", 3, rationals(np.array(metadata["black_level"])), False),
        (50717, "I", 3, (metadata["white_level"],) * 3, False),
        (50718, "2I", 2, (1, 1, 1, 1), False),
        (50719, "I", 2, (0, 0), False),
        (50720, "I", 2, (width, height), False),
        (
            50721,
            "2i",
            9,
            rationals(np.array(metadata["color_matrix_xyz_to_camera"])),
            False,
        ),
        (50728, "2I", 3, rationals(np.array(metadata["as_shot_neutral"])), False),
        (50778, "H", 1, metadata["calibration_illuminant"], False),
        (50829, "I", 4, (0, 0, height, width), False),
        (50936, "s", 0, "NX Ctudio LibRaw matrix research", False),
    ]
    with path.open("xb") as stream:
        tifffile.imwrite(
            stream,
            samples,
            byteorder="<",
            bigtiff=False,
            photometric=34892,
            planarconfig="contig",
            extrasamples=False,
            compression=None,
            rowsperstrip=min(height, 256),
            subfiletype=0,
            metadata=None,
            software="NX Ctudio research",
            extratags=tags,
        )


def roundtrip(path: Path, samples: np.ndarray, metadata: dict) -> dict:
    with tifffile.TiffFile(path) as image:
        page = image.pages[0]
        tiff_differences = int(np.count_nonzero(page.asarray() != samples))
        tags = page.tags
        structure = (
            int(page.photometric) == 34892
            and page.samplesperpixel == 3
            and page.bitspersample == 16
            and 33421 not in tags
            and 33422 not in tags
            and 338 not in tags
            and int(tags[274].value) == metadata["orientation"]
            and tags[50708].value == metadata["model"]
        )
    with rawpy.imread(str(path)) as image:
        image.unpack()
        decoded = image.raw_image[..., :3]
        if decoded.shape != samples.shape:
            raise ValueError("DNG decoded shape differs from the NEFX.")
        differences = int(np.count_nonzero(decoded != samples))
        black = np.array(image.black_level_per_channel[:3])
        source_black = np.array(metadata["black_level"])
        gains = np.array(image.camera_whitebalance[:3])
        neutral = gains[1] / gains
        matrix_error = float(
            np.max(
                np.abs(
                    image.rgb_xyz_matrix[:3] - metadata["color_matrix_xyz_to_camera"]
                )
            )
        )
        neutral_error = float(np.max(np.abs(neutral - metadata["as_shot_neutral"])))
        checks = {
            "linear_raw_structure": bool(structure),
            "tiff_samples_equal": tiff_differences == 0,
            "libraw_samples_equal": differences == 0,
            "black_levels_equal": bool(np.array_equal(black, source_black)),
            "white_level_equal": image.white_level == metadata["white_level"],
            "neutral_equal_within_1e_6": neutral_error < 1e-6,
            "matrix_equal_within_1e_6": matrix_error < 1e-6,
            "orientation_equal": image.sizes.flip == metadata["sizes"]["flip"],
        }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "samples_compared": int(samples.size),
        "tiff_different_samples": tiff_differences,
        "libraw_different_samples": differences,
        "matrix_max_abs_error": matrix_error,
        "neutral_max_abs_error": neutral_error,
    }


def validate(executable: Path, path: Path, stages: Path) -> dict:
    stages.mkdir(exist_ok=False)
    command = [str(executable.resolve()), "-v"]
    for stage in (1, 2, 3):
        command.extend((f"-{stage}", str((stages / f"stage{stage}.tif").resolve())))
    command.append(str(path.resolve()))
    result = subprocess.run(command, capture_output=True, timeout=180, check=False)
    stdout = result.stdout.decode("utf-8", errors="replace")
    stderr = result.stderr.decode("utf-8", errors="replace")
    diagnostics = stdout + "\n" + stderr
    return {
        "command": command,
        "executable_sha256": digest(executable),
        "returncode": result.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "passed": result.returncode == 0
        and "Validation complete" in stdout
        and "***" not in diagnostics,
    }


def check_stages(stages: Path, samples: np.ndarray, metadata: dict) -> dict:
    black = np.array(metadata["black_level"])
    white = metadata["white_level"]
    differences = [0, 0, 0]
    hashes = {}
    for stage in (1, 2, 3):
        path = stages / f"stage{stage}.tif"
        actual = tifffile.memmap(path, mode="r")
        if actual.shape != samples.shape or actual.dtype != np.uint16:
            raise ValueError(f"Unexpected SDK stage {stage} shape or type.")
        for row in range(0, samples.shape[0], 128):
            source = samples[row : row + 128]
            expected = source
            if stage != 1:
                normalized = np.clip(
                    (source.astype(float) - black) / (white - black), 0, 1
                )
                expected = np.floor(normalized * 65535 + 0.5).astype(np.uint16)
            differences[stage - 1] += int(
                np.count_nonzero(actual[row : row + 128] != expected)
            )
        del actual
        hashes[path.name] = digest(path)
    return {
        "normalization": "floor(clip((sample-black)/(white-black),0,1)*65535+0.5)",
        "samples_per_stage": int(samples.size),
        "stage1_different_samples": differences[0],
        "stage2_different_samples": differences[1],
        "stage3_different_samples": differences[2],
        "hashes": hashes,
        "passed": differences == [0, 0, 0],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--validator", type=Path, required=True)
    parser.add_argument("--stages", type=Path, required=True)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    for destination in (args.output, args.report, args.stages):
        if destination.resolve().is_relative_to(repository):
            parser.error("Research outputs must be outside the repository.")
    if args.output.exists() or args.report.exists() or args.stages.exists():
        parser.error("Output, report and stage directory must not already exist.")
    if args.output.resolve() == args.report.resolve():
        parser.error("Output and report must be different paths.")
    report = {
        "evidence_kind": "research_nefx_to_linear_dng_roundtrip",
        "source": str(args.source.resolve()),
        "output": str(args.output.resolve()),
        "tool_sha256": digest(Path(__file__)),
        "decoder": {"rawpy": rawpy.__version__, "libraw": list(rawpy.libraw_version)},
        "passed": False,
        "limitations": [
            "Not an authentic pixel-shift capture or a Lightroom acceptance test.",
            "ColorMatrix1 is the LibRaw camera table, not a Nikon or Adobe profile.",
            "Calibration illuminant is unknown; no second matrix is invented.",
            "WhiteLevel is the 14-bit code limit, not measured sensor saturation.",
            "No previews, private metadata, capture EXIF or Adobe XMP defaults copied.",
            "UniqueCameraModel profile lookup and rendered color remain unverified.",
        ],
    }
    exit_code = 1
    try:
        report["source_sha256"] = digest(args.source)
        samples, metadata = read_source(args.source)
        report["metadata"] = metadata
        write_dng(args.output, samples, metadata)
        report["output_sha256"] = digest(args.output)
        report["roundtrip"] = roundtrip(args.output, samples, metadata)
        report["sdk_validation"] = validate(args.validator, args.output, args.stages)
        if report["sdk_validation"]["passed"]:
            report["sdk_stages"] = check_stages(args.stages, samples, metadata)
        report["source_unchanged"] = digest(args.source) == report["source_sha256"]
        report["passed"] = (
            report["roundtrip"]["passed"]
            and report["sdk_validation"]["passed"]
            and report["sdk_stages"]["passed"]
            and report["source_unchanged"]
        )
        exit_code = 0 if report["passed"] else 1
    except (Exception, KeyboardInterrupt) as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        if isinstance(exc, subprocess.TimeoutExpired):
            report["error"].update(
                stdout=(exc.stdout or b"").decode("utf-8", errors="replace"),
                stderr=(exc.stderr or b"").decode("utf-8", errors="replace"),
            )
        exit_code = 130 if isinstance(exc, KeyboardInterrupt) else 1
    with args.report.open("x") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=True)
        stream.write("\n")
    print(json.dumps({"report": str(args.report), "passed": report["passed"]}))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
