"""Authored multi-model fixtures exercise real LibRaw, not authentic captures."""

import hashlib
import json
import os
import signal
import struct
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import rawpy

from nx_ctudio._tiff import read, serialize
from nx_ctudio.cli import main
from nx_ctudio.merge import digest as merge_digest
from nx_ctudio.merge import merge
from nx_ctudio.nefx import write_nefx
from nx_ctudio.pixel_shift import reconstruct
from nx_ctudio.raw import read_frame

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from ci_fixture import pixels, sequence

MODELS = ("NIKON Z 8", "NIKON Z f", "NIKON Z6_3", "NIKON Z5_2")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("count", [4, 8, 16, 32])
@pytest.mark.parametrize("corrected", [False, True])
def test_native_camera_merge(
    tmp_path: Path, model: str, count: int, corrected: bool
) -> None:
    inputs = sequence(tmp_path, count, model=model)
    hashes = [digest(path) for path in inputs]
    output = tmp_path / "result.nefx"
    report = merge(inputs, output, chromatic_aberration_correction=corrected)
    expected = reconstruct(
        [pixels(i) for i in range(count)],
        (4, 4, 56, 40),
        b"0100" + bytes([(4, 8, 16, 32).index(count) + 1]) + bytes(23),
        100,
    )
    assert report["camera"] == {"make": "NIKON CORPORATION", "model": model}
    assert report["shape"] == list(expected.shape)
    assert report["samples_verified"] == expected.size
    assert report["sha256"] == digest(output)
    assert report["chromatic_aberration_correction"] is corrected
    assert [item["sha256"] for item in report["inputs"]] == hashes
    assert [digest(path) for path in inputs] == hashes
    assert "authentic" in report["limitations"][0]
    with rawpy.imread(str(output)) as decoded:
        np.testing.assert_array_equal(decoded.raw_image[..., :3], expected)
    tree = read(output.read_bytes())
    assert tree.tags[272].payload == model.encode("ascii") + b"\0"
    maker = tree.tags[34665].children[0].tags[37500].children[0]
    source = read(inputs[0].read_bytes())
    original = source.tags[34665].children[0].tags[37500].children[0]
    for code in (0x0C, 0x97):
        assert maker.tags[code].payload == original.tags[code].payload
    assert set(tmp_path.iterdir()) == {*inputs, output}


@pytest.mark.parametrize("model", MODELS)
def test_camera_geometry_is_read_from_source(tmp_path: Path, model: str) -> None:
    inputs = sequence(tmp_path, model=model, shape=(56, 80))
    samples, metadata = read_frame(inputs[0])
    assert samples.shape == (56, 80)
    np.testing.assert_array_equal(samples, pixels(0, (56, 80)))
    assert metadata["crop"] == (4, 4, 72, 48)
    report = merge(inputs, tmp_path / "result.nefx")
    assert report["shape"] == [48, 72, 3]


@pytest.mark.parametrize("change", ["pixels", "replacement"])
@pytest.mark.parametrize("corrected", [False, True])
def test_later_input_change_is_rejected_before_its_first_decode(
    tmp_path: Path, change: str, corrected: bool
) -> None:
    inputs = sequence(tmp_path)
    source = inputs[-1]
    before = source.stat()
    calls = []

    def decode(path: Path):
        calls.append(path)
        result = read_frame(path)
        if path == inputs[0]:
            if change == "pixels":
                tree = read(source.read_bytes())
                raw = tree.tags[330].children[1]
                payload = bytearray(raw.images[273][0])
                payload[0] ^= 1
                raw.images[273] = [bytes(payload)]
                source.write_bytes(serialize(tree))
                os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns + 10**9))
                assert source.stat().st_ino == before.st_ino
            else:
                replacement = tmp_path / "replacement.NEF"
                replacement.write_bytes(source.read_bytes())
                os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
                replacement.replace(source)
                assert source.stat().st_ino != before.st_ino
        return result

    with patch("nx_ctudio.merge.read_frame", side_effect=decode):
        with pytest.raises(ValueError, match="Input changed"):
            merge(
                inputs,
                tmp_path / "result.nefx",
                chromatic_aberration_correction=corrected,
            )
    assert source not in calls
    assert len(calls) == len(set(calls))
    assert set(tmp_path.iterdir()) == set(inputs)


@pytest.mark.parametrize("timing", ["decode", "write", "final_hash", "later_hash"])
def test_restored_input_bytes_do_not_hide_a_source_change(
    tmp_path: Path, timing: str
) -> None:
    inputs = sequence(tmp_path)
    source = inputs[0]
    before = source.stat()
    original_bytes = source.read_bytes()
    changes = []

    def change_and_restore() -> None:
        source.write_bytes(b"changed")
        source.write_bytes(original_bytes)
        os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns + 10**9))
        changes.append(source)

    def decode(path: Path):
        result = read_frame(path)
        if path == source and timing == "decode":
            change_and_restore()
        return result

    def write(*args):
        result = write_nefx(*args)
        if timing == "write":
            change_and_restore()
        return result

    def hash_input(path: Path):
        result = merge_digest(path)
        if (path == source and timing == "final_hash") or (
            path == inputs[-1] and timing == "later_hash"
        ):
            change_and_restore()
        return result

    with (
        patch("nx_ctudio.merge.read_frame", side_effect=decode),
        patch("nx_ctudio.merge.write_nefx", side_effect=write),
        patch("nx_ctudio.merge.digest", side_effect=hash_input),
        pytest.raises(ValueError, match="Input changed"),
    ):
        merge(inputs, tmp_path / "result.nefx")
    assert changes == [source]
    assert source.read_bytes() == original_bytes
    assert set(tmp_path.iterdir()) == set(inputs)


@pytest.mark.parametrize("model", ["NIKON Z 6", "NIKON Z 9", "NIKON Z fc"])
def test_unsupported_camera_fails_before_decode(tmp_path: Path, model: str) -> None:
    source = sequence(tmp_path, model=model)[0]
    with patch("nx_ctudio.raw.rawpy.imread") as decode:
        with pytest.raises(ValueError, match="camera model"):
            read_frame(source)
    decode.assert_not_called()


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("failure", ["ordinary", "he", "balance", "shift", "cfa"])
def test_camera_allowlist_preserves_strict_input_checks(
    tmp_path: Path, model: str, failure: str
) -> None:
    source = sequence(tmp_path, model=model)[0]
    tree = read(source.read_bytes())
    maker = tree.tags[34665].children[0].tags[37500].children[0]
    if failure in ("ordinary", "shift"):
        data = bytearray(maker.tags[0x5E].payload)
        if failure == "ordinary":
            data[4] = 0
        else:
            data[:4] = b"9999"
        maker.set(0x5E, 7, bytes(data))
    elif failure == "he":
        data = bytearray(maker.tags[0x51].payload)
        struct.pack_into("<H", data, 10, 13)
        maker.set(0x51, 7, bytes(data))
    elif failure == "balance":
        maker.set(0x97, 7, b"0803" + bytes(161))
    else:
        tree.tags[330].children[1].set(33422, 1, b"\2\1\1\0")
    source.write_bytes(serialize(tree))
    with patch("nx_ctudio.raw.rawpy.imread") as decode:
        with pytest.raises(ValueError):
            read_frame(source)
    decode.assert_not_called()


@pytest.mark.parametrize("other", MODELS[1:])
def test_mixed_models_never_publish(tmp_path: Path, other: str) -> None:
    inputs = sequence(tmp_path)
    tree = read(inputs[-1].read_bytes())
    tree.set(272, 2, other.encode("ascii") + b"\0")
    inputs[-1].write_bytes(serialize(tree))
    hashes = [digest(path) for path in inputs]
    with pytest.raises(ValueError, match="Sequence metadata differs: model"):
        merge(inputs, tmp_path / "result.nefx")
    assert set(tmp_path.iterdir()) == set(inputs)
    assert [digest(path) for path in inputs] == hashes


@pytest.mark.parametrize("model", MODELS)
def test_cli_reports_camera(tmp_path: Path, capsys, model: str) -> None:
    inputs = sequence(tmp_path, model=model)
    assert (
        main(
            ["merge", *map(str, inputs), "-o", str(tmp_path / "result.nefx"), "--json"]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["camera"]["model"] == model
    assert report["status"] == "ok"


def test_cli_retains_verified_nefx_after_final_publication_signal(
    tmp_path: Path, capsys
) -> None:
    inputs = sequence(tmp_path)
    output = tmp_path / "result.nefx"
    link = os.link
    interrupted = []

    def publish_and_interrupt(source: Path, destination: Path) -> None:
        link(source, destination)
        if destination == output:
            interrupted.append(destination)
            signal.raise_signal(signal.SIGINT)

    with patch("nx_ctudio._atomic.os.link", side_effect=publish_and_interrupt):
        result = main(["merge", *map(str, inputs), "-o", str(output), "--json"])
    report = json.loads(capsys.readouterr().out)
    assert result == 0
    assert report["status"] == "ok"
    assert interrupted == [output]
    completion_warnings = [
        warning
        for warning in report["warnings"]
        if "publication completion failed" in warning
    ]
    assert len(completion_warnings) == 1
    assert "Output published" in completion_warnings[0]
    assert "KeyboardInterrupt" in completion_warnings[0]
    expected = reconstruct(
        [pixels(i) for i in range(4)],
        (4, 4, 56, 40),
        b"0100\x01" + bytes(23),
        100,
    )
    with rawpy.imread(str(output)) as decoded:
        np.testing.assert_array_equal(decoded.raw_image[..., :3], expected)
    assert report["sha256"] == digest(output)
    assert report["samples_verified"] == expected.size
    assert set(tmp_path.iterdir()) == {*inputs, output}


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("matrix_kind", ["missing", "nan", "singular"])
def test_color_matrix_absence_is_explicit_and_model_scoped(
    tmp_path: Path, model: str, matrix_kind: str
) -> None:
    inputs = sequence(tmp_path, model=model)
    original = rawpy.imread
    matrix = np.zeros((4, 3), dtype=np.float32)
    if matrix_kind == "nan":
        matrix[0, 0] = np.nan
    elif matrix_kind == "singular":
        matrix[:3] = 1

    class Decoder:
        def __init__(self, decoded):
            self.decoded = decoded
            self.rgb_xyz_matrix = matrix

        def __getattr__(self, name):
            return getattr(self.decoded, name)

    @contextmanager
    def decode(path):
        with original(path) as decoded:
            yield Decoder(decoded)

    with patch("nx_ctudio.raw.rawpy.imread", side_effect=decode):
        if model == "NIKON Z5_2" and matrix_kind == "missing":
            _, metadata = read_frame(inputs[0])
            assert metadata["color_matrix_xyz_to_camera"] is None
            report = merge(inputs, tmp_path / "result.nefx")
            assert any("no XYZ-to-camera matrix" in s for s in report["limitations"])
        else:
            with pytest.raises(ValueError, match="XYZ-to-camera matrix"):
                read_frame(inputs[0])
