"""Snapshot and evidence gates for the saved NP3 curve replay."""

import hashlib
import importlib.util
import json
import struct
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def replay(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(TOOLS))
    spec = importlib.util.spec_from_file_location(
        "nxc_curve_replay", TOOLS / "verify_picture_curve.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replay):
    payload = bytearray(578)
    payload[:13] = b"I0\0\xff\0\xff\x01\0\x02\0\0\xff\xff"
    data = b"NCP\0" + struct.pack(">II", 2, len(payload)) + payload + bytes(4)
    samples, matrix_dir = tmp_path / "samples", tmp_path / "matrix"
    samples.mkdir()
    matrix_dir.mkdir()
    reports = {}
    for label, directory, count in (("oracle", samples, 5), ("matrix", matrix_dir, 18)):
        cases = []
        for index in range(count):
            name = f"{label}-{index}" + (".NP3" if label == "oracle" else "")
            path = directory / (name if label == "oracle" else f"{name}.NP3")
            path.write_bytes(data)
            cases.append(
                {
                    "name": name,
                    "source_sha256": sha256(data),
                    "returncode": 0,
                    "event": {
                        "status": 0,
                        "guards_intact": True,
                        "table": [int(32767 * i / 256 + 0.5) for i in range(257)],
                    },
                }
            )
        path = tmp_path / f"{label}.json"
        path.write_text(json.dumps({"cases": cases}))
        monkeypatch.setattr(
            replay, f"{label.upper()}_SHA256", sha256(path.read_bytes())
        )
        reports[label] = path
    output = tmp_path / "replay.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_picture_curve.py",
            "--oracle",
            str(reports["oracle"]),
            "--matrix",
            str(reports["matrix"]),
            "--samples",
            str(samples),
            "--matrix-dir",
            str(matrix_dir),
            "--output",
            str(output),
        ],
    )
    return {**reports, "samples": samples, "matrix_dir": matrix_dir, "output": output}


def test_complete_replay_records_source_provenance(replay, inputs, monkeypatch) -> None:
    monkeypatch.chdir(inputs["output"].parent)
    assert replay.main() == 0
    report = json.loads(inputs["output"].read_bytes())
    assert report["passed"] is True
    assert len(report["cases"]) == 23
    assert report["samples_compared"] == 5911
    assert report["oracle_sha256"] == sha256(inputs["oracle"].read_bytes())
    assert report["matrix_sha256"] == sha256(inputs["matrix"].read_bytes())
    assert isinstance(report["git_worktree_dirty"], bool)
    for key, path in (
        ("runner_sha256", Path(replay.__file__)),
        ("generator_sha256", Path(replay.generate_curve.__code__.co_filename)),
        ("parser_sha256", Path(replay.parse.__code__.co_filename)),
    ):
        assert report[key] == sha256(path.read_bytes())
    assert (
        report["git_revision"]
        == subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=TOOLS.parent,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )


@pytest.mark.parametrize("label", ["oracle", "matrix"])
def test_rejects_report_snapshot_different_from_pinned_bytes(
    replay, inputs, monkeypatch: pytest.MonkeyPatch, label: str
) -> None:
    path = inputs[label]
    read_bytes, read_text = Path.read_bytes, Path.read_text
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda self: b'{"cases": []}' if self == path else read_bytes(self),
    )
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda self, *args, **kwargs: (
            '{"cases": []}' if self == path else read_text(self, *args, **kwargs)
        ),
    )
    with pytest.raises(SystemExit) as error:
        replay.main()
    assert error.value.code == 2
    assert not inputs["output"].exists()


def test_rejects_np3_snapshot_different_from_recorded_bytes(
    replay, inputs, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = inputs["samples"] / "oracle-0.NP3"
    changed = bytearray(path.read_bytes())
    changed[-5] = 1  # Stored samples do not influence point-based regeneration.
    read_bytes = Path.read_bytes
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda self: bytes(changed) if self == path else read_bytes(self),
    )
    with pytest.raises(SystemExit) as error:
        replay.main()
    assert error.value.code == 2
    assert not inputs["output"].exists()


@pytest.mark.parametrize(
    "label,count",
    [
        ("oracle", 0),
        ("oracle", 4),
        ("oracle", 6),
        ("matrix", 0),
        ("matrix", 17),
        ("matrix", 19),
    ],
)
def test_requires_exact_case_counts(
    replay, inputs, monkeypatch: pytest.MonkeyPatch, label: str, count: int
) -> None:
    path = inputs[label]
    report = json.loads(path.read_bytes())
    report["cases"] = (report["cases"] * 2)[:count]
    path.write_text(json.dumps(report))
    monkeypatch.setattr(replay, f"{label.upper()}_SHA256", sha256(path.read_bytes()))
    with pytest.raises(SystemExit) as error:
        replay.main()
    assert error.value.code == 2
    assert not inputs["output"].exists()


def test_reads_each_evidence_file_once(replay, inputs, monkeypatch) -> None:
    evidence = {inputs["oracle"], inputs["matrix"]}
    evidence.update(inputs["samples"].glob("*.NP3"))
    evidence.update(inputs["matrix_dir"].glob("*.NP3"))
    reads = Counter()
    original = Path.open

    def tracked_open(path, mode="r", *args, **kwargs):
        if path in evidence and "r" in mode:
            reads[path] += 1
        return original(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracked_open)
    assert replay.main() == 0
    assert reads == Counter(dict.fromkeys(evidence, 1))


def test_duplicate_cases_cannot_satisfy_expected_count(replay, inputs, monkeypatch):
    path = inputs["oracle"]
    report = json.loads(path.read_bytes())
    report["cases"][-1] = report["cases"][0]
    path.write_text(json.dumps(report))
    monkeypatch.setattr(replay, "ORACLE_SHA256", sha256(path.read_bytes()))
    with pytest.raises(SystemExit) as error:
        replay.main()
    assert error.value.code == 2
    assert not inputs["output"].exists()


@pytest.mark.parametrize("value", [False, 0.0, -1, 32768])
def test_rejects_malformed_vendor_sample(replay, inputs, monkeypatch, value):
    path = inputs["oracle"]
    report = json.loads(path.read_bytes())
    report["cases"][0]["event"]["table"][0] = value
    path.write_text(json.dumps(report))
    monkeypatch.setattr(replay, "ORACLE_SHA256", sha256(path.read_bytes()))
    with pytest.raises(SystemExit) as error:
        replay.main()
    assert error.value.code == 2
    assert not inputs["output"].exists()


def test_last_sample_mismatch_produces_failed_report(replay, inputs, monkeypatch):
    path = inputs["matrix"]
    report = json.loads(path.read_bytes())
    report["cases"][-1]["event"]["table"][-1] -= 1
    path.write_text(json.dumps(report))
    monkeypatch.setattr(replay, "MATRIX_SHA256", sha256(path.read_bytes()))
    assert replay.main() == 1
    actual = json.loads(inputs["output"].read_bytes())
    assert actual["passed"] is False
    assert actual["samples_compared"] == 5911
    assert sum(case["differences"] for case in actual["cases"]) == 1


@pytest.mark.parametrize(
    "field,value", [("returncode", 1), ("status", 1), ("guards_intact", False)]
)
def test_rejects_unsuccessful_vendor_cases(
    replay, inputs, monkeypatch, field: str, value: int | bool
) -> None:
    path = inputs["oracle"]
    report = json.loads(path.read_bytes())
    case = report["cases"][0]
    (case if field == "returncode" else case["event"])[field] = value
    path.write_text(json.dumps(report))
    monkeypatch.setattr(replay, "ORACLE_SHA256", sha256(path.read_bytes()))
    with pytest.raises(SystemExit) as error:
        replay.main()
    assert error.value.code == 2
    assert not inputs["output"].exists()


def test_existing_output_is_preserved(replay, inputs) -> None:
    inputs["output"].write_bytes(b"preserve this report")
    with pytest.raises(SystemExit) as error:
        replay.main()
    assert error.value.code == 2
    assert inputs["output"].read_bytes() == b"preserve this report"
