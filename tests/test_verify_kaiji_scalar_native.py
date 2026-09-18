"""Fail-closed checks for saved scalar RGB oracle evidence."""

import copy
import hashlib
import importlib.util
import itertools
import json
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"


@pytest.fixture
def replay(monkeypatch):
    monkeypatch.syspath_prepend(str(TOOLS))
    spec = importlib.util.spec_from_file_location(
        "nxc_scalar_replay", TOOLS / "verify_kaiji_scalar_native.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def oracle(rgb_only=True):
    flags = (1,) if rgb_only else (0, 1)
    cases = []
    for processing, flag, matrix in itertools.product((0, 2), flags, (0, 1)):
        cases.append(
            {
                "processing": processing,
                "flags": flag,
                "matrix": matrix,
                "returncode": 0,
                "passed": True,
                "sha256": hashlib.sha256(bytes(3 * 33**3 * 2)).hexdigest(),
                "events": [
                    {
                        "event": "kaiji_scalar_lattice",
                        "processing": processing,
                        "flags": flag,
                        "matrix": matrix,
                        "selector": 0,
                        "grid": 33,
                        "calls": 33**3,
                        "guards_intact": True,
                        "inputs_unchanged": True,
                        "success": True,
                    }
                ],
            }
        )
    checks = {"processing_modes_equal": True, "sources_unchanged": True}
    for processing in (0, 2):
        for check in ("rgb_matrix_invariant", "rgb_in_range"):
            checks[f"p{processing}_{check}"] = True
        if not rgb_only:
            for check in ("zero_matrix_lab", "initialized_lab_varies"):
                checks[f"p{processing}_{check}"] = True
    return {
        "evidence_kind": "live_vendor_scalar_rgb_capture"
        if rgb_only
        else "live_vendor_scalar_matrix_initialization_controls",
        "output_scope": "rgb" if rgb_only else "rgb_and_lab",
        "passed": True,
        "grid": 33,
        "selector": 0,
        "checks": checks,
        "cases": cases,
    }


@pytest.mark.parametrize("rgb_only", [False, True])
def test_accepts_complete_vendor_evidence(replay, rgb_only):
    report = oracle(rgb_only)
    selected = replay.validate_oracle(report)
    assert len(selected) == 4
    assert {(case["processing"], case["matrix"]) for case in selected} == set(
        itertools.product((0, 2), (0, 1))
    )


def test_accepts_legacy_full_report_with_known_evidence_kind(replay):
    report = oracle(False)
    del report["output_scope"]
    assert len(replay.validate_oracle(report)) == 4


@pytest.mark.parametrize(
    "field,value",
    [
        ("returncode", 99),
        ("returncode", False),
        ("passed", False),
        ("events", []),
        ("events", [None]),
        ("events", [None, None]),
    ],
)
def test_rejects_failed_or_malformed_vendor_cases(replay, field, value):
    report = oracle()
    report["cases"][0][field] = value
    with pytest.raises(ValueError):
        replay.validate_oracle(report)


@pytest.mark.parametrize(
    "field,value",
    [
        ("event", "other"),
        ("processing", 2),
        ("flags", 0),
        ("matrix", 1),
        ("selector", 3),
        ("grid", 32),
        ("calls", 0),
        ("calls", 35936),
        ("guards_intact", False),
        ("guards_intact", 1),
        ("inputs_unchanged", False),
        ("success", False),
    ],
)
def test_rejects_unsafe_or_mismatched_event(replay, field, value):
    report = oracle()
    report["cases"][0]["events"][0][field] = value
    with pytest.raises(ValueError):
        replay.validate_oracle(report)


@pytest.mark.parametrize("field", list(oracle()["cases"][0]["events"][0]))
def test_requires_each_event_field(replay, field):
    report = oracle()
    del report["cases"][0]["events"][0][field]
    with pytest.raises(ValueError):
        replay.validate_oracle(report)


@pytest.mark.parametrize("field", list(oracle(False)["checks"]))
@pytest.mark.parametrize("value", [False, 1, None])
def test_requires_true_checks(replay, field, value):
    report = oracle(False)
    report["checks"][field] = value
    with pytest.raises(ValueError):
        replay.validate_oracle(report)


def test_requires_missing_source_stability_check(replay):
    report = oracle()
    del report["checks"]["sources_unchanged"]
    with pytest.raises(ValueError):
        replay.validate_oracle(report)


@pytest.mark.parametrize(
    "field,value",
    [
        ("grid", 32),
        ("selector", 3),
        ("selector", False),
        ("passed", 1),
        ("output_scope", "other"),
        ("evidence_kind", "other"),
        ("evidence_kind", []),
        ("checks", []),
    ],
)
def test_rejects_unsupported_report(replay, field, value):
    report = oracle()
    report[field] = value
    with pytest.raises(ValueError):
        replay.validate_oracle(report)


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate", "wrong_mode"])
def test_requires_exact_unique_mode_coverage(replay, mutation):
    report = oracle()
    if mutation == "missing":
        report["cases"].pop()
    elif mutation == "extra":
        report["cases"].append(copy.deepcopy(report["cases"][0]))
    elif mutation == "duplicate":
        report["cases"][-1] = copy.deepcopy(report["cases"][0])
    else:
        report["cases"][0]["processing"] = 1
        report["cases"][0]["events"][0]["processing"] = 1
    with pytest.raises(ValueError):
        replay.validate_oracle(report)


def test_rejects_failed_lab_case_even_when_only_rgb_is_replayed(replay):
    report = oracle(False)
    report["cases"][0]["events"][0]["guards_intact"] = False
    with pytest.raises(ValueError):
        replay.validate_oracle(report)


def test_zero_table_has_zero_output_at_lattice_corners_and_center(replay):
    table = replay.ScalarTable(bytes(0x34370))
    for point in itertools.product((0, 16, 32), repeat=3):
        assert table.evaluate(point) == (0, 0, 0)


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    report = oracle()
    table = tmp_path / "table.bin"
    table.write_bytes(bytes(0x34370))
    report["input_hashes"] = {
        str(table.resolve()): hashlib.sha256(table.read_bytes()).hexdigest()
    }
    oracle_path = tmp_path / "report.json"
    output = tmp_path / "replay.json"
    for case in report["cases"]:
        path = tmp_path / f"p{case['processing']}-f1-m{case['matrix']}.bin"
        path.write_bytes(bytes(3 * 33**3 * 2))

    def configure(report):
        snapshot = json.dumps(report).encode()
        oracle_path.write_bytes(snapshot)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "verify_kaiji_scalar_native.py",
                "--oracle",
                str(oracle_path),
                "--oracle-sha256",
                hashlib.sha256(snapshot).hexdigest(),
                "--table",
                str(table),
                "--output",
                str(output),
            ],
        )

    configure(report)
    return report, configure, output


def test_main_rejects_failed_oracle_before_evaluation(replay, inputs, monkeypatch):
    report, configure, output = inputs
    report["cases"][0]["events"][0]["guards_intact"] = False
    configure(report)

    def unexpected(*args, **kwargs):
        pytest.fail("Failed evidence must be rejected before evaluation.")

    monkeypatch.setattr(replay.ScalarTable, "evaluate", unexpected)
    with pytest.raises(SystemExit) as error:
        replay.main()
    assert error.value.code == 2
    assert not output.exists()


def test_complete_replay_of_zero_fixture(replay, inputs):
    _, _, output = inputs
    assert replay.main() == 0
    report = json.loads(output.read_bytes())
    assert report["passed"] is True
    assert report["unique_channel_words"] == 107811
    assert report["comparisons"] == 4
    assert report["different_channel_words"] == 0


def test_rejects_unwritten_rgb_sentinel(replay, inputs):
    report, configure, output = inputs
    data = b"\xff" * (3 * 33**3 * 2)
    case = report["cases"][0]
    (output.parent / "p0-f1-m0.bin").write_bytes(data)
    case["sha256"] = hashlib.sha256(data).hexdigest()
    configure(report)
    with pytest.raises(SystemExit) as error:
        replay.main()
    assert error.value.code == 2
    assert not output.exists()
