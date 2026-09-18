"""Regressions for complete-table coverage and consumed coefficient provenance."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import verify_kaiji_composition as verifier  # noqa: E402


def test_later_truncated_table_is_rejected():
    assert all(verifier.range_mask([(0, bytes(0x34370))], complete=True))
    with pytest.raises(ValueError, match="complete native table"):
        verifier.range_mask([(0, bytes(0x3436C))], complete=True)


@pytest.mark.parametrize(
    "ranges",
    [
        [],
        [(1, bytes(0x34370))],
        [(0, bytes(0x34371))],
        [(0, bytearray(0x34370))],
        [(False, bytes(0x34370))],
    ],
)
def test_invalid_complete_ranges(ranges):
    with pytest.raises(ValueError):
        verifier.range_mask(ranges, complete=True)


def test_coefficient_pin_identifies_consumed_snapshot(tmp_path, monkeypatch):
    path = tmp_path / "coefficients.bin"
    original = b"consumed synthetic coefficients"
    path.write_bytes(original)
    original_hash = hashlib.sha256(original).hexdigest()
    changed_hash = hashlib.sha256(b"changed path bytes").hexdigest()
    output = tmp_path / "report"
    monkeypatch.setattr(verifier, "COEFFICIENT_HASH", original_hash)
    monkeypatch.setattr(
        verifier, "digest", lambda p: changed_hash if p == path else verifier.DLL_HASH
    )
    monkeypatch.setattr(verifier, "dependency_hashes", lambda *_: {})
    monkeypatch.setattr(verifier, "controls", lambda: [])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_kaiji_composition",
            "--dll",
            str(tmp_path / "Rome2.dll"),
            "--probe",
            str(tmp_path / "probe.exe"),
            "--coefficients",
            str(path),
            "--wine-prefix",
            str(tmp_path / "prefix"),
            "--output",
            str(output),
            "--native-complete",
        ],
    )
    assert verifier.main() == 1
    report = json.loads((output / "report.json").read_text())
    assert report["input_hashes"][str(path)] == original_hash
    assert report["sources_unchanged"] is False
