"""Preserve live-oracle evidence when RAW transfer verification is cancelled."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import verify_raw_transfer as verifier  # noqa: E402


@pytest.mark.parametrize("phase", ["probe", "comparison"])
def test_cancellation_saves_failed_report_and_completed_cases(
    tmp_path, monkeypatch, capsys, phase
):
    output = tmp_path / "run"
    dll = tmp_path / "utilities.dll"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_raw_transfer.py",
            "--dll",
            str(dll),
            "--probe",
            str(tmp_path / "probe.exe"),
            "--wine-prefix",
            str(tmp_path / "wine"),
            "--output",
            str(output),
        ],
    )
    monkeypatch.setattr(verifier, "digest", lambda _: verifier.UTILITY_HASH)
    monkeypatch.setattr(verifier, "math_provider", lambda: tmp_path / "libm.so")
    event = {
        "event": "raw_transfer",
        "provider_hex": verifier.windows(dll).encode().hex(),
        "bytes": 262144,
        "unchanged": True,
        "saved": True,
        "released": True,
    }

    def run(*args, **kwargs):
        if phase == "probe":
            raise KeyboardInterrupt
        (output / "vendor.bin").write_bytes(bytes(262144))
        return subprocess.CompletedProcess([], 0, json.dumps(event), "probe diagnostic")

    def native(*, inverse):
        if inverse:
            raise KeyboardInterrupt
        return np.zeros(65536, dtype="<u2")

    monkeypatch.setattr(verifier.subprocess, "run", run)
    monkeypatch.setattr(verifier, "srgb_lut", native)
    assert verifier.main() == 130
    report = json.loads((output / "report.json").read_text())
    assert report["passed"] is False
    assert report["cancelled"] is True
    assert "cancelled" in report["error"].lower()
    assert report["input_hashes"]
    if phase == "probe":
        assert report["cases"] == []
    else:
        assert len(report["cases"]) == 1
        assert report["cases"][0]["gamma"] == 0
        assert report["cases"][0]["different_words"] == 0
        assert report["output_sha256"]
        assert report["returncode"] == 0
        assert report["stderr"] == "probe diagnostic"
    summary = json.loads(capsys.readouterr().out)
    assert summary == {"passed": False, "report": str(output / "report.json")}
