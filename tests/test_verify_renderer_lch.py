"""Coverage and failure accounting for the Lab/Lch export oracle."""

import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import verify_renderer_lch as verifier  # noqa: E402


@pytest.mark.parametrize("selector", [3, 4])
def test_bridge_checks_direction_format_object_and_rectangle(selector):
    rect = [0, 0, 50, 172]
    event = {
        "stage": "cms",
        "rect": rect,
        "format": 0x10060203 if selector == 3 else 0x10060201,
    }
    call = {
        "object": 1000,
        "width": 50,
        "height": 172,
        "kernel_rva": 0x53B640 if selector == 3 else 0x53A2F0,
    }
    state = bytearray(0x748)
    struct.pack_into("<I", state, 0x718, selector)
    struct.pack_into("<Q", state, 0x720, 1000)
    verifier.bridge_contract(event, call, state, rect, selector)
    for target, key, value in (
        (event, "format", 1),
        (event, "rect", [0, 0, 50, 173]),
        (call, "object", 1001),
        (call, "width", 49),
        (call, "kernel_rva", 0),
    ):
        original = target[key]
        target[key] = value
        with pytest.raises(ValueError, match="identity"):
            verifier.bridge_contract(event, call, state, rect, selector)
        target[key] = original
    for offset in (0x718, 0x720):
        changed = state.copy()
        changed[offset] ^= 1
        with pytest.raises(ValueError, match="identity"):
            verifier.bridge_contract(event, call, changed, rect, selector)
    with pytest.raises(ValueError):
        verifier.bridge_contract(event, call, state[:-1], rect, selector)


@pytest.mark.parametrize("inverse", [False, True])
def test_samples_are_deterministic_and_cover_integer_extremes(inverse):
    actual = verifier.samples(inverse)
    assert actual.dtype == np.dtype("<u2") and actual.shape[1] == 3
    np.testing.assert_array_equal(actual, verifier.samples(inverse))
    assert [0, 0, 0] in actual.tolist()
    assert [65535, 65535, 65535] in actual.tolist()


def test_inverse_samples_exercise_every_hue_at_multiple_chromas():
    actual = verifier.samples(True)
    for chroma in (0, 1, 32767, 46341, 65535):
        selected = actual[actual[:, 1] == chroma, 2]
        np.testing.assert_array_equal(np.unique(selected), np.arange(65536))


@pytest.mark.parametrize("inverse", [None, 0, 1, "inverse"])
def test_samples_reject_ambiguous_direction(inverse):
    with pytest.raises(ValueError, match="boolean"):
        verifier.samples(inverse)


def arguments(tmp_path, monkeypatch, output):
    names = ("dll", "probe", "wine-prefix", "lab-profile", "lch-profile", "capture")
    argv = ["verify_renderer_lch.py", "--output", str(output)]
    for name in names:
        argv.extend([f"--{name}", str(tmp_path / name)])
    monkeypatch.setattr(sys, "argv", argv)


@pytest.mark.parametrize("exception,code", [(KeyboardInterrupt, 130), (ValueError, 1)])
def test_partial_evidence_survives_failure(
    tmp_path, monkeypatch, capsys, exception, code
):
    output = tmp_path / "output"
    arguments(tmp_path, monkeypatch, output)

    def fail(args, report):
        report["input_hashes"]["partial"] = "0" * 64
        report["cases"].append({"direction": "lab-lch", "pixels": 1})
        raise exception("Injected failure")

    monkeypatch.setattr(verifier, "verify", fail)
    assert verifier.main() == code
    report = json.loads((output / "report.json").read_text())
    assert report["passed"] is False
    assert report["input_hashes"] == {"partial": "0" * 64}
    assert report["cases"] == [{"direction": "lab-lch", "pixels": 1}]
    assert report.get("cancelled", False) == (code == 130)
    assert json.loads(capsys.readouterr().out)["passed"] is False


@pytest.mark.parametrize("location", ["existing", "repository", "dangling_link"])
def test_reject_unsafe_output(tmp_path, monkeypatch, location):
    output = tmp_path / "output"
    if location == "existing":
        output.mkdir()
    elif location == "repository":
        output = Path(verifier.__file__).resolve().parent / "uncreated-lch-report"
    else:
        output.symlink_to(tmp_path / "missing")
    arguments(tmp_path, monkeypatch, output)
    with pytest.raises(SystemExit) as exc:
        verifier.main()
    assert exc.value.code == 2
