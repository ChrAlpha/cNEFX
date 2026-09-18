"""Dependency-location and provenance checks for vendor research runners."""

import importlib.util
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"


@pytest.fixture
def oracle(monkeypatch):
    monkeypatch.syspath_prepend(str(TOOLS))
    spec = importlib.util.spec_from_file_location(
        "nxc_test_kaiji_oracle", TOOLS / "kaiji_oracle.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resource_is_resolved_beside_executable(oracle, monkeypatch, tmp_path) -> None:
    dll = tmp_path / "vendor" / "Rome2.dll"
    probe = tmp_path / "probes" / "probe.exe"
    expected = {
        dll.parent / "Moon2.dll": oracle.MOON_HASH,
        probe.parent / "prm.bin": oracle.RESOURCE_HASH,
    }
    seen = []

    def digest(path):
        seen.append(path)
        return expected[path]

    monkeypatch.setattr(oracle, "digest", digest)
    assert oracle.dependency_hashes(dll, probe) == {
        str(p): h for p, h in expected.items()
    }
    assert set(seen) == set(expected)


def test_reject_changed_dependency(oracle, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(oracle, "digest", lambda path: "0" * 64)
    with pytest.raises(ValueError, match="dependency hash"):
        oracle.dependency_hashes(tmp_path / "Rome2.dll", tmp_path / "probe.exe")


@pytest.mark.parametrize(
    "event",
    [{}, {"moon_path": None}, {"moon_path": 42}, {"moon_path": "Z:/other/Moon2.dll"}],
)
def test_reject_missing_or_different_loaded_provider(oracle, tmp_path, event) -> None:
    assert not oracle.loaded_provider_matches(event, tmp_path / "Rome2.dll")


def test_accept_windows_case_and_separator_variation(oracle, tmp_path) -> None:
    dll = tmp_path / "Rome2.dll"
    path = ("Z:" + str(dll.parent / "Moon2.dll")).replace("/", "\\").upper()
    assert oracle.loaded_provider_matches({"moon_path": path}, dll)


def test_curve_runner_hashes_imported_implementation(monkeypatch, tmp_path) -> None:
    monkeypatch.syspath_prepend(str(TOOLS))
    spec = importlib.util.spec_from_file_location(
        "nxc_test_kaiji_curve", TOOLS / "verify_kaiji_curve.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    installed_path = tmp_path / "installed" / "picture_curve.py"
    namespace = {}
    exec(
        compile("def generate_curve(payload): return ()", str(installed_path), "exec"),
        namespace,
    )
    monkeypatch.setattr(module, "generate_curve", namespace["generate_curve"])
    seen = []

    def digest(path):
        seen.append(Path(path).resolve())
        return module.DLL_HASH

    class StopAfterHashes(RuntimeError):
        pass

    def stop(dll, probe):
        raise StopAfterHashes

    monkeypatch.setattr(module, "digest", digest)
    monkeypatch.setattr(module, "dependency_hashes", stop)
    monkeypatch.setattr(module, "read_report", lambda *args: ([], "unused"))
    args = ["verify_kaiji_curve"]
    for name in (
        "dll",
        "probe",
        "wine-prefix",
        "samples",
        "oracle",
        "matrix",
        "matrix-dir",
        "output",
    ):
        args.extend((f"--{name}", str(tmp_path / name)))
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(StopAfterHashes):
        module.main()
    assert installed_path in seen
    assert (TOOLS.parent / "src/nx_ctudio/picture_curve.py") not in seen
