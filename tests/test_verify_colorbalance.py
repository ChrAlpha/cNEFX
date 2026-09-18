"""Reject metadata evidence that executes an unpinned dependency location."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from verify_colorbalance import provider_matches  # noqa: E402
from verify_np3_header import windows  # noqa: E402


def test_loaded_provider_must_match_dll_directory(tmp_path):
    dll = tmp_path / "vendor" / "Asteroid11.dll"
    expected = windows(dll.parent / "Strato10.dll")
    assert provider_matches(expected, dll)
    assert provider_matches(expected.replace("\\", "/").upper(), dll)
    assert not provider_matches(windows(tmp_path / "probe" / "Strato10.dll"), dll)
    assert not provider_matches(expected + ".old", dll)


@pytest.mark.parametrize("actual", [None, True, 0, "", "Strato10.dll", {}])
def test_missing_or_malformed_provider_is_rejected(tmp_path, actual):
    assert not provider_matches(actual, tmp_path / "Asteroid11.dll")
