"""Mode3 contrast resampling must round the full ordinate, before narrowing."""

import hashlib
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import kaiji_contrast  # noqa: E402


def coefficients(monkeypatch):
    data = bytearray(422416)
    for bank in (0x35E7E, 0x44362):
        base = bank + 14 * 0x606
        for point in range(257):
            struct.pack_into("<3H", data, base + point * 6, *([1000 - point] * 3))
    struct.pack_into("<257H", data, 0x5B8D6, *([8] * 257))
    result = bytes(data)
    monkeypatch.setattr(
        kaiji_contrast, "COEFFICIENT_HASH", hashlib.sha256(result).hexdigest()
    )
    return result


def test_mode3_rounds_total_ordinate_before_expansion(monkeypatch):
    data = coefficients(monkeypatch)
    sources, expanded = kaiji_contrast.contrast_tables(
        data, 128, 6, model_index=14, output_selector=1
    )
    # At coordinate8,1000 + (999-1000)/2 =999.5 rounds up to1000.
    assert sources == ((1000,) * 257,) * 4
    assert expanded == ((256000,) * 4096,) * 2
    original, _ = kaiji_contrast.contrast_tables(data, 128, 6, model_index=14)
    assert original[0][:3] == (1000, 999, 998)


@pytest.mark.parametrize("selector", [-1, 2, 3, True, 1.0, None])
def test_rejects_unsupported_output_selector(monkeypatch, selector):
    with pytest.raises(ValueError, match="selector"):
        kaiji_contrast.contrast_tables(
            coefficients(monkeypatch), 0, 6, model_index=14, output_selector=selector
        )
