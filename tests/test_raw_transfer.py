"""Bound standard RAW transfer direction, clipping, thresholds and precision."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from raw_transfer import srgb_lut  # noqa: E402
from verify_np3_header import windows  # noqa: E402
from verify_raw_transfer import validate_event  # noqa: E402


@pytest.mark.parametrize("inverse", [False, True])
def test_tables_are_monotone_full_domain_rgb16(inverse):
    lut = srgb_lut(inverse=inverse)
    assert lut.shape == (65536,) and lut.dtype == np.dtype("<u2")
    # The vendor encoder ends at65534 because float32 1.055-0.055 is below1.
    assert lut[0] == 0 and lut[-1] == (65535 if inverse else 65534)
    assert np.all(np.diff(lut.astype(np.int32)) >= 0)
    if inverse:
        assert np.all(lut <= np.arange(65536))
    else:
        assert np.all(lut[:-1] >= np.arange(65535))


def test_piecewise_threshold_and_powf_sensitive_samples():
    encoded = srgb_lut()
    decoded = srgb_lut(inverse=True)
    # These include the threshold neighbors and the double-pow divergence points.
    np.testing.assert_array_equal(
        encoded[[0, 1, 204, 205, 206, 17314, 65535]],
        [0, 12, 2635, 2648, 2661, 36102, 65534],
    )
    np.testing.assert_array_equal(
        decoded[[0, 1, 2650, 2651, 2652, 47587, 65535]],
        [0, 0, 205, 205, 205, 31857, 65535],
    )


@pytest.mark.parametrize("inverse", [0, 1, None, "inverse"])
def test_direction_is_explicit_boolean(inverse):
    with pytest.raises(ValueError, match="boolean"):
        srgb_lut(inverse=inverse)


@pytest.mark.parametrize(
    "field,value",
    [
        ("unchanged", False),
        ("saved", 1),
        ("released", False),
        ("bytes", True),
        ("provider_hex", "ff"),
    ],
)
def test_reject_incomplete_or_wrong_provider_evidence(tmp_path, field, value):
    dll = tmp_path / "utilities.dll"
    event = {
        "event": "raw_transfer",
        "provider_hex": windows(dll).encode().hex(),
        "bytes": 262144,
        "unchanged": True,
        "saved": True,
        "released": True,
    }
    validate_event(event, dll)
    event[field] = value
    with pytest.raises(ValueError):
        validate_event(event, dll)
