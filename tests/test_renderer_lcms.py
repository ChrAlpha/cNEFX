"""Native FFI guards, resource cleanup and final-CMS configuration gates."""

import ctypes
import hashlib
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import renderer_lcms  # noqa: E402
from verify_renderer_lcms import state_contract  # noqa: E402


class Function:
    def __init__(self, callback):
        self.callback = callback

    def __call__(self, *args):
        return self.callback(*args)


@pytest.fixture
def ffi(tmp_path, monkeypatch):
    library = tmp_path / "test-library"
    library.write_bytes(b"stub")
    profile = b"synthetic profile"
    monkeypatch.setattr(
        renderer_lcms, "PROFILE_HASH", hashlib.sha256(profile).hexdigest()
    )
    monkeypatch.setattr(
        renderer_lcms, "LIBRARY_HASH", hashlib.sha256(b"stub").hexdigest()
    )
    state = {"opened": [], "closed": [], "destroyed": [], "rows": [], "fault": None}

    def open_profile(data, size):
        assert data == profile and size == len(profile)
        if state["fault"] == "second_profile" and state["opened"]:
            return 0
        handle = 11 + len(state["opened"])
        state["opened"].append(handle)
        return handle

    def create(source, input_format, target, output_format, intent, flags):
        assert (source, target) == (11, 12)
        assert (input_format, output_format, intent, flags) == (0x4001A, 0x4001A, 0, 0)
        return 0 if state["fault"] == "transform" else 123

    def process(handle, source, target, width):
        assert handle == 123
        state["rows"].append(width)
        ctypes.memmove(target, source, width * 6)
        if state["fault"] == "input":
            ctypes.c_uint16.from_address(source).value ^= 1
        if state["fault"] == "guard":
            ctypes.c_uint16.from_address(target + width * 6).value = 0
        if state["fault"] == "process":
            raise RuntimeError("Injected processing failure")
        if state["fault"] == "cancel":
            raise KeyboardInterrupt

    def close(handle):
        state["closed"].append(handle)
        return state["fault"] != "close"

    class Library:
        cmsOpenProfileFromMem = Function(open_profile)
        cmsCreateTransform = Function(create)
        cmsDoTransform = Function(process)
        cmsDeleteTransform = Function(state["destroyed"].append)
        cmsCloseProfile = Function(close)

    monkeypatch.setattr(renderer_lcms.c, "CDLL", lambda path: Library())
    return library, profile, state


def test_rows_are_guarded_input_is_immutable_and_handles_close(ffi):
    library, profile, state = ffi
    pixels = np.arange(36, dtype="<u2").reshape(12, 3)[::2]
    pixels.flags.writeable = False
    output = renderer_lcms.transform(library, profile, pixels, 2)
    np.testing.assert_array_equal(output, pixels)
    assert output.flags.c_contiguous and not np.shares_memory(output, pixels)
    assert state["rows"] == [2, 2, 2]
    assert state["closed"] == [11, 12] and state["destroyed"] == [123]


@pytest.mark.parametrize(
    "fault",
    ["second_profile", "transform", "input", "guard", "process", "cancel", "close"],
)
def test_failed_native_operations_release_every_created_handle(ffi, fault):
    library, profile, state = ffi
    state["fault"] = fault
    with pytest.raises((ValueError, RuntimeError, KeyboardInterrupt)):
        renderer_lcms.transform(library, profile, np.zeros((4, 3), dtype="<u2"), 2)
    assert state["closed"] == state["opened"]
    assert state["destroyed"] == (
        [] if fault in ("second_profile", "transform") else [123]
    )


@pytest.mark.parametrize("width", [0, -1, True, 3, 5])
def test_reject_invalid_rows_before_loading(ffi, width):
    library, profile, state = ffi
    with pytest.raises(ValueError):
        renderer_lcms.transform(library, profile, np.zeros((4, 3), dtype="<u2"), width)
    assert not state["opened"]


@pytest.mark.parametrize(
    "pixels",
    [
        None,
        np.zeros((0, 3), dtype="<u2"),
        np.zeros((4, 2), dtype="<u2"),
        np.zeros((4, 3), dtype="<f4"),
    ],
)
def test_reject_invalid_pixels_before_loading(ffi, pixels):
    library, profile, state = ffi
    with pytest.raises(ValueError):
        renderer_lcms.transform(library, profile, pixels, 2)
    assert not state["opened"]


def test_profile_and_library_drift_fail_before_loading(ffi):
    library, profile, state = ffi
    pixels = np.zeros((4, 3), dtype="<u2")
    with pytest.raises(ValueError, match="profile"):
        renderer_lcms.transform(library, profile + b"x", pixels, 2)
    library.write_bytes(b"changed")
    with pytest.raises(ValueError, match="build"):
        renderer_lcms.transform(library, profile, pixels, 2)
    assert not state["opened"]


def final_state(tmp_path):
    prefix = tmp_path / "wine"
    profile = prefix / "drive_c/Program Files/Common Files/Nikon/Profiles/NKsRGB.icm"
    state = bytearray(0x2E8)
    for offset, value in (
        (0xA8, "C:/Program Files/Common Files/Nikon/Profiles/NKsRGB.icm"),
        (0x1AC, "Z:" + str(profile)),
    ):
        raw = value.encode("ascii")
        state[offset : offset + len(raw)] = raw
    for offset, value in (
        (0x2B0, 16),
        (0x2B4, 0),
        (0x2B8, 1),
        (0x2C0, 0x10060201),
        (0x2C4, 0x10060201),
    ):
        struct.pack_into("<I", state, offset, value)
    struct.pack_into("<Q", state, 0x2D8, 1234)
    return state, profile, prefix


def test_state_uses_byte_sized_bpc_and_ignores_padding(tmp_path):
    state, profile, prefix = final_state(tmp_path)
    state[0x2BD:0x2C0] = b"pad"
    result = state_contract(state, profile, prefix)
    assert result["intent"] == result["flags"] == 0


@pytest.mark.parametrize(
    "offset", [0xA8, 0x1AC, 0x2B0, 0x2B4, 0x2B8, 0x2BC, 0x2C0, 0x2C4]
)
def test_state_rejects_unsupported_configurations(tmp_path, offset):
    state, profile, prefix = final_state(tmp_path)
    state[offset] ^= 1
    with pytest.raises(ValueError):
        state_contract(state, profile, prefix)


def test_state_requires_complete_paths_and_transform(tmp_path):
    state, profile, prefix = final_state(tmp_path)
    with pytest.raises(ValueError):
        state_contract(state[:-1], profile, prefix)
    state[0x2D8:0x2E0] = bytes(8)
    with pytest.raises(ValueError):
        state_contract(state, profile, prefix)
    state, profile, prefix = final_state(tmp_path)
    state[0xA8:0x1AC] = b"x" * 260
    with pytest.raises(ValueError, match="Unterminated"):
        state_contract(state, profile, prefix)
