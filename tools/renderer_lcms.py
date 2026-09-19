"""Bounded native LittleCMS 2.6 replay of the observed final RGB16 stage."""

from __future__ import annotations

import ctypes as c
import hashlib
from pathlib import Path

import numpy as np

PROFILE_HASH = "49caea94c9d36322910350ee37f1fa09629bed70e01cf615a6863ad3f8d1475e"
LIBRARY_HASH = "14a1fbff2d5897f2e5b05c938a920a0267a65c88a24e0894fc057042afdb26db"


def transform(
    library: Path, profile: bytes, pixels: np.ndarray, width: int
) -> np.ndarray:
    """Run pinned NKsRGB-to-NKsRGB, RGB16, intent0, flags0 with row guards.

    The external Linux library is built from upstream LittleCMS 2.6 commit
    08bbb31ffe5a9348a764ee00cf1b0486f921b379. Other builds need separate validation;
    the system LittleCMS 2.16 does not reproduce the captured Nikon result.
    """
    if (
        type(profile) is not bytes
        or hashlib.sha256(profile).hexdigest() != PROFILE_HASH
    ):
        raise ValueError("Only the pinned Nikon sRGB profile is supported.")
    if (
        not isinstance(pixels, np.ndarray)
        or pixels.ndim != 2
        or pixels.shape[1] != 3
        or pixels.dtype != np.dtype("<u2")
        or not 1 <= len(pixels) <= 1048576
        or type(width) is not int
        or width <= 0
        or len(pixels) % width
    ):
        raise ValueError("Expected bounded packed RGB16 pixels and complete rows.")
    if hashlib.sha256(library.read_bytes()).hexdigest() != LIBRARY_HASH:
        raise ValueError("Only the pinned native LittleCMS 2.6 build is supported.")
    lcms = c.CDLL(str(library.resolve()))
    lcms.cmsOpenProfileFromMem.argtypes = [c.c_void_p, c.c_uint32]
    lcms.cmsOpenProfileFromMem.restype = c.c_void_p
    lcms.cmsCreateTransform.argtypes = [
        c.c_void_p,
        c.c_uint32,
        c.c_void_p,
        c.c_uint32,
        c.c_uint32,
        c.c_uint32,
    ]
    lcms.cmsCreateTransform.restype = c.c_void_p
    lcms.cmsDoTransform.argtypes = [c.c_void_p, c.c_void_p, c.c_void_p, c.c_uint32]
    lcms.cmsDoTransform.restype = None
    lcms.cmsDeleteTransform.argtypes = [c.c_void_p]
    lcms.cmsDeleteTransform.restype = None
    lcms.cmsCloseProfile.argtypes = [c.c_void_p]
    lcms.cmsCloseProfile.restype = c.c_int
    count, words = len(pixels) // width, width * 3
    source = np.full((count, words + 32), 0xA55A, dtype="<u2")
    target = source.copy()
    source[:, 16:-16] = pixels.reshape(count, words)
    original = source.copy()
    profiles, handle = [], None
    try:
        for _ in range(2):
            parsed = lcms.cmsOpenProfileFromMem(profile, len(profile))
            if not parsed:
                raise ValueError("LittleCMS rejected the pinned ICC profile.")
            profiles.append(parsed)
        handle = lcms.cmsCreateTransform(
            profiles[0], 0x4001A, profiles[1], 0x4001A, 0, 0
        )
        if not handle:
            raise ValueError("LittleCMS rejected the bounded RGB16 transform.")
        for index in range(count):
            lcms.cmsDoTransform(
                handle,
                source[index, 16:].ctypes.data,
                target[index, 16:].ctypes.data,
                width,
            )
        if (
            not np.array_equal(source, original)
            or np.any(target[:, :16] != 0xA55A)
            or np.any(target[:, -16:] != 0xA55A)
        ):
            raise ValueError("LittleCMS changed input or row guard bytes.")
        return target[:, 16:-16].copy().reshape(-1, 3)
    finally:
        if handle:
            lcms.cmsDeleteTransform(handle)
        released = [lcms.cmsCloseProfile(parsed) for parsed in profiles]
        if not all(released):
            raise ValueError("LittleCMS profile cleanup failed.")
