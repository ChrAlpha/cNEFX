"""Independent scalar checks for native Nlg spatial-plane recursion."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from nlg_pyramid import spatial_planes


def mirror(i, size):
    if size == 1:
        return 0
    while i < 0 or i >= size:
        i = -i if i < 0 else 2 * size - 2 - i
    return i


def reference(pixels, amounts, previous=None):
    h, w, _ = pixels.shape
    planes = [np.empty((h, w), dtype="<u2") for _ in range(2)]
    for y in range(h):
        for x in range(w):
            r, g, b = map(int, pixels[y, x])
            cb, cr = (
                (512 * b - 339 * g - 173 * r) // 1024,
                (512 * r - 429 * g - 83 * b) // 1024,
            )
            if cr * 339 < cb * 429:
                chroma = cb * 1815 if cr * 173 >= -cb * 512 else -cb * 352 - cr * 731
            else:
                chroma = cr * 1436 if -cr * 512 < cb * 83 else -cb * 352 - cr * 731
            chroma = amounts[0] * (chroma // 1024)
            chroma = min(max((chroma - int(chroma < 0) + 128) // 256, 0), 16383)
            value = min(chroma + (601 * g + 117 * b + 306 * r) // 1024, 16383)
            for i, amount in enumerate(amounts[1:]):
                local = value * amount // 256
                if previous is not None:
                    old = previous[i]
                    by, bx = y // 2, x // 2
                    ny = mirror(by + (1 if y % 2 else -1), len(old))
                    nx = mirror(bx + (1 if x % 2 else -1), old.shape[1])
                    local += (
                        9 * int(old[by, bx])
                        + 3 * int(old[by, nx])
                        + 3 * int(old[ny, bx])
                        + int(old[ny, nx])
                    ) // 16
                planes[i][y, x] = min(local, 16383)
    output = []
    for plane in planes:
        horizontal = np.empty_like(plane)
        final = np.empty_like(plane)
        for y in range(h):
            for x in range(w):
                horizontal[y, x] = (
                    int(plane[y, mirror(x - 1, w)])
                    + 2 * int(plane[y, x])
                    + int(plane[y, mirror(x + 1, w)])
                ) // 4
        for y in range(h):
            for x in range(w):
                final[y, x] = (
                    int(horizontal[mirror(y - 1, h), x])
                    + 2 * int(horizontal[y, x])
                    + int(horizontal[mirror(y + 1, h), x])
                ) // 4
        output.append(final)
    return tuple(output)


@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (9, 1), (3, 3), (7, 8), (33, 22)])
@pytest.mark.parametrize(
    "amounts", [(0, 0, 0), (205, 0, 125), (205, 35, 35), (255, 255, 255)]
)
def test_scalar_reference_without_parent(shape, amounts):
    pixels = np.random.default_rng(sum(shape)).integers(
        0, 16384, (*shape, 3), dtype="<u2"
    )
    expected = reference(pixels, amounts)
    actual = spatial_planes(pixels, amounts)
    for got, want in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("shape", [(3, 3), (4, 4), (7, 8), (8, 7), (17, 19)])
def test_parent_interpolation_saturation_and_odd_dimensions(shape):
    rng = np.random.default_rng(sum(shape))
    pixels = rng.integers(0, 16384, (*shape, 3), dtype="<u2")
    old = tuple(
        rng.integers(0, 16384, ((shape[0] + 1) // 2, (shape[1] + 1) // 2), dtype="<u2")
        for _ in range(2)
    )
    expected = reference(pixels, (205, 71, 18), old)
    for got, want in zip(
        spatial_planes(pixels, (205, 71, 18), old), expected, strict=True
    ):
        np.testing.assert_array_equal(got, want)


def test_constant_luminance_and_plane_order():
    pixels = np.full((3, 5, 3), 4096, dtype="<u2")
    first, second = spatial_planes(pixels, (205, 0, 125))
    np.testing.assert_array_equal(first, 0)
    np.testing.assert_array_equal(second, 2000)


def test_separable_smoothing_truncates_after_each_pass():
    pixels = np.zeros((3, 3, 3), dtype="<u2")
    pixels[:, 1] = np.array([2, 3, 4])[:, None]
    first, _ = spatial_planes(pixels, (0, 255, 0))
    assert first[1, 1] == 0
    assert (1 * 2 + 2 * 4 + 3 * 2) // 16 == 1


def test_readonly_noncontiguous_arrays_are_unchanged():
    pixels = np.arange(7 * 16 * 3, dtype="<u2").reshape(7, 16, 3)[:, ::2]
    pixels.flags.writeable = False
    old = tuple(np.arange(16, dtype="<u2").reshape(4, 4)[::-1] for _ in range(2))
    for plane in old:
        plane.flags.writeable = False
    before = pixels.copy()
    actual = spatial_planes(pixels, (205, 10, 20), old)
    expected = reference(pixels, (205, 10, 20), old)
    for got, want in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(got, want)
        assert not np.shares_memory(got, pixels)
    np.testing.assert_array_equal(before, pixels)


@pytest.mark.parametrize(
    "field,value",
    [
        ("pixels", None),
        ("pixels", np.zeros((0, 3, 3), dtype="<u2")),
        ("pixels", np.zeros((2049, 3, 3), dtype="<u2")),
        ("pixels", np.zeros((3, 3, 4), dtype="<u2")),
        ("pixels", np.full((3, 3, 3), 16384, dtype="<u2")),
        ("pixels", np.zeros((3, 3, 3), dtype=np.float32)),
        ("amounts", [1, 2, 3]),
        ("amounts", (True, 0, 0)),
        ("amounts", (256, 0, 0)),
        ("amounts", (0, -1, 0)),
        ("amounts", (0, 0)),
        ("previous", []),
        ("previous", (np.zeros((2, 2), dtype="<u2"),)),
        ("previous", (np.zeros((2, 2), dtype=np.float32),) * 2),
        ("previous", (np.full((2, 2), 16384, dtype="<u2"),) * 2),
        ("previous", (np.zeros((3, 3), dtype="<u2"),) * 2),
    ],
)
def test_rejects_unverified_inputs(field, value):
    args = dict(
        pixels=np.zeros((3, 3, 3), dtype="<u2"), amounts=(205, 35, 35), previous=None
    )
    args[field] = value
    with pytest.raises(ValueError):
        spatial_planes(**args)


def test_rejects_singleton_parent_interpolation():
    with pytest.raises(ValueError):
        spatial_planes(
            np.zeros((1, 3, 3), dtype="<u2"),
            (0, 0, 0),
            (np.zeros((1, 2), dtype="<u2"),) * 2,
        )
