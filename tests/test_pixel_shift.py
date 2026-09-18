"""Native numerical contracts independent of file decoding and vendor binaries."""

import numpy as np
import pytest

from nx_ctudio._sparse import Selection
from nx_ctudio.pixel_shift import reconstruct

EMPTY_SELECTOR = b"0100" + bytes(24)


def test_four_frame_color_assignment_and_integer_green_average() -> None:
    frames = [np.full((8, 8), value, np.uint16) for value in (100, 201, 300, 400)]
    actual = reconstruct(frames, (2, 2, 2, 2), EMPTY_SELECTOR, 3201)
    expected = np.array(
        [[[100, 300, 300], [201, 200, 400]], [[400, 200, 201], [300, 300, 100]]],
        dtype=np.uint16,
    )
    np.testing.assert_array_equal(actual, expected)
    for frame, value in zip(frames, (100, 201, 300, 400), strict=True):
        assert np.all(frame == value)


def test_eight_frame_vertical_pair_offset_and_truncation() -> None:
    frames = [np.full((8, 8), 100, np.uint16) for _ in range(4)]
    frames += [np.full((8, 8), 201, np.uint16) for _ in range(4)]
    for frame in frames[4:]:
        frame[2:] = 401
    actual = reconstruct(frames, (3, 3, 1, 1), EMPTY_SELECTOR, 3201)
    # Raw rows 2/3 pair with rows 0/1, never with rows 2/3 in the second half.
    np.testing.assert_array_equal(actual, [[[150, 150, 150]]])


def test_sixteen_frame_subpixel_interleave() -> None:
    frames = [np.full((8, 8), 100 * (index // 4 + 1), np.uint16) for index in range(16)]
    actual = reconstruct(frames, (2, 2, 1, 1), EMPTY_SELECTOR, 3201)
    np.testing.assert_array_equal(
        actual,
        [[[100, 100, 100], [200, 200, 200]], [[400, 400, 400], [300, 300, 300]]],
    )


def test_sparse_zero_denominators_and_input_ownership() -> None:
    selector = b"0100" + Selection(12, 16, first_row=4).pack()
    frames = [np.zeros((12, 16), np.uint16) for _ in range(4)]
    actual = reconstruct(frames, (2, 2, 12, 8), selector, 3200)
    assert actual.shape == (8, 12, 3)
    assert not np.any(actual)
    assert all(not np.any(frame) for frame in frames)


def test_sparse_selector_budget_is_checked_before_enumeration() -> None:
    selection = Selection(
        12, 16, middle_count=255, outer_count=255, inner_before=255, inner_after=255
    )
    with pytest.raises(ValueError, match="work budget"):
        selection.coordinates(16, 12)


@pytest.mark.parametrize("crop", [(0, 1, 4, 4), (1, 0, 4, 4), (1, 1, 8, 8)])
def test_invalid_crop_is_rejected(crop: tuple[int, int, int, int]) -> None:
    frames = [np.zeros((8, 8), np.uint16) for _ in range(4)]
    with pytest.raises(ValueError, match="Crop"):
        reconstruct(frames, crop, EMPTY_SELECTOR, 3201)


def test_unsafe_sparse_neighborhood_fails_before_reading() -> None:
    selector = b"0100" + Selection(8, 8, first_row=1).pack()
    frames = [np.zeros((8, 8), np.uint16) for _ in range(4)]
    with pytest.raises(ValueError, match="Unsafe"):
        reconstruct(frames, (1, 1, 6, 6), selector, 3200)


@pytest.mark.parametrize("count", [0, 3, 5, 12])
def test_unsupported_count_fails(count: int) -> None:
    with pytest.raises(ValueError, match="ordered RAW"):
        reconstruct(
            [np.zeros((8, 8), np.uint16)] * count, (1, 1, 6, 6), EMPTY_SELECTOR, 3201
        )
