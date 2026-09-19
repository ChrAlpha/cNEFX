"""Native numerical contracts independent of file decoding and vendor binaries."""

import hashlib
import weakref

import numpy as np
import pytest

from nx_ctudio._sparse import Selection
from nx_ctudio.pixel_shift import output_groups, reconstruct, reconstruct_from_reader

EMPTY_SELECTOR = b"0100" + bytes(24)


@pytest.mark.parametrize(
    "count, outputs",
    [(4, 1), (8, 1), (8, 2), (16, 1), (16, 4), (32, 1), (32, 2), (32, 4), (32, 8)],
)
def test_output_groups_partition_all_captures(count: int, outputs: int) -> None:
    groups = output_groups(count, outputs)
    assert len(groups) == outputs
    assert all(len(group) == count // outputs for group in groups)
    assert sorted(index for group in groups for index in group) == list(range(count))


def test_thirty_two_to_four_uses_cross_half_groups() -> None:
    assert output_groups(32, 4) == (
        (0, 1, 2, 3, 16, 17, 18, 19),
        (4, 5, 6, 7, 20, 21, 22, 23),
        (8, 9, 10, 11, 24, 25, 26, 27),
        (12, 13, 14, 15, 28, 29, 30, 31),
    )


@pytest.mark.parametrize(
    "count, outputs",
    [(4, 2), (8, 4), (16, 2), (32, 3), (32, 0), (12, 1), (8, True), (32.0, 4)],
)
def test_invalid_output_mode_fails(count: int, outputs: int) -> None:
    with pytest.raises(ValueError, match="combination"):
        output_groups(count, outputs)


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


@pytest.mark.parametrize(
    "count, iso, expected",
    [
        (4, 3200, "00c6bfaf3ae40269061c55417136a8a7e91a48cd56e34f0b3ded8809386c570e"),
        (4, 3201, "fdd15054231f1be1a9fd3f6a8a7b4ddad7909a95a7306ad8bc8642ebaaf2d5a1"),
        (8, 3200, "347bf321d896764e6354543a14d772195a384b78db2c0d2bd353e57bc9d8cc41"),
        (8, 3201, "e504c36a171e265f9a5300069bf251d191996c951097efbb3c9bc344ec04eb87"),
        (16, 3200, "66074a4760fe4ae65a7a03f5bf71ca4b0c4f1e2e56082dc3d5a6df0fa52472e2"),
        (16, 3201, "1ea85aed00f49cca995e46f14c0e0f5ef12c517ce3c4b3c23dfbfb8dce896cfd"),
        (32, 3200, "0e023ad1c6b94ee0ed795ff797c19604a1649e6f95ee7b660335881d49059857"),
        (32, 3201, "3d15105fc7346588ad30758ab6ae35a7926a60c693c1d699e310471aa28d63ec"),
    ],
)
def test_reader_releases_inputs_and_preserves_ordered_repairs(
    count: int, iso: int, expected: str
) -> None:
    references = []
    reads = []

    def read(index: int) -> np.ndarray:
        assert index not in reads
        reads.append(index)
        samples = np.random.default_rng(510 + index).integers(
            0, 16384, (18, 24), dtype=np.uint16
        )
        references.append(weakref.ref(samples))
        assert sum(ref() is not None for ref in references) <= 4
        return samples

    selector = (
        b"0100" + Selection(18, 24, first_row=6, inner_step=1, inner_after=5).pack()
    )
    actual = reconstruct_from_reader(count, read, (2, 2, 20, 14), selector, iso)
    assert sorted(reads) == list(range(count))
    assert all(ref() is None for ref in references)
    # Pinned before the streaming change; consecutive repair rows depend on order.
    assert hashlib.sha256(actual.astype("<u2").tobytes()).hexdigest() == expected


@pytest.mark.parametrize("shape", [(9, 10), (10, 9), (9, 9)])
def test_odd_raw_axes_are_outside_the_recovered_domain(shape: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="even"):
        reconstruct(
            [np.zeros(shape, np.uint16)] * 4, (1, 1, 6, 6), EMPTY_SELECTOR, 3201
        )


@pytest.mark.parametrize("failure", ["geometry", "dtype", "range", "decode"])
def test_later_reader_failure_does_not_return_partial_rgb(failure: str) -> None:
    reads = []

    def read(index: int) -> np.ndarray:
        reads.append(index)
        if index == 7:
            if failure == "decode":
                raise OSError("Decode failed.")
            if failure == "geometry":
                return np.zeros((8, 10), np.uint16)
            if failure == "dtype":
                return np.zeros((8, 8), np.int16)
            return np.full((8, 8), 16384, np.uint16)
        return np.zeros((8, 8), np.uint16)

    with pytest.raises((ValueError, OSError)):
        reconstruct_from_reader(16, read, (1, 1, 6, 6), EMPTY_SELECTOR, 3201)
    assert reads == list(range(8))
