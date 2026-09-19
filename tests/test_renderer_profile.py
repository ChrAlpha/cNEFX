"""ICC parameter extraction and mixed-precision inversion boundaries."""

import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from renderer_profile import (  # noqa: E402
    D50,
    inverse_matrix,
    parameters,
    sampled_curve,
)


def profile(matrix=None, words=(0, 32768, 65535), extra=None):
    if matrix is None:
        matrix = np.eye(3, dtype=int) * 65536
    tags = {
        signature: b"XYZ " + bytes(4) + struct.pack(">3i", *column)
        for signature, column in zip((b"rXYZ", b"gXYZ", b"bXYZ"), matrix.T)
    }
    for signature in (b"rTRC", b"gTRC", b"bTRC"):
        tags[signature] = (
            b"curv"
            + bytes(4)
            + struct.pack(">I", len(words))
            + struct.pack(f">{len(words)}H", *words)
        )
    tags.update(extra or {})
    header = bytearray(132 + 12 * len(tags))
    header[8:24] = b"\x02\x20\0\0mntrRGB XYZ "
    header[36:40] = b"acsp"
    header[68:80] = D50
    struct.pack_into(">I", header, 128, len(tags))
    payload = bytearray()
    for index, (signature, value) in enumerate(tags.items()):
        struct.pack_into(
            ">4sII",
            header,
            132 + index * 12,
            signature,
            len(header) + len(payload),
            len(value),
        )
        payload.extend(value)
    result = header + payload
    struct.pack_into(">I", result, 0, len(result))
    return bytes(result)


def test_extracts_colorant_columns_as_row_major_matrix():
    fixed = np.array([[65536, 2048, 8192], [4096, 65536, 16384], [0, 0, 65536]])
    matrix, curves = parameters(profile(fixed))
    np.testing.assert_array_equal(matrix, fixed / 65536)
    assert matrix.dtype == curves.dtype == np.dtype("<f4")
    assert curves.shape == (3, 65536)
    assert curves[0, 0] == 0 and curves[0, -1] == 1


def test_mixed_precision_inverse_matches_recovered_reference():
    fixed = np.array([[28578, 25236, 9375], [14582, 46981, 3972], [912, 6361, 46792]])
    matrix = (fixed / 65536).astype("<f4")
    inverse = inverse_matrix(matrix)
    expected = np.array(
        [
            3.133962392807007,
            -1.6169873476028442,
            -0.4906441867351532,
            -0.9788071513175964,
            1.9161872863769531,
            0.033450618386268616,
            0.07197851687669754,
            -0.22897450625896454,
            1.4055967330932617,
        ],
        dtype="<f4",
    ).reshape(3, 3)
    np.testing.assert_array_equal(inverse, expected)
    assert not np.array_equal(
        inverse, np.linalg.inv(matrix.astype(np.float64)).astype("<f4")
    )
    np.testing.assert_allclose(
        matrix.astype(np.float64) @ inverse, np.eye(3), atol=2e-7
    )


def test_inverse_direction_returns_unquantized_output_curves():
    matrix, curves = parameters(profile(words=(0, 16384, 65535)), inverse=True)
    np.testing.assert_array_equal(matrix, np.eye(3))
    assert curves[0, 16384] == 32767.5
    assert curves[0, 0] == 0 and curves[0, -1] == 65535
    assert np.all(curves[:, 1:] >= curves[:, :-1])


def test_inverse_curve_agrees_with_independent_piecewise_reference():
    words = np.array([0, 3, 35, 411, 32000, 65535], dtype="<u2")
    expected = np.interp(np.arange(65536), words, np.linspace(0, 65535, len(words)))
    result = sampled_curve(words, inverse=True)
    np.testing.assert_allclose(result, expected, atol=0.004, rtol=0)
    np.testing.assert_array_equal(words, [0, 3, 35, 411, 32000, 65535])


def test_forward_curve_expansion_preserves_domains_and_monotonicity():
    words = np.array([0, 10, 2000, 40000, 65535], dtype="<u2")
    actual = sampled_curve(words)
    expected = (
        np.interp(np.arange(65536) / 65535, np.linspace(0, 1, len(words)), words)
        / 65535
    )
    np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=0)
    assert actual[0] == 0 and actual[-1] == 1 and np.all(np.diff(actual) >= 0)


def test_reject_curve_whose_float32_expansion_reverses():
    # The steep first segment leaves a near-flat segment with rounding reversals.
    words = np.array([0, 65534, 65535], dtype="<u2")
    with pytest.raises(ValueError, match="Expanded curve"):
        sampled_curve(words)
    with pytest.raises(ValueError, match="Expanded curve"):
        parameters(profile(words=words))


@pytest.mark.parametrize(
    "offset,value", [(8, 4), (12, ord("s")), (20, ord("L")), (68, 1), (36, 0)]
)
def test_reject_unsupported_headers(offset, value):
    data = bytearray(profile())
    data[offset] = value
    with pytest.raises(ValueError):
        parameters(bytes(data))


@pytest.mark.parametrize(
    "extra",
    [
        {b"A2B0": bytes(12)},
        {b"B2A0": bytes(12)},
        {b"rXYZ": bytes(20)},
        {b"rXYZ": b"XYZ " + bytes(15)},
        {b"rTRC": b"para" + bytes(12)},
    ],
)
def test_reject_unsupported_tag_layouts(extra):
    with pytest.raises(ValueError):
        parameters(profile(extra=extra))


@pytest.mark.parametrize(
    "words", [(0,), (0, 65534), (1, 65535), (0, 0, 65535), (0, 42, 35, 65535)]
)
def test_reject_unsupported_curves(words):
    with pytest.raises(ValueError):
        parameters(profile(words=words))


@pytest.mark.parametrize("value", [np.nan, np.inf, -17, 17])
def test_reject_nonfinite_or_unbounded_inverse(value):
    matrix = np.eye(3, dtype="<f4")
    matrix[0, 0] = value
    with pytest.raises(ValueError):
        inverse_matrix(matrix)


def test_determinant_gate_and_inverse_range():
    for matrix in (
        np.diag([1, 1, 1 / 65536]),
        np.diag([1, 1, 1 / 32]),
        np.eye(3)[[1, 0, 2]],
    ):
        with pytest.raises(ValueError):
            inverse_matrix(matrix.astype("<f4"))
    np.testing.assert_array_equal(
        inverse_matrix(np.diag([1, 1, 1 / 16]).astype("<f4")), np.diag([1, 1, 16])
    )


@pytest.mark.parametrize("inverse", [0, 1, None, "inverse"])
def test_direction_is_boolean(inverse):
    with pytest.raises(ValueError):
        parameters(profile(), inverse=inverse)
