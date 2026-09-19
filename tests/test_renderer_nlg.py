from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from renderer_nlg import Parameters, decode_state, transform_row


def parameters(varying=False):
    return Parameters(
        np.array([[724, 296, 4], [2, 1021, 1], [0, 56, 968]], dtype="<i2"),
        np.stack(
            [np.minimum(np.arange(65536) * k // 32, 32767) for k in (29, 31, 32)]
        ).astype("<u2"),
        np.minimum(np.arange(8192) // 2, 4095).astype("<u2"),
        (
            np.stack([np.arange(4096) // 4 + 512, 1535 - np.arange(4096) // 4])
            if varying
            else np.full((2, 4096), 1024)
        ).astype("<u2"),
        (np.arange(4096) // 8 + 512 if varying else np.full(4096, 1024)).astype("<u2"),
    )


def state_bytes():
    p = parameters(True)
    state = bytearray(0x10F9C0)
    for offset, value in {
        0x62: 256,
        0x66: 38,
        0x68: 48,
        0x6A: 256,
        0x70: 1126,
        0x74: 1228,
        0x76: 4096,
        0x78: 4096,
    }.items():
        struct.pack_into("<H", state, offset, value)
    for offset, value in {0x5C: 1, 0x6C: 205, 0x6D: 4, 0x2EC3: 1}.items():
        state[offset] = value
    arrays = {
        0x4A: p.matrix,
        0xB7D0: p.spatial_curve,
        0xF7D0: np.full(4096, 1024, dtype="<u2"),
        0x197D0: np.full(4096, 1024, dtype="<u2"),
    }
    arrays.update({0x2F7D0 + i * 0x40000: p.curves[i] for i in range(3)})
    arrays.update({k: p.difference_curves[0] for k in (0x117D0, 0x157D0)})
    arrays.update({k: p.difference_curves[1] for k in (0x137D0, 0x177D0)})
    arrays.update({k: p.gain_curve for k in (0x1B7D0, 0x1D7D0)})
    for offset, array in arrays.items():
        state[offset : offset + array.nbytes] = array.tobytes()
    return bytes(state)


def scalar(pixels, first, second, origin, p):
    weights = (
        (225, 286, 1),
        (167, 338, 7),
        (119, 370, 23),
        (81, 382, 49),
        (49, 382, 81),
        (23, 370, 119),
        (7, 338, 167),
        (1, 286, 225),
    )
    output, auxiliary = [], []
    for x, pixel in enumerate(pixels.tolist()):
        rgb = []
        for channel, row in enumerate(p.matrix.tolist()):
            total = sum((a * b + 256) // 512 for a, b in zip(row, pixel, strict=True))
            rgb.append(int(p.curves[channel, (total + 1) // 2]) // 8)
        spatial = []
        for plane in (first.tolist(), second.tolist()):
            coordinate = (origin[0] % 8) + x
            coarse = coordinate // 8
            horizontal = [
                sum(row[coarse + k] * weights[coordinate % 8][k] for k in range(3))
                // 512
                for row in plane
            ]
            value = (
                sum(horizontal[k] * weights[origin[1] % 8][k] for k in range(3)) // 512
            )
            square = min((value + 48) ** 2 // 2048, 8191)
            index = max(square - 38, 0)
            spatial.append(int(p.spatial_curve[index]))
        red, green, blue = rgb
        cb = (blue * 512 - green * 339 - red * 173) // 1024
        cr = (red * 512 - green * 429 - blue * 83) // 1024
        if cr * 339 < cb * 429:
            chroma = cb * 1815 if cr * 173 >= cb * -512 else cb * -352 + cr * -731
        else:
            chroma = cr * 1436 if cr * -512 < cb * 83 else cb * -352 + cr * -731
        luma = min(
            (205 * (chroma // 1024) + 128) // 256
            + (green * 601 + red * 306 + blue * 117) // 1024,
            4095,
        )
        blends = []
        for coarse, amount in zip(spatial, (1126, 1228), strict=True):
            index = luma // 2 - coarse // 2 + 2048
            near, far = map(int, p.difference_curves[:, index])
            value = near * amount // 2048 + far * (2048 - amount) // 2048 - 1024
            blends.append((value * int(p.gain_curve[spatial[0]]) // 1024 + 1024) & 8191)
        gain = min(blends[0] * blends[1] // 1024, 8191)
        output.append([min(v * gain // 128, 32767) * 2 for v in rgb])
        auxiliary.append(min(gain // 16, 255))
    return np.array(output, dtype="<u2"), np.array(auxiliary, dtype="<u2")


@pytest.mark.parametrize("width", [1, 7, 16, 17, 50, 208, 1024])
@pytest.mark.parametrize("origin", [(0, 0), (5, 7), (16, 8)])
def test_matches_scalar_arithmetic(width, origin):
    rng = np.random.default_rng(width + origin[0])
    pixels = rng.integers(0, 32768, (width, 3), dtype=np.uint16)
    coarse_width = ((origin[0] & 7) + width + 7) // 8 + 2
    a, b = rng.integers(0, 4096, (2, 3, coarse_width), dtype=np.uint16)
    p = parameters(True)
    expected = scalar(pixels, a, b, origin, p)
    actual = transform_row(pixels, a, b, origin, p)
    for got, want in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(got, want)


def test_neutral_gain_and_input_immutability():
    p = parameters()
    pixels = np.array([[0, 0, 0], [8, 8, 8], [32767, 32767, 32767]], dtype="<u2")
    original = pixels.copy()
    planes = np.full((3, 3), 2048, dtype="<u2")
    pixels.flags.writeable = planes.flags.writeable = False
    rgb, aux = transform_row(pixels, planes, planes, (0, 0), p)
    expected = (
        np.column_stack([p.curves[c, pixels[:, c]] for c in range(3)]).astype(np.int64)
        >> 3
    ) * 16
    np.testing.assert_array_equal(rgb, np.minimum(expected, 65534).astype("<u2"))
    np.testing.assert_array_equal(aux, [64, 64, 64])
    np.testing.assert_array_equal(pixels, original)
    assert not np.shares_memory(rgb, pixels)


def test_noncontiguous_inputs():
    p = parameters(True)
    pixels = np.arange(18, dtype="<u2").reshape(3, 6)[:, ::2]
    planes = np.arange(18, dtype="<u2").reshape(3, 6)[:, ::2]
    actual = transform_row(pixels, planes, planes, (0, 0), p)
    expected = transform_row(pixels.copy(), planes.copy(), planes.copy(), (0, 0), p)
    for got, want in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(got, want)


def test_spatial_square_clamps_before_black_offset():
    p = parameters(True)
    p.spatial_curve[8154:] = 0
    pixels = np.full((1, 3), 25000, dtype="<u2")
    planes = np.full((3, 3), 4095, dtype="<u2")
    expected = scalar(pixels, planes, planes, (0, 0), p)
    actual = transform_row(pixels, planes, planes, (0, 0), p)
    for got, want in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(got, want)


def test_decode_numeric_state():
    state = state_bytes()
    decoded = decode_state(state)
    expected = parameters(True)
    for name in (
        "matrix",
        "curves",
        "spatial_curve",
        "difference_curves",
        "gain_curve",
    ):
        np.testing.assert_array_equal(getattr(decoded, name), getattr(expected, name))
    decoded.matrix[0, 0] = 0
    assert decode_state(state).matrix[0, 0] == 724


def test_decode_preserves_distinct_difference_curve_pairs():
    state = bytearray(state_bytes())
    for offset in (0x137D0, 0x177D0):
        struct.pack_into("<H", state, offset, 1435)
    decoded = decode_state(bytes(state))
    assert decoded.difference_curves[1, 0] == 1435
    assert decoded.difference_curves[0, 0] != 1435


@pytest.mark.parametrize(
    "offset",
    [
        0x48,
        0x5C,
        0x60,
        0x62,
        0x64,
        0x66,
        0x68,
        0x6A,
        0x6C,
        0x6D,
        0x70,
        0x74,
        0x76,
        0x78,
        0x7A,
        0x7C,
        0x7E,
        0x7F,
        0x2E8D,
        0x2EC3,
        0x309A,
        0x10F7D0,
    ],
)
def test_rejects_unsupported_controls(offset):
    state = bytearray(state_bytes())
    state[offset] ^= 1
    with pytest.raises(ValueError, match="Unsupported"):
        decode_state(bytes(state))


@pytest.mark.parametrize(
    "offset", [0xF7D0, 0x197D0, 0x137D0, 0x157D0, 0x177D0, 0x1D7D0]
)
def test_rejects_unsupported_tables(offset):
    state = bytearray(state_bytes())
    state[offset] ^= 1
    with pytest.raises(ValueError):
        decode_state(bytes(state))


@pytest.mark.parametrize("state", [b"", b"\0" * 0x10F9BF, bytearray(0x10F9C0), None])
def test_rejects_wrong_state_shape(state):
    with pytest.raises(ValueError):
        decode_state(state)


@pytest.mark.parametrize(
    "field,value",
    [
        ("pixels", None),
        ("pixels", np.zeros((0, 3), dtype="<u2")),
        ("pixels", np.zeros((1025, 3), dtype="<u2")),
        ("pixels", np.zeros((1, 4), dtype="<u2")),
        ("pixels", np.ones((1, 3), dtype=np.float32)),
        ("pixels", np.full((1, 3), 32768, dtype="<u2")),
        ("coarse0", np.zeros((3, 2), dtype="<u2")),
        ("coarse1", np.full((3, 3), 4096, dtype="<u2")),
        ("origin", [0, 0]),
        ("origin", (False, 0)),
        ("origin", (-1, 0)),
        ("origin", (16384, 0)),
        ("origin", (0, 16384)),
        ("parameters", None),
    ],
)
def test_rejects_bad_row_inputs(field, value):
    args = dict(
        pixels=np.zeros((1, 3), dtype="<u2"),
        coarse0=np.zeros((3, 3), dtype="<u2"),
        coarse1=np.zeros((3, 3), dtype="<u2"),
        origin=(0, 0),
        parameters=parameters(),
    )
    args[field] = value
    with pytest.raises(ValueError):
        transform_row(**args)


@pytest.mark.parametrize(
    "field", ["matrix", "curves", "spatial_curve", "difference_curves", "gain_curve"]
)
def test_rejects_out_of_range_parameters(field):
    p = parameters()
    getattr(p, field).flat[0] = 32767 if field == "matrix" else 65535
    with pytest.raises(ValueError):
        transform_row(
            np.zeros((1, 3), dtype="<u2"),
            np.zeros((3, 3), dtype="<u2"),
            np.zeros((3, 3), dtype="<u2"),
            (0, 0),
            p,
        )
