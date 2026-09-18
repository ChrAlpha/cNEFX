"""Native NP3 point-curve sampling recovered from the Nikon parameter oracle."""

from __future__ import annotations

import math

CURVE_BYTES = 578
MAX_POINTS = 20
TABLE_SAMPLES = 257


def _natural_derivatives(x: list[float], y: list[float]) -> list[float]:
    """Solve the vendor's natural cubic spline derivative system."""
    count = len(x)
    if count == 2:
        slope = (y[1] - y[0]) / (x[1] - x[0])
        return [slope, slope]
    widths = [x[i + 1] - x[i] for i in range(count - 1)]
    slopes = [(y[i + 1] - y[i]) / widths[i] for i in range(count - 1)]
    derivatives = [slopes[0]]
    for index in range(1, count - 1):
        derivatives.append(
            (widths[index] * slopes[index - 1] + slopes[index] * widths[index - 1])
            / (widths[index - 1] + widths[index])
        )
    derivatives.append(2 * slopes[-1] - derivatives[-1])
    derivatives[0] = 2 * derivatives[0] - derivatives[1]

    lower = [0.0] * count
    upper = [0.0] * count
    rhs = [0.0] * count
    upper[0] = 0.5
    lower[-1] = 0.5
    rhs[0] = (derivatives[0] + derivatives[1]) * 0.75
    rhs[-1] = (derivatives[-2] + derivatives[-1]) * 0.75
    for index in range(1, count):
        if index < count - 1:
            scale = 0.5 / (x[index + 1] - x[index - 1])
            lower[index] = (x[index + 1] - x[index]) * scale
            upper[index] = (x[index] - x[index - 1]) * scale
            rhs[index] = derivatives[index] * 1.5
        divisor = 1.0 - lower[index] * upper[index - 1]
        if index < count - 1:
            upper[index] /= divisor
        rhs[index] = (rhs[index] - lower[index] * rhs[index - 1]) / divisor
    derivatives[-1] = rhs[-1]
    for index in range(count - 2, -1, -1):
        derivatives[index] = rhs[index] - derivatives[index + 1] * upper[index]
    return derivatives


def _sample_spline(
    x: list[float], y: list[float], derivatives: list[float], value: float
) -> float:
    if value <= x[0]:
        return y[0]
    if value >= x[-1]:
        return y[-1]
    interval = 0
    while x[interval + 1] < value:
        interval += 1
    if len(x) == 2:
        return y[0] + (value - x[0]) * (y[1] - y[0]) / (x[1] - x[0])
    width = x[interval + 1] - x[interval]
    t = (value - x[interval]) / width
    a = (x[interval + 1] - value) / width
    return (
        ((2.0 - t) + a) * y[interval + 1] - derivatives[interval + 1] * width * a
    ) * t * t + (
        ((2.0 - a) + t) * y[interval] + derivatives[interval] * width * t
    ) * a * a


def _validate(
    payload: bytes,
) -> tuple[list[float], list[float], float, float, float, float, float]:
    if type(payload) is not bytes or len(payload) != CURVE_BYTES:
        raise ValueError("Expected a 578-byte NP3 curve record.")
    if payload[:2] != b"I0":
        raise ValueError("Unsupported NP3 curve marker.")
    count = payload[8]
    if not 2 <= count <= MAX_POINTS:
        raise ValueError("Curve must contain 2 to 20 points.")
    if any(payload[9 + 2 * count : 64]):
        raise ValueError("Curve padding must be zero.")
    if payload[3] - payload[2] < 3 or payload[5] - payload[4] < 3:
        raise ValueError("Curve input and output levels are too close.")
    if payload[7] > 99:
        raise ValueError("Curve gamma hundredths must be below 100.")
    x = [payload[9 + 2 * i] / 255.0 for i in range(count)]
    y = [payload[10 + 2 * i] / 255.0 for i in range(count)]
    if any(right <= left for left, right in zip(x, x[1:])):
        raise ValueError("Curve point x coordinates must be strictly increasing.")
    input_min = payload[2] / 255.0
    input_max = payload[3] / 255.0
    output_min = payload[4] / 255.0
    output_max = payload[5] / 255.0
    gamma = min(20.0, max(0.05, payload[6] + payload[7] * 0.01))
    return x, y, input_min, input_max, output_min, output_max, gamma


def generate_curve(payload: bytes) -> tuple[int, ...]:
    """Generate the 257-entry uint16 table from an NP3 curve record.

    The stored table bytes at offset 64 are deliberately ignored; the Nikon
    importer regenerates samples from the point and level fields.
    """
    x, y, input_min, input_max, output_min, output_max, gamma = _validate(payload)
    derivatives = _natural_derivatives(x, y)
    table: list[int] = []
    for index in range(TABLE_SAMPLES):
        value = (index / 256.0 - input_min) / (input_max - input_min)
        if value <= x[0]:
            curve = y[0]
        elif value >= x[-1]:
            curve = y[-1]
        else:
            curve = _sample_spline(x, y, derivatives, value)
            if gamma != 1.0:
                normalized = (value - x[0]) / (x[-1] - x[0])
                curve += math.pow(normalized, 1.0 / gamma) - normalized
        curve = min(1.0, max(0.0, curve))
        curve = curve * (output_max - output_min) + output_min
        table.append(int(32767.0 * curve + 0.5))
    return tuple(table)


def generate_table(payload: bytes) -> tuple[int, ...]:
    """Backward-compatible alias for :func:`generate_curve`."""
    return generate_curve(payload)
