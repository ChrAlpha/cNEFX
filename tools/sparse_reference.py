"""Normalized research model of the pinned DLL's four-frame sparse repair.

This is a synthetic oracle comparison model, not a validated camera pipeline.
Keep ordered floating-point comparisons and zero-weight terms: NaNs matter.
"""

from __future__ import annotations

import math
import struct
from collections import Counter
from dataclasses import dataclass


def divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return math.nan if numerator == 0 else math.inf
    return numerator / denominator


def float32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def store(value: float) -> int:
    if math.isnan(value):
        # MINSD preserves the second operand on NaN; CVTTSD2SI yields INT_MIN.
        return 0
    return int(value) if value <= 16383 else 16383


def below_or_unordered(left: float, right: float) -> bool:
    """COMISD followed by JBE includes unordered comparisons."""
    return not (left > right)


@dataclass(frozen=True)
class Selection:
    reference_height: int
    reference_width: int
    half_grid_start: int = 2
    first_row: int = 4
    middle_step: int = 0
    outer_step: int = 0
    inner_step: int = 0
    middle_count: int = 1
    outer_count: int = 1
    inner_before: int = 0
    inner_after: int = 0

    def pack(self) -> bytes:
        return struct.pack(
            "<BB6H3BxBxB3x",
            0,
            0,
            self.reference_height,
            self.reference_width,
            self.half_grid_start,
            self.first_row,
            self.middle_step,
            self.outer_step,
            self.inner_step,
            self.middle_count,
            self.outer_count,
            self.inner_before,
            self.inner_after,
        )

    def coordinates(self, width: int, height: int) -> list[tuple[int, int]]:
        row = self.first_row - math.trunc((self.reference_height - height) / 2)
        lower = math.trunc((self.reference_width - width) / 4)
        coordinates = []
        for outer in range(self.outer_count):
            if outer:
                row += self.outer_step
            for middle in range(self.middle_count):
                if middle:
                    row += self.middle_step
                for inner in range(self.inner_before + self.inner_after + 1):
                    if inner:
                        row += self.inner_step
                    if 0 <= row <= height:
                        column = 1
                        for half in range(1, width // 2 - 1):
                            if (
                                half >= self.half_grid_start // 2
                                and lower <= half <= lower + width // 2
                            ):
                                if half != 1:
                                    column += 2
                                if not (
                                    2 <= row < height - 2 and 1 <= column < width - 1
                                ):
                                    raise ValueError("Unsafe vendor neighborhood.")
                                coordinates.append((row, column))
        return coordinates


def repair_value(
    frame: int,
    current: int,
    mean: float,
    diagonal: int,
    guard: int,
    r: float,
    s: float,
    t: float,
    estimate: float,
    trace: Counter[str],
) -> int:
    """r: diagonal ratio; s: mean ratio; t: displaced same-channel ratio."""
    output = current

    def write(value: float, stage: int, branch: str) -> int:
        trace[f"frame{frame}.stage{stage}.{branch}"] += 1
        return store(value)

    def blend(weight: float, stage: int) -> int:
        # Literal weights avoid changing 1 - 0.7 into a nearby binary value.
        other = {0.3: 0.7, 0.5: 0.5, 0.7: 0.3}[weight]
        return write(current * other + estimate * weight, stage, f"blend{weight}")

    le = below_or_unordered
    if le(s, r + 0.1):
        if r < s:
            if t + 0.1 < r:
                output = blend(0.7, 1)
        elif t + 0.1 < s:
            output = blend(0.3, 1)
    elif le(t, r + 0.1) and le(r, t + 0.1):
        output = blend(0.5, 1)

    def high_displaced() -> int:
        if le(t, r + 0.3):
            if divide(current, mean) < divide(diagonal, mean):
                return blend(0.5, 2)
            trace[f"frame{frame}.stage2.ratio_keep"] += 1
            return current
        if le(mean, 2000) or le(divide(current, mean) + 0.4, divide(diagonal, mean)):
            trace[f"frame{frame}.stage2.intensity_keep"] += 1
            return current
        if le(estimate, current + 500) or estimate + 500 < mean:
            return write(estimate + 0.0, 2, "intensity_replace")
        return write(current + estimate * 0.0, 2, "intensity_zero_weight")

    def low_displaced() -> int:
        if le(1.1, r):
            return blend(0.5, 2)
        if estimate + 2000 < guard or estimate + 300 < diagonal:
            return write(current + estimate * 0.0, 2, "low_ratio_zero_weight")
        return write(estimate + 0.0, 2, "low_ratio_replace")

    near = le(s, t + 0.1)
    if near:
        if le(t, s + 0.1):
            if s < t < r + 0.1:
                output = blend(0.5, 2)
        else:
            output = high_displaced()
    elif frame == 0:
        if le(r, t + 0.1):
            if le(t, r + 0.1):
                value = estimate + 0.0 if le(t, r) else current + estimate * 0.0
                output = write(value, 2, "close_replace" if le(t, r) else "close_keep")
            else:
                output = current
                trace[f"frame{frame}.stage2.far_keep"] += 1
        elif not le(r + 0.1, s):
            output = low_displaced()
    elif frame in (1, 2):
        if t + 0.1 < r:
            if s < r + 0.1:
                output = low_displaced()
        elif r + 0.1 < t:
            output = current
            trace[f"frame{frame}.stage2.far_keep"] += 1
        else:
            value = estimate + 0.0 if le(t, r) else current + estimate * 0.0
            output = write(value, 2, "close_replace" if le(t, r) else "close_keep")
    elif le(r, t + 0.1):
        if le(t, r + 0.1):
            value = estimate + 0.0 if le(t, r) else current + estimate * 0.0
            output = write(value, 2, "close_replace" if le(t, r) else "close_keep")
        else:
            output = current
            trace[f"frame{frame}.stage2.far_keep"] += 1
    elif s < r + 0.1:
        output = low_displaced()
    return output


def repair(
    frames: list[list[int]],
    width: int,
    height: int,
    selection: Selection,
    trace: Counter[str] | None = None,
) -> list[list[int]]:
    if len(frames) != 4 or any(len(p) != width * height for p in frames):
        raise ValueError("Expected four complete RAW planes.")
    if trace is None:
        trace = Counter()
    output = [plane.copy() for plane in frames]
    # (vertical sign, horizontal sign, diagonal frame, ratio frame, displaced frame)
    mappings = ((-1, -1, 2, 1, 3), (-1, 1, 3, 0, 2), (1, 1, 0, 3, 1), (1, -1, 1, 2, 0))
    for row, column in selection.coordinates(width, height):
        index = row * width + column
        for frame, (dy, dx, diag_frame, ratio_frame, far_frame) in enumerate(mappings):
            diagonal_index = index + dy * width + dx
            far_index = index + 2 * dy * width
            current = frames[frame][index]
            diagonal = frames[diag_frame][diagonal_index]
            # The adjacent green pair rotates with each of the four frames.
            horizontal_frame, vertical_frame = ((1, 3), (0, 2), (3, 1), (2, 0))[frame]
            mean = (
                frames[horizontal_frame][index + dx]
                + frames[vertical_frame][index + dy * width]
            ) / 2
            far_mean = (
                frames[diag_frame][far_index + dx] + frames[frame][index + dy * width]
            ) / 2
            r = divide(frames[ratio_frame][diagonal_index], diagonal)
            s = divide(far_mean, mean)
            t = divide(frames[far_frame][far_index], current)
            denominator = float32(
                divide(
                    output[ratio_frame][diagonal_index],
                    output[diag_frame][diagonal_index],
                )
            )
            estimate = divide(output[far_frame][far_index], denominator)
            guard = frames[3][index] if frame == 0 else current
            output[frame][index] = repair_value(
                frame,
                current,
                mean,
                diagonal,
                guard,
                r,
                s,
                t,
                estimate,
                trace,
            )
    return output
