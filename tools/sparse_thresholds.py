"""Controlled neighborhoods straddling recovered sparse-repair thresholds."""

import ast
from fractions import Fraction
from pathlib import Path
from types import FunctionType

import sparse_reference
from sparse_reference import Selection


def cases() -> list[tuple[str, Selection, list[list[int]]]]:
    result = []
    mappings = (
        (-1, -1, 2, 1, 3, 1, 3),
        (-1, 1, 3, 0, 2, 0, 2),
        (1, 1, 0, 3, 1, 3, 1),
        (1, -1, 1, 2, 0, 2, 0),
    )
    for frame, (dy, dx, diagonal, ratio, far, horizontal, vertical) in enumerate(
        mappings
    ):
        for boundary in (
            "s_at_r_plus_0.1",
            "t_plus_0.1_at_r",
            "t_plus_0.1_at_s",
            "t_at_r_plus_0.1",
            "s_at_t_plus_0.1",
            "t_at_s_plus_0.1",
            "t_at_r_plus_0.3",
            "color_ratio_0.4",
            "mean_2000",
            "estimate_delta_500",
            "diagonal_delta_300",
            "guard_delta_2000",
            "diagonal_ratio_1.1",
            "upper_cap",
        ):
            for delta in (-1, 0, 1):
                current = mean = diag = 1000
                guard = 1000
                r, s, t = Fraction(1), Fraction(1), Fraction(1)
                step = Fraction(delta, 1000)
                if boundary == "s_at_r_plus_0.1":
                    s, t = Fraction(11, 10) + step, Fraction(1, 2)
                elif boundary == "t_plus_0.1_at_r":
                    # Equality at s=r+0.1 prevents stage two from replacing the
                    # stage-one blend when t+0.1 is below r.
                    s, t = Fraction(11, 10), Fraction(9, 10) + step
                elif boundary == "t_plus_0.1_at_s":
                    s, t = Fraction(4, 5), Fraction(7, 10) + step
                elif boundary == "t_at_r_plus_0.1":
                    s, t = Fraction(3, 2), Fraction(11, 10) + step
                elif boundary == "s_at_t_plus_0.1":
                    r, s = Fraction(6, 5), Fraction(11, 10) + step
                elif boundary == "t_at_s_plus_0.1":
                    t = Fraction(11, 10) + step
                elif boundary == "t_at_r_plus_0.3":
                    diag, mean = 1500, 2500
                    t = Fraction(13, 10) + step
                elif boundary == "color_ratio_0.4":
                    diag, mean, t = 2000 + delta, 2500, Fraction(7, 5)
                elif boundary == "mean_2000":
                    mean, t = 2000 + delta, Fraction(7, 5)
                elif boundary == "estimate_delta_500":
                    mean, t = 2001, Fraction(3, 2) + step
                elif boundary == "diagonal_delta_300":
                    diag, s, t = 800 + delta, Fraction(3, 4), Fraction(1, 2)
                elif boundary == "guard_delta_2000":
                    diag, guard, s, t = (
                        500,
                        2500 + delta,
                        Fraction(3, 4),
                        Fraction(1, 2),
                    )
                    if frame != 0:
                        current, t = guard, Fraction(500, guard)
                elif boundary == "diagonal_ratio_1.1":
                    r, s, t = Fraction(11, 10) + step, Fraction(3, 4), Fraction(1, 2)
                else:
                    current, mean, diag = 14000, 14000, 16000
                    r, s, t = Fraction(4, 5), Fraction(9, 10), Fraction(11, 10)
                    t += Fraction(delta, current)
                planes = [[1024] * (16 * 12) for _ in range(4)]
                y, x = 4, 5

                def put(
                    plane: int, row: int, column: int, value: int | Fraction
                ) -> None:
                    if value != int(value) or not 0 <= value <= 16383:
                        raise ValueError("Fixture is not a valid 14-bit integer.")
                    planes[plane][row * 16 + column] = int(value)

                put(frame, y, x, current)
                put(diagonal, y + dy, x + dx, diag)
                put(ratio, y + dy, x + dx, r * diag)
                put(horizontal, y, x + dx, mean)
                put(vertical, y + dy, x, mean)
                put(diagonal, y + 2 * dy, x + dx, s * mean)
                put(frame, y + dy, x, s * mean)
                put(far, y + 2 * dy, x, t * current)
                if frame == 0:
                    put(3, y, x, guard)
                result.append(
                    (
                        f"threshold-f{frame}-{boundary}-{delta:+d}",
                        Selection(12, 16),
                        planes,
                    )
                )
    return result


def mutation_checks(
    fixtures: list[tuple[str, Selection, list[list[int]]]],
    observed: list[tuple[int, ...]],
) -> dict:
    """Compare in-memory reference mutants with the unchanged vendor outputs."""
    source_path = Path(sparse_reference.__file__)
    source = source_path.read_text()
    mutations = {}
    for name, operator in (("reversed_gt", ast.Gt), ("inclusive_le", ast.LtE)):
        tree = ast.parse(source)
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "repair_value"
        )
        pattern = ast.parse("t + 0.1 < r", mode="eval").body
        matches = sorted(
            (
                node
                for node in ast.walk(function)
                if isinstance(node, ast.Compare) and ast.dump(node) == ast.dump(pattern)
            ),
            key=lambda node: node.lineno,
        )
        if len(matches) != 2:
            raise ValueError("Expected the two recovered t+0.1<r comparisons.")
        matches[0].ops[0] = operator()
        namespace = vars(sparse_reference).copy()
        exec(
            compile(ast.Module(body=[function], type_ignores=[]), source_path, "exec"),
            namespace,
        )
        mutated = FunctionType(sparse_reference.repair.__code__, namespace)
        results = []
        different_cases = different_samples = 0
        for (case_name, selection, planes), actual in zip(
            fixtures, observed, strict=True
        ):
            expected = mutated(planes, 16, 12, selection, None)
            differences = sum(
                value != expected[frame][pixel]
                for frame in range(4)
                for pixel, value in enumerate(actual[frame * 192 : (frame + 1) * 192])
            )
            different_cases += differences > 0
            different_samples += differences
            if "-t_plus_0.1_at_r-" in case_name:
                frame = int(case_name.split("-")[1][1:])
                pixel = 4 * 16 + 5
                results.append(
                    {
                        "name": case_name,
                        "vendor_target": actual[frame * 192 + pixel],
                        "mutated_target": expected[frame][pixel],
                        "different_samples": differences,
                    }
                )
        mutations[name] = {
            "different_cases": different_cases,
            "different_samples": different_samples,
            "targeted_cases": results,
        }
    targeted = mutations["reversed_gt"]["targeted_cases"]
    below = [case for case in targeted if case["name"].endswith("--1")]
    detected = len(below) == 4 and all(
        case["vendor_target"] != case["mutated_target"] for case in below
    )
    return {
        "comparison": "Stage-one t+0.1<r; the later comparison is unchanged.",
        "method": "Mutate a parsed copy of the reference in memory only.",
        "reversed_predicate_detected_in_each_frame": detected,
        "mutations": mutations,
        "limitation": (
            "An inclusive stage-one comparison is masked at equality by the later "
            "close-replace branch. Distinct triplet outputs alone do not prove "
            "strict-versus-inclusive branch coverage."
        ),
    }
