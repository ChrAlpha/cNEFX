"""Exact Nlg lookup-table generation and bounded numeric controls."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from nlg_tables import tables


def test_vendor_complete_table_vectors():
    corpus = json.loads(
        (Path(__file__).parent / "fixtures/nlg_table_vectors.json").read_text()
    )
    for case in corpus["cases"]:
        result = tables(
            float(case["scale"]), case["base"], case["offset"], bool(case["mode"])
        )
        for name, table in zip(("forward", "inverse"), result, strict=True):
            assert table.shape == (65536,)
            assert table.dtype == np.dtype("<u2")
            assert hashlib.sha256(table.tobytes()).hexdigest() == case[name + "_sha256"]


@pytest.mark.parametrize("half_range", [False, True])
def test_exact_square_points_and_saturation(half_range):
    forward, inverse = tables(1.0, 0, 0, half_range)
    scale = 2 if half_range else 1
    for n in range(100):
        assert forward[scale * n * n] == 128 * n
        assert inverse[128 * n] == scale * n * n
    assert forward[0] == inverse[0] == 0
    assert inverse[-1] == 65535


def test_source_candidate_tables_match_passing_generator_snapshots():
    corpus = json.loads(
        (Path(__file__).parent / "fixtures/nlg_table_vectors.json").read_text()
    )
    reference = corpus["source_candidate"]
    generated = tables(1.0, 0, 191, True)
    assert [hashlib.sha256(v.tobytes()).hexdigest() for v in generated] == reference[
        "sha256"
    ]


def test_scale_is_narrowed_before_forward_and_reciprocal_arithmetic():
    first = tables(1.0, 5, 3, True)
    second = tables(1.0 + 2**-25, 5, 3, True)
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)


def test_result_arrays_are_independent_and_not_shared_between_calls():
    first = tables(1.0, 0, 0, False)
    second = tables(1.0, 0, 0, False)
    assert not np.shares_memory(*first)
    first[0][0] = 17
    assert second[0][0] == 0


@pytest.mark.parametrize(
    "scale", [0.0, -1.0, 1 / 512, 257.0, float("inf"), float("nan"), True, 1, None]
)
def test_rejects_unsupported_scale(scale):
    with pytest.raises(ValueError):
        tables(scale, 0, 0, False)


@pytest.mark.parametrize(
    "base,offset,mode",
    [
        (-1, 0, False),
        (65536, 0, False),
        (True, 0, False),
        (0, -1, False),
        (0, 65536, False),
        (0, 1.5, False),
        (0, 0, 1),
        (0, 0, None),
    ],
)
def test_rejects_unsupported_stored_controls(base, offset, mode):
    with pytest.raises(ValueError):
        tables(1.0, base, offset, mode)
