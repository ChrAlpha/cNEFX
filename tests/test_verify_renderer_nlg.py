"""Reject incomplete evidence and malformed numeric Nlg row snapshots."""

import argparse
import copy
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import verify_renderer_nlg as verifier  # noqa: E402


def events():
    return [
        {
            "event": "nlg_row",
            "index": index,
            "kind": 0,
            "row": row,
            "rect": list(rect),
            "pitch": 1248,
            "advanced": True,
            "saved": True,
        }
        for index, (rect, row) in enumerate(
            ((rect, row) for rect in verifier.RECTANGLES for row in verifier.ROWS), 1
        )
    ]


def snapshots(width=208):
    event = events()[0 if width == 208 else 6]
    job = bytearray(0x3A0)
    struct.pack_into("<2Q", job, 0x338, width * 172, width * 172)
    struct.pack_into("<Q", job, 0x350, 0xDEADBEEF)
    struct.pack_into("<i4i", job, 0x358, event["row"], *event["rect"])
    struct.pack_into("<Q", job, 0x380, 0x12340000)
    struct.pack_into("<I", job, 0x390, 1248)
    rings = {}
    coarse = (width + 7) // 8
    for offset in verifier.RINGS:
        spatial = offset in (0x110, 0x198)
        columns = coarse if spatial else width
        pitch = ((columns + (2 if spatial else 0) + 31) // 32) * 32
        count = 3 if spatial or offset == 0 else 1
        struct.pack_into("<Q", job, offset + 0x28, count)
        struct.pack_into("<3I", job, offset + 0x60, columns, 1, pitch)
        struct.pack_into("<3i", job, offset + 0x78, 0, 2 if spatial else 0, 0)
        rings[offset] = np.arange(count * pitch, dtype="<u2").reshape(count, pitch)
    after = bytearray(job)
    struct.pack_into("<Q", after, 0x380, 0x12340000 + 1248)
    return event, job, after, rings


def test_row_events_accept_reordered_complete_set():
    assert verifier.row_events(events()[::-1]) == events()


@pytest.mark.parametrize("change", ["missing", "duplicate", "empty", "unstructured"])
def test_row_events_reject_incomplete_and_duplicate_captures(change):
    rows = events()
    changed = {
        "missing": rows[:-1],
        "duplicate": rows + rows[:1],
        "empty": [],
        "unstructured": [None],
    }[change]
    with pytest.raises(ValueError, match="Nlg"):
        verifier.row_events(changed)


@pytest.mark.parametrize(
    "field,value",
    [
        ("index", True),
        ("index", 0),
        ("index", 13),
        ("index", 2),
        ("kind", True),
        ("kind", 1),
        ("kind", 0.0),
        ("row", True),
        ("row", 2),
        ("rect", [False, 0, 208, 172]),
        ("rect", [0, 0, 207, 172]),
        ("pitch", True),
        ("pitch", 1248.0),
        ("saved", 1),
        ("advanced", False),
        ("extra", 0),
    ],
)
def test_row_events_reject_wrong_types_fields_and_geometry(field, value):
    rows = events()
    rows[0][field] = value
    with pytest.raises(ValueError, match="Nlg"):
        verifier.row_events(rows)


def test_original_lifecycle_filter_preserves_unknown_and_failing_events():
    core = [
        {"event": "render_started", "result": -1},
        {"event": "sdk_log", "code": 2},
        {"event": "nlg_unrecognized"},
    ]
    traces = [{"event": event} for event in verifier.INSTRUMENTATION]
    assert verifier.original_events(core + traces) == core


@pytest.mark.parametrize("width", [50, 208])
def test_job_geometry_and_allocated_rings(width):
    event, before, after, rings = snapshots(width)
    assert verifier.job_geometry(before, after, event) == (width, (width + 7) // 8)
    for offset, values in rings.items():
        decoded = verifier.ring_values(
            before, values.tobytes(), offset, width, (width + 7) // 8
        )
        retained = (width + 7) // 8 + 2 if offset in (0x110, 0x198) else width
        np.testing.assert_array_equal(decoded, values[:, :retained])


@pytest.mark.parametrize("offset", [0x110, 0x198])
def test_coarse_ring_reorders_by_both_numeric_indices(offset):
    _, before, _, rings = snapshots()
    struct.pack_into("<i", before, offset + 0x78, 2)
    struct.pack_into("<i", before, offset + 0x80, 2)
    actual = verifier.ring_values(before, rings[offset].tobytes(), offset, 208, 26)
    np.testing.assert_array_equal(actual, rings[offset][[1, 2, 0], :28])


@pytest.mark.parametrize("offset", [0, 0x350, 0x358, 0x35C, 0x388, 0x390])
def test_job_rejects_mutations_outside_output_pointer(offset):
    event, before, after, _ = snapshots()
    after[offset] ^= 1
    with pytest.raises(ValueError, match="stability"):
        verifier.job_geometry(before, after, event)


@pytest.mark.parametrize("offset", [0x338, 0x340, 0x358, 0x35C, 0x390])
def test_job_rejects_consistent_but_wrong_numeric_geometry(offset):
    event, before, after, _ = snapshots()
    before[offset] ^= 1
    after[offset] ^= 1
    with pytest.raises(ValueError, match="geometry"):
        verifier.job_geometry(before, after, event)


@pytest.mark.parametrize("advance", [0, 1247, 1249, -1248])
def test_job_rejects_incorrect_output_pointer_advance(advance):
    event, before, after, _ = snapshots()
    struct.pack_into("<Q", after, 0x380, 0x12340000 + advance)
    with pytest.raises(ValueError, match="advance"):
        verifier.job_geometry(before, after, event)


@pytest.mark.parametrize("before_size,after_size", [(927, 928), (928, 929)])
def test_job_requires_complete_snapshots(before_size, after_size):
    with pytest.raises(ValueError, match="complete"):
        verifier.job_geometry(bytes(before_size), bytes(after_size), events()[0])


@pytest.mark.parametrize("offset", verifier.RINGS)
@pytest.mark.parametrize(
    "field,value",
    [
        (0x28, 4),
        (0x60, 1),
        (0x64, 2),
        (0x68, 321),
        (0x78, -1),
        (0x7C, 7),
        (0x80, 3),
    ],
)
def test_ring_rejects_invalid_allocation_and_geometry(offset, field, value):
    _, before, _, rings = snapshots()
    struct.pack_into("<i", before, offset + field, value)
    with pytest.raises(ValueError, match="ring"):
        verifier.ring_values(before, rings[offset].tobytes(), offset, 208, 26)


@pytest.mark.parametrize("difference", [-2, 2])
def test_ring_rejects_truncated_or_extended_snapshot(difference):
    _, before, _, rings = snapshots()
    raw = rings[0x110].tobytes()
    raw = raw[:difference] if difference < 0 else raw + bytes(difference)
    with pytest.raises(ValueError, match="snapshot size"):
        verifier.ring_values(before, raw, 0x110, 208, 26)


def test_ring_ignores_captured_allocation_pointers():
    _, before, _, rings = snapshots()
    expected = verifier.ring_values(before, rings[0x110].tobytes(), 0x110, 208, 26)
    struct.pack_into("<Q", before, 0x110 + 0x18, 0xFFFFFFFFFFFFFFFF)
    actual = verifier.ring_values(before, rings[0x110].tobytes(), 0x110, 208, 26)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("mapping", ["input_hashes", "output_hashes"])
def test_verify_rejects_current_file_hash_drift(tmp_path, monkeypatch, mapping):
    source = tmp_path / "recorded.bin"
    source.write_bytes(b"changed")
    capture = {
        "passed": True,
        "sources_unchanged": True,
        "cases": [{"returncode": 0}],
        "input_hashes": {},
        "output_hashes": {},
    }
    capture[mapping][str(source)] = hashlib.sha256(b"original").hexdigest()
    path = tmp_path / "capture.json"
    raw = json.dumps(capture).encode()
    path.write_bytes(raw)
    monkeypatch.setattr(verifier, "CAPTURE_HASH", hashlib.sha256(raw).hexdigest())
    with pytest.raises(ValueError, match="hash drift"):
        verifier.verify(argparse.Namespace(capture=path), {"input_hashes": {}})


def arguments(tmp_path, monkeypatch, output):
    argv = ["verify_renderer_nlg.py", "--output", str(output)]
    for name in ("capture", "vendor", "wine-prefix"):
        argv.extend([f"--{name}", str(tmp_path / name)])
    monkeypatch.setattr(sys, "argv", argv)


@pytest.mark.parametrize("exception,code", [(KeyboardInterrupt, 130), (ValueError, 1)])
def test_failure_and_cancellation_preserve_partial_evidence(
    tmp_path, monkeypatch, capsys, exception, code
):
    output = tmp_path / "output"
    arguments(tmp_path, monkeypatch, output)
    partial = {"index": 1, "rgb": {"different_words": 3}}

    def fail(args, report):
        report["replays"].append(copy.deepcopy(partial))
        raise exception("Injected failure")

    monkeypatch.setattr(verifier, "verify", fail)
    assert verifier.main() == code
    report = json.loads((output / "report.json").read_text())
    assert report["passed"] is False
    assert report["replays"] == [partial]
    assert report.get("cancelled", False) == (code == 130)
    assert json.loads(capsys.readouterr().out)["passed"] is False


@pytest.mark.parametrize("location", ["existing", "repository", "dangling_link"])
def test_output_must_be_new_external_directory(tmp_path, monkeypatch, location):
    output = tmp_path / "output"
    if location == "existing":
        output.mkdir()
    elif location == "repository":
        output = Path(verifier.__file__).resolve().parent / "uncreated-nlg-report"
    else:
        output.symlink_to(tmp_path / "missing")
    arguments(tmp_path, monkeypatch, output)
    with pytest.raises(SystemExit) as exc:
        verifier.main()
    assert exc.value.code == 2


def grid_snapshots():
    descriptor = bytearray(0x80)
    struct.pack_into("<2I", descriptor, 0x60, 1032, 688)
    struct.pack_into("<I", descriptor, 0x7C, 1)
    plane = (np.arange(1032 * 688) % 4096).astype("<u2").reshape(688, 1032)
    return descriptor, plane


def grid_events():
    return [{"event": "nlg_grid", "index": i, "saved": True} for i in (1, 2)]


def test_grid_events_require_exact_complete_capture():
    verifier.grid_events(grid_events()[::-1])
    for changed in ([], grid_events()[:1], grid_events() * 2):
        with pytest.raises(ValueError, match="grid capture"):
            verifier.grid_events(changed)


@pytest.mark.parametrize(
    "field,value",
    [
        ("index", True),
        ("index", 1.0),
        ("index", 0),
        ("index", 3),
        ("saved", 1),
        ("saved", False),
        ("extra", 0),
    ],
)
def test_grid_events_reject_wrong_types_fields_and_failures(field, value):
    rows = grid_events()
    rows[0][field] = value
    with pytest.raises(ValueError, match="grid capture"):
        verifier.grid_events(rows)


def test_grid_events_remain_lifecycle_visible_in_original_capture():
    original = [{"event": "nlg_grid"}, {"event": "unknown"}, {"event": "closed"}]
    assert verifier.original_events(original) == original
    assert verifier.original_events(original, grids=True) == original[1:]


def test_grid_planes_use_logical_order_without_pointer_access():
    descriptor, plane = grid_snapshots()
    struct.pack_into("<Q", descriptor, 0x48, 0xFFFFFFFFFFFFFFFF)
    struct.pack_into("<I", descriptor, 0x70, 687)
    actual = verifier.grid_values(descriptor, plane.tobytes())
    np.testing.assert_array_equal(actual, plane)


@pytest.mark.parametrize(
    "offset,value",
    [
        (0x60, 0),
        (0x60, 1033),
        (0x64, 0),
        (0x64, 689),
        (0x70, 688),
        (0x70, 0xFFFFFFFF),
        (0x7C, 0),
    ],
)
def test_grid_rejects_invalid_geometry_or_missing_borrow(offset, value):
    descriptor, plane = grid_snapshots()
    struct.pack_into("<I", descriptor, offset, value)
    with pytest.raises(ValueError, match="descriptor"):
        verifier.grid_values(descriptor, plane.tobytes())


@pytest.mark.parametrize(
    "kind", ["short_descriptor", "long_descriptor", "short", "long"]
)
def test_grid_rejects_incomplete_or_extended_snapshots(kind):
    descriptor, plane = grid_snapshots()
    raw = plane.tobytes()
    if kind == "short_descriptor":
        descriptor = descriptor[:-1]
    elif kind == "long_descriptor":
        descriptor += b"\x00"
    elif kind == "short":
        raw = raw[:-2]
    else:
        raw += bytes(2)
    with pytest.raises(ValueError, match="size mismatch"):
        verifier.grid_values(descriptor, raw)


@pytest.mark.parametrize("value", [4096, 65535])
def test_grid_rejects_out_of_range_value_even_outside_sampled_coordinates(value):
    descriptor, plane = grid_snapshots()
    plane[1, 1] = value
    with pytest.raises(ValueError, match="bounded range"):
        verifier.grid_values(descriptor, plane.tobytes())


def test_grid_scale_requires_both_exact_scale32_values():
    state = bytearray(0x10F9C0)
    struct.pack_into("<2d", state, 0x10F950, 32.0, 32.0)
    verifier.grid_scale(state)
    with pytest.raises(ValueError, match="scale32"):
        verifier.grid_scale(state[:-1])
    for offset in (0x10F950, 0x10F958):
        for value in (0.0, 31.0, 32.00000001, float("nan"), float("inf")):
            changed = bytearray(state)
            struct.pack_into("<d", changed, offset, value)
            with pytest.raises(ValueError, match="scale32"):
                verifier.grid_scale(changed)


def tile_stages():
    return {
        index: {"stage": name, "rect": list(rect), "format": 0x10060201}
        for (source, target), rect in zip(verifier.TILE_PAIRS, verifier.RECTANGLES)
        for index, name in ((source, "lut"), (target, "nlg"))
    }


def test_tile_pairs_enforce_actual_upstream_source_and_both_tile_rectangles():
    stages = tile_stages()
    for (source, target), rect in zip(verifier.TILE_PAIRS, verifier.RECTANGLES):
        assert verifier.tile_geometry(stages, source, target) == rect
    with pytest.raises(ValueError, match="upstream tile pair"):
        verifier.tile_geometry(stages, 2543, 2547)


@pytest.mark.parametrize("index", [2545, 2546, 2547, 2548])
@pytest.mark.parametrize(
    "field,value",
    [
        ("stage", "nr2"),
        ("rect", [0, 1, 208, 172]),
        ("format", 0x10020001),
    ],
)
def test_tile_pairs_reject_stage_identity_or_geometry_mismatch(index, field, value):
    stages = tile_stages()
    stages[index][field] = value
    source, target = next(pair for pair in verifier.TILE_PAIRS if index in pair)
    with pytest.raises(ValueError, match="upstream stage"):
        verifier.tile_geometry(stages, source, target)


def test_unrecognized_capture_hash_is_rejected_before_artifact_access(tmp_path):
    path = tmp_path / "capture.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="capture hash drift"):
        verifier.verify(argparse.Namespace(capture=path), {"input_hashes": {}})
