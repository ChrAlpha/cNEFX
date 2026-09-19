"""Check bounded pyramid replay integration and reject incomplete evidence."""

import copy
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import verify_nlg_pyramid as verifier


def test_parameter_order_and_first_level():
    raw = bytes(0xAF) + bytes([1, 205, 125, 0, 0, 1, 0])
    assert verifier.level_parameters(raw, True) == (205, 0, 125)
    raw = bytes(0xAF) + bytes([1, 205, 35, 71, 1, 1, 0])
    assert verifier.level_parameters(raw, False) == (205, 71, 35)


@pytest.mark.parametrize("offset", [0xAF, 0xB3, 0xB4, 0xB5])
def test_rejects_unrecovered_branches(offset):
    raw = bytearray(bytes(0xAF) + bytes([1, 205, 35, 35, 1, 1, 0]))
    raw[offset] ^= 1
    with pytest.raises(ValueError):
        verifier.level_parameters(bytes(raw), False)


@pytest.mark.parametrize("size", [0, 181, 183])
def test_rejects_truncated_and_extended_parameters(size):
    with pytest.raises(ValueError):
        verifier.level_parameters(bytes(size), True)


def test_rgb_scale_parameters_require_all_three_enabled_bytes():
    raw = bytearray(0xB6)
    raw[0xAC:0xAF] = b"\x01\x01\x01"
    verifier.rgb_parameters(bytes(raw))
    for offset in (0xAC, 0xAD, 0xAE):
        for value in (0, 2, 255):
            changed = raw.copy()
            changed[offset] = value
            with pytest.raises(ValueError, match="scale-transition"):
                verifier.rgb_parameters(bytes(changed))
    for changed in (raw[:-1], raw + b"\x00"):
        with pytest.raises(ValueError, match="scale-transition"):
            verifier.rgb_parameters(bytes(changed))


def fixture(channels=3):
    event = dict(
        width=5, height=3, channels=channels, saved=True, bounded=True, valid=True
    )
    descriptor = bytearray(0x88)
    struct.pack_into("<Q", descriptor, 0x28, 3 * channels)
    struct.pack_into("<3I", descriptor, 0x60, 5, 3, 32)
    struct.pack_into("<I", descriptor, 0x78, 2)
    pixels = np.arange(3 * channels * 5, dtype="<u2").reshape(3, channels, 5)
    return event, descriptor, pixels


@pytest.mark.parametrize("channels", [1, 3])
def test_plane_layout_already_has_logical_row_order(channels):
    event, descriptor, pixels = fixture(channels)
    actual = verifier.plane_values(event, descriptor, pixels.tobytes())
    expected = pixels.transpose(0, 2, 1)
    np.testing.assert_array_equal(
        actual, expected[:, :, 0] if channels == 1 else expected
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("width", True),
        ("width", 0),
        ("width", 2049),
        ("height", 2049),
        ("channels", 2),
        ("saved", False),
        ("bounded", False),
        ("valid", False),
    ],
)
def test_rejects_invalid_plane_event(field, value):
    event, descriptor, pixels = fixture()
    event[field] = value
    with pytest.raises(ValueError):
        verifier.plane_values(event, descriptor, pixels.tobytes())


@pytest.mark.parametrize(
    "offset,value", [(0x28, 8), (0x60, 6), (0x64, 4), (0x68, 4), (0x78, 3)]
)
def test_rejects_invalid_descriptor(offset, value):
    event, descriptor, pixels = fixture()
    struct.pack_into("<I", descriptor, offset, value)
    with pytest.raises(ValueError):
        verifier.plane_values(event, descriptor, pixels.tobytes())


@pytest.mark.parametrize("change", [-2, 2])
def test_rejects_incomplete_plane_payload(change):
    event, descriptor, pixels = fixture()
    raw = pixels.tobytes()
    with pytest.raises(ValueError):
        verifier.plane_values(
            event, descriptor, raw[:change] if change < 0 else raw + bytes(change)
        )


@pytest.mark.parametrize(
    "events", [[], [{"event": "nlg_pyramid", "index": 1, "result": 0}]]
)
def test_rejects_missing_or_failed_pyramid_calls(events):
    with pytest.raises(ValueError, match="sequence"):
        verifier.pyramid_events(events)


def arguments(tmp_path, monkeypatch, output):
    argv = ["verify_nlg_pyramid.py", "--output", str(output)]
    for name in ("capture", "vendor", "wine-prefix"):
        argv += ["--" + name, str(tmp_path / name)]
    monkeypatch.setattr(sys, "argv", argv)


@pytest.mark.parametrize("error,code", [(ValueError, 1), (KeyboardInterrupt, 130)])
@pytest.mark.parametrize("group", ["levels", "downsample", "reconstruction"])
def test_partial_evidence_survives_failure(
    tmp_path, monkeypatch, capsys, error, code, group
):
    output = tmp_path / "output"
    arguments(tmp_path, monkeypatch, output)
    partial = dict(index=7, slot=14, different_words=1)

    def fail(args, report):
        report[group].append(copy.deepcopy(partial))
        raise error("Injected failure")

    monkeypatch.setattr(verifier, "verify", fail)
    assert verifier.main() == code
    result = json.loads((output / "report.json").read_text())
    assert result["passed"] is False
    assert result[group] == [partial]
    assert json.loads(capsys.readouterr().out)["passed"] is False


@pytest.mark.parametrize("location", ["existing", "repository", "link"])
def test_no_overwrite_or_repository_output(tmp_path, monkeypatch, location):
    output = tmp_path / "output"
    if location == "existing":
        output.mkdir()
    elif location == "repository":
        output = Path(verifier.__file__).parent / "uncreated-pyramid-output"
    else:
        output.symlink_to(tmp_path / "missing")
    arguments(tmp_path, monkeypatch, output)
    with pytest.raises(SystemExit) as exc:
        verifier.main()
    assert exc.value.code == 2


SMALL_TOTALS = {
    "downsample": (10, 8184),
    "reconstruction": (10, 32736),
    "inputs": (20, 5456),
    "levels": (24, 21840),
    "shared_grids": (4, 16384),
}


def small_capture():
    planes, parameters = {}, {}
    for offset in (0, 12):
        for level, width in enumerate((64, 32, 16, 8, 4), start=2):
            index = offset + level
            planes[index, "output", 13] = np.full(
                (width, width, 3), 100 + index, dtype="<u2"
            )
            planes[index, "output", 12] = np.full(
                (width // 2, width // 2, 3), 100 + index, dtype="<u2"
            )
        for level, width in enumerate((2, 4, 8, 16, 32, 64), start=7):
            index = offset + level
            planes[index, "output", 13] = np.full(
                (width, width, 3), 400 + index, dtype="<u2"
            )
            for slot in (14, 15):
                planes[index, "output", slot] = np.zeros((width, width), dtype="<u2")
            if level > 7:
                for slot in (9, 10):
                    planes[index, "input", slot] = np.zeros(
                        (width // 2, width // 2), dtype="<u2"
                    )
            if level < 12:
                planes[index, "input", 6] = np.full(
                    (width * 2, width * 2, 3), 300 + index, dtype="<u2"
                )
                planes[index, "input", 7] = planes[index, "output", 13]
                planes[index, "output", 11] = np.full(
                    (width * 2, width * 2, 3), 400 + index, dtype="<u2"
                )
        for level in range(2, 13):
            raw = bytearray(0xB6)
            raw[0xAC:0xB0] = b"\x01\x01\x01\x01"
            raw[0xB3] = int(level != 7)
            raw[0xB4] = 1
            parameters[offset + level] = bytes(raw)
    grids = [np.zeros((64, 64), dtype="<u2") for _ in range(2)]
    return planes, parameters, grids


def replay_small_capture(monkeypatch):
    planes, parameters, grids = small_capture()
    report = {key: [] for key in SMALL_TOTALS}
    report.update(sources_unchanged=True, capture_providers=list(range(28)))
    monkeypatch.setattr(verifier, "EXPECTED_TOTALS", SMALL_TOTALS)
    verifier.replay_pyramids(
        lambda index, phase, slot: planes[index, phase, slot],
        parameters.__getitem__,
        grids,
        report,
    )
    return report


def test_replay_integrates_both_rgb_sequences_and_recursive_spatial_checks(monkeypatch):
    assert verifier.EXPECTED_TOTALS == {
        "downsample": (10, 1418970),
        "reconstruction": (10, 5674710),
        "inputs": (20, 945980),
        "levels": (24, 3786044),
        "shared_grids": (4, 2840064),
    }
    report = replay_small_capture(monkeypatch)
    verifier.summarize(report)
    assert report["passed"] is True
    for group, (count, words) in SMALL_TOTALS.items():
        assert len(report[group]) == count
        assert report[group + "_words"] == words
    assert [r["index"] for r in report["downsample"]] == [
        2,
        3,
        4,
        5,
        6,
        14,
        15,
        16,
        17,
        18,
    ]
    assert [r["index"] for r in report["reconstruction"]] == [
        7,
        8,
        9,
        10,
        11,
        19,
        20,
        21,
        22,
        23,
    ]


@pytest.mark.parametrize("group", tuple(SMALL_TOTALS))
@pytest.mark.parametrize("change", ["omitted", "duplicate", "word_count", "pixels"])
def test_replay_rejects_incomplete_counts_and_mismatches(monkeypatch, group, change):
    report = replay_small_capture(monkeypatch)
    if change == "omitted":
        report[group].pop()
    elif change == "duplicate":
        report[group].append(copy.deepcopy(report[group][0]))
    elif change == "word_count":
        report[group][0]["compared_words"] -= 1
    else:
        report[group][0]["different_words"] = 1
    verifier.summarize(report)
    assert report["passed"] is False


@pytest.mark.parametrize("change", ["input_drift", "providers"])
def test_replay_preserves_input_and_provider_checks(monkeypatch, change):
    report = replay_small_capture(monkeypatch)
    if change == "input_drift":
        report["sources_unchanged"] = False
    else:
        report["capture_providers"].pop()
    verifier.summarize(report)
    assert report["passed"] is False


@pytest.mark.parametrize("index", [2, 6, 7, 11, 14, 18, 19, 23])
def test_rgb_parameter_rejection_keeps_prior_comparisons(index):
    planes, parameters, grids = small_capture()
    raw = bytearray(parameters[index])
    raw[0xAD] = 0
    parameters[index] = bytes(raw)
    report = {key: [] for key in SMALL_TOTALS}
    with pytest.raises(ValueError, match="scale-transition"):
        verifier.replay_pyramids(
            lambda i, phase, slot: planes[i, phase, slot],
            parameters.__getitem__,
            grids,
            report,
        )
    assert not any(
        e["index"] == index
        for key in ("downsample", "reconstruction")
        for e in report[key]
    )
    if index > 2:
        assert report["downsample"][0]["index"] == 2
