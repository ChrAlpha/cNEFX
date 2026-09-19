"""Adversarial gates for saved complete CMS0 composition evidence."""

import copy
import json
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import verify_renderer_cms as verifier  # noqa: E402


def traces():
    return [
        {"event": "stage_taps", "installed": True},
        {
            "event": "stage_pixels",
            "index": 477,
            "stage": "nlg",
            "object": 1234,
            "format": 0x10060201,
            "rect": [1040, 0, 1248, 208],
            "stride": 1248,
            "saved": True,
            "error": False,
        },
        {
            "event": "inner_pixels",
            "index": 691,
            "object": 5678,
            "format": 0x10060204,
            "width": 208,
            "height": 208,
            "kernel_rva": 0x53A2F0,
            "saved": [1, 1],
            "error": False,
        },
        {
            "event": "inner_pixels",
            "index": 1804,
            "object": 6789,
            "format": 0x10060201,
            "width": 208,
            "height": 208,
            "kernel_rva": 0x53B640,
            "saved": [1, 1],
            "error": False,
        },
    ]


def test_stage_identity_does_not_depend_on_trace_order():
    events = traces()
    assert verifier.stage_events(events) == verifier.stage_events(events[::-1])


@pytest.mark.parametrize(
    "field,value",
    [
        ("index", True),
        ("index", 691),
        ("index", -1),
        ("object", 0),
        ("format", 1),
        ("stride", True),
        ("stride", 1247),
        ("stride", 2**27),
        ("stage", "unknown"),
        ("rect", [0, 0, 1, 208]),
        ("rect", [0, 0, 208, 0]),
        ("rect", [-1, 0, 208, 208]),
        ("rect", [0, 0, 4097, 208]),
        ("rect", [False, 0, 208, 208]),
        ("rect", [0, 0, 4096, 4096]),
        ("rect", [0, 0]),
        ("saved", False),
        ("saved", 1),
        ("error", True),
        ("error", 0),
    ],
)
def test_reject_unsupported_stage(field, value):
    events = traces()
    events[1][field] = value
    with pytest.raises(ValueError):
        verifier.stage_events(events)


@pytest.mark.parametrize("position", [2, 3])
@pytest.mark.parametrize(
    "field,value",
    [
        ("error", True),
        ("saved", [1, 0]),
        ("width", True),
        ("height", 1),
        ("kernel_rva", 0),
        ("object", 0),
    ],
)
def test_reject_failed_excluded_inner_calls(position, field, value):
    events = traces()
    events[position][field] = value
    with pytest.raises(ValueError):
        verifier.stage_events(events)


@pytest.mark.parametrize("change", ["missing", "duplicate", "boolean", "empty"])
def test_reject_incomplete_trace_installation(change):
    events = traces()
    if change == "missing":
        events = events[1:]
    elif change == "duplicate":
        events.append(copy.deepcopy(events[0]))
    elif change == "boolean":
        events[0]["installed"] = 1
    else:
        events = []
    with pytest.raises(ValueError):
        verifier.stage_events(events)


def states():
    cms, lut, matrix = bytearray(0x748), bytearray(0x8E0), bytearray(0x890)
    struct.pack_into("<I", cms, 0x718, 6)
    struct.pack_into("<Q", cms, 0x720, 5678)
    struct.pack_into("<Q", cms, 0x730, 6789)
    struct.pack_into("<Q", lut, 0, 0x100B2A478)
    struct.pack_into("<Q", matrix, 0, 0x100B2A868)
    struct.pack_into("<I", lut, 0x22C, 16)
    struct.pack_into("<I", lut, 0x578, 0x60204)
    lut[0x3D0:0x3D3] = b"AAA"
    struct.pack_into("<3I", lut, 0x3D4, *((0x41800080,) * 3))
    struct.pack_into("<3Q", lut, 0x3E0, 12675, 195, 3)
    for offset in (0x230, 0x2B0, 0x330, 0x3F8, 0x478, 0x4F8):
        struct.pack_into("<I", lut, offset, 6)
    matrix[0x230] = 1
    for base, kind in ((0x3F0, 5), (0x558, 6)):
        for channel in range(3):
            struct.pack_into("<I", matrix, base + 0x78 * channel, kind)
    return [cms, lut, matrix]


def test_cms_padding_and_relocation_are_not_compared_as_parameters():
    cms, lut, matrix = states()
    verifier.cms_contract(cms, lut, matrix, 5678)
    cms[0x71C:0x720] = b"junk"
    lut[0x3D3] = 255
    struct.pack_into("<Q", lut, 0, 0x200B2A478)
    struct.pack_into("<Q", matrix, 0, 0x200B2A868)
    verifier.cms_contract(cms, lut, matrix, 5678)


@pytest.mark.parametrize(
    "position,offset",
    [
        (0, 0x10),
        (0, 0x11),
        (0, 0x12),
        (0, 0x600),
        (0, 0x718),
        (0, 0x720),
        (1, 0),
        (1, 0x22C),
        (1, 0x578),
        (1, 0x3D0),
        (1, 0x3D4),
        (1, 0x3E0),
        (1, 0x230),
        (1, 0x4F8),
        (2, 0),
        (2, 0x230),
        (2, 0x888),
        (2, 0x3F0),
        (2, 0x558),
    ],
)
def test_reject_changed_cms_contract(position, offset):
    data = states()
    data[position][offset] ^= 1
    with pytest.raises(ValueError):
        verifier.cms_contract(*data, 5678)


@pytest.mark.parametrize("position", [0, 1, 2])
def test_reject_truncated_cms_states(position):
    data = states()
    data[position] = data[position][:-1]
    with pytest.raises(ValueError):
        verifier.cms_contract(*data, 5678)


def source_report():
    return {
        "evidence_kind": "native_source_controls_vs_live_sdk_top",
        **{
            name: True
            for name in (
                "passed",
                "sources_unchanged",
                "native_decryption",
                "picture_import",
                "originals_only",
                "kaiji_tables",
                "prepared_grid",
                "renderer_profile",
            )
        },
        "cases": [
            {
                "index": 0,
                "returncode": 0,
                "passed": True,
                "variant": "original",
                "source_sha256": "1" * 64,
                "np3_sha256": "2" * 64,
            }
        ],
    }


@pytest.mark.parametrize(
    "field",
    [
        "passed",
        "sources_unchanged",
        "native_decryption",
        "picture_import",
        "originals_only",
        "kaiji_tables",
        "prepared_grid",
        "renderer_profile",
    ],
)
def test_reject_unverified_source_report(field):
    report = source_report()
    captured = copy.deepcopy(report["cases"][0])
    assert verifier.source_case(report, captured) == captured
    report[field] = 1
    with pytest.raises(ValueError):
        verifier.source_case(report, captured)


@pytest.mark.parametrize("field", ["source_sha256", "np3_sha256"])
def test_reject_cross_run_source_control_mismatch(field):
    report = source_report()
    captured = copy.deepcopy(report["cases"][0])
    captured[field] = "3" * 64
    with pytest.raises(ValueError, match="differs between runs"):
        verifier.source_case(report, captured)


def test_source_report_excluded_cases_must_succeed():
    report = source_report()
    report["cases"].append({**report["cases"][0], "index": 1, "returncode": 2})
    with pytest.raises(ValueError, match="Failed"):
        verifier.source_case(report, report["cases"][0])


@pytest.mark.parametrize("value", [None, 0, "unknown", "A" * 64])
def test_matching_invalid_source_hashes_are_rejected(value):
    report = source_report()
    report["cases"][0]["source_sha256"] = value
    with pytest.raises(ValueError, match="Invalid source photograph"):
        verifier.source_case(report, copy.deepcopy(report["cases"][0]))


def arguments(tmp_path, monkeypatch, output):
    names = (
        "capture",
        "source-report",
        "coefficients",
        "modes",
        "blend-calibration",
        "decoding-table",
        "template",
        "vendor",
        "wine-prefix",
    )
    argv = ["verify_renderer_cms.py", "--output", str(output)]
    for name in names:
        argv.extend([f"--{name}", str(tmp_path / name)])
    monkeypatch.setattr(sys, "argv", argv)


@pytest.mark.parametrize("exception,code", [(KeyboardInterrupt, 130), (ValueError, 1)])
def test_partial_report_survives_cancellation_or_failure(
    tmp_path, monkeypatch, capsys, exception, code
):
    output = tmp_path / "output"
    arguments(tmp_path, monkeypatch, output)

    def fail(args, report):
        report["input_hashes"]["partial"] = "0" * 64
        report["replays"].append({"nlg": 477})
        raise exception("Injected failure")

    monkeypatch.setattr(verifier, "verify", fail)
    assert verifier.main() == code
    report = json.loads((output / "report.json").read_bytes())
    assert report["passed"] is False
    assert report["input_hashes"] == {"partial": "0" * 64}
    assert report["replays"] == [{"nlg": 477}]
    assert report.get("cancelled", False) == (code == 130)
    assert json.loads(capsys.readouterr().out)["passed"] is False


@pytest.mark.parametrize("location", ["existing", "repository", "dangling_link"])
def test_reject_unsafe_output(tmp_path, monkeypatch, location):
    output = tmp_path / "output"
    if location == "existing":
        output.mkdir()
    elif location == "repository":
        output = Path(verifier.__file__).resolve().parent / "uncreated-cms-report"
    else:
        output.symlink_to(tmp_path / "missing")
    arguments(tmp_path, monkeypatch, output)
    with pytest.raises(SystemExit) as error:
        verifier.main()
    assert error.value.code == 2


def test_hash_drift_is_reported_before_native_generation(tmp_path, monkeypatch):
    output = tmp_path / "output"
    arguments(tmp_path, monkeypatch, output)
    artifact = tmp_path / "input.bin"
    artifact.write_bytes(b"modified")
    (tmp_path / "capture").write_text(
        json.dumps({"input_hashes": {str(artifact): "0" * 64}})
    )
    (tmp_path / "source-report").write_text("{}")
    assert verifier.main() == 1
    report = json.loads((output / "report.json").read_bytes())
    assert report["error"]["message"] == "Recorded evidence hash drift."
    assert report["generation"] == {}
