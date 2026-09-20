# NX Ctudio

Agent-oriented repository map for native Linux Nikon Z8 pixel-shift synthesis.
The package is `nx-ctudio`, the import namespace is `nx_ctudio`, and the CLI is
`nxc`. Read [AGENTS.md](AGENTS.md) before editing.

## Scope and acceptance state

The active product path merges complete ordered Z8 sequences into NEFX. It is
experimental: authentic capture/reference validation and Lightroom acceptance
remain open. Passing fixtures or decoding a generated NEFX does not close them.
The recorded scope is [research/project-scope.json](research/project-scope.json).

| Implemented path | Boundary |
| --- | --- |
| Native merge | Complete 4, 8, 16, or 32-frame sequences; nine output modes below. No Wine, Nikon DLL, or ExifTool dependency in this path. |
| Source decoding | Observed lossless-compressed 14-bit Z8 NEF layout, strict metadata and geometry checks, rawpy/LibRaw decoding. HE/HE* is unsupported. |
| NEFX writing | Packed14 RGB, selected Nikon metadata/CRC updates, private RGB preview, retained source JPEGs, exact decoded-sample verification. |
| Lateral chromatic correction | Optional source-metadata radial correction before crop; disabled by default; requires supported matching parameters. |
| Inspection | ExifTool metadata and SHA-256 only; no RAW decoding or sequence validation. |
| Historical code | Linear DNG is secondary; Picture Control, Flexible Color, rendering, and `picture-curve` are outside active development. |

HLG, non-FX image areas, other firmware, and broader correction parameters remain
unvalidated. Source Adobe XMP defaults are preserved without vendor-equivalent
updates; default rendered appearance is a separate limitation from merged samples.
This repository is not a CLI clone of NX Studio.

## Environment

Run commands from the repository root on Linux with Python 3.12+ and `uv`.
[pyproject.toml](pyproject.toml) defines the package and tools;
[uv.lock](uv.lock) pins dependency resolution.

```sh
uv sync --locked
uv run --locked nxc --version
uv run --locked nxc merge --help
```

Runtime dependencies are NumPy, rawpy, and tifffile. The default development group
adds pytest and Ruff. ExifTool must be on `PATH` for `nxc inspect` and
`tests/test_inspect.py`; Debian/Ubuntu provide it as `libimage-exiftool-perl`.
Output-directory publication requires Linux libc `renameat2` and filesystem
support for `RENAME_NOREPLACE`.

RAW datasets and vendor assets are not bundled. Historical reports commonly point
to `/workspaces/LumaForge/nxc-data/`, a workstation-local external artifact tree.
It is not a runtime dependency or an assumed path on a fresh checkout. Resolve
required artifacts from the relevant research record and verify their hashes.

## Source routing

The merge path is:

```text
cli.main -> merge / merge_outputs
  -> raw.read_frame (metadata, identity, decode, source hash)
  -> pixel_shift.output_groups + reconstruct_from_reader
       -> _sparse repair and optional radial correction
  -> nefx.write_nefx -> _tiff streaming serialization
  -> rawpy exact decoded RGB comparison
  -> complete-sequence validation + final source hash checks
  -> _atomic no-replace publication
```

This is the responsibility flow; decoding, validation, and writing interleave
across output groups in `merge.py`.

| Read or change | Source | Focused tests |
| --- | --- | --- |
| CLI/report orchestration and sequence identity | [cli.py](src/nx_ctudio/cli.py), [merge.py](src/nx_ctudio/merge.py) | `tests/test_merge.py`, `tests/test_merge_outputs.py` |
| Nikon source metadata and decoding | [raw.py](src/nx_ctudio/raw.py) | `tests/test_raw_metadata.py`, `tests/test_merge.py` |
| Output grouping, phase mapping, integer reconstruction | [pixel_shift.py](src/nx_ctudio/pixel_shift.py), [_sparse.py](src/nx_ctudio/_sparse.py) | `tests/test_pixel_shift.py` |
| Source-metadata radial correction | [radial.py](src/nx_ctudio/radial.py) | `tests/test_radial.py` |
| NEFX tags, packing, preview, TIFF layout | [nefx.py](src/nx_ctudio/nefx.py), [_tiff.py](src/nx_ctudio/_tiff.py) | `tests/test_nefx.py`, `tests/test_tiff_stream.py` |
| Atomic publication, cancellation, warnings | [_atomic.py](src/nx_ctudio/_atomic.py) | `tests/test_atomic.py`, `tests/test_merge_outputs.py` |
| Read-only ExifTool inspection | [inspect.py](src/nx_ctudio/inspect.py) | `tests/test_inspect.py` |

`tools/` contains standalone probes and replay utilities, not product modules.
`research/` contains structured findings and external artifact references.
Many files in those directories and `tests/` concern retained historical work;
their presence does not expand the current scope.

## CLI contract

Always pass explicit files in capture order. A directory or a shell glob is not
an ordered sequence contract. The merger requires consecutive capture numbers,
consistent observed metadata, and distinct input files; it verifies source hashes
again before publication. Inputs must remain unchanged throughout the run.

Replace the absolute example paths with real captures. The output parent must
already exist. The destination file or directory must not exist, including an
empty directory or dangling symlink.

Four frames to one NEFX:

```sh
uv run --locked nxc merge \
  /data/capture4/shot-01.NEF /data/capture4/shot-02.NEF \
  /data/capture4/shot-03.NEF /data/capture4/shot-04.NEF \
  --output /data/results/merged.nefx --json
```

Eight frames to a complete two-file output set:

```sh
uv run --locked nxc merge \
  /data/capture8/shot-01.NEF /data/capture8/shot-02.NEF \
  /data/capture8/shot-03.NEF /data/capture8/shot-04.NEF \
  /data/capture8/shot-05.NEF /data/capture8/shot-06.NEF \
  /data/capture8/shot-07.NEF /data/capture8/shot-08.NEF \
  --outputs 2 --output-directory /data/results/merge-set --json
```

Add `--chromatic-aberration-correction` to enable supported source-metadata
correction. `--output` requires a `.nefx` suffix and one output. Output counts
above one require `--output-directory`; a directory can also contain one output.
`--outputs` defaults to `1`.

| Capture count | Allowed `--outputs` | Per-output crop dimensions |
| --- | --- | --- |
| 4 | 1 | 1x width and height |
| 8 | 1, 2 | 1x width and height |
| 16 | 1, 4 | 2x for one output; 1x for four |
| 32 | 1, 2, 4, 8 | 2x for one or two outputs; 1x for four or eight |

Groups are consecutive except 32-to-4: it pairs four-frame blocks at capture
offsets `0/16`, `4/20`, `8/24`, and `12/28`. Each output inherits metadata from
its group's first source. See `pixel_shift.output_groups` for exact index tuples.

With `--json`, merge writes its report to stdout:

- Success: `status`, `backend`, `output_format`, `sequence_validation`, ordered
  `inputs` with hashes and shot numbers, correction setting, `limitations`, and
  `warnings`.
- Single output: `output`, `sha256`, `shape` in height/width/channels order, and
  `samples_verified`.
- Output set: `output_directory`, `output_count`, `outputs`, total
  `samples_verified`, and zero-based `input_indices` for each output. The directory
  contains `merged-01.nefx` through `merged-NN.nefx` and `manifest.json`.
- Runtime failure: `status: "error"`, `backend: "native"`, and `error` with `code`
  and `message`. Codes are `merge_failed` or `cancelled`.

Merge exits `0` on success, `1` on runtime failure, `130` on caught cancellation,
and `2` on argument errors. Argument errors use argparse stderr, even with
`--json`. Merge reports currently have no `schema_version` field. Check the exit
code, status, and warnings; parse JSON only when a report was emitted.

Published output can survive a parent-directory sync or temporary-cleanup warning.
These are success reports with warnings, not grounds to delete or overwrite the
destination. The directory manifest is written before final publication and cannot
include warnings raised afterward; the final stdout report includes them.

Metadata inspection is a separate command:

```sh
uv run --locked nxc inspect /data/capture4/shot-01.NEF --json
```

Its report has `schema_version: 1`, a `files` array with per-file status, and
`raw_decode` / `sequence_validation` set to `not_performed`. Exit codes are `0`
for all files inspected, `1` for file failures, `2` for dependency/argument errors,
and `130` for caught cancellation. Inspection success is not merge eligibility.

## Verification routing

After environment setup, this focused suite covers the active synthesis path
without capture datasets or vendor binaries; the inspection tests need ExifTool:

```sh
uv run --locked pytest -q \
  tests/test_pixel_shift.py tests/test_merge.py tests/test_merge_outputs.py \
  tests/test_atomic.py tests/test_radial.py tests/test_nefx.py \
  tests/test_tiff_stream.py tests/test_raw_metadata.py tests/test_inspect.py
```

For a narrow edit, select the relevant files from the routing table. Run Ruff only
on changed Python files; for example, when changing `pixel_shift.py` and its tests:

```sh
uv run --locked ruff check src/nx_ctudio/pixel_shift.py tests/test_pixel_shift.py
uv run --locked ruff format --check src/nx_ctudio/pixel_shift.py tests/test_pixel_shift.py
git diff --check
```

External-data verification is a separate evidence tier:

| Tool | Required input and output | What it establishes |
| --- | --- | --- |
| [verify_native_core.py](tools/verify_native_core.py) | Saved passing pipeline reports; `--report NEW_JSON`; optional `--radial-correction` | Native numerical core versus hash-checked saved synthetic vendor outputs. |
| [verify_native_cli.py](tools/verify_native_cli.py) | Saved pipeline reports; `--output NEW_DIRECTORY`; optional `--chromatic-aberration-correction` | CLI replay against saved vendor outputs using synthetic identity copies. |
| [verify_native_modes.py](tools/verify_native_modes.py) | `--matrix MATRIX_JSON --manifest FIXTURE_JSON --output NEW_DIRECTORY` | All nine uncorrected modes versus the saved synthetic vendor matrix. |

Run these with `uv run --locked python tools/TOOL.py --help` before preparing a
replay. Reports reference external files, often with absolute paths. The CLI and
mode replay tools patch shutter metadata on synthetic copies: they are not
authentic-capture acceptance runners. Keep their output outside the repository.
Some other historical tools contain workstation-specific paths; inspect them
before execution.

## Evidence and next work

| Question | Start here |
| --- | --- |
| Current scope and release gaps | [project-scope.json](research/project-scope.json) |
| Nine-mode replay, publication, isolated wheel, resource measurements | [pixel-shift-native-modes.json](research/pixel-shift-native-modes.json) |
| Reconstruction and sparse repair | [pixel-shift-algorithm.json](research/pixel-shift-algorithm.json), [pixel-shift-sparse-pipeline.json](research/pixel-shift-sparse-pipeline.json) |
| NEFX layout, metadata, and writer evidence | [pixel-shift-nefx-writer.json](research/pixel-shift-nefx-writer.json), [pixel-shift-serialization.json](research/pixel-shift-serialization.json) |
| Correction and compression limits | [pixel-shift-native-radial.json](research/pixel-shift-native-radial.json), [pixel-shift-compression.json](research/pixel-shift-compression.json) |
| Previous capture-data search | [pixel-shift-capture-search.json](research/pixel-shift-capture-search.json) |

Research records are dated evidence, not live verification. Earlier records can
contain gaps resolved by later work; reconcile them with current scope, source,
and the native-modes record. Full-resolution repeated-RAW measurements are
synthetic resource observations, not authentic image-quality results or universal
memory bounds.

The next acceptance work requires authentic ordered lossless 14-bit FX Z8 captures
and matching NX Studio NEFX references across the supported modes and correction
settings, followed by version-pinned Lightroom import, Develop, and export checks.
Preserve original files and record their provenance and hashes. Run the native
merger without metadata patching; compare decoded samples, dimensions, selected
metadata, previews, and downstream behavior as distinct checks. Record the exact
application versions and settings with the results.

Until those inputs are available, address concrete synthesis defects and explicitly
requested maintenance. Do not infer acceptance from LibRaw/Adobe SDK decoding,
restart exhausted public searches without a new lead, or reopen general renderer
development to fill an unrelated validation gap.
