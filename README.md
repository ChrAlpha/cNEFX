# cNEFX

Native Nikon pixel-shift synthesis to NEFX, implemented as a C++20 library and
CLI. Linux is the primary platform; native builds also cover macOS and Windows.
The product has one command, `cnefx merge`, and no Python or vendor runtime.

**Experimental:** authentic capture/reference comparisons and version-pinned
Lightroom acceptance remain open. Synthetic agreement and successful LibRaw
decoding do not establish those acceptance requirements.

## Build and run

A C++20 compiler and CMake 3.21+ are required for the presets. The default build
fetches the hash-pinned, unmodified LibRaw 0.22.3 source and links it statically.
External codecs, LCMS, OpenMP, examples and vendor SDK integrations are disabled.
No Python environment is needed to build or run the executable.

```sh
cmake --preset release
cmake --build --preset release
ctest --preset release
./build/release/cnefx --version
./build/release/cnefx merge --help
```

The default presets build with two parallel jobs. Multi-configuration generators
such as Visual Studio place the executable in `build/release/Release/`. On Windows
use `cnefx.exe`. An offline source archive can be supplied with
`-DCNEFX_LIBRAW_ARCHIVE=/absolute/path/LibRaw-0.22.3.tar.gz`; the same SHA-256 check
applies. `-DCNEFX_USE_SYSTEM_LIBRAW=ON` instead selects LibRaw 0.22+ through
pkg-config for development; standalone packaging uses the pinned build.

Verified native archives are uploaded by
[Native CI](https://github.com/ChrAlpha/cNEFX/actions/workflows/ci.yml).
Choose an artifact from the required branch/revision and check its accompanying
`.sha256` file. Historical `nxc` Python releases predate this implementation.

| Platform | Archive | CI system |
| --- | --- | --- |
| Linux x86_64 | `cnefx-<version>-linux-x86_64.tar.gz` | Ubuntu 22.04 |
| Linux ARM64 | `cnefx-<version>-linux-arm64.tar.gz` | Ubuntu 24.04 |
| macOS Apple silicon | `cnefx-<version>-macos-arm64.tar.gz` | macOS 15 |
| Windows x86_64 | `cnefx-<version>-windows-x86_64.zip` | Windows Server 2022 |

Each archive contains `bin/cnefx` (or `bin/cnefx.exe`), build provenance in
`share/cnefx/build-info.json`, and LibRaw license files. The executable runs
without the Python reference package or ExifTool on `PATH`. Preserve the license
files when redistributing an archive. Builds are unsigned and not notarized;
compatibility with older systems is not established by this CI matrix.

The examples below use `cnefx` on `PATH`. Substitute the executable's full path
when running from a build or extracted archive.

## Scope and acceptance

The active product accepts complete ordered 4, 8, 16 or 32-frame sequences from
Z8, Zf, Z6III and Z5II, producing one NEFX or a complete output directory. It keeps
strict model, sequence, compression, CFA, geometry and calibration checks.
There is no camera-identity override. The recorded scope is
[project-scope.json](research/project-scope.json).

| Boundary | Implemented behavior |
| --- | --- |
| Source decoding | Observed lossless-compressed 14-bit Nikon NEF layouts, decoded by LibRaw from the descriptor used for integrity checks. HE/HE* is unsupported. |
| Reconstruction | Integer phase mapping, pair averaging, ordered sparse repair and all nine output modes below. |
| Optional correction | Source-metadata red/blue radial correction before crop; disabled by default and requires supported matching parameters. |
| NEFX writing | Packed14 RGB, Nikon metadata/CRC updates, private RGB preview and preserved source JPEGs. Every output sample is read back and checked before publication. |
| Publication | Private staging beside the destination; no replacement of existing files or directories; complete output sets become visible together. |
| Reference archive | Python synthesis, DNG, inspection, Picture Control and rendering remain under `reference/` for verification and historical research. They are outside the product runtime and active feature scope. |

| Camera | Exact TIFF Model | Evidence boundary |
| --- | --- | --- |
| Z8 | `NIKON Z 8` | Saved synthetic Nikon-output replays and authored native tests. |
| Zf | `NIKON Z f` | Recorded ordinary-RAW layout/decode evidence and authored native tests. |
| Z6III | `NIKON Z6_3` | Recorded ordinary-RAW layout/decode evidence and authored native tests. |
| Z5II | `NIKON Z5_2` | Recorded ordinary-RAW layout/decode evidence and authored native tests. |

Authentic acceptance remains open for every model. Four-model fixture coverage
uses real LibRaw decoding but does not establish per-model NX Studio parity.
Source white balance and ColorBalance tags are preserved. If LibRaw lacks the
Z5II color matrix, the report records that limitation; synthesis does not render
RAW data or substitute calibration.

HLG, non-FX image areas, other firmware and broader correction parameters remain
unvalidated. Source Adobe XMP defaults are preserved without vendor-equivalent
updates; default rendered appearance is separate from merged-sample correctness.
A readable NEFX with matching samples is not byte-identical Nikon serialization.
Wine and Nikon DLLs are external research oracles only. GUI, WASM and general RAW
rendering are outside the current product scope.

## CLI contract

Pass explicit files in capture order. A directory or a shell glob is not an
ordered-sequence contract. Capture numbers must be consecutive and observed
sequence metadata must agree. Duplicate paths, hard links and changed inputs are
rejected. Source SHA-256 hashes are checked again before publication.

The output parent must already exist. The destination must not exist, including
an empty directory or a dangling symlink. Four frames to one NEFX:

```sh
cnefx merge \
  /data/capture4/shot-01.NEF /data/capture4/shot-02.NEF \
  /data/capture4/shot-03.NEF /data/capture4/shot-04.NEF \
  --output /data/results/merged.nefx --json
```

Eight frames to a complete two-file output set:

```sh
cnefx merge \
  /data/capture8/shot-01.NEF /data/capture8/shot-02.NEF \
  /data/capture8/shot-03.NEF /data/capture8/shot-04.NEF \
  /data/capture8/shot-05.NEF /data/capture8/shot-06.NEF \
  /data/capture8/shot-07.NEF /data/capture8/shot-08.NEF \
  --outputs 2 --output-directory /data/results/merge-set --json
```

Add `--chromatic-aberration-correction` to enable supported source correction.
`--output` requires a `.nefx` suffix and one output. Multiple outputs require
`--output-directory`; a directory can also contain one output. `--outputs`
defaults to `1`. Use `--` before input paths that begin with a hyphen.

| Capture count | Allowed `--outputs` | Per-output crop dimensions |
| --- | --- | --- |
| 4 | 1 | 1x width and height |
| 8 | 1, 2 | 1x width and height |
| 16 | 1, 4 | 2x for one output; 1x for four |
| 32 | 1, 2, 4, 8 | 2x for one or two outputs; 1x for four or eight |

Groups are consecutive except 32-to-4: it pairs four-frame blocks at zero-based
capture offsets `0/16`, `4/20`, `8/24` and `12/28`. Each output inherits metadata
from its group's first source.

With `--json`, stdout carries a report with `schema_version: 1`:

- Success includes `status: "ok"`, `backend: "native"`, `output_format: "NEFX"`,
  program/LibRaw versions, `camera`, `sequence_validation`, ordered `inputs` with
  hashes and shot numbers, correction setting, `limitations` and `warnings`.
- A single output has `output`, `sha256`, `shape` in height/width/channels order,
  and `samples_verified`.
- An output set has `output_directory`, `output_count`, `outputs`, total
  `samples_verified` and zero-based `input_indices` for each output. It contains
  `merged-01.nefx` through `merged-NN.nefx` and `manifest.json`.
- Runtime failure has `status: "error"` and `error` with `code` and `message`.
  Codes are `merge_failed` or `cancelled`.

Exit codes are `0` for success, `1` for runtime failure, `130` for caught
cancellation and `2` for argument errors. Argument errors use stderr even with
`--json`. SIGINT/SIGTERM cancellation is cooperative during merge work.

Single-file publication uses a no-replace hard link on POSIX and an exclusive
move on Windows. Directory publication uses Linux `renameat2(RENAME_NOREPLACE)`,
macOS `renamex_np(RENAME_EXCL)` or Windows exclusive move. The filesystem must
support the required operation; unsupported publication fails without exposing
a partial output set.

A parent-directory sync or temporary-cleanup failure after publication produces
a success report with warnings and retains the valid output. The directory
manifest is written before publication and cannot contain later warnings; the
final stdout report includes them. Check status, exit code and warnings before
planning a retry.

## Source routing

The reusable public API is [core.hpp](include/cnefx/core.hpp) and
[merge.hpp](include/cnefx/merge.hpp). CMake consumers can use `add_subdirectory`
and link `cnefx::cnefx`; no stable binary ABI or installed SDK is promised.
Cancellation and progress callbacks run synchronously on the caller's thread.

| Responsibility | Source | Native checks |
| --- | --- | --- |
| Grouping, phase mapping, sparse repair, radial correction | `src/core.cpp`, `src/sparse.cpp`, `src/radial.cpp` | `tests/cpp/core_test.cpp`, `tools/verify_cpp_core.py` |
| Descriptor ownership, SHA-256, bounded TIFF graph, publication | `src/io.cpp`, `src/sha256.cpp`, `src/tiff.cpp` | `tests/cpp/io_test.cpp`, format/CLI lifecycle checks |
| Strict Nikon metadata and RAW decode | `src/metadata.cpp`, `src/raw.cpp` | `tests/cpp/format_test.cpp`, `tools/verify_cpp_format.py` |
| NEFX packing, Nikon blocks, CRCs, previews | `src/nefx.cpp` | Format differential checks and saved vendor replays |
| Sequence orchestration and reports | `src/merge.cpp`, `src/report.cpp` | `tests/cpp/merge_test.cpp`, `tools/verify_cpp_cli.py` |
| CLI argument handling and signals | `src/cli.cpp` | CLI differential, real cancellation and package smoke checks |
| Build, pinned dependency and packaging | `CMakeLists.txt`, `cmake/`, `.github/workflows/ci.yml` | Four native platforms, ASan/UBSan, empty-PATH package checks |

The numerical core is independent of decoding and capture identity. Each capture
index is decoded once; reconstruction keeps at most four source planes, or two
for paired modes. Sparse repair retains original values only at affected sites.
Decoder buffers, output RGB and radial correction add to the working set, so this
is not a universal peak-memory bound.

`tools/` holds probes and verification runners; `research/` indexes evidence.
`reference/nx_ctudio/` preserves the independent Python implementation. It is
installed only in the development environment as `cnefx-reference`, with no
`nxc` console entry point. Historical files and tests do not expand product scope.

## Verification

Read [AGENTS.md](AGENTS.md) before editing. Native unit tests require only the
C++ build. Differential checks additionally use Python 3.12+ and the locked `uv`
environment; NumPy, rawpy and tifffile are reference dependencies only.

```sh
uv sync --locked
uv run --locked python tools/ci.py verify --build-dir build/release
cmake --preset sanitize
cmake --build --preset sanitize
ctest --preset sanitize
uv run --locked python tools/ci.py verify --build-dir build/sanitize --config Debug
```

`verify` runs 128 numerical cases, 32 format roundtrips plus 14 rejections, and
36 four-model/nine-mode CLI cases with failure injections. Reports are saved in
`build/<preset>/verification/`. It refuses to overwrite that directory: preserve
it before rerunning, or invoke an individual `tools/verify_cpp_*.py` runner with a
fresh `--work-directory`. The sanitizer preset instruments both cNEFX and LibRaw.

Run only affected Python reference tests for reference changes. Format modified
C++ paths with `.clang-format`, use Ruff on modified Python paths, and run
`git diff --check`. Native CI checks changed text for ASCII and packages each
platform after unit and differential checks. To package a local release build:

```sh
uv run --locked python tools/ci.py package --build-dir build/release
```

The package check verifies checksums/provenance, extracts the archive, and runs
4-to-1 and 32-to-4 synthesis with an empty `PATH`.

External evidence requires separately supplied, hash-checked artifacts:

| Runner | Evidence |
| --- | --- |
| `tools/verify_native_modes.py --executable build/release/cnefx` | Nine modes against the saved synthetic Nikon output matrix. |
| `tools/verify_native_cli.py --executable build/release/cnefx` | Saved synthetic vendor output with optional radial correction. |
| `tools/verify_firmware_reference.py` | Integer lattice/pair/green arithmetic against the pinned external firmware project; excludes sparse repair and serialization. |
| `tools/benchmark_cpp.py` | Bounded Linux full-resolution C++/Python resource comparison and exact decoded agreement on hash-pinned synthetic copies. Requires GNU time and prlimit. |

Inspect each runner's `--help` and record explicit time, memory and concurrency
limits before expensive work. Reports pin revisions, binaries, inputs and
references. Retain failures and partial evidence. The native replay tools patch
identity metadata on synthetic copies; they are not authentic-capture runners.

The correctness reference is
[NikonFirmwarePatch/z8pixelshift](https://github.com/XinmiaoZhao/NikonFirmwarePatch/tree/13c3709a35c58f9cf6fc9c1043705712074651eb/z8pixelshift),
pinned to `13c3709a35c58f9cf6fc9c1043705712074651eb`. Its code is an external test
oracle, not a product dependency. Sparse repair, radial correction and container
behavior are also compared against retained Python and saved Nikon outputs.

## Evidence and remaining acceptance

| Question | Record |
| --- | --- |
| Product scope and release gaps | [project-scope.json](research/project-scope.json) |
| C++ migration, validation, portability and resource results | [pixel-shift-native-modes.json](research/pixel-shift-native-modes.json), `cpp_refactor_20261029` |
| Model layouts and ordinary-RAW evidence | [pixel-shift-camera-support.json](research/pixel-shift-camera-support.json) |
| Reconstruction and sparse repair | [pixel-shift-algorithm.json](research/pixel-shift-algorithm.json), [pixel-shift-sparse-pipeline.json](research/pixel-shift-sparse-pipeline.json) |
| NEFX format and writer | [pixel-shift-nefx-writer.json](research/pixel-shift-nefx-writer.json), [pixel-shift-serialization.json](research/pixel-shift-serialization.json) |
| Radial correction and compression | [pixel-shift-native-radial.json](research/pixel-shift-native-radial.json), [pixel-shift-compression.json](research/pixel-shift-compression.json) |
| Previous capture-data search | [pixel-shift-capture-search.json](research/pixel-shift-capture-search.json) |

These are dated records, not live verification. Large datasets, binaries and
reports remain outside Git. Historical workstation paths under
`/workspaces/LumaForge/nxc-data/` are artifact references, not runtime dependencies.
Verify availability and hashes before relying on them.

Acceptance still requires complete authentic lossless 14-bit FX sequences for
each enabled model, with matching NX Studio NEFX references across modes and
correction settings. Preserve original provenance and hashes, run without
metadata patching, and compare samples, dimensions, metadata and previews
separately. Then record version-pinned Lightroom import, Develop and export
results. Repeated ordinary RAW at full resolution measures synthetic resource
use and sample agreement, not authentic image quality.
