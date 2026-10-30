# Coding Agent Contract

This is the repository instruction entry point and development reference.
Read [README.md](README.md) for installation and usage; use the commands, source
map and verification guidance below for engineering work. All repository text,
comments, commit messages, and new documentation must be English ASCII.

## Mission and scope

- Build cNEFX as a C++20 library and thin CLI exclusively for native Nikon
  pixel-shift synthesis to NEFX, prioritizing Linux. Target Z8, Zf, Z6III and Z5II;
  add models only with official shooting support and recorded file-format
  evidence. Keep per-model acceptance
  distinct from synthetic coverage.
- Treat [research/project-scope.json](research/project-scope.json) as the recorded
  scope decision. The current user request takes precedence over recorded scope.
- Keep Wine and Nikon DLLs outside the product runtime. They are external research
  oracles, not dependencies, fallbacks, or distributable assets.
- Keep Linear DNG, inspection, Picture Control, Flexible Color and rendering
  outside the active scope. Recover historical implementations from Git tags;
  do not retain duplicate product implementations as reference code.
- Use the pinned, unmodified static LibRaw build. Optional system LibRaw is for
  development; external codecs, GUI frameworks and vendor SDKs are not required.
- Retain experimental status until authentic capture/reference comparisons and
  version-pinned downstream application acceptance establish the missing evidence.

## Start a task

1. Check the requested outcome, working directory, branch, HEAD, and dirty paths:

   ```sh
   git status --short --branch
   git rev-parse HEAD
   git diff --stat
   git diff --cached --stat
   ```

2. Read `research/project-scope.json`, `CMakeLists.txt`, and the relevant source
   and tests. Read `pyproject.toml` for verification tooling. Use the source
   routing table below before exploring historical research.
3. If `agent-handoff` is available, run `agent-handoff show --json`. Reconcile its
   objective and saved state with the current request. Saved checks and unchanged
   Git metadata do not establish fresh verification.
4. Identify the module boundary, observable behavior, relevant checks, and file
   budget before editing. Preserve unrelated changes and staged content.
5. For a synthesis continuation, inspect the acceptance gaps in the README. A
   concrete defect or new authentic data can justify work; an old blocked handoff
   does not justify repeating exhausted searches or expanding into rendering.

## Implementation invariants

| Boundary | Required behavior |
| --- | --- |
| Input identity | Skip only captures without active pixel-shift metadata. Group the rest by Nikon burst identity and order each sequence by shot number, as NX Studio does; never infer order from file names or timestamps. Preserve complete sequence size, consecutive shot numbers, matching sequence metadata and immutable source hashes. Reject duplicate files and hard links. |
| RAW decoding | Keep `metadata.cpp` and `raw.cpp` strict about supported Nikon metadata, compression, CFA, geometry, and crop. Read each source once from its owned descriptor; hash, parse and decode those same bytes. Do not patch real capture metadata or relax validation to accept an unsupported file. |
| Numerical core | Keep reconstruction independent of file decoding and capture identity. Preserve integer arithmetic, truncation, ordered sparse repair, phase mapping, and the special 32-to-4 grouping. Disable floating-point contraction and fast math for calibrated arithmetic. |
| Resource ownership | Preserve one decode per capture index and bounded source-plane retention. Keep LibRaw state on the heap with callbacks, streams and source bytes alive through its destruction. Source bytes, decoder buffers, final RGB and radial correction consume additional memory. Run callbacks only on the caller's thread; workers poll a stop flag, and outputs must not depend on the job count. |
| NEFX serialization | Preserve required source metadata and JPEG payloads, update recovered Nikon blocks and CRCs, and verify decoded RGB samples exactly before publication. Do not equate this with byte-identical Nikon output. |
| Publication | Stage privately beside each destination; publish without replacement only after every output is validated. Pre-publication failures must not expose any output. A directory publishes all outputs at once; separate files publish in order, and a later failure must report the outputs already published. |
| Post-publication failure | Retain published output and report parent-sync or cleanup warnings. Do not delete a valid output or imply that retrying the same destination is safe. |
| CLI | Keep orchestration thin. Preserve versioned JSON reports, process exit codes, cancellation behavior, limitations, and warnings. Argument errors use stderr, not JSON reports. |

Do not introduce a vendor backend, Python dependency or external subprocess into
the merge runtime. Keep reusable computation in `src/`, public contracts in
`include/cnefx/`, bounded probes in `tools/` and small native cases in `tests/cpp/`.
Keep fixture helpers separate from the product. Saved expectations must come
from independently verified evidence, never the executable being tested.

## Verification and evidence

- Use CMake/CTest and the pinned LibRaw source for the product. Use the locked
  `uv` environment and Ruff configuration for native verification runners.
  Follow the development commands below and preserve existing synthesis evidence.
- For behavior changes, reproduce the failure with a focused test when practical.
  Run affected CTest cases first, then the core/format/CLI regression suites
  for changes spanning decoding, reconstruction, writing or publication. Run the
  ASan/UBSan preset for native ownership, bounds and numerical changes.
- Format changed C++ with `.clang-format`; lint and format only modified Python
  paths with Ruff. There is no configured Python typechecker. Do not weaken or
  skip a failing test to obtain a passing result.
- Keep evidence classes separate: unit fixtures, saved expectation checks,
  saved vendor-output replays,
  live vendor oracle runs, authentic capture comparisons, and downstream
  application acceptance. Label repeated ordinary RAW and metadata-patched inputs
  as synthetic, including at full resolution.
- For measured claims, record the command, code revision, dependency/tool versions,
  input and reference hashes, mode, correction settings, result, and artifact path.
  Compare decoded samples and metadata separately from rendered appearance.
- Before expensive full-resolution or vendor runs, set explicit time, memory, and
  concurrency limits. Record failures and partial evidence; do not silently retry
  indefinitely or claim a historical report was rerun.
- Keep RAW captures, NEFX/DNG outputs, DLLs, installers, large reports, and generated
  fixtures outside Git. Existing `research/*.json` records index external evidence
  and its hashes; verify availability and hashes before relying on those artifacts.
- Use existing JSON research records for durable findings. Create additional
  Markdown only when explicitly requested; keep these two entry documents aligned
  when a requested change affects their contracts.

## Development commands

Build and run native tests with a C++20 compiler and CMake 3.21+:

```sh
cmake --preset release
cmake --build --preset release
ctest --preset release
./build/release/cnefx --version
```

The presets build with two parallel jobs. Multi-configuration generators such
as Visual Studio place the executable in `build/release/Release/`; on Windows
use `cnefx.exe`. The default build fetches hash-pinned, unmodified LibRaw 0.22.3
and links it statically. External codecs, LCMS, OpenMP, examples and vendor SDK
integrations are disabled. For offline builds, configure with
`-DCNEFX_LIBRAW_ARCHIVE=/absolute/path/LibRaw-0.22.3.tar.gz`; the same SHA-256
check applies. `-DCNEFX_USE_SYSTEM_LIBRAW=ON` selects LibRaw 0.22+ through
pkg-config for development; standalone packaging uses the pinned build.

Native verification runners require Python 3.12+ and the locked `uv`
environment. NumPy, rawpy and tifffile are development dependencies only.

```sh
uv sync --locked
uv run --locked python tools/ci.py verify --build-dir build/release
cmake --preset sanitize
cmake --build --preset sanitize
ctest --preset sanitize
uv run --locked python tools/ci.py verify --build-dir build/sanitize --config Debug
```

`verify` runs numerical, format, four-model/nine-mode CLI, multi-sequence and
failure-injection checks. Reports go to `build/<preset>/verification/`. Preserve that directory
before rerunning: the command refuses to overwrite it. Individual
`tools/verify_cpp_*.py` runners accept a fresh `--work-directory`. The sanitizer
preset instruments both cNEFX and LibRaw.

[native-golden.json](tests/fixtures/native-golden.json) pins independently
verified sample and TIFF-tree hashes from before the Python implementation was
removed. Never regenerate expectations from the executable being tested.
Sample, metadata and image-payload checks remain synthetic regression evidence.

Package a release build after verification:

```sh
uv run --locked python tools/ci.py package --build-dir build/release
```

Every build regenerates `build-info.json` with the current revision and
working-tree state. The package check requires that revision to equal `HEAD`,
verifies checksums, extracts the archive, and runs 4-to-1 and 32-to-4 synthesis
with an empty `PATH`. Rebuild after a commit before packaging.

[Native CI](.github/workflows/ci.yml) packages Linux x86_64 on Ubuntu 22.04,
Linux ARM64 on Ubuntu 24.04, macOS ARM64 on macOS 15, and Windows x86_64 on
Windows Server 2022. Archives contain `bin/cnefx` (or `bin/cnefx.exe`),
`share/cnefx/build-info.json`, the project `LICENSE` and LibRaw licenses;
preserve those licenses when redistributing. Builds are unsigned and not
notarized. The CI matrix does not establish compatibility with older systems.

CI also assembles the `cnefx-cli` npm package: `tools/ci.py npm` unpacks the
four verified archives into `native/<target>/`, `pnpm pack` adds the
[launcher](npm/cli.ts), and the packed launcher runs through `pnx`. The launcher
forwards arguments, exit codes and POSIX signals; on Windows the native process
receives Ctrl+C from the shared console.

To release, keep the versions in `CMakeLists.txt` and `package.json` equal, merge
to master, then push the matching tag (`git tag v0.1.0 && git push origin v0.1.0`).
[Release](.github/workflows/release.yml) reruns CI, creates the GitHub release
with the native archives and `.sha256` files used by mise, and publishes npm
through trusted publishing; failed jobs can be rerun, and an already published
npm version is skipped. npm only accepts a trusted publisher for an existing
package, so publish the first version once by hand from a CI run of the tagged
commit, then register the workflow:

```sh
gh run download <run-id> --name npm-package --dir build/npm
npm publish build/npm/cnefx-cli-0.1.0.tgz
npm trust github cnefx-cli --file release.yml --repo ChrAlpha/cNEFX --allow-publish
```

A new trusted publisher expires unless a workflow publish succeeds within two
days. Before tagging after such a lapse, replace it: find its ID with
`npm trust list cnefx-cli`, run `npm trust revoke cnefx-cli --id=<id>`, then
repeat the `npm trust github` command above.

## Source routing and API

The public API is [core.hpp](include/cnefx/core.hpp) and
[merge.hpp](include/cnefx/merge.hpp). CMake consumers can use `add_subdirectory`
and link `cnefx::cnefx`, which exposes only `include/`. A subproject builds no
tests, installs nothing and keeps the consumer's runtime library. No stable
binary ABI or installed SDK is promised.

| Responsibility | Source | Native checks |
| --- | --- | --- |
| Grouping, phase mapping, sparse repair, radial correction | `src/core.cpp`, `src/sparse.cpp`, `src/radial.cpp` | `tests/cpp/core_test.cpp`, `tools/verify_cpp_core.py` |
| Descriptor ownership, SHA-256, bounded TIFF graph, publication | `src/io.cpp`, `src/sha256.cpp`, `src/tiff.cpp` | `tests/cpp/io_test.cpp`, format/CLI lifecycle checks |
| Strict Nikon metadata and RAW decode | `src/metadata.cpp`, `src/raw.cpp` | `tests/cpp/format_test.cpp`, `tools/verify_cpp_format.py` |
| NEFX packing, Nikon blocks, CRCs, previews | `src/nefx.cpp` | Saved format expectations and vendor replays |
| Input discovery, sequence orchestration and reports | `src/merge.cpp`, `src/report.cpp` | `tests/cpp/merge_test.cpp`, `tools/verify_cpp_cli.py`, `tools/verify_cpp_sequences.py` |
| CLI argument handling and signals | `src/cli.cpp` | CLI regression, cancellation and package smoke checks |
| Build, packaging and release | `CMakeLists.txt`, `cmake/`, `package.json`, `npm/`, `.github/workflows/` | Four native platforms, ASan/UBSan, empty-PATH package and npm launcher checks |

The numerical core is independent of decoding and capture identity. Each capture
index is decoded once; reconstruction retains at most four source planes, or two
for paired modes. Sparse repair retains original values only at affected sites.
Compressed source bytes, concurrent decoder buffers, output RGB and radial
correction add to the working set; this is not a universal peak-memory bound.
Callbacks run on the caller's thread; workers stop at their next poll after
failure or cancellation.

Inputs are files or directories; a directory contributes its immediate `.NEF`
files. Sequences follow first appearance and each is reconstructed with its own
ISO, sparse selection, radial metadata and decoder calibration. AE bracketing
needs no special handling: each exposure is a separate burst. `outputs` applies
to each sequence. Output groups within a sequence are consecutive except 32-to-4:
it pairs four-frame blocks at zero-based capture offsets `0/16`, `4/20`, `8/24`
and `12/28`. Each output inherits metadata from its group's first source.

## CLI reports and publication

With `--json`, stdout carries a report with `schema_version: 2`:

- Success includes `status: "ok"`, `backend: "native"`, `output_format: "NEFX"`,
  program/LibRaw versions, `camera`, `sequence_validation`, correction setting,
  `sequence_count`, `inputs` in synthesis order with hashes, shot numbers and
  `sequence_index`, `skipped` paths with reasons, `limitations` and `warnings`.
- Every report has `output_count`, `outputs` and total `samples_verified`. Each
  output has `output`, `sha256`, `shape` in height/width/channels order,
  `samples_verified`, `sequence_index` and zero-based `input_indices`.
- Outputs default to `<first capture stem>_merged.NEFX` beside that capture, or
  `_merged_01` through `_merged_NN` for several outputs per sequence. `--output`
  names the only output. `--output-directory` adds `output_directory` and holds
  the same names plus `manifest.json`.
- Runtime failure has `status: "error"` and `error` with `code` and `message`.
  Codes are `merge_failed` or `cancelled`.

Exit codes are `0` for success, `1` for runtime failure, `130` for caught
cancellation and `2` for argument errors. Argument errors use stderr even with
`--json`. SIGINT/SIGTERM cancellation is cooperative during merge work.

Publication uses Linux `renameat2(RENAME_NOREPLACE)`, macOS
`renamex_np(RENAME_EXCL)` or a Windows move without replacement. If a POSIX
filesystem lacks exclusive rename, a single file uses a no-replace hard link.
A directory requires exclusive rename; unsupported publication fails without
exposing a partial output set. Separate output files are verified together, then
published one by one; cancellation stops publication only before the first.

Parent-directory sync or temporary-cleanup failure after publication produces
success with warnings and retains the output. The directory manifest precedes
publication and cannot contain later warnings; the final stdout report does.
Check status, exit code and warnings before planning a retry.

## Checkpoint and delivery

1. Keep each change to one intent. Prefer files below 500 lines; avoid unrelated
   refactoring, dependency churn, or historical-research cleanup.
2. Stop editing at eight dirty files and make a verified checkpoint before touching
   more files. Preserve user-owned changes throughout.
3. Inspect the actual diff and run `git diff --check` plus scoped checks. For a
   non-trivial change, obtain a fresh review of the behavior, constraints, diff,
   and evidence. Distinguish review of reported results from independent reruns.
4. Commit the smallest coherent verified unit by default unless the user forbids
   commits. Stage explicit paths, inspect the index, and preserve unrelated staged
   changes. Use the existing `feat:`, `fix:`, `research:`, or `docs:` commit style.
5. Do not bypass hooks or signing, add AI co-authorship metadata, force-push, reset
   destructively, or delete branches without explicit authorization. Never read or
   write `.env` files, credentials, or secrets.
6. Update an existing handoff after a meaningful checkpoint using its current
   revision. Preserve unresolved synthesis acceptance gaps when completing an
   ancillary task such as documentation. Use the home long-task policy when
   available; keep task state out of new tracked Markdown files.
7. Continue remaining authorized work after a checkpoint. Report changed paths,
   commit, commands and outcomes, and any unresolved acceptance requirement.

Read-only queries and necessary reversible technical operations can proceed within
the requested scope. Before any interaction representing the user to other people
(PRs, reviews, comments, issues, messages, mentions, merges, or request edits),
prepare the exact target and content and obtain explicit confirmation. Routine
technical pushes and CI do not grant permission for those interactions.
