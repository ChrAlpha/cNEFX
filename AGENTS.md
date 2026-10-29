# Coding Agent Contract

This is the repository instruction entry point. Read [README.md](README.md) for
commands, module routing, and evidence locations. All repository text, comments,
commit messages, and new documentation must be English ASCII.

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
- Keep Python, Linear DNG, inspection, Picture Control, Flexible Color and
  rendering in `reference/` and research tools. They are verification references
  or historical research, not product runtime features or an active backlog.
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
   and tests. Read `pyproject.toml` for reference or harness work. Use the README
   routing tables before exploring historical research.
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
| Input identity | Preserve explicit capture order, complete sequence size, consecutive shot numbers, matching sequence metadata, and immutable source hashes. Reject duplicate files and hard links. |
| RAW decoding | Keep `metadata.cpp` and `raw.cpp` strict about supported Nikon metadata, compression, CFA, geometry, and crop. Decode from the same owned descriptor used for hashing. Do not patch real capture metadata or relax validation to accept an unsupported file. |
| Numerical core | Keep reconstruction independent of file decoding and capture identity. Preserve integer arithmetic, truncation, ordered sparse repair, phase mapping, and the special 32-to-4 grouping. Disable floating-point contraction and fast math for calibrated arithmetic. |
| Resource ownership | Preserve one decode per capture index and bounded source-plane retention. Keep LibRaw state on the heap with callbacks and streams alive through its destruction. Decoder buffers, final RGB and radial correction consume additional memory. |
| NEFX serialization | Preserve required source metadata and JPEG payloads, update recovered Nikon blocks and CRCs, and verify decoded RGB samples exactly before publication. Do not equate this with byte-identical Nikon output. |
| Publication | Stage privately beside the destination; publish without replacement only after validation. Pre-publication failures must not expose partial output sets. |
| Post-publication failure | Retain published output and report parent-sync or cleanup warnings. Do not delete a valid output or imply that retrying the same destination is safe. |
| CLI | Keep orchestration thin. Preserve versioned JSON reports, process exit codes, cancellation behavior, limitations, and warnings. Argument errors use stderr, not JSON reports. |

Do not introduce a vendor backend, Python dependency or external subprocess into
the merge runtime. Keep reusable computation in `src/`, public contracts in
`include/cnefx/`, bounded probes in `tools/` and small native cases in `tests/cpp/`.
The retained Python oracle in `reference/nx_ctudio/` must remain independent enough
to detect numerical or format regressions in the C++ implementation.

## Verification and evidence

- Use CMake/CTest and the pinned LibRaw source for the product. Use the locked
  `uv` environment and existing pytest/Ruff configuration for differential checks
  and research. Follow the README commands and preserve existing evidence.
- For behavior changes, reproduce the failure with a focused test when practical.
  Run affected CTest cases first, then the core/format/CLI differential suites
  for changes spanning decoding, reconstruction, writing or publication. Run the
  ASan/UBSan preset for native ownership, bounds and numerical changes.
- Format changed C++ with `.clang-format`; lint and format only modified Python
  paths with Ruff. There is no configured Python typechecker. Do not weaken or
  skip a failing test to obtain a passing result.
- Keep evidence classes separate: unit fixtures, saved vendor-output replays,
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
