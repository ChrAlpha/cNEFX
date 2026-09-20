# Coding Agent Contract

This is the repository instruction entry point. Read [README.md](README.md) for
commands, module routing, and evidence locations. All repository text, comments,
commit messages, and new documentation must be English ASCII.

## Mission and scope

- Build native Linux Nikon Z8 pixel-shift synthesis with NEFX as the primary output.
- Treat [research/project-scope.json](research/project-scope.json) as the recorded
  scope decision. The current user request takes precedence over recorded scope.
- Keep Wine and Nikon DLLs outside the product runtime. They are external research
  oracles, not dependencies, fallbacks, or distributable assets.
- Keep Linear DNG secondary. Existing Picture Control, Flexible Color, rendering,
  and `picture-curve` code is retained research, not an active feature backlog.
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

2. Read `research/project-scope.json`, `pyproject.toml`, and the relevant source and
   tests. Use the README routing tables before exploring historical research.
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
| RAW decoding | Keep `raw.py` strict about supported Nikon metadata, compression, CFA, geometry, and crop. Do not patch real capture metadata or relax validation to accept an unsupported file. |
| Numerical core | Keep reconstruction independent of file decoding and capture identity. Preserve integer arithmetic, truncation, ordered sparse repair, phase mapping, and the special 32-to-4 grouping. |
| Resource ownership | Preserve one read per capture index and bounded source-plane retention. The final RGB buffer still consumes memory; radial correction has a larger working set. |
| NEFX serialization | Preserve required source metadata and JPEG payloads, update recovered Nikon blocks and CRCs, and verify decoded RGB samples exactly before publication. Do not equate this with byte-identical Nikon output. |
| Publication | Stage privately beside the destination; publish without replacement only after validation. Pre-publication failures must not expose partial output sets. |
| Post-publication failure | Retain published output and report parent-sync or cleanup warnings. Do not delete a valid output or imply that retrying the same destination is safe. |
| CLI | Keep orchestration thin. Preserve JSON reports, process exit codes, cancellation behavior, limitations, and warnings. Argparse errors are not JSON reports. |

Do not introduce a vendor backend, external subprocess, or import from `tools/`
into the merge runtime. `inspect` has a separate, explicit ExifTool dependency.
Keep reusable computation in `src/nx_ctudio/`; put bounded probes in `tools/` and
small regression cases in `tests/`.

## Verification and evidence

- Use the locked `uv` environment and existing pytest/Ruff configuration. Follow
  the README commands; do not introduce another package manager or test framework.
- For behavior changes, reproduce the failure with a focused test when practical.
  Run affected tests first; use the focused synthesis suite for changes spanning
  decoding, reconstruction, writing, or publication.
- Lint and format only modified Python paths. There is no configured typechecker;
  do not add one as incidental cleanup. Do not weaken or skip a failing test to
  obtain a passing result.
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
