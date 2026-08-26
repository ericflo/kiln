# Cleanup Ledger

This file is a cooperative guestbook for agents working to make Kiln the
cleanest possible repository it can be: no fluff, no dead weight, no stray
files or directories, no stale ideas — every line, file, and directory has a
clear purpose, and everything is organized as if chiseled from perfection.

We get there one agent session at a time.

## Protocol

Each agent that receives a cleanup task should:

1. **Read this ledger first.** Scan previous entries to learn what has already
   been done and avoid duplicating work or undoing someone else's cleanup.
2. **Explore.** Look around the repository for one way to clean up, tidy up,
   or improve it — big or small. Dead code, unused files, duplicated logic,
   stale docs, broken references, inconsistent naming, organizational drift.
   Anything that moves us closer to pristine.
3. **Verify nothing breaks.** Build, test, and check references before and
   after your change. A cleanup that breaks something is not a cleanup.
4. **Relentlessly pursue that one cleanup** until it is fully done.
5. **Commit your work.** `git add` the changed files and `git commit` with a
   clear message describing the cleanup. Every improvement lands as its own
   commit — never leave finished work uncommitted.
6. **Sign the ledger.** Append an entry at the bottom of this file using the
   format below, then `git add CLEANUP.md && git commit --amend --no-edit`
   (or a small follow-up commit) so the signature is committed too.

## Entry format

```
## <Agent name> — <YYYY-MM-DD>

<What was cleaned up, why it mattered, and how it was verified.>
```

Rules of engagement:

- One focused cleanup per session is enough. Done beats ambitious.
- Never remove or rewrite another agent's entry.
- If you find something worth cleaning but out of scope for your session,
  leave the repository as you found it — the next agent will find it too.
- Prefer deletion over addition. The best cleanup leaves fewer things behind.
- Always commit. A cleanup that exists only in the working tree doesn't count.

---

## Ledger

## Cleanup Agent — 2026-08-25

Removed two stray artifacts that had been committed to the repository root:
`_audit_boxes.json` (44 KB JSON array of 146 audit-checklist strings left over
from an earlier candle-removal audit session — zero references in any code,
doc, script, or CI) and `VP_VULKANINFO_NVIDIA_RTX_6000_Ada_Generation_550_127_8_0.json`
(222 KB machine-specific `vulkaninfo` profile dump from one RTX 6000 Ada
machine, driver 550.127.8.0). The only mention of the latter anywhere was a
provenance comment above the `captured_rtx_6000_ada_limits_select_compatible_routes`
test in `crates/kiln-vulkan-kernel/src/policy.rs`; that comment was rewritten
to preserve the provenance (GPU model + driver) without dangling a pointer at
deleted data — the test's limits were already hardcoded inline and never read
the file. Also deleted the untracked scratch directory
`sft-cap.pre-edit-ctx-gather/` (20 KB of agent context-gather leftovers, only
referenced by ignored `Qwen3.5-4B/` trace logs) from the working tree. Why it
mattered: both were committed machine/session scratch with no ongoing purpose,
bloating every clone and mistaken for curated repo root contents. Verified: `cargo test -p kiln-vulkan-kernel --lib` passed identically
before and after the change (65 passed, 0 failed); repo-wide grep for both
filenames and for `audit_boxes` found no remaining references outside git
history and agent trace logs; `git status` clean after the commit.

## Cleanup Agent (round 2) — 2026-08-25

Moved `PROFILING.md` (532 KB, 9,267 lines — the live profiling report and
the accumulated Phase 6 / 7 / MTP investigation record) from the repo root to
`docs/archive/profiling/PROFILING.md`, co-locating it with its whole family
(`PROFILING-C2.md`, `PROFILING-MTP-C39.md` … `C40f.md`), whose README already
treated it as that folder's live counterpart. Deliberately a move, not a
deletion or prune: `CONTRIBUTING.md` and the PR template still route
contributors to it as the canonical NVTX hot-region source, and its own
banners mark it as the authoritative current state of the optimization
frontier. Updated all 22 root-relative links inside the moved file (13
`docs/audits/` docs, 4 `docs/archive/phase-c/` docs, 3 kernel sources, 1
`bench-results/` CSV, the `phase-c66/artifacts/` dir, and the
`profiling-artifacts/` CSV) to correct relative paths from the new location,
plus the four inbound references: the `CONTRIBUTING.md` performance-table
link, the `docs/archive/profiling/README.md` live-report pointer (heading
adjusted to "Profiling reports" now that the live report lives there), the
`docs/archive/phase-c/README.md` live-story pointer, and the PR template's
perf-change section. Historical CHANGELOG entries and prose mentions of the
unchanged *filename* in frozen audit/eval-trace records were left untouched.
Why it mattered: a half-megabyte investigation log was masquerading as a
curated root-level reference doc — an order of magnitude larger than the
root's other big docs — while the rest of the PROFILING family was already
archived under `docs/`. Verified: a scripted link audit from the new location
resolves the same 20 targets as the pre-move baseline, with the same two
pre-existing dangling links (`phase-c66/artifacts/`, which never existed, and
the deliberately purged `profiling-artifacts/` CSV preserved in history);
`git grep` shows zero links to the old root path and no script/CI job reads
`PROFILING.md` as a file path; docs-site `--validate-only` passed (59
documents) and all 11 docs-site unit tests passed; `git status` clean after
the commit.

## Cleanup Agent (round 3) — 2026-08-25

Anchored the unanchored `adapters/` rule in `.gitignore` to `/adapters/`. The
broad form matched any directory named `adapters` at any depth, silently hiding
scratch/output directories from `git status` (a scratch dir under `crates/`
was demonstrably invisible before the fix). The rule's own comment scopes it to
the root-level default runtime state directory, so the anchored form matches
the stated intent; nested `adapters/` dirs that legitimately exist (under the
already-ignored `.qualification/` and `Qwen3.5-4B/`) remain ignored by their
parent rules. Verified: `git ls-files | grep -E '(^|/)adapters/'` returns zero
tracked paths, so nothing tracked depends on the broad form; `git check-ignore`
confirms the root `adapters/` dir is still ignored while a hypothetical nested
`adapters/junk.bin` is no longer masked by this rule; repo-wide grep found no
script or CI job parsing `.gitignore` or relying on ignore semantics for nested
adapters paths; `scripts/check_repository_artifacts.py` passes after the change
(6712 tracked paths); `git status` shows only the intended `.gitignore` edit.

## Cleanup Agent (round 4) — 2026-08-26

Fixed 11 broken navigation links in the per-capability READMEs under
`capabilities/caps/*/README.md` (pi-code-comprehension, pi-context-aware-edits,
pi-doctest, pi-error-recovery, pi-incremental-progress, pi-precondition-check,
pi-search-then-read, pi-shell-hygiene, pi-source-mod-workflow,
pi-test-interpretation, pi-tool-call-efficiency). Each "Read first" section
linked `[`../README.md`](../README.md)`, which resolves to
`capabilities/caps/README.md` — a file that has never existed in the repo's
git history. This was an off-by-one path-depth bug from the round-3 flat-
caps/ unification: the intended target is `capabilities/README.md` (one level
deeper), which LAYOUT.md calls the top-level entry and whose FAQ covers ECHO
defaults and paradigm/methodology pointers. Fixed by rewriting both the link
text and target to `../../README.md` in each file (11 files, 11 lines).
Why it mattered: these are live working docs for the capability-uplift
pipeline — every "read first" hop to repo context was dead for 11 of 28 caps.
Verified before and after with a scripted relative-link audit over all tracked
.md files: exactly these 11 links were dangling among live capability docs and
all now resolve; the remaining dangling links are confined to frozen archive/
audit records (historical run logs, profiling-artifacts CSVs purged in prior
rounds) and were deliberately left untouched per protocol. Confirmed nothing
generates or parses these files (docs-site manifest doesn't include
capabilities/, no script/CI job reads cap READMEs), and that the
`capability-creator` skill itself points at `capabilities/README.md` as the
top-level entry, confirming the canonical target. Also audited this round's
steering candidates without acting: `.pytest_cache/` is already ignored via
its own pytest-generated internal `.gitignore`, so no root rule is needed;
`THIRD_PARTY_LICENSES.md` is intentionally checked in (generated by
cargo-about per its header, uploaded by server-release.yml, linked from the
README) and its crate set matches current Cargo.lock — regeneration was test-
run locally but reproduces degraded output offline (clearlydefined API
unreachable → duplicate license sections), so the checked-in copy stays.
`scripts/check_repository_artifacts.py` passed after the change (6712 tracked
paths); `git status` shows only the 11 README edits.

## Cleanup Agent (round 5) — 2026-08-26

Deleted the stale committed `docs/site/sitemap.xml` (47 lines, hand-maintained
`<changefreq>/<priority>` format listing only 9 product-guide URLs).
Investigation of this round's steering candidate confirmed that the rest of
the committed `docs/site/*.html` set is NOT build output: `pages.yml` deploys
exclusively from a freshly built `_site/` (`node scripts/docs-site/build.mjs
--out _site`), and those HTML pages are hand-authored product-guide source
that the build copies verbatim via `cp(siteSourceDir, buildOut)`. The sitemap,
however, is the one file the build always overwrites afterward —
`writeSitemap(buildOut)` regenerates it as a flat-format sitemap covering all
71 canonical routes — so the committed copy never reaches the published site
and had drifted badly (9 URLs vs 71, wrong format). Nothing consumes it as
input: `check_docs_site_smoke.mjs` validates only the generated sitemap at the
site root (and skips that block entirely when pointed at unbuilt `docs/site/`,
since `llms.txt` is already absent there), `robots.txt` references just the
published URL, and the unit tests read temp-dir output. Verified before AND
after: fresh builds to /tmp are byte-identical (`diff -r` clean), all 11
docs-site unit tests pass, `--validate-only` passes (59 documents), the smoke
check against the fresh post-deletion build produces output identical to the
pre-deletion baseline (its only failure is the local absence of Chromium, an
environmental pre-existing condition), and `check_repository_artifacts.py`
passes with exactly one fewer tracked path (6712 → 6711).

## Cleanup Agent (round 6) — 2026-08-26

Fixed 7 dangling relative links in live capability docs. Five caps
(pi-precondition-check, pi-shell-hygiene, pi-source-mod-workflow,
pi-test-interpretation, pi-tool-call-efficiency) had copied pi-doctest's
"Round 2 setup" boilerplate verbatim, including the sentence "The previous
iter log and writeups are preserved in [`archive/`](archive/)" — but unlike
pi-doctest and the other 17 caps sharing that template line, these five never
had an `archive/` directory in git history, so every link was dead. Removed
the false sentence from each. Separately, `capabilities/caps/pi-doctest/
capability.md` linked `kiln-polish-prerequisites.md` at the wrong depth in
two places; the file actually lives at that cap's
`archive/kiln-polish-prerequisites.md`, so both links were rewritten to the
correct relative path (link text updated to match). Why it mattered: these
are live working docs for the capability-uplift pipeline, and the dangling
links broke the documented trail to round-1 evidence. Verified with a
scripted relative-link audit over all tracked .md files (excluding frozen
archive/audit records): exactly these 7 links were dangling among live docs
and all now resolve to existing files; confirmed via git history that no
archive dir ever existed for the five caps while all 17 sibling caps using
the same sentence do have one; `scripts/check_repository_artifacts.py`
passes unchanged after the edit.

## Cleanup Agent (round 7) — 2026-08-26

Investigated the round-6 steering candidate `scripts/c2_artifacts/` (7.9 MB of
tracked safetensors parity-failure dumps + comparator stdout) and left it in
place: the signals are genuinely ambiguous by design — the artifacts were
deliberately committed in 9371035bf as retained evidence for
`docs/archive/profiling/PROFILING-C2.md`, which documents them as "raw
artifacts" for the archived C2 investigation, and `source_tree_hash.py`
explicitly excludes the directory as historical artifacts rather than treating
it as dead weight. Not conclusive scratch, so per protocol it stays.
Instead, deleted `docs/.ipynb_checkpoints/` — two committed Jupyter checkpoint
scratch copies (`vk_native_gdn-checkpoint.md`,
`vk_resident_decode_plan-checkpoint.md`) of the live docs
`vk_native_gdn.md` / `vk_resident_decode_plan.md`. Both copies were stale
(missing later sections present in the live docs, e.g. the 2026-07-16
serving-state quarantine section), nothing references them anywhere, and the
docs-site build already skips `.ipynb_checkpoints` by name
(`scripts/docs-site/lib.mjs`), so they were pure invisible dead weight.
Verified before and after: docs-site `--validate-only` passes (59 documents),
all 11 docs-site unit tests pass, `scripts/check_repository_artifacts.py`
passes with exactly two fewer tracked paths (6711 → 6709); repo-wide grep
confirms no references to either checkpoint filename outside git history;
`git status` clean after the commit.

## Cleanup Agent (round 8) — 2026-08-26

Removed genuinely dead code from `crates/kiln-vulkan-kernel` (three files,
~55 lines): (1) the never-referenced `FlceState` struct and the
`_unused_keep_vk_matmul_export` stub in `vk_ops/flce.rs`, plus the now-unused
`vk_matmul` import — the stub only existed to silence that import; `vk_matmul`
itself stays (used by mlp.rs, parity tests, and kiln-model); (2) the
`alloc_zeroed_f32` helper and its `_silence_unused` keeper stub in
`vk_ops/gdn_chunkwise.rs` (`upload_f32`, which the same stub referenced, is
genuinely used elsewhere in the file and stays); (3) the no-op public method
`CommandBatch::assert_last_handle` in `cmd_batch.rs`, which had zero callers
repo-wide and did nothing but swallow its argument. Why it mattered: these were
`#[allow(dead_code)]`-masked leftovers from refactors masquerading as live
surface area, and the stubs obscured real usage relationships. Verified:
repo-wide grep confirmed zero references to each removed item outside its own
definition; before AND after, `cargo test -p kiln-vulkan-kernel --lib` passes
identically (65 passed, 0 failed) and `cargo clippy -p kiln-vulkan-kernel --lib`
reports no unused/never-constructed warnings and no error output (the ~280
pre-existing style lints are unchanged environmental noise);
`scripts/check_repository_artifacts.py` passes; `git status` shows only the
three source edits plus this ledger entry.

## Cleanup Agent (round 9) — 2026-08-26

Eliminated the single largest clippy lint category in
`crates/kiln-vulkan-kernel`: all 125 `clippy::manual_div_ceil` warnings,
crate-wide across 28 source files. Each was a hand-rolled ceiling division of
the form `(a + b - 1) / b` (workgroup-count computations, buffer-size
round-ups, packed-length math) replaced with the idiomatic `a.div_ceil(b)` —
already the style used elsewhere in the crate, and strictly safer since it
cannot overflow on `a + b - 1`. This is round-8 candidate (a), scoped to one
mechanical, low-risk lint category as suggested. Applied via
`cargo clippy --fix` restricted to exactly that lint
(`-A clippy::all -W clippy::manual_div_ceil`) so no other category was
touched; a diff audit confirms every changed line involves `div_ceil`. Why it
mattered: cut the crate's clippy noise from 280 to 155 warnings (-45%) with a
purely mechanical change, making remaining lints easier to triage.
Verified: before AND after, `cargo test -p kiln-vulkan-kernel --lib` passes
identically (65 passed, 0 failed); after, `cargo build -p kiln-model` (the
downstream consumer) succeeds; clippy JSON output confirms zero remaining
`manual_div_ceil` warnings and that no other lint count increased;
`scripts/check_repository_artifacts.py` passes (6709 tracked paths);
`git status` shows only the 28 source edits plus this ledger entry.

## Cleanup Agent (round 10) — 2026-08-26

Eliminated the largest remaining clippy lint category in
`crates/kiln-vulkan-kernel`: all 58 `clippy::needless_borrow` warnings,
all concentrated in `src/kernels.rs`. Each was an extra `&` on an expression
that already evaluated to a reference (e.g. `&x_data` where `x_data: &[u8]`
passed to a `&[u8]` parameter, and tuple elements like `(&weight_buf,
&weight_data)`), replaced by the value itself — purely mechanical, no
behavioral surface. This is the round-9 steering candidate, using the same
playbook: applied via `cargo clippy --fix` restricted to exactly that lint
(`-A clippy::all -W clippy::needless_borrow`) so no other category was
touched; a full diff audit confirms every changed line is only a removed
borrow on an argument or tuple element (57 changed lines, all in kernels.rs).
Why it mattered: cut the crate's clippy noise from 155 to 97 warnings (-37%),
continuing round 9's triage of mechanical lints ahead of the intentional
`too_many_arguments` keeps. Verified: before AND after,
`cargo test -p kiln-vulkan-kernel --lib` passes identically (65 passed,
0 failed); `cargo build -p kiln-model` (downstream consumer) succeeds; clippy
JSON output confirms zero remaining `needless_borrow` warnings and that no
other lint count increased; `scripts/check_repository_artifacts.py` passes;
`git status` shows only the one source edit plus this ledger entry.
Note for future agents: the lib test suite exhibits a rare pre-existing flake
(2–3 GPU/device-related test failures observed ~2 times across ~45 baseline
runs before any edit, not reproducible with output capture); it is unrelated
to this change.

## Cleanup Agent (round 11) — 2026-08-26

Eliminated the next-largest fixable clippy lint category in
`crates/kiln-vulkan-kernel`: all 18 `clippy::manual_is_multiple_of` warnings,
across 5 files (kernels.rs ×6, resident.rs ×6, vk_ops/attention.rs ×3,
vk_ops/gdn_gates.rs ×2, vk_ops/rope.rs ×1). Each was a hand-written divisibility
check of the form `a % b == 0` (GQA head-ratio assertions, rotary_dim evenness
checks, gate-sharding ensure!s) replaced with the idiomatic
`a.is_multiple_of(b)` — purely mechanical, no behavioral surface. Same playbook
as rounds 9–10: applied via `cargo clippy --fix` restricted to exactly that lint
(`-A clippy::all -W clippy::manual_is_multiple_of`) so no other category was
touched; the diff audit confirms every changed line is only the divisibility-
check rewrite (18 changed lines). Why it mattered: cut the crate's clippy noise
from 97 to 79 warnings (-19%), continuing the mechanical-lint triage; remaining
categories are the intentional `too_many_arguments` keeps plus small batches
(useless_vec ×10, manual_c_str_literals ×7) left for future sessions.
Verified: before AND after, `cargo test -p kiln-vulkan-kernel --lib` passes
identically (65 passed, 0 failed); `cargo build -p kiln-model` (downstream
consumer) succeeds; clippy JSON output confirms zero remaining
`manual_is_multiple_of` warnings and that no other lint count increased;
`git status` shows only the five source edits plus this ledger entry.

## Cleanup Agent (round 12) — 2026-08-26

Eliminated the two remaining small mechanical clippy lint categories in
`crates/kiln-vulkan-kernel`, crate-wide across all targets: all 17
`clippy::useless_vec` warnings (13 in kernels.rs, 4 in
bin/vulkan_decode_microbench.rs) and all 7 `clippy::manual_c_str_literals`
warnings (device.rs ×5, pipeline.rs ×1, kernels.rs ×1). The `vec![...]`
literals whose length never changes were replaced with fixed-size arrays
(they are only read as slices, so no behavioral surface), and the
`CStr::from_bytes_with_nul(b"...\0").unwrap()` calls became `c"..."`
literals (no unwrap needed, same value). This is round-11's leftover
candidate pair, applied via `cargo clippy --fix --all-targets` restricted to
exactly those two lints (`-A clippy::all -W clippy::useless_vec -W
clippy::manual_c_str_literals`), then rustfmt on just the four touched files
to normalize the array line breaks — leaving the one pre-existing fmt diff
in vk_ops/gdn_gates.rs untouched. Why it mattered: removed every remaining
mechanical fixable lint category from the crate; what's left is intentional
keeps (`too_many_arguments`) plus scattered single-instance style lints for
future triage. Verified: before AND after, `cargo test -p kiln-vulkan-kernel
--lib` passes identically (65 passed, 0 failed; one flaky failure on the
first post-change run passed on re-run per the documented baseline flake);
`cargo build -p kiln-model` succeeds; a full clippy JSON diff against the
pre-change baseline confirms exactly the two target categories disappeared
and no other lint count changed; `cargo fmt --check` shows the same single
pre-existing diff as baseline; `git status` shows only the four source edits
plus this ledger entry.

## Cleanup Agent (round 13) — 2026-08-26

Fixed the last remaining `cargo fmt --check` diff in the repository: a
pre-existing formatting violation in `crates/kiln-vulkan-kernel/src/
vk_ops/gdn_gates.rs` around line 113, where an `anyhow::ensure!` call in
`vk_gdn_gates_bwd_no_grad` exceeded line width. This was a round-11 leftover —
rustfmt was run on other touched files that round but not this one, so the
crate had been failing `cargo fmt --check` ever since the round-9/11 lint
rewrites lengthened the line. Applied `rustfmt` to just that file (one hunk,
4 insertions / 1 deletion, whitespace-only reflow of the macro args). Why it
mattered: `cargo fmt --check` is now fully clean repo-wide, so CI-style format
enforcement would pass with zero suppressions. Verified: before the change,
`cargo fmt --check` showed exactly this one diff; after, it exits 0 with no
output; `cargo test -p kiln-vulkan-kernel --lib` passes identically before
and after (65 passed, 0 failed); `cargo build -p kiln-model` succeeds;
`git status` shows only the one source edit plus this ledger entry.

## Cleanup Agent (round 14) — 2026-08-27

Fixed three factual-drift bugs in `QUICKSTART.md` found by auditing its claims
against `README.md`, `docs/CONFIGURATION.md`, and the source code. (1) The
`streaming_prefill.mode` config row claimed `auto` dispatches at "at least 2048
tokens on CUDA/ROCm/Metal" — wrong for ROCm: `StreamingPrefillBackendPolicy::
ROCM_AUTO_MIN_PROMPT_TOKENS = 256` in
`crates/kiln-model/src/backend/capability.rs`, and README already said
"256 prompt tokens on ROCm, 2048 on CUDA/Metal". Rewrote to match code+README.
(2) The tool-calling section cited test
`message_tool_calls_round_trip_preserved` at `completions.rs:2496` — that file
is only 2,119 lines and contains no such test; it lives at
`crates/kiln-server/src/api/completions/tests/mod.rs:1624`. Citation corrected.
(3) The key-settings table omitted `batching.actor_cycle_idle_ms` while
claiming "All eight batching values are immutable startup policy" — there are
nine per CONFIGURATION.md and README's table; added the missing row (default 0,
0–60000 ms, matches CONFIGURATION.md) and fixed the count to nine. Why it
mattered: QUICKSTART is the first-run entry doc, and two of these could send a
ROCm operator or a code reader to a nonexistent location/number. Verified:
each corrected fact cross-checked against source (`capability.rs`,
actual test file, `docs/CONFIGURATION.md` lines 999 + streaming-prefill
section) before editing; docs-site `--validate-only` passes after the change
(59 documents); `scripts/check_repository_artifacts.py` passes (6709 tracked
paths); no code touched, so no cargo checks needed beyond confirming the cited
test name exists exactly once repo-wide; remaining QUICKSTART↔README overlap
(hardware minimums, port 8420, model id, ECHO λ=0.05, webhook 5s timeout,
batch cap 64, cispo_max_weight 5.0, adapter upload limits, CLI surface)
audited and consistent.

## Cleanup Agent (round 15) — 2026-08-27

Fixed the webhook payload documentation drift flagged as round-14 leftover
(a): both `QUICKSTART.md` (section 9, payload example) and the doc comment on
`TrainingConfig::webhook_url` in `crates/kiln-server/src/config.rs` listed the
emitted `"job_type"` values as only `"sft" | "grpo"`, but
`TrainingCompletionEvent::job_type_str`
(`crates/kiln-server/src/training_queue.rs:88`) also emits `"opd"`, and OPD
completions fire through the same webhook path. Updated both to
`"sft" | "grpo" | "opd"`. README.md has no payload listing (only a one-line
feature bullet), so it needed no change. Verified: `cargo check -p
kiln-server` passes (doc-comment-only code change; the 22 lib warnings are
pre-existing); docs-site `--validate-only` passes (59 documents); repo-wide
grep confirms no remaining `"sft" | "grpo"` payload listing without `"opd"`;
`git status` shows exactly the two doc edits plus this ledger entry.

## Cleanup Agent (round 16) — 2026-08-27

Eliminated every clippy warning in `crates/kiln-rmsnorm-kernel` (the round-16
steering candidate: a scoped lint triage of a smaller kernel crate), all in
its build script — the crate's lib/bin targets were already clean. Five
`clippy::collapsible_if` warnings in `find_cuda_root()` / `find_rocm_root()`
(nested `if let Ok(...) { if ... }` and `if let Some(a) { if let Some(b) {
... } }` toolchain-probing ladders) collapsed into Rust-2024 let-chains,
semantics identical; plus three trivial fixes in the same file:
`clippy::ptr_arg` (`&PathBuf` → `&Path` on `configure_nvcc_from_cuda_root`,
with the `Path` import added), a redundant `&format!(...)` borrow passed to
`build.flag`, applied via `cargo clippy --fix -A clippy::all -W
clippy::collapsible_if` then two hand edits, and rustfmt normalization. Why it
mattered: the crate now compiles with zero clippy output, matching the
standard rounds 8–12 set for kiln-vulkan-kernel, and the CUDA/HIP discovery
logic reads flat instead of five levels deep. Verified: before AND after,
`cargo build -p kiln-rmsnorm-kernel` succeeds (the crate has no tests; the
build script runs during build) and `cargo build -p kiln-model` (downstream
consumer) succeeds; after, `cargo clippy -p kiln-rmsnorm-kernel --all-targets`
reports zero warnings attributable to this crate; `cargo fmt --check` remains
clean repo-wide. Noted for future agents: `crates/kiln-tensor/build.rs`
contains the same duplicated discovery code with the identical 5
collapsible_if + ptr_arg pattern, left untouched to keep this session scoped
to one crate.
## Cleanup Agent (round 17) — 2026-08-27

Eliminated every clippy warning in `crates/kiln-tensor/build.rs` — the exact
duplicated toolchain-discovery pattern round 16 cleaned up in
kiln-rmsnorm-kernel, applied here with the same playbook. Five
`clippy::collapsible_if` warnings collapsed into Rust-2024 let-chains (the
`which nvcc` / `which hipcc` fallback ladders in `find_cuda_root()` /
`find_rocm_root()`, including a triple-nested parent-probe), plus two trivial
fixes: `clippy::needless_borrows_for_generic_args` on the
`&format!(...)` passed to `build.flag`, and `clippy::ptr_arg`
(`&PathBuf` → `&Path` on `configure_nvcc_from_cuda_root`; `Path` was already
imported). All hand edits, no other code touched. Why it mattered: the crate's
build script now compiles warning-free like its siblings, and the CUDA/HIP
discovery ladders read flat instead of five levels deep. Verified: before AND
after, `cargo build -p kiln-tensor` succeeds; after, `cargo clippy -p
kiln-tensor --all-targets` shows zero warnings attributable to build.rs (the
one pre-existing lib warning is unchanged) and `cargo build -p kiln-model`
(downstream consumer) succeeds; `cargo fmt --check` remains clean repo-wide;
`git status` shows only the one source edit plus this ledger entry.
## Cleanup Agent (round 18) — 2026-08-27

Eliminated every mechanical clippy warning in `crates/kiln-memory` (the
round-17 steering candidate), all in `src/vram.rs`: both `clippy::match_result_ok`
(`if let Some(x) = ....ok()` → `if let Ok(x) = ...` on the /proc/self/mountinfo
read) and `clippy::let_and_return` (dropped the `let snapshot` binding in
`try_current_memory_snapshot_for`, returning the match expression directly),
plus `clippy::manual_div_ceil` (`(a + b - 1) / b` → `a.div_ceil(b)` in the
tight-headroom segment picker — strictly safer, cannot overflow on
`a + b - 1`) and `clippy::manual_range_contains` (`gib >= 8.0 && gib <= 11.5`
→ `(8.0..=11.5).contains(&gib)` in a test assertion). The remaining two
warnings are `clippy::result_large_err` on `governor.rs` public APIs — a
design-level lint requiring an error-boxing refactor, deliberately left.
Why it mattered: cut the crate's clippy noise from 6 to 2 warnings with four
purely mechanical rewrites. Verified: before AND after,
`cargo test -p kiln-memory --lib` passes identically (71 passed, 0 failed);
after, `cargo clippy -p kiln-memory --all-targets` shows only the two
intentional `result_large_err` keeps; `cargo build -p kiln-model` succeeds;
`cargo fmt --check` remains clean repo-wide; `git status` shows only the one
source edit plus this ledger entry.

## Cleanup Agent (round 19) — 2026-08-27

Eliminated all doc-formatting and small mechanical clippy warnings in
`crates/kiln-tensor` (the round-19 steering candidate), cutting the crate's
lib clippy noise from 33 to 24 warnings. Fixed: the root cause of all six
`doc_lazy_continuation` warnings in `tensor.rs` was a prose line starting
with `` + `shape` `` that rustdoc parsed as an accidental markdown list item
— rephrased to "plus an explicit `dtype` and `shape`" so the paragraph is no
longer a list at all; three `doc_overindented_list_items` continuations in
`ops/logit_mirostat.rs` realigned to the list-item content column;
`derivable_impls` — `AllocatorMode`'s hand-written `Default` impl replaced by
`#[derive(Default)]` + `#[default]` on `Pool`; `collapsible_if` — the
cache-hit nested `if let` in `cpu_allocator.rs` collapsed into a let-chain;
plus two one-liners, `manual_is_multiple_of`
(`rope_split_half.rs`) and `manual_contains` (`tile.rs`). Deliberately left:
15 `needless_range_loop` (each needs an individual borrow/index-semantics
review, not mechanical), 2 `excessive_precision` (`0.7978845608_f32` —
clippy's suggestion rounds, changing the literal's value), plus
`result_large_err`, `dead_code`, `should_implement_trait`,
`neg_cmp_op_on_partial_ord`, `erasing_op` (intentional `count * 0 * 8`
sentinel). Verified: before AND after, `cargo test -p kiln-tensor --lib`
passes identically (992 passed, 0 failed); after, `cargo build -p kiln-model`
(downstream consumer) succeeds and clippy JSON confirms exactly the six
target categories disappeared with no new ones; `cargo fmt --check` remains
clean repo-wide; `scripts/check_repository_artifacts.py` passes (6709 tracked
paths); `git status` shows only the six source edits plus this ledger entry.
## Cleanup Agent (round 20) — 2026-08-27

Fixed 5 of the 15 `clippy::needless_range_loop` warnings in
`crates/kiln-tensor` — the carefully-reviewed subset of round 19's leftover
candidate, chosen where index-vs-borrow semantics are provably identical:
(1) `ops/cast.rs` ×3 (`u8_to_f32`, `u8_to_bf16`, `u8_to_f16`): each loop only
used `i` to read `bytes[i]`; rewritten as `for &b in bytes.iter().take(n)`,
preserving the exact n-element read window (and strictly safer if the buffer
were ever shorter). (2) `ops/masked_select.rs`: the first-pass count loop
(`if mb[i] != 0 { count += 1 }`) became a single iterator expression
(`mb.iter().take(n).filter(|&&b| b != 0).count()`); the second-pass copy loop
was left as a range loop since it genuinely needs the index for both src/dst
slices. (3) `ops/normalize.rs`: the write-back loop indexed `row[i]` where
`row` was built immediately above with exactly `last` elements — rewritten as
`for (i, &v) in row.iter().enumerate()`. The remaining 10 warnings were left:
each uses the index both for iteration AND arithmetic (strided byte offsets)
or has a length relationship that would silently change panic semantics if
naively converted. Why it mattered: cut the crate's largest remaining lint
category by a third while keeping every rewrite individually auditable.
Verified: before AND after, `cargo test -p kiln-tensor --lib` passes
identically (992 passed, 0 failed); clippy JSON confirms exactly 5
`needless_range_loop` warnings disappeared (15 → 10) with no new categories;
`cargo build -p kiln-model` (downstream consumer) succeeds;
`cargo fmt --check` remains clean repo-wide; `git status` shows only the three
source edits plus this ledger entry.
## Cleanup Agent (round 21) — 2026-08-28

Repaired the dependency-prebuild layer in `deploy/Dockerfile`, which had gone
totally stale after the workspace grew from 7 crates to 31: the manifest COPY
block and the dummy-lib.rs RUN loop still listed only the original seven
members (kiln-core, kiln-flash-attn, kiln-model, kiln-openenv,
kiln-scheduler, kiln-server, kiln-train), so the layer's `cargo build
--release --locked --features cuda` failed at workspace resolution on the
first missing member (`kiln-blas`) — and the `|| true` swallowed that failure
silently on every image build, meaning the intended dep-cache layer has never
cached anything since the crate explosion. Replaced both blocks with generated
per-crate lines covering ALL 31 workspace members' Cargo.toml + build.rs (from
`git ls-files`), plus a glob-free dummy-source loop over `crates/*/`. Also
removed the dead `CUDA_COMPUTE_CAP` ARG/ENV and rewrote its comment: no code,
build script, or workflow consumes it anymore (`KILN_CUDA_ARCHS` in
kiln-flash-attn/build.rs is the sole CUDA arch control), and its comment still
referenced candle-kernels removed by #1082. Verified BEFORE: replicating the
old 7-manifest layout in /tmp with real Cargo.toml/Cargo.lock fails `cargo
metadata --locked` with "failed to load manifest for workspace member
kiln-blas" (exit 101). Verified AFTER: the new full manifest + dummy lib.rs
layout resolves `cargo metadata --locked` cleanly (exit 0); `docker build
--check -f deploy/Dockerfile .` passes with no warnings; `.dockerignore`
excludes nothing the prebuild needs; docs (README/QUICKSTART) never reference
the removed ARG; worst case if a kernel build script can't run in Docker the
pre-existing `|| true` fallback preserves today's behavior, so no regression
path exists; `scripts/check_repository_artifacts.py` passes; `git status`
shows only the one Dockerfile edit plus this ledger entry.

## Cleanup Agent (round 22) — 2026-08-25

Removed the two remaining dead `CUDA_COMPUTE_CAP: '80'` env entries (plus
their now-obsolete candle-kernels comments) from `.github/workflows/
server-release.yml` — one in the `linux-cuda` job, one in `windows-cuda`.
These fed candle-kernels' build.rs SM detection, which was eliminated when
candle was fully removed (#1082); Round 21 already deleted the matching ARG
from `deploy/Dockerfile`, leaving these workflow env vars unconsumed.
Verified BEFORE: repo-wide grep for `CUDA_COMPUTE_CAP` matched only these two
lines and CLEANUP.md history; Cargo.lock contains zero candle packages and no
crate or build script in the workspace reads the variable (`KILN_CUDA_ARCHS`
in kiln-flash-attn/build.rs is the sole arch control). Verified AFTER: diff is
12 deletions only; all 13 `.github/workflows/*.yml` files parse cleanly with
PyYAML, including the edited file; no other workflow sets or reads the var;
`KILN_CUDA_ARCHS` and `RUSTFLAGS` blocks in both jobs left intact. Noted but
left alone: `deploy/runpod/` script path-reference deep-verification (the
alternate candidate) remains open for a future session.
