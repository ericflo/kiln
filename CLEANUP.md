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
## Cleanup Agent (round 23) — 2026-08-28

Completed the two-rounds-unclaimed steering candidate (a): deep-verified every
path/reference in `deploy/runpod/` against the current repo layout, then fixed
the one defect found. Verified as correct (no action needed): Dockerfile COPY
sources all exist; `.github/workflows/runpod-image.yml` exists, builds from
context `deploy/runpod`, and its heartbeat contract-key assertions
(`uptime_s load_1m gpu0_util_pct workspace_target_mtime build_logs`) all match
keys actually emitted by `kiln-heartbeat.sh`; `scripts/setup-build-cache.sh`
exists and its SCCACHE_VERSION default (v0.9.1) matches the Dockerfile ARG;
the `kiln-bench` binary exists (`kiln-server/src/bench.rs`) and every flag
`kiln-smoke-check.sh` passes (`--model-path --skip-training --latency-only
--max-output-tokens --prompt-tokens --paged`) is parsed by it; the referenced
crate `kiln-gdn-kernel` exists; README version claims (sccache 0.9.1,
PyTorch 2.4.1 cu124) match Dockerfile pins; `kiln-setup.sh`'s self-help range
(`sed -n '2,29p'`) covers exactly its header through the env-file note.
The defect: `kiln-smoke-check.sh`'s `usage()` printed only lines 2–33 of its
header comment, so `-h` truncated the exit-code documentation after code 1 —
codes 2 (known-bad sccache pattern), 3 (unknown failure), 4 (cli misuse), and
124 (timeout), which are precisely what an operator needs mid-triage, were
invisible. Fixed by extending the range to `'2,41p'` (through the end of the
header's cache-policy note), matching how `kiln-setup.sh`'s range ends at its
own header boundary. Why it mattered: the exit codes are the script's API;
`--help` was lying about them by omission.
Verified: `bash -n` passes on the edited script; running `--help` now shows
all six documented exit codes plus the never-modifies-cache note; confirmed no
script, test, or CI step parses or depends on the previous 32-line help output
(workflow only runs heartbeat checks); no other file changed.

## Cleanup Agent (round 24) — 2026-08-27

Audited the `desktop/` directory (never touched by a prior round) and removed
dead code: `installer::discover_latest_version` in `desktop/src/installer.rs`,
a `pub async fn` that no caller ever used — the app's update flow exclusively
calls `discover_latest_version_and_body`, which wraps the same
`discover_asset` call and returns the version plus release notes. The dead
function generated a `dead_code` warning on every `cargo check`/build, and its
doc contract ("Returns None when...") was duplicated across both wrappers.
Folded the distinguishing-cases guidance (`supports_auto_install`) into the
surviving function's doc comment so nothing documented by the deleted stub was
lost. Why it mattered: first desktop audit; this was the crate's only dead-code
warning, and removing it leaves just two pre-existing `deprecated`
`shell().open` warnings (migrating to tauri-plugin-opener would add a new
dependency — out of scope for a cleanup).
Verified: repo-wide grep confirms zero remaining references to
`discover_latest_version` outside git history (the `_and_body` variant is the
only consumer-facing API); `cargo check` before showed 3 warnings including
`function discover_latest_version is never used`, after shows exactly the 2
pre-existing deprecated-method warnings; `cargo test` passes 161/161;
`node scripts/check_desktop_ui_smoke.mjs` and
`node scripts/check_runtime_defaults.mjs` pass unchanged.
## Cleanup Agent (round 25) — 2026-08-25

Deleted the three #1082-era agent-orchestration prompt scratch files in
`scripts/`: `metal_bridge_removal_workflow.js` (130 lines),
`metal_flip_workflow.js` (201), and `metal_gemm_design_workflow.js` (117).
Each was a one-shot multi-agent workflow template (phases + embedded String.raw
prompt text) drafted to drive the candle-removal effort — bridge removal in
metal.rs, kt-native kernel flips, and Metal GEMM backend design. That effort
has fully landed: Cargo.lock contains zero candle packages,
`crates/kiln-model/src/backend/metal.rs` has no `candle_core` surface or
`kt_logits_to_candle` bridges, and kiln-tensor now owns `metal_matmul.rs` plus
a metal_fwd matmul op — so the prompts describe work that no longer exists.
Repo-wide grep found zero references to any of the three filenames outside git
history (not even from the historical bench-results/ docs that reference other
candle-audit scripts, which were therefore left in place as generators of
checked-in artifacts); nothing enumerates or imports scripts/*.js generically.
Why it mattered: 448 lines of completed-effort scratch masquerading as live
scripting surface. Verified before AND after:
`python3 scripts/check_source_parsing_tests.py` (0 tests, 0 reads) and
`python3 scripts/check_repository_artifacts.py` pass with exactly three fewer
tracked paths (6709 → 6706); `scripts/qualification/
validate_retained_evidence.sh` passes untouched (all receipts OK); `git
status` shows only the three deletions plus this ledger entry.
## Cleanup Agent (round 26) — 2026-08-28

Audited this round's steering candidates and fixed the one drift found: the
`deny.toml` header comment claimed CI enforcement ("a GitHub Actions job
running `cargo deny check`") was "a follow-up task", but enforcement has been
live for some time — `.github/workflows/ci.yml` runs `EmbarkStudios/
cargo-deny-action` with `command: check --all-features`, gated to
manifest/lockfile/policy changes by `scripts/ci_rust_scope.py`
(`dependency_policy` output), and CONTRIBUTING.md §131 already instructs
contributors to run `cargo deny check --all-features` locally. Rewrote the
header to point at the actual CI step and the local command instead of a
phantom TODO. Audited the other candidates and left them healthy:
`contracts/` — all four generators (`generate_artifact_schema`,
`generate_eval_schema`, `generate_observability_schema`,
`generate_control_plane_schema`) pass their `--check` modes, plus
`check_config_schema.py` (117 canonical fields) and
`check_http_api_contract.py` (111 paths) all green; `kiln.example.toml` is
schema-validated by the latter checker with zero drift; `rust-toolchain.toml`
(1.96.1 + rustfmt) matches CI's fmt gate; `qualification/` —
`validate_retained_evidence.sh` passes (all receipts OK) and its unittest
suite passes 754 tests (1 skip); a repo-wide relative-link audit over live
root/capability docs surfaced only frozen-archive and generated-file links
(out of scope per protocol). Why it mattered: deny.toml is the dependency-
policy contract; its header misdescribed the project's own enforcement state
and pointed new contributors at work that was already done.
Verified: diff is comment-only (4 insertions, 2 deletions in one comment
block — no TOML keys touched, so no cargo/deny behavior surface);
`python3 scripts/check_repository_artifacts.py` passes (6706 tracked paths);
grep confirms no remaining "follow-up task" text in deny.toml and that
ci.yml's cargo-deny step and CONTRIBUTING.md's instructions match the new
header; THIRD_PARTY_LICENSES.md's license overview cross-checked against the
deny.toml allow list (all nine licenses present); `git status` shows only
deny.toml plus this ledger entry.
## Cleanup Agent (round 27) — 2026-08-29

Deleted `docs/audits/MACOS_QWEN35_4B_FASTEST_LOG.md` — a 20,526-line raw
terminal-session transcript of the 2026-05-03 macOS Qwen3.5-4B optimization
pass (full command inputs + outputs for experiments E001–E0xx). This round's
steering primary candidate. Investigation confirmed it is redundant retained
evidence, not load-bearing: (1) zero references to its filename anywhere
tracked — no doc, script, CI workflow, receipt, or manifest links to it; (2)
the compact reviewed summary already exists alongside it
(`MACOS_QWEN35_4B_FASTEST_SHORTLOG.md`, 2,279 lines covering every experiment
with tests/checks passed), satisfying ARTIFACT_RETENTION.md's requirement that
"a reviewer can understand and validate without replaying an entire terminal
session"; (3) the per-experiment compact JSON receipts in
`MACOS_QWEN35_4B_FASTEST_artifacts/` (982 tracked files) remain untouched as
the digest-level evidence; (4) the raw siblings from that same session
(the `.stderr.log`/`.prom` captures under the same `_artifacts/` dir) were
already purged by `docs/audits/removed-raw-artifacts-2026-07-13-v1.json`,
establishing clear prior intent that this session's RAW output does not belong
in Git — the big transcript log is exactly that category per the policy line
"Do not check in raw server logs ... traces". Git history preserves every byte.
Why it mattered: −20,526 lines, the single largest LOC win available, removing
a raw session dump masquerading as an audit document. Verified BEFORE and
AFTER: `scripts/check_repository_artifacts.py` passes both times (6706 → 6705
tracked paths); `scripts/qualification/validate_retained_evidence.sh` passes
both times (all receipts OK — none hash or locate this file); post-deletion
repo-wide grep finds zero remaining references outside git history and ignored
agent traces; `git status` shows only the deletion plus this ledger entry.

## Cleanup Agent (round 28) — 2026-08-29

Deleted `docs/audits/WSL_CUDA_QWEN35_4B_FASTEST_LOG.md` — the 1,605-line
(75 KB) raw WSL2/CUDA terminal-session transcript of the 2026-05-09
Qwen3.5-4B throughput optimization pass, Round 27's flagged sibling run
through the identical playbook. Confirmed redundant retained evidence, not
load-bearing: (1) zero tracked references — no doc, script, CI workflow,
receipt, or manifest links to its filename; (2) the reviewed compact summary
`WSL_CUDA_QWEN35_4B_FASTEST_SHORTLOG.md` remains alongside it and covers every
outcome of that session (rejected CUDA-graph experiments incl. the
dxgkio/libcuda SIGSEGV findings, accepted fused RMSNorm dispatch with exact
latency/throughput numbers, training smoke result), satisfying
ARTIFACT_RETENTION.md's reviewability requirement; (3) the policy explicitly
forbids checking in raw traces/logs ("Do not check in raw server logs ...
traces"), placing this transcript squarely in the purged category per the
removed-raw-artifacts manifest precedent. Git history preserves every byte.
Why it mattered: −1,605 lines of raw machine output masquerading as an audit
doc, continuing round 27's raw-log purge. Verified BEFORE and AFTER:
`scripts/check_repository_artifacts.py` passes both times (6705 → 6704
tracked paths); `scripts/qualification/validate_retained_evidence.sh` passes
both times (all receipts OK); post-deletion `git grep` for the filename finds
zero remaining references; `git status` shows only the deletion plus this
ledger entry. Noted for future sessions (same playbook applies):
`docs/audits/vulkan-strix-halo-2026-05-09-gpu-decode-log.md` (8,608 lines)
and `docs/audits/vulkan-strix-halo-optimization-log.md` (3,706 lines) are
similar raw session logs whose shortlog siblings
(`vulkan-strix-halo-2026-05-09-gpu-decode-shortlog.md`,
`vulkan-strix-halo-shortlog.md`) exist.

## Cleanup Agent (round 29) — 2026-08-29

Deleted the two remaining raw Vulkan Strix Halo session logs flagged by
round 28, running rounds 27–28's playbook once more: (1)
`docs/audits/vulkan-strix-halo-2026-05-09-gpu-decode-log.md` (8,608 lines —
the 2026-05-09 decode-pass experiment log A001–A0xx) and (2)
`docs/audits/vulkan-strix-halo-optimization-log.md` (3,706 lines — the
2026-05-03 first-class-Vulkan optimization pass E0xx). Confirmed redundant
retained evidence, not load-bearing: zero tracked references to either
filename outside CLEANUP.md itself; both reviewed compact shortlog siblings
remain in place (`vulkan-strix-halo-2026-05-09-gpu-decode-shortlog.md`, 903
lines, per-experiment table with verdicts; `vulkan-strix-halo-shortlog.md`,
135 lines), satisfying ARTIFACT_RETENTION.md's reviewability requirement,
whose policy line "Do not check in raw server logs ... traces" places these
session-scale dumps in the purged category per the removed-raw-artifacts
manifest precedent. Git history preserves every byte. Also updated the
decode shortlog's header: its "the detailed log is <path>" pointer was
rewritten into a provenance note explaining the removal and pointing at the
standing manifest precedent, so no dangling reference remains.
Why it mattered: −12,314 lines of session-scale raw logs masquerading as
audit docs, completing round 27–28's raw-log purge. Verified BEFORE and
AFTER: `scripts/check_repository_artifacts.py` passes both times (6704 →
6702 tracked paths); `scripts/qualification/validate_retained_evidence.sh`
passes both times (exit 0, all receipts OK — none hash or locate these
files); post-deletion `git grep` for both filenames finds zero remaining
references; `git status` shows only the two deletions, the shortlog header
edit, and this ledger entry.
## Cleanup Agent (round 30) — 2026-08-29

Deleted the redundant PR #1383 interim eval checkpoint: the directory
`docs/audits/pr1383-qwen35-base-production-tool-call-eval-1000-2026-05-25-partial/`
(9 tracked JSON files, 37,879 lines / ~1.8 MB) and its superseded checkpoint doc
`pr1383-qwen35-base-production-tool-call-eval-1000-2026-05-25-partial.md`
(123 lines). This was this round's steering primary candidate; the named
`flce_phase_*_raw_2026-04-29.log` files no longer exist anywhere (tracked or
on disk), so exploration pivoted to the nearest live equivalent. Confirmed
redundancy before deleting: a per-file md5sum comparison shows all 9 remaining
partial-dir files are byte-identical to their counterparts in the final
`...-2026-05-25/` evidence directory, which additionally contains shards 09–11
and `aggregate_metrics.json` — i.e. the partial dir is a strict subset whose
every byte survives verbatim in the sibling dir. Its raw log siblings
(`base_eval_shard_*.log`, `trace_suite2.log`, `materialize_errors.log`) were
already purged by the 2026-07-13 cleanup and are recorded only in
`removed-raw-artifacts-2026-07-13-v1.json`, which stays untouched as the
exact-bytes retention locator. The `-partial.md` checkpoint itself declares
"This is not the final result doc. Shards 09–11 remain"; the final audit doc
(`...-2026-05-25.md`) supersedes it with the full 269/775 result and never
references the partial run. Why it mattered: −38k lines of duplicated
interim evidence masquerading as a second eval result. Verified BEFORE and
AFTER: `scripts/check_repository_artifacts.py` passes both times (6702 →
6692 tracked paths); `scripts/qualification/validate_retained_evidence.sh`
passes both times (all receipts OK); post-deletion `git grep` for
`2026-05-25-partial` matches only CLEANUP.md and the standing manifest;
`git status` shows only the ten deletions plus this ledger entry.
## Cleanup Agent (round 31) — 2026-08-30

Ran the rounds 9–12 mechanical-lint playbook over `crates/kiln-server` and
its workspace dependency crates (kiln-eval, kiln-resource, kiln-opd-loss-
kernel, kiln-blas, kiln-core, kiln-scheduler) — surfaces untouched by prior
lint campaigns. Fixed every machine-applicable instance of
`clippy::collapsible_if` (~28 sites, nested `if let`/`if` ladders collapsed
into Rust-2024 let-chains), plus `manual_div_ceil`, `manual_is_multiple_of`,
`manual_contains`, `unnecessary_sort_by` (`sort_by(|a,b| a.cmp(b))` →
`sort_by_key`), `while_let_on_iterator`, and the trivial one-offs clippy's
fixer surfaced: six hand-written `Default` impls replaced by
`#[derive(Default)]` (+`#[default]` on enums), `new_without_default`
(`EvalQueue`/`Metrics` got `impl Default = Self::new()`), and two redundant
borrows on `&format!()`. Net −101 lines. Restored four imports the fixer had
removed as "lib-unused" that are actually used under `#[cfg(test)]`
(`Path`, `VramSource` ×2, `TokenPhaseDurations`) after the test target caught
it. Why it mattered: cut the kiln-server build's clippy diagnostics from 78
to 29 (−63%), leaving only judgment-call categories (needless_range_loop ×10,
too_many_arguments keeps, result_large_err, etc.). Verified: `cargo check -p
kiln-server -p kiln-eval -p kiln-core -p kiln-scheduler` clean; `cargo test
--lib -p kiln-server -p kiln-eval -p kiln-core -p kiln-scheduler` all green
(103+239+12+1189 passed, 0 failed); `cargo check -p kiln-model` (downstream)
clean; `cargo fmt --all --check` clean; clippy JSON confirms zero remaining
target-category warnings; `scripts/check_repository_artifacts.py` passes;
diff audit confirms only mechanical rewrites plus the four import restores.
## Cleanup Agent (round 32) — 2026-08-30

Fixed six stale candle-era claims about the Metal backend in the live root
docs — README.md (×3), QUICKSTART.md (×2), and BENCHMARKS.md (×3 sites in
three blocks). #1082 fully removed candle from the workspace: Cargo.lock
contains zero `candle-core`/`candle-nn`/`candle-metal-kernels` packages, and
the Metal lane now runs on kiln's own substrate (`kiln-tensor/src/metal_rt/`
+ `metal_kernels.rs`, JIT-compiling MSL via objc2-metal's
`new_library_with_source`; attention runs the native fused MLX-style SDPA
MSL kernels, GDN uses native fused dispatches with portable fallbacks). Yet
the docs still said "Metal backend via candle" (README build table), "macOS
drives the candle-metal build" (README desktop section), "runs via the
candle-metal backend" (README phase-history paragraph), "`candle-metal-kernels`
JIT-compiles MSL" (QUICKSTART ×2, BENCHMARKS), "via candle-metal"
(BENCHMARKS macOS section), and "Kiln's Metal backend uses
`candle_nn::ops::sdpa` ... portable candle composition" (BENCHMARKS run
section). All rewritten to describe the current native Metal path; README's
phase-history line keeps its v0.2.0 historical clause with an explicit
"(candle removed in #1082)" note. Historical narrative elsewhere (the
BENCHMARKS broadcast_matmul fix story at line ~552) was left untouched since
it correctly describes the pre-fix candle era. Also audited but left alone:
docs/METAL_INTEGRATION.md and docs/metal-types-objc2-swap-plan-2026-05-28.md
still reference candle APIs — both are dated migration-pattern records whose
text explicitly frames candle as present-at-writing; flagged for a future
decision on archiving them under docs/archive/.
Verified: each corrected fact cross-checked against code before editing
(Cargo.lock package list, kiln-tensor Cargo.toml feature comments,
metal_rt/device.rs + metal_kernels.rs `new_library_with_source`, metal.rs
module surface); repo-wide grep confirms zero remaining stale claims in live
docs; docs-site `--validate-only` passes (59 documents);
`scripts/check_repository_artifacts.py` passes (6692 tracked paths); no code
touched. Diff is 11 insertions / 11 deletions across the three docs.

## Cleanup Agent (round 33) — 2026-08-25

Archived the two candle-era Metal docs flagged by Round 32:
`docs/METAL_INTEGRATION.md` and
`docs/metal-types-objc2-swap-plan-2026-05-28.md` moved to
`docs/archive/metal/` (git mv, history preserved). Both are dated #1082
migration-pattern records whose work is fully landed — candle-core/candle-nn
are completely removed from the workspace and the Metal substrate is
kt-native objc2-metal throughout — so their present-tense candle API
descriptions no longer reflect the codebase. Added a
`docs/archive/metal/README.md` stating the archival rationale and completion
status, and rewrote all 10 sibling links inside the moved plan doc from
`./...` to `../../...` so they resolve to the companion STOP/status docs that
remain under `docs/`. Updated the one live code reference: the rustdoc
pointer in `crates/kiln-tensor/src/metal_storage.rs`
(`metal_sdpa_last_axis`) now cites the archived path. No other inbound
references exist: repo-wide grep for both filenames hits only CLEANUP.md,
git internals, agent traces, and that one code comment; neither file appears
in `docs/site/docs-manifest.json`, scripts, or CI. Verified: scripted link
audit of all three files in their new location resolves every relative link
(0 MISSING); `cargo check -p kiln-tensor --lib` passes with only the
pre-existing `cuda_tag` dead-code warning, identical before and after.
## Cleanup Agent (round 34) — 2026-08-30

Archived all 19 dated candle-era coordination records under `docs/` (the
Round-33 steering primary candidate) into a new `docs/archive/candle-removal/`
directory: CANDLE_REMOVAL_PLAN.md, candle-removal-status-2026-05-28-pm.md,
inject-grad-flip-blocked-2026-05-28.md, all seven issue-1082-* plan/checklist/
roadmap/audit docs, and the eight per-crate STOP/status docs (kiln-flce,
kiln-rmsnorm, kiln-server, kiln-train-deps, kt-tape-substrate, lora-bwd,
metal-cargo-toml, opd-loss, rmsnorm-caller). All are fully landed #1082
records — candle is completely removed from Cargo.lock and every backend runs
kt-native — so none qualified as still-live. Inventory classification: 19 of
19 archived; zero live-doc consumers; inbound references found only in code
comments, two archived Metal docs, and archived PROFILING.md, all rewritten.
Link rewrites: within moved files, the boilerplate "current behavior" links to
./CONFIGURATION.md / ./NATIVE_SFT_PROFILE.md deepened to ../../ (20 links),
and backtick `docs/<member>.md` prose mentions updated to the archive path;
inbound: 12 crate files' comment/doc paths (kiln-autograd, kiln-flce-kernel
incl. its Cargo.toml, kiln-opd-loss-kernel incl. opd_topk_kl.cu ×2,
kiln-rmsnorm-kernel, kiln-tensor/metal_storage.rs ×2, kiln-train ×4),
docs/archive/metal/{metal-types-objc2-swap-plan,README}.md (10 links + 2 prose
paths → ../candle-removal/), and PROFILING.md's three CANDLE_REMOVAL_PLAN
pointers. Added docs/archive/candle-removal/README.md stating the archival
rationale and completion status, mirroring round 33's metal/ precedent.
Why it mattered: ~5,900 lines of completed-effort coordination scratch no
longer masquerade as live top-level docs; docs/ root drops from 60 to 41
entries with the #1082 story consolidated next to its metal/ sibling.
Verified: scripted relative-link audit over all 19 moved files + both edited
archive docs resolves every link (0 MISSING); repo-wide git grep for each
filename finds no un-updated reference outside CLEANUP.md/git internals/
protected .qualification search indices; `cargo check` on all six affected
crates passes with only pre-existing warnings identical to baseline
(cuda_tag dead_code, grpo dead_code); docs-site --validate-only passes
(59 documents — none of these files were ever in docs-manifest.json);
scripts/check_repository_artifacts.py passes; git status shows exactly the
19 renames, 15 modified files, new README, and this ledger entry.

## Cleanup Agent (round 35) — 2026-08-30

Closed two small flagged loops. (1) The pre-existing `cuda_tag` dead_code
warning in `crates/kiln-tensor/src/ops/scalar.rs` (flagged rounds 17 & 33):
investigation showed the method IS used, but only from `cuda_fwd` and
`rocm_fwd`, both behind feature gates — so default builds warn. Fixed by
cfg-gating the method itself with `#[cfg(any(feature = "cuda",
feature = "rocm"))]` plus a comment explaining why, rather than a blanket
`#[allow(dead_code)]`. The gate exactly matches its call sites' cfgs.
(2) The dangling prose pointer in
`docs/archive/candle-removal/CANDLE_REMOVAL_PLAN.md` to
`docs/kiln-tensor-metal-allocator-stop-2026-05-27.md` (flagged round 34),
which was never committed: annotated the list entry as struck-through with a
note that no such file exists in git history, preserving the historical
record without implying a live path.
Why it mattered: kiln-tensor now builds with zero warnings by default, and
the archived plan no longer dangles a pointer at a file that never existed.
Verified: `cargo check -p kiln-tensor --lib` is warning-free after the change
(baseline had exactly this one warning); the `--features cuda` check fails
identically on the stashed baseline due to no local CUDA toolkit
(pre-existing environment limitation, not this change); `cargo test -p
kiln-tensor --lib` passes 992/992; git status shows only these two files
plus this ledger entry.
## Cleanup Agent (round 36) — 2026-08-30

Fixed five stale candle-era comments in live operational files — the CI/build
surface analog of Round 32's doc sweep, which covered README/QUICKSTART/
BENCHMARKS but not these. All are comment-only (no behavioral surface):
(1) workspace `Cargo.toml` default-members comment claimed kiln-kt-bridge has
a "default-on `candle` feature" that candle-free consumers can opt out of —
#1082 removed candle from the crate entirely (`crates/kiln-kt-bridge/Cargo.toml`
itself says "candle fully removed ... kt-native only", no such feature exists);
rewritten to match. (2) `.github/workflows/server-release.yml` MSVC step said
"candle-kernels compiles .cu files with nvcc" — candle-kernels is gone; our own
kernel crates (kiln-blas et al.) do the nvcc compilation now. (3) Same file's
Windows DLL-bundling list justified nvrtc64_120_0.dll as "used by
candle-kernels JIT" — it is actually linked by cudarc's `nvrtc` feature
(enabled in the workspace manifest), so bundling stays correct and the
attribution was fixed. (4) `.github/workflows/perf-regression-nightly.yml`
said the bench's `generic` trainer "dispatches to the BackendRuntime+candle
path" — rewritten to the shared kt-native BackendRuntime path. (5)
`.github/workflows/ci.yml`'s Metal job attributed its `--test-threads=1`
serialization to a panic "in candle-metal 0.10.2's MetalDevice::new"; kept the
serialization but marked the attribution historical (pre-#1082). Verified:
every claim cross-checked before editing (kt-bridge Cargo.toml features,
kiln-blas csrc/*.cu + build.rs nvcc usage, cudarc nvrtc feature in Cargo.toml,
no remaining candle packages in Cargo.lock); all three edited YAML files parse
cleanly with PyYAML; `cargo metadata --locked` resolves cleanly after the
manifest comment edit (comment-only, so no rebuild needed);
`scripts/check_repository_artifacts.py` passes; diff audit confirms every
changed line is inside a comment block.

## Cleanup Agent (round 37) — 2026-08-30

Fixed three stale/false claims in the live perf-gate operational files (the
SFT nightly regression gate) — the same candle-era drift class as rounds
32/36, found by auditing the gate surface end to end. (1)
`.github/workflows/perf-regression-nightly.yml`'s bench-env comment claimed
"`native` asks for cuda_native_sft_train via the env knob the bench reads" —
false twice over: no code reads `KILN_CUDA_NATIVE_TRAINING` anymore
(docs/CONFIGURATION.md marks it an obsolete "legacy CUDA-native selector"),
and `native_route_enabled()` (`crates/kiln-model/src/backend/capability.rs`)
is false on every backend, so both matrix legs run the identical shared
kt-native BackendRuntime path. Rewrote the comment to say exactly that,
keeping the env line itself (harmless) and noting the native/generic rows now
serve as cross-check baselines. (2)
`bench-results/regression/sft_generic_a6000_baseline.json`'s comment still
described the `generic` trainer as "the BackendRuntime+candle-autograd path"
with "`native` routes through `generic` after #1071" — both superseded;
rewritten to match current dispatch reality. (3)
`bench-results/check_sft_train_regression.py`'s usage docstring cited a
nonexistent baseline file `regression/sft_train_a6000_baseline.json`; pointed
at the real `sft_native_a6000_baseline.json`.
Flagged for future sessions, deliberately NOT acted on this round: the §9.9
OPD bench gate is dead post-#1082 — commit 4f04c8a50 deleted
`crates/kiln-opd-loss-kernel/examples/bench_opd_topk_kl.rs`, but
`.github/workflows/opd-bench-gate.yml`'s cuda-bench job still runs that exact
example, `bench-results/check_opd_regression.py` parses its candle-column
output format that nothing produces anymore (`bench_opd_topk_kl_vk.rs` prints
an entirely different format), and `scripts/opd_phase0_pod_validation.sh`
would fail at its opd_kernel_bench phase; only the fake-data gate-self-test
job still passes. Deleting/rewiring contradicts the live grand-plan §9.9
(doc cites both baseline JSONs), so it needs an owner decision (re-wire the
vk example + re-capture baselines vs retire the gate).
Verified: every claim checked against code before editing (grep: zero
readers of KILN_CUDA_NATIVE_TRAINING; capability.rs native_route_enabled
always false; workflow matrix semantics); PyYAML parses the edited workflow;
both regression JSONs parse and `check_sft_train_regression.py` runs clean
end-to-end against them (py_compile + live invocation exercising parse →
null-baseline error path); grep confirms no remaining candle-era claims in
any of the three files; `scripts/check_repository_artifacts.py` passes (6694
tracked paths); diff is comment/docstring-only plus one JSON string value.
## Cleanup Agent (round 38) — 2026-08-30

Fixed the stale candle-era comment cluster in kernel-crate `kt_api.rs`
surfaces (this round's steering candidate a), the largest of which was
kiln-flce-kernel still describing itself as a Phase 7 *prep* surface:
`kt_api.rs` claimed "today's Phase A/B forward+backward run on
`candle_core::Tensor` ops", that the kt entries were only a future
"migration target" for external integrations, and pointed ~17 times across
lib.rs/kt_api.rs/kt_tape.rs at the deleted module
`kiln_train::flce_candle_shim` (its `FlceMatmulProvider`, `shim_envelope_ok`,
the parity oracle, and the `KILN_FLCE_PHASE_A` escape hatch) as if live —
all false post-#1082: kiln-train/src/lib.rs records flce_candle_shim was
deleted and FLCE is kt-native via these very kt entries; no reader of
KILN_FLCE_PHASE_A or FlceMatmulProvider exists outside this crate's own
comments. Rewritten to present-day reality (crate 100% candle-free; kt-typed
entries are the production path) with explicitly historical framing where
the narrative is kept. Same class of fix in kiln-conv1d-kernel/src/kt_api.rs
(header said candle-typed functions "remain in place; Phase 7 deletes them"
and supports_kt doc awaited their deletion — they were already removed per
that crate's own updated lib.rs) and kiln-flash-attn/src/kt_api.rs
(FlashAttnError doc justified its design by "so Phase 7 can delete candle",
which has happened). Comment-only changes; zero code touched.
Verified: every claim cross-checked before editing (grep for
flce_candle_shim/KILN_FLCE_PHASE_A/FlceMatmulProvider repo-wide;
kiln-train imports DEFAULT_CHUNK_SIZE directly from kiln_flce_kernel);
baseline-vs-after `cargo check -p kiln-flce-kernel -p kiln-conv1d-kernel
-p kiln-flash-attn` fails identically (pre-existing environmental cudarc
build-script failure — no local CUDA toolkit, same as round 35's baseline);
comment edits are syntactically valid per `cargo fmt --check` on all three
crates (clean) and full-workspace fmt stays clean;
scripts/check_repository_artifacts.py passes (6694 tracked paths);
remaining flce_candle_shim mentions are all explicitly historical;
git status shows only the five source edits plus this ledger entry.
## Cleanup Agent (round 39) — 2026-08-31

Rewrote `crates/kiln-blas/README.md` — Round 38's flagged leftover and the
last stale candle-era surface in the crate. The README predated Phase 2.1
and #1082 in four ways, all verified against the crate itself before
erediting: (1) the file-layout diagram claimed Cargo.toml "depends on
candle-core (cuda) + half + cc" — actual deps are optional cudarc (the sole
CUDA substrate per the manifest's own #1082 comment), half, kiln-resource,
serde/serde_json, with build-dependency cc; (2) "What it measures" called
cublasGemmEx "the locked-in candle path" and cited a file in the deleted
`vendor/candle-core/` tree (`src/cuda_backend/mod.rs:2625`) as a mirror —
no vendor/candle-core exists anywhere on disk or in git; reframed as the
pre-#1082 baseline dispatch with explicit historical framing; (3) the header
said only the probe ships and "Phase 2 fills in the production matmul path"
— Phase 2.1 already shipped AlgoCache, WorkspacePool, the BackendMatmul
trait, and the feature-gated CublasLtMatmulHandle that kiln-tensor's
cuda_matmul.rs actually dispatches through; header + new "Why the probe is
kept" section now describe the real three-layer split; (4) the layout tree
was missing src/{algo_cache,workspace_pool,backend_matmul,cublaslt_handle}.rs,
tests/cublaslt_handle_smoke.rs, and csrc/cublaslt_matmul.cu entirely.
Comment-only doc rewrite; no code touched, no build possible locally (nvcc)
or needed. Verified: every claim cross-checked before editing (Cargo.toml,
build.rs feature gates + both .cu compiles, src/lib.rs module docs,
kiln-tensor/src/cuda_matmul.rs dispatch); repo-wide grep confirms no
remaining unqualified candle-present claims in the crate's README;
`scripts/check_repository_artifacts.py` passes (6694 tracked paths);
`git status` shows only the README edit plus this ledger entry.
Noted but left alone: cublaslt_probe.cu and examples/cublaslt_mlp_probe.rs
retain their own historical "candle path" comments (accurate as history,
matching this README's reframing); kiln-tensor metal_storage.rs/method_api.rs/
operators.rs still cite deleted vendor/candle-core paths in comments —
same class as round 33's precedent, left for a future session.

## Cleanup Agent (round 40) — 2026-08-31

Rewrote three present-tense citations of the deleted `vendor/candle-core/`
tree in kiln-tensor source comments — Round 39's flagged leftover: (1)
`crates/kiln-tensor/src/metal_storage.rs` cited `vendor/candle-core/src/
metal_backend`'s `RESOURCE_OPTIONS` constant as if on disk; reframed as
candle-core's upstream `metal_backend`, with the pre-#1082 vendor tree
named explicitly as history; (2) `crates/kiln-tensor/src/method_api.rs`
claimed method signatures "are matched against `vendor/candle-core/src/
tensor.rs`"; reframed to candle-core's upstream `tensor.rs` API surface
captured by that vendor snapshot; (3) `crates/kiln-tensor/src/operators.rs`
said "`vendor/candle-core/src/tensor.rs` defines a `bin_trait!` macro";
reframed as the upstream file captured pre-#1082. Comment-only changes;
zero code touched. Verified: baseline-vs-after `cargo test -p kiln-tensor
--lib` identical (992 passed, 0 failed both sides); repo-wide grep shows no
remaining vendor-path citations outside CLEANUP.md itself, the explicitly
historical cublaslt_probe.cu header (kept per round 39), and two phase-7
guard scripts whose vendor-path string matching is their own concern.
Noted but left alone: `scripts/audit-candle-usage.sh` still documents an
excluded vendored tree and `scripts/phase7-migrate-candle-bail.py` matches
on `vendor/candle-core` paths — inert post-deletion, candidate for a future
session.

## Cleanup Agent (round 41) — 2026-09-01

Archived `docs/vk-harmonization/` (12 files, ~5,079 lines — this round's
steering primary candidate a) to `docs/archive/vk-harmonization/` following
the rounds 33–34 playbook. Confirmed the archive trigger before moving: the
entire PR1–PR7 series landed on main via PR #1441 (`feat/vk-tape-harmonization`),
including PR6 (`3b226d620`, orchestration flip routing Vulkan SFT/GRPO/OPD
through shared trainer.rs/opd.rs) and PR7 (`a909d46ff`, deletion of the legacy
fork `vk_train.rs`/`vk_forward.rs`/server opt-out family — verified gone from
the tree), so the specs, test scaffolds, review notes, and soak handoff are a
completed-effort coordination record whose present-tense "SPEC ONLY — not
implemented" statuses no longer describe reality. Moved with `git mv`
(history preserved); added an archive README stating the rationale and
completion status (mirroring round 33's metal/ and round 34's candle-removal/
precedents). Link fixes: the one relative markdown link in PR4-spec.md
deepened to `../../`; three in-archive prose `docs/vk-harmonization/...`
mentions updated to the archived path; both inbound live references rewritten
(`crates/kiln-model/tests/vk_tape_record_proof.rs:326` comment, now noting
PR1–PR7 landed via #1441, and `docs/vulkan-train-harmonization-plan.md:279`,
which stays live as the authoritative plan). Why it mattered: ~5k lines of
landed-effort planning no longer masquerade as pending work at docs/ root.
Verified: scripted relative-link audit over all moved markdown files resolves
every link (0 MISSING); repo-wide git grep finds no un-updated
`vk-harmonization` reference outside CLEANUP.md, the plan doc's updated
pointer, and the archived dir itself; `cargo fmt --check -p kiln-model` clean
(the only .rs change is a comment); scripts/check_repository_artifacts.py
passes (6694 tracked paths, same count — moves not deletions); git status
shows exactly the 11 renames, new README, two reference edits, and this
ledger entry.

## Cleanup Agent (round 42) — 2026-09-02

Archived `docs/vulkan-train-harmonization-plan.md` (318 lines — Round 41's
flagged leftover, this round's steering primary candidate) to
`docs/archive/vk-harmonization/vulkan-train-harmonization-plan.md`, completing
the vk-harmonization consolidation started last round. Confirmed the archive
trigger before moving: the full PR1–PR7 series is landed on main via PR #1441,
including PR6 (`3b226d620`, orchestration flip) and PR7 (`a909d46ff`, deletion
of the legacy fork — `crates/kiln-train/src/vk_train.rs` and
`crates/kiln-model/src/vk_forward.rs` verified gone from the tree), so the
plan's present-tense "PR1/PR2 implemented; PR3–PR7 specced" status and its
pre-landing file:line reference index no longer describe reality. Moved with
`git mv`; added a landed-status banner at the top of the plan (explicitly
marking its file:line references historical) and updated the archive README's
closing pointer, which had called the plan "authoritative" and live. Link
fixes: all five inbound references were inside the archive itself — the four
spec docs' parent-plan pointers rewritten from `docs/...` / `../../...`
prose/link forms to sibling-relative paths, and README.md now links the plan
as an archived record. Zero references outside the archive remain (repo-wide
git grep), and the file was never in docs/site/docs-manifest.json.
Why it mattered: the last live surface of a fully landed effort no longer
masquerades as pending design work at the docs/ root; the whole #1082 Vulkan
harmonization story now lives in one directory.
Verified: scripted check that every rewritten link target exists in
`docs/archive/vk-harmonization/` (all resolve); repo-wide git grep for
`vulkan-train-harmonization-plan` matches only the archived dir;
`scripts/check_repository_artifacts.py` passes (6695 tracked paths, same count
— move not deletion); no code touched, no cargo checks needed;
`git status` shows exactly the rename, five archive edits, and this ledger
entry.
## Cleanup Agent (round 43) — 2026-09-02

Fixed a false-negative in `scripts/audit-substrate-status.sh` discovered while
evaluating this round's steering candidates: the Phase 0.3 row ("determinism
stance (PROFILING.md section)") probed for `PROFILING.md` at the repo root,
but Round 2 relocated that file to `docs/archive/profiling/PROFILING.md` — so
the dashboard has reported "64/65 deliverables shipped" with Phase 0.3 as
`[ ] todo` ever since, even though the deliverable shipped long ago. Updated
the row's path to the archived location with a comment explaining why, and
dropped its stale trailing comment claiming a "deeper check via grep below"
that does not exist anywhere in the script. Steering candidates audited and
left alone: (a) `docs/audits/pr1383-qwen35-base-production-tool-call-
eval-1000-2026-05-25/` is genuinely protected — its 13 files are cited by the
final audit doc `...-eval-1000-2026-05-25.md`, which survived Round 30's
partial-dir deletion; (b) both `bench-results/regression/*.json` baselines
are consumed by `check_sft_train_regression.py` (Round 37 already fixed the
docstring drift) — no stale entries; (c) desktop/ deep audit: `cargo check`
shows only the two documented pre-existing tauri-plugin-shell deprecations,
no dead code surfaced.
Why it mattered: the substrate dashboard's whole point is an accurate
shipped/todo census; it has been lying by one row since the Round 2 move.
Verified BEFORE and AFTER: human mode went from "64/65" with exactly one todo
row (0.3) to "65/65 deliverables shipped"; `--json` mode's unshipped list
went from `[phase 0.3]` to empty with all other rows byte-identical;
`--help` still works; `bash -n` passes; git status shows only the script edit
plus this ledger entry.

## Cleanup Agent (round 44) — 2026-09-02

Deleted the orphaned candle-era Phase 7 migration-tooling trio (this round's
steering primary candidate): `bench-results/phase7-removal-plan.md`
(36 lines — a plan citing 1,845 candle call sites that no longer exist), its
generator `scripts/phase7-candle-removal-plan.py` (215 lines), and the
closely-coupled `scripts/phase7-migrate-candle-bail.py` (a dry-run rewriter
for a 493-site `candle_core::bail!` migration that already happened; Round 40
had flagged it inert). Orphan status confirmed: #1082 fully removed candle,
and repo-wide grep over all tracked files showed the trio's only citers were
each other plus one hand-written prose bullet in
`bench-results/substrate-status.md`; neither file appears in
`audit-substrate-status.sh`'s probe ROWS (Phase 0.1 checks only
`audit-candle-usage.sh` + `candle-api-surface.csv`, which stay untouched per
the standing directive) or in any script/CI/doc surface; remaining grep hits
are ignored agent traces and frozen eval-trace JSONs. The substrate-status.md
bullet was rewritten to explicitly historical framing ("served the #1082
candle removal and were deleted once candle was fully removed; see git
history") so no live pointer dangles.
Why it mattered: −251 lines of completed-effort migration tooling whose own
plan described a codebase state that no longer exists.
Verified BEFORE and AFTER: `bash scripts/audit-substrate-status.sh` reports
65/65 deliverables shipped both times (human + --json modes);
`scripts/check_repository_artifacts.py` passes both times (6692 → 6689
tracked paths); post-deletion `git grep` for all three filenames matches
nothing outside CLEANUP.md; `git status` shows exactly the three deletions,
one doc edit, and this ledger entry.

## Cleanup Agent (round 45) — 2026-09-02

Fixed CLI-flag drift in the never-before-audited live docs `docs/VIGNETTES.md`
and `docs/TRAJECTORY_TURN_THROUGHPUT.md`. Vignette 2's §10.14 five-command
pipeline documented flags that do not exist on the actual `kiln` CLI:
`kiln judge distill --sessions/--judge-name/--rank` (real: `--url`, `--name`,
`--teacher`), `kiln self-improve --sessions/--output-name/--post-eval` (real:
`--url`, `--agent`, `--judge`, `--no-crisp`; no eval flag, sessions come from
the pi capture dir automatically), and `kiln judge drift-check
--sample-size 50` (real: `--url/--judge/--teacher`; sample size is not a
flag). All three command invocations rewritten to the real surface with prose
adjusted to match the CLI/API help text (`--kiln-url` in step 1 is a genuine
alias for `--url` and stays). Separately, TRAJECTORY_TURN_THROUGHPUT.md said
"all eight batching values are immutable" — there are nine per
`crates/kiln-server/src/config.rs`'s BatchingConfig + rendezvous family and
QUICKSTART.md's own table; corrected to nine. Audited and left healthy: the
rest of VIGNETTES.md (endpoints /v1/teachers, /v1/recipes/run,
/v1/distill/pump, /v1/adapters/distill_merge, /v1/library/publish/{name},
/v1/adapters/{name}/receipt all verified present; frontier-pump recipe exists;
drift-check's honest-501 note matches ApiError::drift_check_not_implemented)
and LONG_CONTEXT_GRPO_BENCH.md (example binary + all cited flags and record
fields exist).
Why it mattered: VIGNETTES is the runnable recipe doc for the grand plan's
closing vignettes; every Bob-vignette command would have failed at argument
parsing as written.
Verified: every rewritten flag cross-checked against `crates/kiln-server/
src/cli.rs` (JudgeCommands, SelfImprove, DriftCheck variants) and
`run_judge`/`run_self_improve` request bodies before editing; docs-site
`--validate-only` passes (59 documents); repo grep confirms no remaining
phantom flags outside frozen plan docs; no code touched; `git status` shows
only the two doc edits plus this ledger entry.

## Cleanup Agent (round 46) — 2026-09-02

Audited the never-before-touched `desktop/` Tauri crate for unused
dependencies, dead config keys, and UI↔backend command drift, per this
round's steering candidate (b). Findings: all ten tauri plugins in Cargo.toml
are registered in main.rs and permissioned in capabilities/default.json; every
`tauri.conf.json` key is live; every one of the 30 `invoke("…")` calls across
the six ui/*.html windows resolves to a command registered in the
`generate_handler![]` list; CHANGELOG.md / Cargo.toml / tauri.conf.json all
agree on version 0.2.16; flate2/tar/sha2/semver/toml/reqwest are all genuinely
used by installer/settings/hf_download. One real find: the `dirs = "5"`
dependency was declared but referenced nowhere in desktop/src, build.rs, or
the capabilities config — removed it from `desktop/Cargo.toml`; `cargo check`
regenerated `desktop/Cargo.lock`, dropping the orphaned dirs/dirs-sys chain
(−88 lines).
Why it mattered: an unused dependency silently ships its full platform
backend (dirs-sys, plus the only remaining reason for that windows-sys entry)
into every desktop build and lockfile audit.
Verified: baseline `cargo check --tests` passed before the change; after,
`cargo check --tests` passes with identical warnings and `cargo test`
passes 161/161 in the desktop workspace; `git grep dirs` finds no remaining
references outside the lockfile history.

## Cleanup Agent (round 47) — 2026-09-02

Removed the dead Tauri command `open_kiln_ui_in_browser` from `desktop/src/
main.rs` (~26 lines) and its entry in the `generate_handler![]` registration
list, plus the paired dead event emit `menu://open-in-browser` in `desktop/src/
tray.rs`'s ITEM_OPEN_IN_BROWSER handler (including its now-unused
`app_for_emit` clone). This resolves Round 46's candidate (a), with a key
correction to its premise: of the three flagged commands, `path_info` and
`fetch_update_sha256` are NOT dead — both are invoked from
`desktop/ui/settings.html` (lines 864 and 1683 respectively). Only
`open_kiln_ui_in_browser` had zero callers: no UI page invokes it, no test
references it, and the tray's open-in-browser menu item uses its own private
`tray::open_kiln_ui_in_default_browser`, making the command an unreachable
duplicate path. The `menu://open-in-browser` emit it accompanied fired an
event with zero listeners anywhere in any UI window (verified against every
`listen(` call across all six ui/*.html files). Why it mattered: a registered
command is public IPC surface reachable by any compromised/buggy webview;
deleting it shrinks the attack surface while removing dead code.
Verified: exhaustive invoke-name census over all six ui/*.html windows lists
30 commands, none named `open_kiln_ui_in_browser`; baseline before the change
was `cargo check --tests` clean with 2 warnings and `cargo test` 161/161; after,
`cargo check --tests` clean with exactly 1 warning (the pre-existing
deprecated `shell().open` in tray.rs — the removed command's own duplicate
deprecated-call warning disappears as a bonus), `cargo test` still 161/161;
`node scripts/check_desktop_ui_smoke.mjs` and
`node scripts/check_runtime_defaults.mjs` pass unchanged; repo-wide grep finds
zero remaining references to either removed identifier. Noted for a future
session: tray.rs also emits `menu://view-logs`, which likewise has no listener
in any UI page — left alone this round to keep scope tight.

## Cleanup Agent (round 48) — 2026-08-25

Removed the dead `menu://view-logs` tray emit in `desktop/src/tray.rs`
(Round 47's flagged candidate). The `ITEM_LOGS` menu branch already opens
the logs window directly via `open_logs_window`; the accompanying
`app_handle.emit("menu://view-logs", ())` fired an event with zero listeners:
an exhaustive census of every `listen(` / event-subscription call across all
Rust sources and every ui/*.html window found the only `menu://` listener in
the entire codebase is `menu://check-updates` in dashboard.html. Verified:
`cargo check` on kiln-desktop passes before and after with identical output
(the single pre-existing deprecated `shell().open` warning is unrelated);
repo-wide grep finds no remaining reference to `view-logs`; the `emit` import
in tray.rs remains used by other live emits (`open-dashboard`,
`open-settings`, `check-updates`). No other code touched.

## Cleanup Agent (round 49) — 2026-09-02

Removed 18 unused crate dependencies across 11 workspace manifests, found by
an exhaustive scan (`cargo metadata --no-deps` dependency names vs. every
`.rs` file in each crate including `build.rs`, hyphen→underscore normalized):
`chrono` (kiln-core), `memmap2` (kiln-tensor), `half` (kiln-rocblas),
`thiserror` (kiln-autograd, kiln-graph-metal, kiln-graph-cuda,
kiln-graph-vulkan, kiln-param, kiln-scheduler), and `kiln-tensor` + `half` +
`serde` + `serde_json` (kiln-mps and kiln-vulkan-blas). Every flagged name had
zero occurrences anywhere in its crate's sources — the kiln-mps /
kiln-vulkan-blas entries are Phase 2.2/2.3 scaffolding deps pre-declared for
code that never landed, the same pattern as the #1082 candle-core removal
documented in those manifests. Regenerated Cargo.lock (−17 lines; no external
packages dropped since other crates still use them). Why it mattered: unused
deps inflate compile times and lockfile audits and mislead readers about what
each crate actually needs. Verified: baseline `cargo check` passed on all 11
crates (plus `--features probe` for kiln-mps and `--features vulkan` for
kiln-vulkan-blas) before the change, identical result after; `cargo test`
passes on kiln-core (103+0f), kiln-tensor (992+0f), kiln-autograd (272+0f),
kiln-param (66+0f), kiln-scheduler (12+0f), kiln-rocblas (23+0f);
`cargo check -p kiln-server` (largest dependents) passes before and after with
identical output (one pre-existing warning); full `--workspace` check fails
only inside cudarc's build script (`nvcc` not installed) — reproduced
identically on a pristine stash, i.e. environmental, not caused by this
change.
## Cleanup Agent (round 50) — 2026-09-02

Repaired the drifted `contracts/production-file-budget-v1.json` exception ceilings for the five kiln-server files that round 31's clippy campaign (commit `fed6f8e92`) resized without updating the contract — found by running this round's steering candidate (a), a file-by-file contracts/ audit beyond the four generators already verified in round 26, which surfaced `scripts/check_production_file_budget.py` failing on the pre-change baseline. The policy demands *exact* reviewed ceilings (any headroom or exceedance is an error), so five entries were stale: api/training.rs 6613→6612, cli.rs 6331→6329, config.rs 11313→11252 (all three shrank from lint rewrites) and state.rs 8384→8387, training_queue.rs 7960→7962 (both grew past their caps). All other contract checkers were run first to scope the audit: runtime-env-direct-reads, source-parsing-test-inventory, OpenEnv contract, config schema (117 fields), thinking-budget schema+conformance, and runtime-defaults all pass; all 15 contracts/ files have live consumers. Why it mattered: `.github/workflows/repository-hygiene.yml` runs this checker, so CI's hygiene gate was red purely due to un-maintained ceiling numbers after an otherwise-clean lint campaign.
Verified BEFORE: checker failed with exactly the five findings above on the untouched tree. AFTER: `check_production_file_budget.py` passes (647 files, 5000-line default, 14 reviewed exceptions); the checker's own unittest suite (`test_production_file_budget.py`, 6 tests) passes; `scripts/check_repository_artifacts.py` passes unchanged (6692 tracked paths); diff is exactly 5 max_lines values, rationales untouched; no code touched.

## Cleanup Agent (round 51) — 2026-09-02

Repaired a red `cargo clippy` baseline: `cargo clippy -p kiln-server` (and any
build that compiles kiln-tensor under clippy) failed on the untouched tree with
a deny-by-default `erasing_op` error at
`crates/kiln-tensor/src/ops/nonzero.rs:64` — `vec![0u8; count * 0 * 8]` in the
rank-0 scalar branch, an expression that always yields zero bytes. Rewrote it
as `Vec::new()` and removed the adjacent dead statement
`if is_nonzero(dtype, &cpu, 0)? { /* empty body */ }`, which was a redundant
second predicate evaluation whose result was discarded. Added two scalar-path
tests (`nonzero_scalar_zero` → shape [0,0], `nonzero_scalar_nonzero` → shape
[1,0]) since the rank-0 branch previously had zero coverage; expectations match
the branch's documented contract (shape [N, 0] with N∈{0,1}, empty flat data).
Why it mattered: clippy is deny-on-error for this lint class, so every clippy
run touching kiln-tensor was failing before any of this session's work;
the dead `if` also doubled an allocation-free predicate call for no reason.
Verified BEFORE: baseline `cargo test -p kiln-tensor --lib` 992 passed /
0 failed; baseline `cargo clippy -p kiln-server` errored exactly once with
`erasing_op` at nonzero.rs:64. AFTER: lib tests 994 passed / 0 failed (incl.
the two new tests), `cargo test -p kiln-tensor --test new_ops_parity`
19 passed, `cargo clippy -p kiln-server` completes with zero errors (only the
pre-existing judgment-call warnings remain), `check_production_file_budget.py`
passes (647 files; nonzero.rs is not an exception file),
`check_repository_artifacts.py` passes (6692 tracked paths). Noted but left:
~29 remaining clippy warnings across kiln-tensor/kiln-eval/kiln-core/
kiln-memory/opd-loss build script are all judgment-call categories
(too-many-arguments, needless_range_loop in hand-rolled kernels, etc.) —
steering round (b) reviewed them; none are safe-mechanical wins.

## Cleanup Agent (round 52) — 2026-09-03

Purged the last candle-era references from the live build-cache operator
tooling and fixed a stale pointer in `.gitignore`. (1) `scripts/setup-build-cache.sh`'s restore-count `find` still matched `-name 'candle-flash-attn-*'` alongside `kiln-flash-attn-*`; the external candle-flash-attn dependency was removed by #1082 (Cargo.lock has zero candle packages, and every build dir under `target/release/build/` is named `kiln-flash-attn-<hash>`), so the branch could only ever match nothing. Dropped it. (2) `scripts/push-build-cache.sh` — the producer half of the same B2 flash-attn artifact cache that setup pulls (`build-cache/kiln/${ARCH}/artifacts/flash-attn/`) — had the same dual-pattern `find` plus a comment claiming support for "candle-flash-attn (external dep)"; pattern dropped and comment replaced with an explicit historical note citing #1082. Both scripts stay in the repo: push-build-cache.sh is unreferenced by CI/docs but is the intentional manual counterpart to setup-build-cache.sh's live pull path (used by deploy/runpod per round 23). (3) `.gitignore`'s node-tooling comment pointed at `scripts/capture-screenshots.mjs`, which does not exist; corrected to the actual file `scripts/capture-desktop-screenshots.mjs`.
Why it mattered: the two scripts are the documented RunPod cold-build path; both still described a dependency that no longer exists, and the .gitignore pointer sent readers to a nonexistent script.
Verified BEFORE: `bash -n` on both scripts; baseline find with old dual pattern vs new single pattern over the real `target/release/build/` returns byte-identical directory lists (8 kiln-flash-attn dirs), proving no behavioral change today; `git status` clean. Verified AFTER: `bash -n` passes on both scripts; find-equivalence check still holds; restored-file count logic exercises cleanly (36 files); repo-wide grep confirms the only remaining `candle-flash-attn` mention is the intentional historical note; `.gitignore` target filename exists as tracked path; `scripts/check_repository_artifacts.py` passes; diff is exactly 3 files, comment/pattern-only.
## Cleanup Agent (round 53) — 2026-09-03

Repaired a false positive in `scripts/check_docs_site_smoke.mjs` that made the
Pages CI's own smoke checker unrunnable against the committed source site: any
static hub page linking into the build-generated `docs/<slug>/` tree (e.g. the
`<a href="docs/">Documentation</a>` nav entry present in all eight top-level
pages, plus per-page `docs/<slug>/` deep links) failed with "broken local href"
because `docs/site/docs/` only exists after `scripts/docs-site/build.mjs`
emits it — so `node scripts/check_docs_site_smoke.mjs` exited 1 on an untouched
tree before ever reaching its Chromium stage. Fixed by adding a generated-target
branch to `validateDocsSiteLocalLinks`: when a resolved target is missing, lies
under `<siteRoot>/docs/`, and `KILN_DOCS_REQUIRE_GENERATED` is not set (i.e.
we're validating source, not a built site), the reference is validated against
`docs/site/docs-manifest.json` instead of the filesystem — the hub link passes,
`docs/<slug>/` must match a manifest slug, and `#fragment` anchors into Markdown
documents are verified against `slugifyHeading`-derived heading IDs of the
document's committed `.md` source (imported from `scripts/docs-site/lib.mjs`,
side-effect free). Built-site CI runs (`KILN_DOCS_SITE_ROOT` +
`KILN_DOCS_REQUIRE_GENERATED=true`) are unchanged: targets exist there, so the
existing filesystem path handles them as before. Deliberate bad-reference
injection tests confirm the new branch still catches unknown slugs and wrong
anchors ("no manifest document with slug nope", "missing heading #zzz-wrong in
docs/NATIVE_SFT_PROFILE.md").
Why it mattered: the checker guarding the published site could only be run in
CI against a prebuilt tree; locally it reported a healthy site as broken, and
anyone adding a legitimate docs/ link got a false failure with no way to pass.
Verified BEFORE: static-only run fails exactly on `api.html: broken local href
docs/`; scripted sweep shows all 206 raw missing-target hrefs resolve into
generated `docs/site/docs/` (zero genuine broken non-generated links). AFTER:
`KILN_DOCS_SMOKE_STATIC_ONLY=true node scripts/check_docs_site_smoke.mjs` exits
0; built-site mode also exits 0 after a real `/tmp/site-build` build; negative
injection tests both fail with the new precise messages; injected edits
reverted (`git diff` clean on api.html); `npm test --prefix scripts/docs-site`
passes 11/11 (lib.mjs import surface intact); full smoke still blocked locally
only by absent Chromium, identical to baseline. Diff is one file,
43 insertions / 3 deletions.

## Cleanup Agent (round 54) — 2026-09-03

Removed the stale "TODO Phase 4" from `crates/kiln-vulkan-kernel/src/
vk_ops/conv1d.rs` — this round's steering candidate (c), a TODO/FIXME audit of
crates/*/src. The module doc claimed the autograd-aware training path
`vk_causal_conv1d_pre_silu_no_grad` was still to be added ("is added
(TODO Phase 4). For now we expose:"), but the work landed long ago: the
function is implemented in the same file and `vk_causal_conv1d` dispatches to
it whenever any input requires grad, making it the production GDN training
path. The header's "For now we expose" list also omitted both live functions.
Rewrote the comment to present tense listing all three exported functions;
comment-only, zero code touched. Other TODO candidates audited and left: the
~20 `TODO(#1082, phase 4 Metal/Vulkan): implement` fallback stubs in
kiln-tensor/src/ops/* are genuinely unimplemented backend methods (each body
still returns None/errors), and model_dispatch.rs's phase-2 continuous-batching
graph-capture TODO matches reality (batch-1-only capture).
Why it mattered: the TODO told readers a shipped training path was missing.
Verified: `cargo fmt --check -p kiln-vulkan-kernel` clean;
`cargo test -p kiln-vulkan-kernel --lib` passes 65/65 (first run hit the
documented ~2-in-45 baseline flake with 3 failures; re-run green);
grep confirms no remaining "TODO Phase 4" anywhere in crates/;
`git status` shows only the one source edit plus this ledger entry.
