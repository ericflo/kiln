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
