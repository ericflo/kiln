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
