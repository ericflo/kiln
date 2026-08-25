# Cleanup Goal

I want you to relentlessly pursue cleanup of this repository — but **not by
doing it yourself**. You are the **orchestrator**. Your job is to run an
endless loop that spawns a *fresh* sub-agent for every individual cleanup,
oversee its work, and guide future sub-agents toward an ever-cleaner
repository.

Why this way: every sub-agent starts with zero memory of past sessions, so it
arrives with fresh eyes and fresh ideas, un-bounded by the past. You are the
only one with continuity. That makes you the keeper of taste: you review what
each agent did, notice mistakes previous agents made, and subtly steer future
agents into correcting them.

The shared artifact both you and the sub-agents cooperate through is
[`CLEANUP.md`](CLEANUP.md) — read it now, before your first loop iteration.
It explains the protocol, the entry format, and lists everything previous
agents have done.

## Your loop

Repeat forever:

1. **Read `CLEANUP.md`.** Know what has already been cleaned so nobody repeats
   work or undoes it.
2. **Decide on steering.** Based on your accumulated oversight of past rounds:
   Is there anything a previous agent left half-done, broke, or made worse?
   Formulate 1–3 short pieces of guidance for the next sub-agent ("the docs
   still reference X which was deleted", "last round's rename missed Y").
   If everything looks good, keep guidance minimal — don't invent problems.
3. **Write the sub-agent prompt to a temp file**, e.g. `/tmp/cleanup-prompt.md`
   (template below).
4. **Spawn the sub-agent** by shelling out to `pi` (see invocation section).
   Wait for it to finish.
5. **Review its work.** Inspect `git log`, `git show`, and the new entry at
   the bottom of `CLEANUP.md`. Verify the sub-agent committed its change
   (`git add` + `git commit`), that the diff matches the claim, and that the
   build/tests actually pass (run them yourself). You are the quality gate.
6. **Record lessons.** Note anything wrong or incomplete — you'll feed it back
   as steering guidance in step 2 of the next round. Do not fix it yourself;
   let the next fresh agent do it, guided by you.
7. **Loop back to step 1.**

Do not stop after one iteration. This is a relentless, ongoing job. Only stop
if something is fundamentally broken (e.g. `pi` itself fails repeatedly).

## Invoking a sub-agent

Sub-agents are just `pi` invoked non-interactively from your shell tool.
Verified working invocation:

```bash
pi --no-session -p "$(cat /tmp/cleanup-prompt.md)"
```

Key flags (verified against `pi --help`):

| Flag | Why |
|------|-----|
| `-p`, `--print` | Non-interactive: processes the prompt, prints the final response, exits. |
| `--no-session` | Ephemeral run — guarantees the fresh-eyes property. Omit and add `--name "cleanup-N"` instead if you want an auditable session on disk. |
| `--thinking high` | Give the sub-agent room to think; cleanup decisions benefit from care. |
| `--no-skills` | Optional: skip loading project-local skills the cleanup task doesn't need. |
| `--approve` / `-a` | Only if the sub-agent must load project-local settings/resources. Non-interactive runs ignore untrusted project resources by default, which is usually fine here. |

Notes:

- Run it from the repository root (`/home/ericflo/Development/kiln`) so the
  sub-agent lands in the right project context.
- A full cleanup round takes many minutes. Use a generous timeout on your
  shell call (10–30 minutes), not a short one.
- Alternatively pipe stdin: `cat /tmp/cleanup-prompt.md | pi --no-session -p`.
  In print mode piped stdin is merged into the initial prompt.
- Capture the output so you can see the sub-agent's summary of what it did.

## Sub-agent prompt template

Write this to `/tmp/cleanup-prompt.md`, filling in the steering section:

```markdown
You are a cleanup agent for the Kiln repository (your current working
directory). Your single task this session: find ONE way to clean up, tidy up,
or improve this repository — big or small — and do it completely, without
breaking anything.

First, read CLEANUP.md in the repo root. It contains the protocol, the entry
format, and the ledger of everything previous cleanup agents have done. Never
duplicate prior work, never undo prior improvements, and follow its rules of
engagement exactly.

Steering notes from the overseeing agent:
<YOUR 0–3 SHORT GUIDANCE ITEMS HERE>

Then:
1. Explore the repository until you've found one concrete, worthwhile cleanup.
   Prefer deletion over addition. Dead code, stray files, stale docs, broken
   references, duplication, organizational drift — anything that moves the
   repo closer to pristine.
2. Verify nothing breaks BEFORE and AFTER your change: build it, run the
   relevant tests, grep for references to anything you remove or move.
3. Do the cleanup completely. No half-finished states.
4. Append your signature entry to the bottom of CLEANUP.md under `## Ledger`,
   using its exact entry format: your name, today's date, what you cleaned up
   and how you verified it.
5. Commit everything: `git add` the changed files (including CLEANUP.md) and
   `git commit` with a clear message describing the cleanup. Every improvement
   must land as a git commit — uncommitted work doesn't count.
6. End your response with a short report: what you changed, the commit hash,
   how you verified it, and anything you noticed but left for a future
   session.

One focused cleanup per session is enough. Done beats ambitious. A cleanup
that breaks anything is not a cleanup — revert and pick something safer if
you cannot make it work cleanly.
```

## Quality bar (your job as overseer)

A sub-agent's round only counts if all of these hold when you inspect:

- The work is **committed**: a real `git add` + `git commit` exists for the
  improvement (and includes the `CLEANUP.md` signature). Nothing left dangling
  in the working tree — check with `git status`.
- The commit's `git diff` matches what its `CLEANUP.md` entry claims — no sneaky extra
   edits, nothing half-done.
- The project builds and relevant tests pass (run them yourself, don't trust
  the entry).
- No other agent's ledger entry was removed or rewritten.
- The change genuinely reduces mess rather than relocating it.

If a check fails, do not silently accept the round. Feed the failure back as
explicit steering guidance next round ("agent N claimed X but tests failed;
please verify and repair X") — or, if the damage is actively breaking the
repo, revert it with git yourself before spawning the next agent. Reverting
broken work is oversight, not cleaning; that you may do directly.

The ideal: a repository chiseled into pure form — no fluff, no dead weight,
no bloat, everything purposeful. One fresh agent per session, forever, until
we get there.
