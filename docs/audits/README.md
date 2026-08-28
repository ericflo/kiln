# Audits

Frozen audit receipts plus local raw evidence from optimization, eval, and
release investigations. Each `.md` receipt documents one hypothesis,
preflight, or eval at a point in time; its verdicts are load-bearing context
for later phase decisions, so re-verify against current code before acting on
them.

## Receipts vs raw evidence

- **Receipts (tracked):** every `.md` file in this directory and its
  subdirectories — audit reports, shortlogs, preflights, verdict summaries.
- **Raw evidence (local-only):** raw run captures, probe dumps, candidate
  request/response JSON, sweep summaries, terminal transcripts, and patch
  diffs. These are ignored by `.gitignore` and untracked from git (wave-1
  docs-tree overhaul); they remain in git history and on local disks.

Policy (`.gitignore`): *"Raw benchmark, serving, metrics, and profiler
output. Retain compact receipts, summaries, manifests, and hashes instead."*

## Raw-evidence homes

- `pr1383-qwen35-base-production-tool-call-eval-2026-05-24/` and
  `pr1383-qwen35-base-production-tool-call-eval-1000-2026-05-25/` —
  per-shard eval result JSON + suite reports for the PR #1383 production
  tool-call evals. Reports:
  `pr1383-qwen35-base-production-tool-call-eval-2026-05-24.md`,
  `pr1383-qwen35-base-production-tool-call-eval-1000-2026-05-25.md`, and
  `pr1383-production-eval-speed-plan-2026-05-26.md`.
- `MACOS_QWEN35_4B_FASTEST_artifacts/` — per-experiment JSON receipts,
  terminal transcripts, and patch diffs from the 2026-05-03 macOS
  Qwen3.5-4B fastest-decode pass. Reports:
  `MACOS_QWEN35_4B_FASTEST_SHORTLOG.md` (reviewed summary) plus the seven
  `e44*_…_summary.md` files inside the artifacts directory.

## Load-bearing reports (cited from code)

- `security-audit-v0.1.md` — kiln-server API threat audit (cited in
  `kiln-server` completions/batch.rs and training_queue.rs).
- `PHASE10_LORA_PRECISION_STUDY.md` — LoRA precision study §5 (cited in
  `kiln-model` lora_loader.rs and `kiln-train` lora_parameters.rs).
- `PHASE10_MODE_B_TRACE.md` — FLCE phase-B trace (cited in
  `kiln-flce-kernel` lib.rs).

Two receipts (`pr1383-qwen35-base-production-tool-call-eval-2026-05-24.md`
and `phase7-h15b-stratified-c29-v2.md`) link raw JSON captures that only
exist locally; on GitHub those links 404 by design.
