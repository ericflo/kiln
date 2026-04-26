# Audits & Preflights

Frozen audit and preflight reports. Each file documents a single hypothesis or
preflight check at a specific point in time — the verdicts here are
load-bearing for later phase decisions but the surrounding code may have moved
on. Read these to understand *why* a decision was made, then verify the
current state against the code before acting on it.

## Phase 10 (Liger-Kernel port)

- `PHASE10_LIGER_AUDIT.md` — Liger kernel port audit; ranks fused linear
  cross-entropy as the top win, kills RMSNorm/LayerNorm as duplicates/absent.
- `PHASE10_FLCE_PREFLIGHT.md` — VRAM measurement preflight gating the FLCE
  port; `(a+b+c)/peak_VRAM ≥ 30%` at T=16384 → GREEN.

## MTP (Phase B — multi-token prediction debug)

- `MTP_PHASE_B12.md` — layer-31 drift bisect; verdict: drift is benign bf16
  accumulation, not a correctness bug.

## Phase 7 (DX + perf-frontier closeout)

MTP α-acceptance investigation:

- `phase7-mtp-acceptance-state-of-play.md` — state-of-play across the
  H-hypothesis chain; H-table, current α, decision tree.
- `phase7-h15a-marlin-determinism.md` — H15a: Marlin pack determinism
  correlation with α drift.
- `phase7-h15b-stratified-c29-v2.md` — H15b: stratified C29 v2 reject-row
  probe.
- `phase7-h15c-vllm-alpha-microbench.md` — H15c: vLLM α microbench as
  external-reference upper bound vs kiln (`vllm_mtp_unsupported`).
- `phase7-h16-external-alpha-options-audit.md` — H16: doc-only enumeration of
  8 external-α reference candidates with pre-registered decision rule.
- `phase7-h17-sglang-alpha-microbench.md` — H17: SGLang α microbench (class
  exists and dispatches, fails native runtime on A6000).
- `phase7-h17b-vllm-020-alpha-microbench.md` — H17b: vLLM v0.20.0 α microbench
  retest (free pre-step).
- `phase7-h18-hf-transformers-alpha-reference.md` — H18: hand-rolled HF
  transformers MTP α reference vs kiln.

Prefix cache:

- `phase7-prefix-cache-reuse-ab.md` — real prefix-cache reuse A/B
  measurement; flat cache, append-only suffix variants.
- `phase7-sglang-radix-audit.md` — kiln radix prefix cache vs SGLang
  RadixAttention; structural comparison and what kiln is missing.

## GDN (Gated DeltaNet kernel)

- `gdn-vllm-audit.md` — vLLM `fused_recurrent_gated_delta_rule` vs kiln
  `kiln-gdn-kernel`; verdict: no portable bounded micro-port win for kiln on
  A6000 under CUDA graphs.

## Phase 9 (release prep)

- `security-audit-v0.1.md` — first formal security audit of the kiln-server
  HTTP API surface (adapters, training, completions, health, metrics) against
  twelve threat classes; 1 HIGH (adapter delete/load name validation), 3
  MEDIUM (queue cap, composition stack cap, default listen), 4 LOW, 4 NONE.
