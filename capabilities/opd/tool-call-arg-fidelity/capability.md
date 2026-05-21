# Capability: tool-call-arg-fidelity

## Description
A coding agent that decides to call a tool (file_read, run_bash, web_search,
list_files, etc.) must emit a JSON tool-call object whose arguments are
well-formed and conform to the tool's schema. The 4B today often produces
JSON with subtle defects: a missing required field, a stringified int, an
extra "ignore_errors" key the tool doesn't accept, or a syntax error from
a trailing comma. Downstream the agent harness rejects these calls.

This is structured-output adjacent — same family as code-symbol-extraction
(capability #2, hit 97% skip-rate ceiling). Mitigation: use rollout
max_tokens=384 (well over typical tool-call length of ~60 tokens) so OPD
has loss landscape room per the rollout-sizing finding from capability #3.

Concrete failure modes the 4B exhibits:
- JSON syntax errors (trailing comma, missing brace, single quotes)
- Missing required fields (`file_read` call without `path`)
- Wrong types (`"offset": "10"` instead of `"offset": 10`)
- Hallucinated extras (`"ignore_errors": true` on a tool with no such arg)
- Sometimes wraps the tool call in prose ("I'll call file_read like so: {...}")

## Base model
Qwen3.5-4B (kiln serve on http://localhost:8420)

## Teacher
`vllm/qwen3.6-27b-awq` at http://localhost:8002 (AWQ-INT4, max_logprobs=64)

## Rubric

| Sub-score | Weight | What it measures |
|-----------|--------|-------------------|
| `parses` | 0.10 | Strict JSON.loads succeeds on the response (or first JSON block extracted). Low weight; guards the floor. |
| `required_fields` | 0.40 | Of the tool's required fields, what fraction are present. **Target sub-score.** |
| `type_correctness` | 0.30 | Of the fields the call includes, what fraction have the right type per schema. |
| `no_extra_fields` | 0.20 | 1.0 if no fields outside the schema; linearly penalized. Catches hallucinated args. |

Composite = `0.10 × parses + 0.40 × required_fields + 0.30 × type_correctness + 0.20 × no_extra_fields`.
Direction: higher is better.

Each eval prompt declares the tool schema in the system prompt (so rubric
and model share the contract). The rubric parses the first JSON object,
scoring each sub-score independently — so a response that fails `parses`
can still partially score via best-effort extraction.

## Baseline (filled by headroom.py after iter 0)
| Sub-score | Weight | Baseline | Headroom (w×(1−b)) |
|-----------|--------|----------|---------------------|
|           |        |          |                     |
| **Total** |        |          | **<sum>**           |

## Target sub-score
**`required_fields`** (expected: largest headroom — this is the dominant
failure mode the 4B exhibits). Will confirm via baseline.

## Hypothesis log
| iter | slug | family | composite | comp Δ | target Δ | verdict |
|------|------|--------|-----------|--------|----------|---------|
| 0 | baseline | — | 0.8509 | — | — | required_fields=0.700 = target |
| 1 | h1-r16-6ep | H1 | **0.8939** | +4.30pp | +16.67pp | ✓ confirmed (kept; best for this capability) |
| 2 | h9-asym | H9 | — | — | — | ? killed mid-run, no adapter (pre-checkpoint) |
| 3 | h9-asym-ckpt | H9 | 0.7917 (best ckpt) / 0.343 (final) | -10.22pp | — | ✗ falsified (EOS collapse from over-training) |

## Dead ends
- max_tokens=128 rollout budget on a structured-output capability — too
  narrow; OPD finds nothing useful to do (cap #3 lesson; applied here
  upfront with max_tokens=384).
- H9 asymmetric at 6-epoch dosage on this prompt set: over-shoots into
  EOS collapse. Mechanism is real (skip 93.6% → 0%) but produces a
  worse adapter than symmetric. Future H9 here would need fewer
  effective steps and/or a softer teacher prefix.

## Open questions
- Does H9-short (2 epochs, ~52 steps) land near iter 1 or near ckpt-75?
  Ckpt-75 of iter 3 scored 0.792 at roughly equivalent training amount.
- Would a strict-only-JSON `parses` (no best-effort extraction) make
  the EOS collapse hit the floor faster and protect iter 1's score?
  (rubric Goodhart hole identified in iter 3 verdict; logged to
  kiln-polish.)

## Closeout (iter 3)
Best kept adapter: **iter 1 `toolcall-h1-r16-6ep`, composite 0.8939**
(+4.30pp vs baseline). Three meta-wins worth more than the +4.30pp:

1. **Asymmetric teacher conditioning** shipped as
   `OpdPrompt.teacher_extra_messages` (commits `46087c4e`, `626f03b2`).
   First real-world use confirmed the mechanism (skip 93.6%→0%) and
   surfaced its known failure mode (EOS collapse from over-training).
2. **Periodic OPD checkpointing** shipped as
   `OpdConfig.checkpoint_interval` (commit `96774c99`). First real use
   saved 6 intermediate adapters; the best of them (ckpt-75 = 0.792)
   is what we eval against, not the catastrophic final (0.343).
3. **EOS-collapse failure mode characterised** — the OPD paper
   literature's "flawed prefix trap" observed concretely. Mitigation
   directions logged to kiln-polish.

Moving to capability #5 (diff/patch fluency) with the eval-anti-shortcut
lesson applied upfront — strict parses, no best-effort credit for
"valid content followed by garbage."

## Checkpoints
(every 3rd iter)


## Round 2 setup

This cap was normalized to the round-2 layout on 2026-05-21. The previous
iter log and writeups are preserved in [`archive/`](archive/). The
`capability.jsonl` starts empty for the new round.

### Kiln features the new round uses

- `kiln adapter verify` (#4) — adapter loadability + behavioral check.
- `cuda_*` trainer `--install-adapter-dir` / `--install-adapter-name` (#5) —
  atomic install into the registry; no more `output/adapter/` symlink bugs.
- `train_receipt.json` (#8) — the canonical per-run artifact with kiln SHA,
  data hashes, hyperparameters, LoRA delta norms, and ECHO metrics.
- `cuda_grpo_ablation --dry-run` (#9) — pre-GPU validation of data, masks,
  base-adapter shape, and saturated-reward warnings.
- `kiln trajectory inspect` (#10) — Rust-native mask + token-count
  diagnostic; replaces the Python `lib/pi_trajectory.py` for new code.
- ECHO observability in receipt (#12) — env-token CE, action-token count,
  warning-prefix masked-out byte count.
- `kiln serve --eval-mode` (#15) — deterministic, no thinking, no
  per-request adapter drift.
- `--adapter-smoke-test` (#19) — post-train base-vs-adapter logit-delta check.
- `--filter-var-min` (#22) — official strong-signal filtering.
- `kiln eval-adapter --seeds N` (#33) — multi-seed paired-eval driver wrapped
  by `capability.oracle.sh`.
- `adapter_manifest.json` + `kiln adapter restore` (#36) — replaces ad-hoc B2
  backup scripts.

### Workflow

```bash
./capability.oracle.sh                     # baseline (no adapter)
./run_iter.sh h1-default-recipe            # first training iter
./run_iter.sh h2-lower-lr                  # subsequent
```

See [`run_iter.sh`](run_iter.sh) for the full pipeline.

## Round 2 improvement plan
Round 1 status: **OPD cap with multiple iters attempted**.

### Round 2 plan

1. **Run after `code-symbol-extraction` canary passes** to confirm the
   OPD trainer fix transferred.
2. **Adversarial calibration cases.** Tool-call argument fidelity is
   exactly where the §0 cheat-by-guessing pattern bites. Include in
   calibration: well-formed args, args with one swapped field,
   args with extra hallucinated fields, args missing required fields.
3. **Cross-validate with `pi-tool-call-efficiency` eval cap.** A
   tool-call-arg-fidelity adapter that hurts tool-call efficiency
   (more retries, more arg-rebuild loops) is a Pyrrhic win.
