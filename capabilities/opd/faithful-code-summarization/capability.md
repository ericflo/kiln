# Capability: faithful-code-summarization

## Description
The model is shown a short code snippet (3–30 lines, any common language) and asked
to write a brief summary of what it does. **Faithful** means: every named entity
the summary mentions (function names, class names, variable names) must actually
appear in the code, and the summary must not invent behavior the code does not
contain. The summary should be concise (under ~150 words) and readable.

Today's 4B baseline gets the gist right but routinely (a) renames functions in
summaries ("the `process` function" when the code defines `handle`), (b) invents
helper functions the code doesn't define, and (c) attributes behavior to the
snippet that lives elsewhere ("then it logs to a file" when no logging is there).

OPD against a 27B teacher should pull the 4B closer to the teacher's grounded
behavior at the states the 4B itself visits.

## Base model
Qwen3.5-4B (kiln serve on http://localhost:8420)

## Teacher
`vllm/qwen3.6-27b-awq` at http://localhost:8002
AWQ-INT4 (Q4), `--max-logprobs 64`, `--gpu-memory-utilization 0.45`,
`--enforce-eager`, `--max-model-len 4096`.

## Rubric

| Sub-score | Weight | What it measures |
|-----------|--------|-------------------|
| `parses` | 0.20 | Response is readable text: ≥20 chars, contains spaces, no malformed control chars, no obvious gibberish. |
| `entity_recall` | 0.20 | Of the named entities in the code (functions, classes, top-level constants), what fraction does the summary mention by name? |
| `entity_precision` | 0.40 | Of the named entities the summary mentions, what fraction actually appears in the code? This is the **anti-hallucination** sub-score. Heavy weight. |
| `concise` | 0.20 | Word count in `[20, 150]` (full credit), with a soft penalty outside that band. |

Composite = `0.20 × parses + 0.20 × entity_recall + 0.40 × entity_precision + 0.20 × concise`.
Direction: higher is better. Composite is in `[0, 1]`.

## Baseline

| Sub-score | Weight | Baseline | Headroom (w×(1−b)) |
|-----------|--------|----------|---------------------|
| parses | 0.20 | 1.0000 | 0.0000 |
| entity_recall | 0.20 | 0.9500 | 0.0100 |
| entity_precision | 0.40 | 1.0000 | 0.0000 |
| concise | 0.20 | 1.0000 | 0.0000 |
| **Total movable** | | | **0.0100** |

**Headroom-gate fired: total movable headroom (0.01) is well under the
0.05 floor.** The skill (§0) halts the session before training. The 4B
already produces accurate, grounded, concise summaries on these 5–15
line snippets; OPD has nothing to teach here.

This capability would be OPD-able with a harder rubric (longer snippets
where the model genuinely drifts, stricter entity match requiring
called-function coverage, or an explicit anti-claim n-gram check). For
this session, we abandon it and move to capability #2.

## Target sub-score
<filled after baseline; `entity_precision` expected given the 4B's known hallucination pattern>

## Hypothesis log
| iter | slug | family | composite | comp Δ | target Δ | verdict |
|------|------|--------|-----------|--------|----------|---------|
|      |      |        |           |        |          |         |

## Dead ends
(none yet)

## Open questions
(none yet)

## Checkpoints
(every 3rd iter, write a brief progress summary here under `### Checkpoint at iter N`)


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
Round 1 status: **OPD cap, less developed**.

### Round 2 plan

1. **Mature the rubric first.** Round-1 cap is sparse; before training,
   build `calibration/{good,bad}.jsonl` with at least 10/10 separated
   examples. Round-2 LAYOUT now makes rubric_sanity.py mandatory.
2. **Teacher pool larger than student baseline.** Faithful
   summarization requires the teacher to be reliably more faithful
   than the student. Spot-check 20 teacher outputs against ground
   truth before running OPD.
3. **Compose with `pi-code-comprehension`.** Code comprehension
   (round-1 +12.9pp big win) produces structured summaries; this cap
   refines the *fidelity* of those summaries. Sequential composition
   in the integration track.
