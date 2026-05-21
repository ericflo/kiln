# Capability: code-symbol-extraction

## Description
The model is shown a code snippet (30–80 lines, Python/Rust/Go/JS/TS) and asked
to list every top-level **defined** symbol — functions, classes, structs,
enums, traits, type aliases — one symbol per line, **nothing else**. No
explanations, no markdown, no commentary.

The 4B baseline routinely (a) adds prose commentary the eval can't parse,
(b) misses nested or less-common symbol kinds (e.g. Rust `trait`, Go
struct methods on a receiver), and (c) over-recalls by listing local
variables or imported names.

A teacher (27B AWQ) does this cleanly and concisely. OPD should pull the
4B toward the teacher's compact symbol-only output.

## Base model
Qwen3.5-4B (kiln serve on http://localhost:8420)

## Teacher
`vllm/qwen3.6-27b-awq` at http://localhost:8002 (AWQ-INT4, max_logprobs=64)

## Rubric

| Sub-score | Weight | What it measures |
|-----------|--------|-------------------|
| `parses` | 0.15 | Response is non-empty text, no gibberish. |
| `format_compliance` | 0.15 | Output is symbol-name-per-line, no markdown bullets, no prose. Lines must be plain identifiers (allow `name (kind)` form). |
| `symbol_recall` | 0.35 | Fraction of ground-truth symbols the model listed. |
| `symbol_precision` | 0.35 | Fraction of listed names that are actually ground-truth symbols. |

Composite = `0.15 × parses + 0.15 × format_compliance + 0.35 × recall + 0.35 × precision`.
Direction: higher is better. Composite is in `[0, 1]`.

The two heavy weights (recall+precision = 0.70) make hallucinated symbols
hurt as much as missed symbols. This is intentional: it forces the model
to balance saying-enough against not-making-things-up.

## Baseline

| Sub-score | Weight | Baseline | Headroom (w×(1−b)) |
|-----------|--------|----------|---------------------|
| parses | 0.15 | 1.0000 | 0.0000 |
| format_compliance | 0.15 | 1.0000 | 0.0000 |
| symbol_recall | 0.35 | 1.0000 | 0.0000 |
| symbol_precision | 0.35 | 0.8039 | 0.0686 |
| **Total movable** | | | **0.0686** |

Composite: **0.9314** (headroom 0.069 above the 0.05 floor — proceed).

## Target sub-score

**`symbol_precision`** owns 100% of movable headroom. The 4B over-recalls — typical failure is listing imported names alongside defined ones (e.g. `EventEmitter` when the snippet does `extends EventEmitter`). OPD against the 27B should suppress this.

## Hypothesis log
| iter | slug | family | composite | comp Δ | target Δ | verdict |
|------|------|--------|-----------|--------|----------|---------|
| 1 | h1-r16-6ep | H1 | 0.9370 | +0.0056 | +0.0160 | ? Inconclusive (97% skip rate; bump epochs) |
| 2 | h1-r16-12ep | H1 | 0.9370 | +0.0056 | 0.0000 | ✗ Falsified — same 7 effective steps as iter 1 |

## Closing summary

**Capability #2 closed after 2 H1 iterations.** Best result: iter 1
(`symbol-h1-r16-6ep`) at composite 0.9370 (+0.6% over baseline 0.9314),
capturing 8.2% of the 0.0686 movable headroom. iter 2 (12 epochs) produced
literally identical eval scores to iter 1 because the same 7 prompts trained
both times — the other 33 always EOS at sample time regardless of epoch.

**Root cause found (kiln OPD trainer bug).** The rollout-prompt construction
in `opd_train` builds `prompt_only` from `orig_input_ids[..prompt_end]` where
`prompt_end` is the first `label_mask=true` position. The label_mask spans
the entire `<|im_start|>assistant\n...<|im_end|>` block, so `prompt_only`
stops *before* the assistant cue marker. The student samples from a context
where it hasn't seen the assistant role marker and emits EOS immediately
~97% of the time on terse-list capabilities (vs ~87% on free-form ones).

The `KILN_OPD_GEN_PROMPT_SUFFIX=1` env var is the half-fix from v6 of the
JSON-schema run, but it uses raw `tokenizer.encode()` which doesn't resolve
`<|im_start|>` to its special-token id. The proper fix renders the suffix
via `apply_chat_template_full_with_options(enable_thinking=false)`.

**Next session plan.** Land the kiln fix first (commit it to kiln-train),
then re-run iter 1 of this capability to validate the fix lifts skip rate
from 0.97 to <0.5 and unlocks more effective training steps. Expect
composite to reach 0.96+ at that point.

The skill itself worked exactly as designed: the verdict gate caught the
zero-Δ on iter 2 cleanly, forced honest documentation of the structural
finding, and gave kiln a clear fix-target rather than a vague "OPD didn't
help here" complaint.


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
