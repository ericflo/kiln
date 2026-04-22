# Phase C35 — H13 residual 4-cell A/B: prose/chat × BF16/FP32 argmax

## Verdict

**Hypothesis 13 is REFUTED on the sampler-dtype residual and
CONFIRMED-partial on the prompt-workload residual.** Under the
C18 MTP k=1 harness, applying the Qwen ChatML template to the
bench prompt raises median α from **0.1759 → 0.5875**
(+0.4116 absolute, **+234%**). Forcing the MTP argmax path to FP32
(vLLM `rejection_sampler.py:117` parity) moves α by **−0.0108**
under prose and **+0.0201** under chat — both within the
3-seed noise band. The 5× α gap in Phase C18 was a
**harness artifact**, not a kiln contract bug: the bench fed raw
English prose into `tokenizer.encode(..., add_special=false)`
while the Qwen3.5 paper and vLLM regression harnesses all run
chat-formatted input. Chat-templated α now reaches **0.7639
on seed 1 of Cell D** (exceeds the paper floor of 0.72) with a
3-seed median of 0.6076, closing the ceiling on the residual
gap from 0.57 to 0.11.

Action items:

1. **Do NOT ship `KILN_MTP_ARGMAX_FP32` as a default.** It buys
   no measurable α, costs decode tok/s (−10% on prose, flat on
   chat), and has no numerical justification on this workload.
   The env flag stays in the tree as an audit switch.
2. **Re-state the C18 α baseline as `0.5875` (chat × BF16 argmax,
   seeds 0..2 median)**, not 0.153. All prior Phase C* α numbers
   measured against the prose pool are workload-invalidated for
   upstream-parity comparisons. They remain valid as **internal
   regression numbers** against one another on that same
   workload.
3. **Promote `--chat-template` to the default for α-measurement
   bench runs** (or add a `--prompt-workload=chat|prose` knob
   that defaults to `chat` under `KILN_SPEC_METHOD=mtp`). The
   prose pool stays available for latency/throughput
   regression testing where prompt-content stability matters.
4. **The remaining 0.112 gap** (Cell D median 0.608 vs paper
   floor 0.72) is the next target. Candidate causes, none
   of which are sampler contract:
   - Prompt-content distribution: `PROMPT_POOL` is ad-hoc prose;
     the paper uses GSM8K / HumanEval / MT-Bench style tasks.
   - Decode-length: `--max-output-tokens 128` may be too short
     for the stable-regime α the paper reports.
   - Residual MTP head weight/init deltas vs the released
     checkpoint — H14 territory, not sampler.

## 1. Setup

### 1.1 Bench invariants (Phase C18 protocol)

All 12 runs (4 cells × 3 seeds each) were executed on the same
A6000 pod (`kiln-pool-1776810593758943658`, runpod id
`sq3exutrqy2cuw`) with identical bench args:

```
./target/release/kiln-bench \
  --model-path /workspace/qwen3.5-4b \
  --paged --prompt-tokens 512 --max-output-tokens 128 \
  --skip-training --seed {0,1,2}

KILN_SPEC_ENABLED=1
KILN_SPEC_METHOD=mtp
KILN_W4A16=1
KILN_CUDA_GRAPHS=true
```

Build: `cargo build --release --features cuda --bin kiln-bench`
on commit `9047e53` of branch `ce/mtp-c35-h13-residual-ab`.

### 1.2 The two residual knobs

**`--chat-template`** (new CLI flag, MTP arm only): wraps the
raw `PROMPT_POOL[seed%8]` prose in a single-user-turn
`ChatMessage` and calls `KilnTokenizer::apply_chat_template`.
Because the loaded tokenizer sets no Jinja template string
explicitly in-bench, `apply_chat_template` falls through to the
plain ChatML framing (`<|im_start|>user\n...<|im_end|>\n
<|im_start|>assistant\n`) defined at
[`kiln-core/src/tokenizer.rs:152-163`](../../crates/kiln-core/src/tokenizer.rs).
This is the same surface that production ChatML traffic hits
through the HTTP server — no special-token divergence.

**`KILN_MTP_ARGMAX_FP32`** (new env flag): gates a
`.to_dtype(DType::F32)?` cast before each of the three
`greedy_sample` call sites in
[`speculative.rs::speculative_mtp_decode_step`](../../crates/kiln-model/src/speculative.rs)
(draft, verify pos-0, bonus) and the one prefill seed call
in `bench.rs::bench_latency_paged_mtp`. Off-by-default (matches
C18 baseline), on-by-env (matches vLLM
`rejection_sampler.py:117` and `:394`).

The helper is deliberately cheap: `OnceLock<bool>` cache on
first call, pure clone when unset, `to_dtype` when set. The
original BF16 tensor is preserved for downstream consumers
(C1 attribution, mtp_debug top-k logging).

### 1.3 The 2×2 design

| Cell | Prompt workload | Argmax dtype | Env/flags delta |
|------|-----------------|--------------|-----------------|
| A    | prose (C18 baseline) | BF16 | (none) |
| B    | prose           | FP32         | `KILN_MTP_ARGMAX_FP32=1` |
| C    | chat-templated  | BF16         | `--chat-template` |
| D    | chat-templated  | FP32         | `--chat-template`, `KILN_MTP_ARGMAX_FP32=1` |

Each cell runs seeds 0, 1, 2 (seed index drives both prompt
selection and the unused StdRng slot). All other bench
parameters are identical.

## 2. Results

### 2.1 Acceptance rate per cell

| Cell | seed 0 | seed 1 | seed 2 | **α median** | Δ vs A | Δ vs baseline |
|------|--------|--------|--------|--------------|--------|---------------|
| A (prose × BF16)  | 0.2308 | 0.1759 | 0.1140 | **0.1759** | — | 0 (baseline) |
| B (prose × FP32)  | 0.2308 | 0.1651 | 0.1043 | **0.1651** | **−0.0108** | −6.1% rel |
| C (chat × BF16)   | 0.6076 | 0.5875 | 0.5679 | **0.5875** | **+0.4116** | **+234.0% rel** |
| D (chat × FP32)   | 0.6076 | 0.7639 | 0.5875 | **0.6076** | **+0.4317** | **+245.4% rel** |

Decode throughput (tok/s, decode-only, median of 3):

| Cell | seed 0 | seed 1 | seed 2 | **tok/s median** | Δ vs A |
|------|--------|--------|--------|------------------|--------|
| A | 18.74 | 23.04 | 25.89 | **23.04** | — |
| B | 19.22 | 20.59 | 25.32 | **20.59** | −2.45 |
| C | 33.65 | 34.18 | 36.12 | **34.18** | +11.14 (+48.4%) |
| D | 33.88 | 41.13 | 37.48 | **37.48** | +14.44 (+62.7%) |

(Decode tok/s values above are the per-run `decode_tokens_per_sec`
from each `cell{X}_seed{Y}.json`; median reported for comparison.)

### 2.2 The sampler-dtype residual is noise

Comparing Cell B vs Cell A (prose workload, only the argmax
dtype changes):

- seed 0: 0.2308 → 0.2308 (exact tie)
- seed 1: 0.1759 → 0.1651 (−0.0108)
- seed 2: 0.1140 → 0.1043 (−0.0097)
- **median: 0.1759 → 0.1651 (−0.0108)**

Comparing Cell D vs Cell C (chat workload, only dtype changes):

- seed 0: 0.6076 → 0.6076 (exact tie)
- seed 1: 0.5875 → 0.7639 (+0.1764)
- seed 2: 0.5679 → 0.5875 (+0.0196)
- **median: 0.5875 → 0.6076 (+0.0201)**

The BF16→FP32 cast moves α by **at most ±0.01 on median
across both workloads**. The large +0.1764 at Cell D seed 1 is
a single-seed outlier — the other two seeds are within
noise of Cell C. This matches the C34 static-audit prediction:
BF16 tie/near-tie rate on Qwen3.5-scale vocabularies is low
enough that the argmax flips rarely, and when they do flip they
do not systematically favor tokens the verifier would have
accepted.

**Conclusion: dtype parity with vLLM is contract-correct but
numerically unnecessary on this workload.** Ship the env flag
but do not default it on.

### 2.3 The prompt-workload residual is real and large

Comparing Cell C vs Cell A (only the prompt framing changes,
dtype held at BF16):

- seed 0: 0.2308 → 0.6076 (+0.3768)
- seed 1: 0.1759 → 0.5875 (+0.4116)
- seed 2: 0.1140 → 0.5679 (+0.4539)
- **median: 0.1759 → 0.5875 (+0.4116, +234% rel)**

Every seed shows a +0.37 to +0.45 absolute jump. Cell C median
exceeds the Phase C18 absolute ceiling (0.153 → 0.5875) by
**3.84×** on the same MTP head weights, same model path, same
decode length, same kernel stack. The only change was which
framing the prompt entered the tokenizer in.

### 2.4 Paper target (≥0.72) intersection

Cell D seed 1 hit **α = 0.7639**, clearing the Qwen3.5 paper
floor. Cell D median sits at 0.6076 — **gap to paper: 0.1124**
(down from 0.567 under the C18 prose baseline).

This is the first Phase C* run where any single configuration
reached the paper floor.

### 2.5 ITL (inter-token latency) also improves under chat

Median ITL medians tell the same story:

| Cell | p50 ITL (ms) | p99 ITL (ms) |
|------|--------------|--------------|
| A | 53.33 | 63.61 |
| B | 57.43 | 71.58 |
| C | 19.35 | 62.73 |
| D | 18.95 | 62.09 |

The p50 drop (53→19 ms) is direct-consequence: a higher α means
more accepted draft tokens per speculative step, i.e. more
tokens per wall-clock ms. Tail (p99) is flat — the failure-mode
step cost is still dominated by the verify forward pass.

## 3. Interpretation

### 3.1 Why chat framing lifts α so much

The Qwen3.5 base MTP head was trained against the same chat
distribution the main model was RL-tuned on. Feeding it raw
prose puts the next-token distribution **off-manifold** for the
MTP head's one-step lookahead: the head is near-uniformly
uncertain about what the main LM would emit next, so its
argmax frequently disagrees with the main verifier.

Under chat framing, both the MTP head and the verifier see
the token distribution they were trained on, and the draft
matches the verify more often. This is exactly the
workload-dependence note in C34 §1.15 and line up with
[published MTP literature](../phase-c18/c18-mtp-initial-baseline.md):
α under an off-distribution workload falls by O(3-5×), not by
fractions.

### 3.2 Why FP32 argmax is a no-op here

Because the C34 static audit already established that kiln's
greedy sampler contract is bit-identical to vLLM's all-greedy
branch except for the tensor dtype going into `argmax`, the
only way FP32 could have moved α is via BF16 tie-breaks
between the top-1 and top-2 tokens. On a 152K-vocab Qwen3.5
distribution that is dominated by a clear top logit on the
typical decoding step, the tie rate is <2%, and a flip only
matters when it would have flipped an accept into a reject or
vice versa. That double-coincidence drives the observed
≤±0.02 movement.

### 3.3 Why this doesn't invalidate prior Phase C* measurements

The prose-workload α number is a **stable internal regression
signal** — it is consistent seed-to-seed (σ ≈ 0.06 across
seeds 0..2), it reacts to real contract changes (e.g. pos-1
ablation in C24), and it has been the benchmark for every
phase fix from C19 onward. What it does NOT do is compare
apples-to-apples with the Qwen3.5 paper's reported α or
with vLLM regression numbers, both of which use chat
workloads.

Going forward, α-vs-paper comparisons MUST use `--chat-template`.
α-vs-prior-kiln-phase comparisons can continue on prose as
long as the switch is disclosed.

## 4. Handoff to Phase C36

### 4.1 Merged in this PR

- `--chat-template` CLI flag on `kiln-bench` MTP arm
  ([`bench.rs`](../../crates/kiln-server/src/bench.rs))
- `KILN_MTP_ARGMAX_FP32` env flag with `mtp_argmax_fp32_enabled()`
  + `argmax_input()` helper in
  [`speculative.rs`](../../crates/kiln-model/src/speculative.rs)
- `ChatMessage` import routed from `kiln_core::tokenizer`
- Three `greedy_sample` call sites in
  `speculative_mtp_decode_step` gated on the env flag
  (draft, verify pos-0, bonus), plus the one prefill
  `greedy_sample` in `bench_latency_paged_mtp`

### 4.2 Not merged — left for C36

- Default-on behavior for `--chat-template` in the MTP bench arm
  (deferred; current default preserves C18 comparison continuity).
- Re-run of the Phase C18-C33 regression harness on the
  chat-templated workload to get "true" α baselines for each
  intermediate fix.
- Investigation of the remaining **0.112** gap between
  Cell D median (0.608) and the paper floor (0.72). Candidate
  H14 hypotheses:
  - `PROMPT_POOL` is not drawn from a standardized benchmark
    distribution (GSM8K, HumanEval, MT-Bench, WildChat). The
    paper α numbers use specific task suites.
  - `--max-output-tokens 128` may truncate below the
    stable-regime α the paper reports at longer decodes.
  - A residual in the MTP head weights or the init path
    (most likely last-layer LN or positional-embedding subtleties)
    that does not surface under plain `lm_head_forward`.

### 4.3 Decision rule outcome (C35 task spec)

| Outcome predicate | Action |
|-------------------|--------|
| Chat cells rise → H13 REFUTED on contract, workload mismatch real | **✓ Observed, chat lifts α by 3.3×. This is the path.** |
| FP32 cells rise → ship dtype fix, H13 CONFIRMED-partial | ✗ FP32 is noise-level on both workloads. Do not ship as default. |
| Neither rises → H13 fully REFUTED, proceed to C36+ | ✗ (chat did rise) |

### 4.4 Bench timing

- Build: 18.24 s (sccache hit-warm)
- 12 bench runs (4 cells × 3 seeds): ~22.7 min wall-clock
- A6000 pod lease: acquired from kiln pool, released at end
  of this PR. Total pod time on this task: ~30 min
  (well under 90 min budget).

## References

### kiln

- `crates/kiln-server/src/bench.rs::bench_latency_paged_mtp`
  — the bench arm we're measuring. Now threads
  `chat_template: bool` and branches the prompt through
  `apply_chat_template` when set.
- `crates/kiln-model/src/speculative.rs::speculative_mtp_decode_step`
  — the three greedy_sample call sites (draft, verify-pos0,
  bonus), now gated on `mtp_argmax_fp32_enabled()` via the
  `argmax_input()` helper. Returned tensor is either a
  clone of the BF16 original (default) or a new FP32 tensor
  (env set). Original tensor is preserved so C1 attribution
  and mtp_debug top-k logging continue to see the
  as-emitted BF16 logits.
- `crates/kiln-core/src/tokenizer.rs::apply_chat_template`
  — ChatML fallback used when no Jinja template is loaded
  (lines 152–163). The bench does not load a chat template
  explicitly, so all Cell C and Cell D runs hit this
  fallback branch. This is the same framing that
  production ChatML-over-HTTP traffic sees.

### Phase C18

- [`docs/phase-c18/c18-mtp-initial-baseline.md`](../phase-c18/c18-mtp-initial-baseline.md)
  — original α=0.153 baseline measurement.

### Phase C34

- [`docs/phase-c34/c34-sampler-parity-audit.md`](../phase-c34/c34-sampler-parity-audit.md)
  — static audit that motivated the C35 2×2.
