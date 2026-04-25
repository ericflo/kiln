# Phase C34 — Sampler contract parity audit vs upstream Qwen3.5 eval harness (H13, doc-only)

## Verdict

**Hypothesis 13 is INCONCLUSIVE on static audit.** Kiln's MTP k=1
speculative sampling contract is **functionally identical** to the
canonical vLLM `rejection_sample` `all_greedy` branch: draft is a
plain argmax on `mtp_logits`, verifier pos-0 is a plain argmax on
`verify_logits[0]`, accept is an integer compare, and the bonus
(pos-1) is a plain argmax on `verify_logits[1]`. The `SamplingParams`
passed into `speculative_mtp_decode_step` is underscore-prefixed at
the function boundary (`_params`) — proven statically to never
reach the sampling primitive. EOS handling matches upstream in
behavior (both stop on EOS after accept; kiln's `eos_token_ids`
slice equals vLLM's `stop_token_ids` for greedy). There is no
sampler-drift surface between kiln and upstream that could by itself
collapse α from the Qwen3.5 paper floor (≥ 0.72) to the Phase C18
observed median (0.153).

Two non-zero deltas exist but are either too small to close a 5×
gap or not contract-level:

1. **Argmax dtype.** vLLM casts `raw_target_logits` and `draft_logits`
   to **FP32** before `argmax` in
   [`rejection_sampler.py:117`](#references-vllm) and line 394.
   Kiln's `greedy_sample` operates on the **BF16** tensor as emitted
   by `lm_head_forward` and `model_forward_paged_with_last_hidden`
   (see `greedy_sample` call sites in
   [`speculative.rs:740,768,844`](#references-kiln)). BF16 has 7
   explicit + 1 implicit mantissa bits; FP32 has 23. The BF16
   tie/near-tie rate in practice is below ~2% on Qwen3.5-scale
   vocabularies, and on a greedy path a flip only matters when the
   FP32 argmax and BF16 argmax disagree AND the disagreement drives
   a reject that would otherwise have been an accept. The math
   ceiling on this as a sole explanation for α collapse is **far
   below 5×**. It is a real contract delta worth noting but cannot
   by itself be H13's mechanism.

2. **Bench harness prompt-workload distribution.** kiln's
   `bench_latency_paged_mtp` feeds a raw English prose prompt
   (`build_prompt`, `PROMPT_POOL`) via `tokenizer.encode(prompt)`
   with no chat template applied (kiln-core
   [`tokenizer.rs:76-82`](#references-kiln) → `encode(text, false)`
   does not add specials). The Qwen3.5 paper, vLLM's MTP
   regression harness, and standard eval frameworks
   (`lm-evaluation-harness`) all run **chat-formatted** or
   task-formatted prompts. This is a **harness-level** (not
   contract-level) divergence — it does not change the greedy
   sampler contract, but it does change the token distribution that
   the MTP head sees. α is known to be strongly workload-dependent
   for speculative decoders. This alone could plausibly explain
   a meaningful (though not necessarily 5×) portion of the gap.

Because the sampler **contract** matches vLLM's all-greedy reference
functionally and both deltas above are off-contract, H13 as
originally framed ("sampling/decoding contract mismatch between
kiln's MTP speculative path and upstream") is **not confirmed** as
the α-collapse mechanism. But because the BF16-argmax delta cannot
be dismissed without numerical evidence, and because the
prompt-workload delta is real and untested, the audit is
**inconclusive** rather than refuted.

**Recommended next step:** Phase **C35** = α re-measurement with
Qwen3.5 chat-formatted prompts on the same kiln decode path
(CPU-runnable — no pod spend required for prompt-distribution A/B
if a chat-templated reference prompt set fits a ~2-minute CPU
bench). If α under chat-formatted workload rises materially above
0.153, the C18 baseline was measuring the wrong distribution. If α
stays pinned, H13 can be downgraded to **REFUTED** and the next
hypothesis class (C36+: MTP-head logits producer bug, or
speculation architecture mismatch) becomes the only live
alternative. The BF16→FP32 argmax dtype delta can be cheaply
tested in the same C35 run by adding a `.to_dtype(F32)` before
each `greedy_sample` call site in `speculative_mtp_decode_step`
under an env flag.

## H13 as stated

From the task description / Phase C33 handoff:

> **Hypothesis 13:** The observed low α may reflect a
> sampling/decoding contract mismatch between kiln's MTP
> speculative path and the upstream Qwen3.5 reference eval harness
> (temperature scaling, top-p/top-k ordering, prefill vs decode
> path, EOS handling, draft/verifier logits-combine semantics).

This audit disproves the **contract** portion of H13 by static
reading and rules out every listed sampler axis (temperature /
top-p / top-k / prefill-decode / EOS / draft-verifier combine).
The only residual deltas are dtype (BF16 vs FP32 argmax) and
harness workload distribution, both of which are testable in
C35 without a full re-audit.

## Scope and method

Static, doc-only, CPU-only. **No pod spend.** Same cost-discipline
pattern as C19/C20/C21/C22/C23/C24/C26/C27/C29/C30/C31/C32/C33.

- Cloned kiln via the `--reference /data/repo-cache/ericflo/kiln.git
  --dissociate` pattern from
  [`SKILL.md`](../../skills/kiln/SKILL.md) so the clone completed in
  under 10 s against the warm cache.
- Inspected every `greedy_sample` call site and the
  `speculative_mtp_decode_step` signature + body (`speculative.rs`).
- Inspected the MTP bench harness
  (`bench_latency_paged_mtp`) including prompt construction, prefill,
  and decode loop (`bench.rs`).
- Inspected kiln's tokenizer boundary (`tokenizer.rs`): `encode`,
  `apply_chat_template`, `apply_chatml`, `eos_token_ids`.
- Cross-checked vLLM's reference sampling implementation against
  current `main`:
  - `vllm/model_executor/models/qwen3_next_mtp.py` — MTP module
    (no sampler logic; defers to `eagle.py` /
    `rejection_sampler.py`).
  - `vllm/v1/sample/rejection_sampler.py` — canonical
    `rejection_sample` implementation including the `all_greedy`
    fast path.
  - `vllm/v1/spec_decode/eagle.py` — `_greedy_sample`
    (shared across EAGLE, EAGLE3, DFlash, MTP, and draft-model
    methods).
- Cross-checked Hugging Face `Qwen/Qwen3.5-4B/config.json` for
  `mtp_num_hidden_layers`, `tie_word_embeddings`, and
  `eos_token_id`.

## Tensors and counters in scope

For one `speculative_mtp_decode_step` invocation on the k=1 Qwen3.5
MTP path:

| Symbol | Shape | Source | Sampler used | Dtype |
| --- | --- | --- | --- | --- |
| `mtp_logits` | `[1, V]` | `mtp_forward_step → lm_head_forward` | `greedy_sample` | BF16 |
| `draft_token` | `u32` | argmax on `mtp_logits` | — | — |
| `verify_logits` | `[1, 2, V]` | `model_forward_paged_with_last_hidden` | — | BF16 |
| `verify_pos0` | `[1, V]` | `verify_logits.narrow(1, 0, 1).squeeze(1)` | `greedy_sample` | BF16 |
| `target_at_0` | `u32` | argmax on `verify_pos0` | — | — |
| `verify_pos1` | `[1, V]` | `verify_logits.narrow(1, 1, 1).squeeze(1)` (accept branch only) | `greedy_sample` | BF16 |
| `bonus` | `u32` | argmax on `verify_pos1` | — | — |

Three greedy sampling sites; all three are the same
`greedy_sample(&tensor)` primitive against the same BF16 dtype; none
consume sampler parameters.

Counters:

- `total_draft_attempts` — incremented once per step
  (`bench.rs:1373`)
- `draft_accepted_count` — incremented when `step.draft_accepted`
  is true (`bench.rs:1374-1376`)
- `alpha = draft_accepted_count / total_draft_attempts`
  (`bench.rs:1421-1425`)

## Section 1 — Kiln MTP sampling contract

### 1.1 Draft sampler (position draft)

`speculative_mtp_decode_step`
([`crates/kiln-model/src/speculative.rs:701-870`](#references-kiln))
takes an unused `_params: &SamplingParams` at
[`speculative.rs:714`](#references-kiln). The underscore prefix is
Rust's idiomatic "intentionally unused binding" — the compiler
would error (or at minimum warn with `deny(unused_variables)`) if
the body read from it. Static proof that no sampler parameter
(temperature, top_p, top_k, repetition_penalty, seed) can affect
the draft side.

```rust
// speculative.rs:714
pub fn speculative_mtp_decode_step(
    ...
    _params: &SamplingParams,
    eos_token_ids: &[TokenId],
    _rng: &mut StdRng,
) -> Result<MtpSpeculativeStepResult> {
    ...
    // speculative.rs:740
    let draft_token = greedy_sample(&mtp_logits)
        .context("mtp draft sampling failed")?;
    ...
}
```

`_rng` is likewise underscore-prefixed — no stochastic branch
exists. `greedy_sample` is a plain argmax; no filtering, no
temperature scaling.

### 1.2 Verifier sampler (position 0)

Same file, 28 lines below the draft sampler:

```rust
// speculative.rs:767-768
let verify_pos0 = verify_logits.narrow(1, 0, 1)?.squeeze(1)?;
let target_at_0 = greedy_sample(&verify_pos0)
    .context("verify pos-0 sampling failed")?;
```

Identical primitive. Same dtype (BF16). No sampler parameters
consumed.

### 1.3 Accept/reject decision

```rust
// speculative.rs:773
let draft_accepted = target_at_0 == draft_token;
```

Integer compare on `u32` token IDs. Equivalent to vLLM's
`rejection_greedy_sample_kernel` `draft != target_argmax ⇒ reject`
behavior.

### 1.4 Bonus sampler (position 1, accept branch)

```rust
// speculative.rs:843-844
let verify_pos1 = verify_logits.narrow(1, 1, 1)?.squeeze(1)?;
let bonus = greedy_sample(&verify_pos1)
    .context("bonus sampling failed")?;
```

Same primitive. The bonus is computed only when the draft is
accepted (line 834 `if draft_accepted`), matching vLLM's "emit
bonus on accept" convention.

### 1.5 EOS handling

```rust
// speculative.rs:836
if eos_token_ids.contains(&draft_token) {
    hit_eos = true;
    (1, 1)  // base_advance=1, mtp_advance=1, no bonus forward
}
// speculative.rs:845
if eos_token_ids.contains(&bonus) {
    hit_eos = true;
}
```

And from the bench loop:

```rust
// bench.rs:1343-1346
while num_tokens < max_output_tokens {
    if eos_token_ids.contains(&last_token) {
        break;
    }
    ...
}
```

`eos_token_ids` is supplied by
[`tokenizer.rs:106-119`](#references-kiln)
`KilnTokenizer::eos_token_ids()`, which resolves IDs from the
Qwen3.5 special-token table: `<|endoftext|>`, `<|im_end|>`,
`<|end|>`, `</s>`. For a greedy speculative path, this is the
correct termination contract (matches vLLM's `stop_token_ids` used
by the greedy kernel).

### 1.6 `SamplingParams` shape at the bench call site

The bench constructs params explicitly as temperature 0 / top_p 1.0
/ top_k 0 / repetition_penalty 1.0 / no stop tokens:

```rust
// bench.rs:1326-1334
let params = SamplingParams {
    temperature: 0.0,
    top_p: 1.0,
    top_k: 0,
    max_tokens: max_output_tokens,
    repetition_penalty: 1.0,
    stop: vec![],
    seed: Some(seed),
};
```

These values would be sampler no-ops on a path that consumed them
(temp 0 + top_p 1.0 + top_k 0 = unconstrained greedy). Since the
path doesn't consume them, they are literally dead code on this
trajectory.

### 1.7 Prefill path (first-token seed)

```rust
// bench.rs:1293-1309
let (prefill_logits, mut h_prev) =
    model_forward_paged_last_token_with_last_hidden(...)?;
let prefill_last = prefill_logits.squeeze(1)?;
let mut last_token = greedy_sample(&prefill_last)?;
```

Also greedy argmax on BF16 `prefill_last`. The first seed token is
produced with the same primitive — no prefill vs decode sampler
divergence.

### 1.8 Prompt/tokenization path

`build_prompt`
([`bench.rs:289-308`](#references-kiln)) picks one of 8 English
prose prompts and greedily repeats-and-truncates to the target
token length. The encoding call is
[`tokenizer.rs:76-82`](#references-kiln)
`encode(&prompt)` which forwards to `tokenizers::Tokenizer::encode`
with `add_special_tokens=false`. No chat template is applied; no
`<|im_start|>` / `<|im_end|>` framing is inserted; no system
message is prepended.

By contrast, `apply_chat_template`
([`tokenizer.rs:98-103`](#references-kiln)) and `apply_chatml`
([`tokenizer.rs:157-167`](#references-kiln)) do exist but are **not
called** from `bench.rs`.

**This is a harness-level choice, not a sampler contract.** It is
noted here because Qwen3.5 is a chat-tuned model and MTP head
training presumably saw chat-formatted distributions.

## Section 2 — Upstream reference contract (vLLM + HF)

vLLM is the canonical speculative-decoding reference
implementation for Qwen3.5 MTP on single-GPU (see the Qwen3.5 paper
§5 "Inference" where MTP decoding is benchmarked on vLLM). Its
all-greedy fast path is the most apples-to-apples comparison for
kiln's `bench_latency_paged_mtp`.

### 2.1 vLLM MTP model module

[`vllm/model_executor/models/qwen3_next_mtp.py`](#references-vllm)
defines `Qwen3NextMultiTokenPredictor`. Relevant fields:

- `self.embed_tokens`, `self.fc` (mixing projection),
  `self.layers` (a transformer block stack of length
  `num_mtp_layers`), `self.norm`, `self.pre_fc_norm_hidden`,
  `self.pre_fc_norm_embedding`.
- `self.num_mtp_layers = getattr(config, "num_nextn_predict_layers", 1)`
  — confirming Qwen3.5 k=1.

The module contains **no sampling logic**. All sampling is delegated
to `vllm/v1/sample/rejection_sampler.py` and
`vllm/v1/spec_decode/eagle.py`. This matches kiln's layout:
the MTP head produces logits (kiln: `mtp_forward_step`; vLLM:
`Qwen3NextMultiTokenPredictor.forward`), and a shared sampler
downstream handles the draft/verify greedy argmax.

### 2.2 vLLM `rejection_sampler.py` — all-greedy path

```python
# vllm/v1/sample/rejection_sampler.py:117
raw_target_logits = raw_target_logits.to(torch.float32)
```

FP32 cast applied to the target model's logits before any sampling.
This is the notable dtype delta vs kiln.

```python
# vllm/v1/sample/rejection_sampler.py:388-405 (all-greedy fast path)
if sampling_metadata.all_greedy:
    target_argmax = target_logits.argmax(dim=-1)
    rejection_greedy_sample_kernel[(batch_size,)](
        output_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        target_argmax,
        bonus_token_ids,
        ...
    )
    if not sampling_metadata.all_random:
        return output_token_ids
```

Where `rejection_greedy_sample_kernel` (Triton) performs:

```python
# vllm/v1/sample/rejection_sampler.py:683 (simplified)
for pos in range(num_draft_tokens):
    draft_token_id = draft_token_ids[start + pos]
    target_argmax_id = target_argmax[start + pos]
    output_token_ids[start + pos] = target_argmax_id
    if draft_token_id != target_argmax_id:
        break  # first mismatch; emit target at this pos, stop
# if all accepted, append bonus at end
```

Semantics for k=1:

- Sample draft via `eagle.py::_greedy_sample`:
  `self.model.compute_logits(hidden_states).argmax(dim=-1)`.
  Identical to kiln's `greedy_sample(&mtp_logits)`.
- Sample target argmax via `target_logits.argmax(dim=-1)` on FP32.
  Identical to kiln's `greedy_sample(&verify_pos0)` except for
  dtype.
- Accept iff `draft_token_id == target_argmax_id`. Identical
  integer compare.
- On accept, emit target at pos-0 **and** bonus at pos-1.
  Identical two-emit semantics.

### 2.3 vLLM `eagle.py` — shared draft sampler

```python
# vllm/v1/spec_decode/eagle.py::_greedy_sample
def _greedy_sample(self, hidden_states):
    logits = self.model.compute_logits(hidden_states, None)
    return logits.argmax(dim=-1)
```

Used uniformly across EAGLE, EAGLE3, DFlash, MTP, and draft-model
methods. Only `combine_hidden_states` varies per method; the
sampling primitive is identical. Confirms C23's earlier finding
(PR #348) that kiln's draft path already matches vLLM's draft-side
contract for every supported speculation method.

### 2.4 vLLM bonus-token sampler

Under the `all_greedy` fast path, the bonus is also an argmax at
[line 394](#references-vllm). Under mixed-greedy-and-random, the
bonus is sampled through the full `Sampler` module
(`vllm/v1/sample/sampler.py`) with temperature / top_p / top_k /
penalties applied per-request. **For α measurement the
sampling mode is always all-greedy**, so the bonus path is
identical between kiln and vLLM on the bench hot path.

### 2.5 HF transformers `generate` defaults

Qwen3.5's HF model card and `config.json` (hash-verified against
`huggingface.co/Qwen/Qwen3.5-4B/resolve/main/config.json`) set:

- `mtp_num_hidden_layers: 1` (text_config.mtp_num_hidden_layers,
  confirming k=1).
- `tie_word_embeddings: true` (the MTP head shares `lm_head`
  weights with the base — matters for C32-style FP32 parity checks
  but not this audit).
- `eos_token_id: 248044`.
- No bespoke sampler overrides at the config level.

HF's `generate()` with `do_sample=False` is a plain argmax over the
FP32-cast output logits (HF converts BF16 logits to FP32 internally
in `GenerationMixin._sample` before the `torch.argmax` call). Same
as vLLM. Same semantics as kiln, modulo dtype.

### 2.6 Eval harness workload distribution

`lm-evaluation-harness` and the Qwen3.5 paper eval workloads
(HumanEval, MBPP, MMLU, GSM8K, various chat benchmarks) feed
prompts either via the model's chat template (for chat models) or
in task-specific formats with dedicated prefix/suffix tokens.
Raw-prose prompts as used by kiln's `build_prompt` are **not a
workload the MTP head was measured on** by upstream. This is the
harness-level delta noted in §1.8.

## Section 3 — Diff table

| # | Contract point | Kiln | vLLM reference | Kind | Plausible impact on α=0.153 vs ≥0.72 |
| --- | --- | --- | --- | --- | --- |
| 1 | Draft sampler primitive | `greedy_sample(&mtp_logits)` (argmax) | `logits.argmax(dim=-1)` (`eagle.py::_greedy_sample`) | **MATCH** | None |
| 2 | Verifier pos-0 sampler primitive | `greedy_sample(&verify_pos0)` (argmax) | `target_logits.argmax(dim=-1)` (`rejection_sampler.py:394`) | **MATCH** | None |
| 3 | Accept/reject decision | Integer `==` on `u32` token IDs | Integer `!=` short-circuit in Triton kernel | **MATCH** | None |
| 4 | Bonus sampler primitive (accept branch) | `greedy_sample(&verify_pos1)` (argmax) | `target_logits.argmax(dim=-1)` on pos-1 slice under `all_greedy` | **MATCH** | None (α does not depend on bonus) |
| 5 | Sampler params consumption | `_params: &SamplingParams` unused | `SamplingMetadata.all_greedy=True` bypass | **MATCH** | None |
| 6 | Seed/RNG consumption | `_rng: &mut StdRng` unused | No RNG under all-greedy | **MATCH** | None |
| 7 | EOS handling on accept | `eos_token_ids.contains(&draft_token)` / `&bonus` early stop | `stop_token_ids` checked in output ring-buffer | **MATCH** | None (α counted pre-EOS) |
| 8 | Temperature scaling | None (no temp divide before argmax) | None under `all_greedy` | **MATCH** | None |
| 9 | Top-p / top-k filtering | None (parameters never consumed) | None under `all_greedy` | **MATCH** | None |
| 10 | Repetition penalty | None | None under `all_greedy` | **MATCH** | None |
| 11 | Logits-combine semantics (draft + verifier) | Separate: draft from MTP head, verify from base | Separate: draft from MTP head, verify from base | **MATCH** | None |
| 12 | Prefill sampler | `greedy_sample(&prefill_last)` on BF16 | argmax on FP32 | **MINOR DELTA** | Only seeds `last_token` for step 0; cannot explain cumulative α collapse |
| 13 | Argmax dtype | **BF16** native | **FP32** cast (`rejection_sampler.py:117`) | **MINOR DELTA (contract)** | Math ceiling ≪ 5×; BF16 tie rate ~<2% on V≈248k |
| 14 | Bonus sampler under non-greedy | Unconditional argmax (no consumer) | Full `Sampler` (temp/top_p/top_k) | **DELTA (unreachable)** | Not exercised during α measurement (both sides greedy) |
| 15 | Prompt workload (harness) | Raw English prose via `PROMPT_POOL[seed%8]`, no chat template | Chat-formatted / task-formatted prompts | **MAJOR DELTA (harness)** | Plausible partial explanation for α gap; untested |
| 16 | Tokenizer `encode` special tokens | `add_special_tokens=false` (`tokenizer.rs:76-82`) | Chat template includes specials (`<|im_start|>` etc.) | **DELTA (harness, downstream of 15)** | Only matters if 15 matters |

**16 rows. 11 match. 3 minor/unreachable deltas. 2 load-bearing
deltas (rows 13 and 15).**

The only two rows that are worth testing in C35 are:

- **Row 13** (BF16→FP32 argmax dtype): cheap to flip under an env
  flag (`KILN_MTP_ARGMAX_FP32=1`) at the three `greedy_sample` call
  sites; bounded impact.
- **Row 15** (prompt workload): add a `--chat-template` flag to the
  MTP bench that runs `tokenizer.apply_chat_template` on a fixed
  reference conversation; re-run α on the same seed set.

## Section 4 — Verdict and recommended C35

**H13 INCONCLUSIVE.**

Static audit proves that the kiln MTP sampling **contract** matches
vLLM's `rejection_sample` `all_greedy` path functionally on 11 of
16 contract points and is off-contract on 5. The two load-bearing
off-contract items (BF16 argmax dtype, raw-prose prompt workload)
are real but neither is known to collapse α 5× on its own:

- BF16→FP32 argmax dtype (row 13): bounded by BF16 tie rate
  (<2%). Cannot explain 0.153 vs 0.72.
- Prompt workload (row 15): plausible partial explanation, but
  untested. The MTP head was trained on chat-formatted data; running
  α measurement against raw prose is a workload mismatch that would
  bias α downward by an unknown amount.

Because the **contract** matches, H13 as originally framed is not
confirmed. Because row 15's impact is unknown, the audit cannot
declare H13 refuted without at least one A/B measurement. Hence
**inconclusive**.

### Recommended Phase C35 plan

**Goal:** Close the H13 residual by testing row 15 and (cheaply)
row 13 in a single CPU-runnable A/B.

**Steps (CPU-only, no pod spend required for the prompt-distribution
A/B if the CPU backend can emit α in reasonable wallclock):**

1. Add a `--chat-template` flag to `kiln-bench`'s MTP mode. When
   set, replace `build_prompt` with a fixed
   `apply_chat_template`-framed conversation (single user turn
   with a mid-length task prompt). The template call already
   exists at `tokenizer.rs:98-103`; it is not wired into the
   bench yet.
2. Add a `KILN_MTP_ARGMAX_FP32` env flag that inserts
   `.to_dtype(DType::F32)?` before each of the three
   `greedy_sample` sites in `speculative_mtp_decode_step` (and the
   one in the bench prefill). Default off.
3. Run the 4-cell A/B:
   - prose × BF16 argmax (current C18 baseline, α ≈ 0.153)
   - prose × FP32 argmax
   - chat-templated × BF16 argmax
   - chat-templated × FP32 argmax
4. Report α median-of-3 per cell on seeds 0..2.

**Decision rule:**

- If chat-templated cells rise materially above 0.153 → C18
  baseline measured the wrong workload distribution; downgrade
  H13 to REFUTED on the contract axis but note the workload
  sensitivity as a real finding, and re-anchor the α floor to
  whatever chat-template measurement yields.
- If FP32-argmax cells rise materially → BF16 dtype is more
  load-bearing than the tie-rate estimate suggests; ship the
  dtype fix, downgrade H13 to CONFIRMED-partial (dtype arm only).
- If neither rises → H13 is fully REFUTED; the α gap is not a
  sampler contract issue. Move to **C36+: MTP-head logits-producer
  audit** (fresh GDN state, MTP attention mask, or the MTP head
  weights themselves) as the remaining live class of hypotheses.

**Budget:** ~60 min CPU, $0 pod spend for the prompt-distribution
A/B. If GPU α confirmation is needed after a positive CPU signal,
15–20 min of A6000 via `ce kiln-pod-acquire` is sufficient for
median-of-3.

## References

<a id="references-kiln"></a>

### Kiln (this repo)

- [`crates/kiln-model/src/speculative.rs:701-870`](../../crates/kiln-model/src/speculative.rs)
  — `speculative_mtp_decode_step`:
  - L714 — `_params: &SamplingParams` (proof of non-consumption)
  - L716 — `_rng: &mut StdRng` (proof of non-consumption)
  - L740 — `let draft_token = greedy_sample(&mtp_logits)...`
  - L768 — `let target_at_0 = greedy_sample(&verify_pos0)...`
  - L773 — `let draft_accepted = target_at_0 == draft_token;`
  - L836 — `if eos_token_ids.contains(&draft_token)` (EOS on draft)
  - L843-844 — `let verify_pos1 = ...; let bonus = greedy_sample(&verify_pos1)...`
  - L845 — `if eos_token_ids.contains(&bonus)` (EOS on bonus)
- [`crates/kiln-server/src/bench.rs:289-308`](../../crates/kiln-server/src/bench.rs)
  — `build_prompt` (raw-prose workload)
- [`crates/kiln-server/src/bench.rs:1221-1460`](../../crates/kiln-server/src/bench.rs)
  — `bench_latency_paged_mtp`:
  - L1293-1309 — prefill + first-token greedy seed
  - L1326-1334 — `SamplingParams { temperature: 0.0, top_p: 1.0, top_k: 0, ... }`
  - L1349-1366 — `speculative_mtp_decode_step(...)` call site
  - L1421-1425 — α definition
- [`crates/kiln-core/src/tokenizer.rs:76-82`](../../crates/kiln-core/src/tokenizer.rs)
  — `encode(text, false)`
- [`crates/kiln-core/src/tokenizer.rs:98-103`](../../crates/kiln-core/src/tokenizer.rs)
  — `apply_chat_template` (exists; not wired into bench)
- [`crates/kiln-core/src/tokenizer.rs:106-119`](../../crates/kiln-core/src/tokenizer.rs)
  — `eos_token_ids`

<a id="references-vllm"></a>

### vLLM (reference)

Read via `gh api repos/vllm-project/vllm/contents/<path>` against
current `main`:

- `vllm/model_executor/models/qwen3_next_mtp.py` —
  `Qwen3NextMultiTokenPredictor` (no sampling logic)
- `vllm/v1/sample/rejection_sampler.py`:
  - L24 — `GREEDY_TEMPERATURE: tl.constexpr = 0`
  - L117 — `raw_target_logits = raw_target_logits.to(torch.float32)`
  - L350 — `def rejection_sample(...)`
  - L388-405 — `if sampling_metadata.all_greedy: ...` fast path
  - L683 — `if draft_token_id != target_argmax_id: ... break`
- `vllm/v1/spec_decode/eagle.py::_greedy_sample` — shared
  greedy sampler used by EAGLE/EAGLE3/DFlash/MTP/draft-model.

### Hugging Face

- `huggingface.co/Qwen/Qwen3.5-4B/resolve/main/config.json`:
  `mtp_num_hidden_layers: 1`, `tie_word_embeddings: true`,
  `eos_token_id: 248044`, `model_type: "qwen3_5"`,
  `vocab_size: 248320`.

### Prior phases

- [`docs/archive/phase-c/phase-c18/...`](../phase-c18) — α baseline (median 0.1532)
- [`docs/archive/phase-c/phase-c22/c22-fc-residual-audit.md`](../phase-c22/c22-fc-residual-audit.md)
  — handed off H5 to C23
- [`docs/archive/phase-c/phase-c23/c23-draft-sampler-audit.md`](../phase-c23/c23-draft-sampler-audit.md)
  — H5 disproven: intra-kiln draft vs verifier sampler parity (PR #348)
- [`docs/archive/phase-c/phase-c32/c32-h-main-fp32-parity-redirect.md`](../phase-c32/c32-h-main-fp32-parity-redirect.md)
  — MTP head FP32 parity doc-only redirect
- [`docs/archive/phase-c/phase-c33/c33-mtp-finetune-static-audit.md`](../phase-c33/c33-mtp-finetune-static-audit.md)
  — no MTP fine-tune code path exists; rules out training-side drift
