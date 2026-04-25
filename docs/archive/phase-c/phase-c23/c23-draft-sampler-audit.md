# Phase C23 — MTP draft sampler / temperature / penalty drift audit (hypothesis 5, doc-only redirect)

## Verdict

**Hypothesis 5 is DISPROVEN on static audit. No code change. No pod
spend.** Kiln's MTP draft sampler, verifier pos-0 sampler, and
verifier bonus pos-1 sampler are all the same primitive
(`greedy_sample`) called against the same dtype tensor with no
sampler parameters threaded through. The `SamplingParams` argument
to `speculative_mtp_decode_step` is underscore-prefixed
(`_params`) at the function boundary, proving statically that
temperature, top-k, top-p, repetition penalty, presence/frequency
penalty, logit processors, and seed/RNG cannot affect the draft
side at all. There is no sampler-drift surface between draft and
verifier — both are bit-identical greedy argmax, which matches
vLLM's documented `_greedy_sample` contract for EAGLE/MTP/DFlash
draft and the `all_greedy=True` branch of `rejection_sample` on the
verifier. Hand off the residual α gap (C18 median 0.1532, floor
0.5) to **H3 hardware A/B** as the last cheap discriminator
(`abs_pos = base_pos + mtp_pos` vs `abs_pos = base_pos - 1 +
mtp_pos`, ~25 min A6000, ~$5). If H3 also inconclusive, escalate to
full MTP weight reload + bit-level verifier/draft logit diff on a
small calibration set.

## C22 hypothesis as stated

From [`docs/archive/phase-c/phase-c22/c22-fc-residual-audit.md`](../phase-c22/c22-fc-residual-audit.md)
("Next: kick off Phase C23 (H5)"):

> Open a Phase C23 task to audit the draft-side sampler against the
> verifier-side sampler and the vLLM EAGLE/MTP drafter reference.
> Same $0-pod CPU-only static audit pattern as C19/C20/C21/C22. If
> H5 also disproves cleanly, run the H3 hardware A/B (~$5, ~25 min
> A6000) as the last cheap discriminator before escalating.

The C18 hypothesis queue framed H5 as:

> **Draft-side sampler / temperature / penalty drift.** If the
> draft head and the verifier head are sampling logits with even
> mildly different temperatures, top-k cutoffs, top-p cutoffs, or
> penalty stacks, the draft will systematically diverge from the
> verifier and α will collapse — even if every preceding tensor is
> bit-identical.

The audit below disproves this on static reading: kiln does not
apply any sampler at all to the draft logits beyond an on-device
argmax, the verifier applies the same primitive against the same
shape, and vLLM's reference behaves identically.

## Tensors and counters in scope

For one `speculative_mtp_decode_step` invocation:

| Symbol | Shape | Source | Meaning |
| --- | --- | --- | --- |
| `mtp_logits` | `[1, V]` | `mtp_forward_step` → `lm_head_forward` | MTP draft head logits at draft pos |
| `draft_token` | `u32` | `greedy_sample(&mtp_logits)` | One draft token id |
| `verify_logits` | `[2, V]` | base model forward over `[real_token, draft_token]` | Verifier logits across 2 positions |
| `verify_pos0` | `[1, V]` | `verify_logits.narrow(0, 0, 1)` | Verifier slice at pos-0 (real-token verification) |
| `target_at_0` | `u32` | `greedy_sample(&verify_pos0)` | Verifier's argmax at pos-0; compared against `draft_token` |
| `verify_pos1` | `[1, V]` | `verify_logits.narrow(0, 1, 1)` | Verifier slice at pos-1 (bonus-token slot) |
| `bonus` | `u32` | `greedy_sample(&verify_pos1)` | Verifier's bonus token (used iff draft was accepted) |

Three sampling sites; all three identical primitive; none consume
sampler parameters.

## Evidence

### 1. Draft sampler is greedy argmax with `_params` provably unused

[`crates/kiln-model/src/speculative.rs:533-577`](#references-kiln)
defines `speculative_mtp_decode_step`. The signature accepts a
`SamplingParams` argument but binds it to `_params`, the standard
Rust convention for "argument intentionally unused":

```rust
pub fn speculative_mtp_decode_step(
    model: &Model,
    state: &mut SpeculativeState,
    next_token: u32,
    _params: &SamplingParams,                  // <-- underscore = unused
    sequence_id: u64,
) -> Result<SpeculativeMtpStepOutput> {
    ...
    let draft_token = greedy_sample(&mtp_logits)
        .context("mtp draft sampling failed")?;   // line 577
    ...
}
```

The function body never references `_params`, so the rustc
unused-variable lint would fire on the non-prefixed name. The
underscore prefix is a static, compile-checked guarantee that no
draft-side branch reads temperature, top-k, top-p, repetition
penalty, presence/frequency penalty, seed, or any other field of
`SamplingParams`. The doc comment immediately above the function
([`speculative.rs:533-536`](#references-kiln)) states this
explicitly: "Greedy-only path (temperature == 0). The stochastic
rejection-sampling variant is a follow-up; this implementation
takes the minimum viable path."

### 2. Verifier pos-0 sampler is the same primitive

[`crates/kiln-model/src/speculative.rs:600-606`](#references-kiln):

```rust
let verify_pos0 = verify_logits.narrow(0, 0, 1)?;
let target_at_0 = greedy_sample(&verify_pos0)
    .context("verify pos-0 sampling failed")?;   // line 604
```

Identical function (`greedy_sample`), identical input shape (`[1,
V]`), identical dtype, no sampler params consumed. The acceptance
predicate at [`speculative.rs:610-615`](#references-kiln) is a
direct integer compare:

```rust
let accepted = target_at_0 == draft_token;
```

If the two argmaxes match, accept the draft; otherwise reject. No
softmax, no temperature scaling, no top-k/top-p/penalty, no
probability ratio. This is precisely the all-greedy branch of
speculative decoding — the simplest possible verifier rule and the
one with zero sampler-drift surface.

### 3. Bonus sampler is the same primitive

[`crates/kiln-model/src/speculative.rs:692-695`](#references-kiln):

```rust
let verify_pos1 = verify_logits.narrow(0, 1, 1)?;
let bonus = greedy_sample(&verify_pos1)
    .context("bonus sampling failed")?;   // line 695
```

Same primitive again. The bonus token is only emitted when the
draft was accepted at pos-0; on rejection the verifier's
`target_at_0` becomes the recovery token instead.

### 4. The `greedy_sample` primitive is on-device argmax with no parameters

[`crates/kiln-model/src/sampling.rs:51-66`](#references-kiln):

```rust
pub fn greedy_sample(logits: &Tensor) -> Result<u32> {
    let argmax = logits.argmax(D::Minus1)?;
    let argmax = argmax.flatten_all()?.to_dtype(DType::U32)?;
    let argmax = argmax.to_vec1::<u32>()?;
    argmax
        .first()
        .copied()
        .context("argmax produced empty tensor")
}
```

Single-line semantics: argmax over the vocab dimension, materialise
the scalar token id, return it. No parameters, no RNG, no
penalties, no kernels other than candle's stock argmax. Deterministic
across runs; deterministic across draft and verifier call sites.
The full sampler `sample_with_params` exists in the same file
([`sampling.rs:68-...`](#references-kiln)) but is NOT called by any
MTP path.

### 5. Bench harness builds an all-greedy `SamplingParams` that would be greedy even if consumed

[`crates/kiln-server/src/bench.rs:1069-1077`](#references-kiln):

```rust
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

Every numeric field is the greedy-equivalent identity:
`temperature: 0.0` (no scaling), `top_p: 1.0` (no nucleus cut),
`top_k: 0` (no top-k cut), `repetition_penalty: 1.0` (no penalty).
Even on a hypothetical future kiln where `_params` is consumed,
this construction collapses to greedy argmax by definition. The
`seed` field is non-`None` but immaterial for argmax.

The bench then calls `speculative_mtp_decode_step(&model, &mut
state, next_token, &params, sequence_id)` at
[`bench.rs:1092-1108`](#references-kiln); the callee discards
`&params` per item (1) above.

For comparison, the "Off" baseline plain decode at
[`bench.rs:454`](#references-kiln) also uses
`next_token = greedy_sample(&logits)?;` — the same primitive on
the same dtype tensor. Both arms of the A/B harness sample from
identical distributions; α cannot move because of sampler asymmetry
between baseline and MTP arms either.

### 6. The skip-layer speculative path *does* consume `SamplingParams`, but it's a different function

[`crates/kiln-server/src/bench.rs:849`](#references-kiln) and
[`crates/kiln-model/src/speculative.rs`](#references-kiln) (function
`speculative_decode_step`, distinct from
`speculative_mtp_decode_step`) accept and consume `SamplingParams`.
This is not the MTP path under test in C18-C23 and does not affect
the MTP α measurement. It is mentioned only to head off the
question "but doesn't kiln plumb sampler params into speculative
decode somewhere?" — yes, in the *skip-layer* path, which is a
different draft model, different verifier glue, and orthogonal to
the MTP residual α gap.

### 7. vLLM's MTP/EAGLE/DFlash drafter is also greedy argmax, also discards request `SamplingParams`

[`vllm/v1/spec_decode/eagle.py:398-402`](#references-vllm)
(`EagleProposer._greedy_sample`):

```python
def _greedy_sample(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """Greedy-sample draft tokens from hidden states."""
    if self.use_local_argmax_reduction:
        return self.model.get_top_tokens(hidden_states)
    return self.model.compute_logits(hidden_states).argmax(dim=-1)
```

Called at [`eagle.py:484`](#references-vllm) for the single-token
draft path:

```python
draft_token_ids = self._greedy_sample(sample_hidden_states)
```

And at [`eagle.py:507`](#references-vllm) for the multi-token
extend path. The tree-attention extend variant uses
`logits.argmax(dim=-1).view(batch_size, -1)` directly at
[`eagle.py:1000`](#references-vllm) and
[`eagle.py:1138`](#references-vllm). Every code path is `argmax`
on the draft logits, with no `temperature` / `top_k` / `top_p` /
penalty argument anywhere in the call chain. The request-side
`SamplingParams` never reaches the drafter.

### 8. vLLM's verifier (`rejection_sample`) all-greedy branch matches kiln's verifier rule

[`vllm/v1/sample/rejection_sampler.py:350-405`](#references-vllm):

```python
def rejection_sample(
    draft_token_ids: torch.Tensor,
    ...
    target_logits: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    ...
    if sampling_metadata.all_greedy:
        is_greedy = None
    else:
        is_greedy = sampling_metadata.temperature == GREEDY_TEMPERATURE
    if not sampling_metadata.all_random:
        # Rejection sampling for greedy sampling requests.
        target_argmax = target_logits.argmax(dim=-1)
        rejection_greedy_sample_kernel[(batch_size,)](
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            target_argmax,
            bonus_token_ids,
            is_greedy,
            max_spec_len,
            ...
        )
        if sampling_metadata.all_greedy:
            return output_token_ids
```

The all-greedy path computes `target_logits.argmax(dim=-1)` and
compares against the precomputed `draft_token_ids`. No temperature
scaling, no top-k filter, no top-p filter, no penalty stack — same
contract as kiln's `target_at_0 == draft_token` predicate. The
fused triton kernel `rejection_greedy_sample_kernel` is a faster
implementation of the same equality check, not a different rule.

### 9. There is no model-level sampler hook in vLLM's MTP head

A sparse-clone grep over `vllm/model_executor/models/qwen3_next_mtp.py`
returns no `SamplingParams`, no `compute_logits` overrides that
shape distributions, no per-position penalty hooks, and no
sampler-aware logit processors specific to the MTP head. The MTP
head produces raw logits via `lm_head` (weight-tied to
`embed_tokens`), and the drafter's `_greedy_sample` argmaxes
those — exactly as kiln does via `lm_head_forward` →
`greedy_sample`.

## Parameter parity table

| Sampler axis | kiln draft | kiln verifier pos-0 | kiln bonus pos-1 | vLLM EAGLE/MTP draft | vLLM verifier (all_greedy) | Drift surface? |
| --- | --- | --- | --- | --- | --- | --- |
| Sampling rule | argmax | argmax | argmax | argmax | argmax | None |
| Temperature | n/a (not consumed) | n/a (not consumed) | n/a (not consumed) | n/a (greedy is unconditional) | n/a in all_greedy branch | None |
| Top-k | n/a | n/a | n/a | n/a | n/a | None |
| Top-p | n/a | n/a | n/a | n/a | n/a | None |
| Repetition penalty | n/a | n/a | n/a | n/a | n/a | None |
| Presence penalty | n/a | n/a | n/a | n/a | n/a | None |
| Frequency penalty | n/a | n/a | n/a | n/a | n/a | None |
| Logit processors | none | none | none | none | none in greedy path | None |
| RNG / seed | not used (deterministic) | not used (deterministic) | not used (deterministic) | not used (deterministic) | not used (deterministic) | None |
| Dtype before argmax | model output dtype (BF16) | model output dtype (BF16) | model output dtype (BF16) | model output dtype | model output dtype | None |
| Vocab dimension | last (`D::Minus1`) | last (`D::Minus1`) | last (`D::Minus1`) | last (`dim=-1`) | last (`dim=-1`) | None |
| Scalar emitted | u32 token id | u32 token id | u32 token id | int64 token id (compared as int) | int64 token id | None |

Every row is `n/a` or "None" because both implementations skip the
sampler entirely in greedy mode. There is no axis on which kiln's
draft and verifier could disagree by configuration. There is no
axis on which kiln and vLLM could disagree by design.

## What this rules out

This audit closes off the entire H5 hypothesis surface:

- **Asymmetric temperature** between draft and verifier: impossible
  — neither side scales logits.
- **Asymmetric top-k / top-p** between draft and verifier:
  impossible — neither side filters logits.
- **Asymmetric repetition penalty** between draft and verifier:
  impossible — neither side applies penalty.
- **Hidden logit processor on the MTP head only**: impossible —
  the MTP path runs `lm_head_forward` → `greedy_sample` with no
  intermediate processor.
- **Seed/RNG divergence**: impossible — argmax is deterministic.
- **Dtype mismatch between draft argmax and verifier argmax**:
  both arms argmax over the model's native logits dtype (BF16)
  and produce a u32 token id; no precision asymmetry.
- **vLLM behaves differently here**: vLLM's drafter and verifier
  both use unconditional argmax in greedy mode, matching kiln.

## What this does NOT rule out (handoff to H3)

This audit only proves that the *sampling* primitive cannot
explain α=0.1532. The remaining low-cost hypothesis is **H3
(rotary position-embedding off-by-one in the MTP draft step)**:

- Suspected drift: `abs_pos = base_pos + mtp_pos` (kiln current)
  vs `abs_pos = base_pos - 1 + mtp_pos` (alternative reading of
  the HF reference, parked from C21).
- Cost to discriminate: ~25 min A6000 wall-clock, ~$5 pod spend.
- Method: build with the alt formula behind a kill-switch, run a
  single MTP α bench at the C18 baseline configuration, compare
  median α against C18's 0.1532 floor.
- Decision: if the alt formula recovers α materially toward 0.5,
  ship it and close the residual gap; if it does not move α,
  H3 is also disproven and the next step is the bit-level
  verifier/draft logit diff on a calibration set (much higher cost,
  ~few hours engineering + ~$15-30 pod time).

This is the cheapest remaining discriminator. No new hypothesis
should be added ahead of H3.

## Why we didn't take a pod for this

C19, C20, C21, C22 all established the same pattern: when the
hypothesis is about a discrete parameter or wiring claim that can
be read directly from the source on both sides (kiln + vLLM), the
audit can be conclusive on a static read with zero pod spend. H5
fits this pattern exactly — sampler parameters either flow through
the function signature or they don't, and Rust's underscore-prefix
convention plus the explicit doc comment make "they don't" a
compile-checked invariant.

The Case-A path (drift found → fix + α-recovery bench) was reserved
for a real wiring asymmetry. None exists.

## References (kiln)

Audited at HEAD `0c92131` (PR #347, Phase C22 merged 2026-04-22):

- `crates/kiln-model/src/speculative.rs:533-577` —
  `speculative_mtp_decode_step` signature with `_params:
  &SamplingParams` (intentionally unused) and the draft
  `greedy_sample(&mtp_logits)` call.
- `crates/kiln-model/src/speculative.rs:600-606` —
  verifier pos-0 `greedy_sample(&verify_pos0)` call.
- `crates/kiln-model/src/speculative.rs:610-615` —
  acceptance predicate `target_at_0 == draft_token` (no
  probability ratio, no temperature scaling).
- `crates/kiln-model/src/speculative.rs:692-695` —
  bonus `greedy_sample(&verify_pos1)` call.
- `crates/kiln-model/src/sampling.rs:51-66` — the `greedy_sample`
  primitive; on-device argmax over the last dim, materialise
  scalar u32, no parameters.
- `crates/kiln-model/src/sampling.rs:68-...` —
  `sample_with_params` (full sampler, NOT called by any MTP path).
- `crates/kiln-server/src/bench.rs:1069-1077` — MTP bench builds
  `SamplingParams { temperature: 0.0, top_p: 1.0, top_k: 0,
  repetition_penalty: 1.0, ... }` and passes it to
  `speculative_mtp_decode_step` (which discards it).
- `crates/kiln-server/src/bench.rs:1092-1108` — the call site.
- `crates/kiln-server/src/bench.rs:454` — "Off" baseline plain
  decode also uses `greedy_sample(&logits)`; same primitive as
  draft and verifier.
- `crates/kiln-server/src/bench.rs:849` — skip-layer
  `speculative_decode_step` does consume `SamplingParams`; this
  is a different speculative path (not MTP) and is mentioned only
  to disambiguate.

## References (vLLM)

Sparse-cloned at `vllm-ref/` (read-only, CPU-only):

- `vllm/v1/spec_decode/eagle.py:398-402` — `_greedy_sample`:
  unconditional `argmax(dim=-1)` on the draft logits, no sampler
  params.
- `vllm/v1/spec_decode/eagle.py:484` — single-token draft call
  site `draft_token_ids = self._greedy_sample(sample_hidden_states)`.
- `vllm/v1/spec_decode/eagle.py:507` — multi-token extend draft
  call site (same `_greedy_sample`).
- `vllm/v1/spec_decode/eagle.py:1000,1138` — tree-attention
  extend path; raw `logits.argmax(dim=-1)`.
- `vllm/v1/sample/rejection_sampler.py:350-405` —
  `rejection_sample`: in the `all_greedy` branch,
  `target_argmax = target_logits.argmax(dim=-1)` and the fused
  `rejection_greedy_sample_kernel` compares against
  `draft_token_ids` directly. No temperature, no top-k, no
  top-p, no penalty.
- `vllm/model_executor/models/qwen3_next_mtp.py` — no
  model-level sampler hooks; MTP head emits raw logits via
  `lm_head` for the drafter to argmax.

## Next: H3 hardware A/B (the last cheap discriminator)

Open a Phase C24 task to discriminate H3:

1. Behind kill-switch `KILN_MTP_ROTARY_POS_MINUS_ONE` (or
   similar), implement the alternative `abs_pos = base_pos - 1 +
   mtp_pos` formula at the rotary position computation site
   identified in C21.
2. Acquire one A6000 from the pod pool: `ce kiln-pod-acquire
   --gpu-type 'NVIDIA RTX A6000'`.
3. Build with `KILN_CUDA_ARCHS=86 cargo build --release --features
   cuda --bin kiln-bench`.
4. Run the C18 baseline MTP bench unmodified (control), then run
   the same bench with `KILN_MTP_ROTARY_POS_MINUS_ONE=1` set
   (treatment). 3 runs each, median α.
5. Decision: if treatment median α is materially closer to 0.5
   than the C18 0.1532 floor, ship the fix as Case A (PR with
   bench evidence). If treatment median α is within noise of
   control, H3 is disproven and escalate to bit-level
   verifier/draft logit diff on calibration set.

Wall-clock budget for C24: 60 min / $20.
