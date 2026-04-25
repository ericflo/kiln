# Phase C33 — MTP head fine-tune budget + gradient-flow static audit

**Hypothesis (H12):** kiln's MTP head fine-tune recipe (loss weighting,
optimizer schedule, frozen-vs-trainable parameter set, gradient flow)
materially diverges from the upstream Qwen3.5 MTP training spec, in a
way that caps α at 0.153 regardless of the inference-side correctness
established by C15…C32.

**Verdict: H12 is N/A. Kiln has no MTP fine-tune path.** The MTP head
weights only ever come from the published Qwen3.5-4B checkpoint and
are never updated by kiln's SFT or GRPO loops. No code change in this
PR. Doc-only redirect.

**$0 pod spend. CPU-only static audit.**

## Why this is the next step after C32

C32 (PR #359) closed the Class B inference-side α-collapse audit:

- C15 (#338) found a 2.0–2.4× kiln/HF magnitude ratio on `h_main`.
- C18 (#341) fixed it at `forward.rs:5097-5138` (one-RMSNorm-behind on
  the LM-head path); α 0.000 → 0.153 (4.7×).
- C29 (#355) ran 49 paired kiln/HF dumps post-C18: **100% top-1 match,
  median Jaccard@5/@10/@20 = 1.0**.
- C25/C26/C28/C31 cleared the LM-head, weight reload, h_prev contract,
  and head trio.
- C30 cleared accept/reject + KV-rollback math.
- C32 confirmed h_main parity transitively.

The remaining α gap (0.153 actual → ≥0.72 published Qwen3.5 ceiling)
cannot live on the inference path under this evidence. C32's handoff
named three options for C33:

1. **MTP head fine-tune budget verification** ← this audit
2. Speculative-decoding sampler contract verify-side vs draft-side parity
3. Accept α=0.153 as kiln-native ceiling and reframe the goal

Option 1 is the highest-leverage starting point: if H12 is supported,
it pinpoints the actual root cause; if N/A or refuted, it definitively
rules out the entire training-side surface and lets us choose between
options 2 and 3 with confidence.

## Static-audit checklist (kiln vs upstream)

### Headline finding

`grep -E "mtp|MTP|MtpWeights|mtp_layers|mtp_fc|mtp_norm|num_predict_tokens|MtpConfig|multi_token" -r crates/kiln-train/src`
returns **exactly one match** in the entire `kiln-train` crate:

```
crates/kiln-train/src/trainer.rs:1627:            mtp: None,
```

That single line is in the `tiny_weights()` test helper (`#[cfg(test)]`
scope) which constructs a stub `GpuWeights` for unit tests. It is not
a training-time reference to MTP. **The training crate makes zero use of
MTP weights, MTP loss, MTP loss positions, MTP optimizer scope, MTP
warmup, or MTP target modules.**

### 1. Trainable parameter set

`crates/kiln-train/src/trainer.rs:35-43`:

```rust
const DEFAULT_TARGET_MODULES: &[&str] = &[
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
];
```

`TrainableLoraParams::initialize` (`trainer.rs:76-162`) walks
`weights.layers` (the **base-model** transformer layers) and creates
LoRA A/B `Var` pairs for each module in `DEFAULT_TARGET_MODULES`. It
does not walk `weights.mtp` and never references `mtp.fc`,
`mtp.pre_fc_norm_embedding`, `mtp.pre_fc_norm_hidden`, `mtp.layer.*`,
or `mtp.final_layernorm`.

`TrainableLoraParams::all_vars` (`trainer.rs:200-220`) returns only
the LoRA `Var`s for the seven base-model module names above. This is
the exact set passed to `loss.backward()` and updated by `sgd_step` /
`sgd_step_from_map` (`trainer.rs:1015-1066`).

**Verdict for C32 handoff item:** kiln's trainable parameter set is
LoRA-only over base-model attention/MLP projections. `mtp.fc`,
`mtp.norm`, and the MTP transformer layer's projections receive no
gradient updates. `lm_head` is tied weights (`embed_tokens.t()`,
`crates/kiln-model/src/weights.rs:321-322`) and is also frozen — both
because `embed_tokens` is loaded as a plain `Tensor` (not a `Var`,
`crates/kiln-model/src/forward.rs`), and because `lm_head` is not in
`DEFAULT_TARGET_MODULES`.

### 2. Loss weighting (MTP loss vs main-token CE loss)

There is no MTP loss in kiln. `cross_entropy_loss` (`trainer.rs:938`)
is invoked over `model_forward_head(...)` logits (next-token CE only).
There is no second loss term for MTP positions, no `mtp_loss_weight`
config, no scaling factor. Verdict: **no MTP loss to weight.**

### 3. Loss positions (single position vs multi-position)

`tokenize_for_training` (`trainer.rs:860`) produces a `label_mask`
covering assistant-response tokens. `cross_entropy_loss` (and its FLCE
counterpart `fused_linear_cross_entropy`) compute per-position CE on
the **next** token at each masked position — i.e. the standard
single-token-prediction loss on `lm_head` logits. No MTP-position loss
is computed. Verdict: **no MTP positions to score.**

### 4. Optimizer scope

The optimizer is plain SGD (`sgd_step`, `sgd_step_from_map`,
`trainer.rs:1015-1066`) iterating over `params.all_vars()`. Single
param group, single learning rate. There is no separate group for MTP
parameters because no MTP parameters are trainable. Verdict: **no MTP
optimizer scope.**

### 5. Warmup / LR schedule

There is no separate MTP-head warmup. The SFT loop uses a constant
learning rate (`config.learning_rate`, `trainer.rs:300-460`) for the
LoRA `Var`s. GRPO is identical (`trainer.rs:485-720`). Verdict: **no
MTP warmup to mismatch.**

### 6. Gradient flow into `lm_head`

`lm_head` is **not** a separately stored tensor. It is tied to
`embed_tokens` via the cached transpose `embed_tokens_t`
(`crates/kiln-model/src/weights.rs:228-230` and `321-322`,
`crates/kiln-model/src/forward.rs:4156-4165`). `embed_tokens` and
`embed_tokens_t` are loaded as plain `Tensor`s in
`crates/kiln-model/src/forward.rs` (no `Var` in the entire
`kiln-model` crate — `grep "candle_core::Var\|use.*Var" crates/kiln-model/`
returns nothing). Therefore `lm_head` is **frozen by construction**
during kiln SFT/GRPO. Even if someone added MTP loss positions, the
gradient path would terminate at the LoRA modules; nothing would flow
into `lm_head`. Verdict: **no MTP→`lm_head` gradient path; `lm_head`
is frozen on the existing main-token path too.**

### 7. Initialization of `mtp.fc`

The loader (`crates/kiln-model/src/loader.rs:569-624`) detects the
published `mtp.*` tensor set in the safetensors checkpoint and
populates `MtpWeights { fc, pre_fc_norm_embedding, pre_fc_norm_hidden,
layer, final_layernorm }` (`crates/kiln-model/src/weights.rs:232-245`)
directly from the checkpoint bytes. There is no
re-initialization-from-`lm_head` codepath, no Kaiming-init codepath,
and no checkpoint-resume-with-modified-MTP codepath. Verdict: **MTP
weights come from the published Qwen3.5 checkpoint as-is.**

### 8. Token sampling for MTP loss

N/A — there is no MTP loss, so there is no MTP target sequence to
sample with teacher-forcing or otherwise.

### Bonus: LoRA on the MTP path at inference

`crates/kiln-model/src/forward.rs:4440` (immediately above
`mtp_forward_step`) carries the explicit comment:

> *(small fraction of per-step cost). LoRA is not applied to MTP.*

So even if someone trained a base-model LoRA adapter via SFT, the
MTP head would not see it during draft-time inference. The MTP path
**always** runs the published Qwen3.5 weights, period.

## Structural conclusion

Kiln's MTP head behavior is fully determined by the published Qwen3.5-4B
checkpoint. No kiln-side code change to training, optimizer, loss, or
inference can move the MTP draft distribution. Combined with the C29
finding (100% top-1 match between kiln MTP logits and HF reference
MTP logits across 49 paired dumps), this means:

> **kiln's α = 0.153 is the published Qwen3.5-4B MTP head's α on
> kiln's decode prompt distribution and acceptance harness, equivalent
> to what HF reference would give on the same inputs.**

The 0.72 figure cited as the Qwen3.5 paper floor must come from a
different evaluation harness — different prompt distribution,
different temperature/sampling settings, or a different verifier
acceptance rule. None of these are training-side.

## Cross-reference: upstream is also inference-only

vLLM's `vllm/model_executor/models/qwen3_next_mtp.py` carries the
docstring `"""Inference-only Qwen3Next MTP model."""` on line 3.
vLLM, like kiln, consumes the published MTP head as-is and ships no
fine-tune path. SGLang follows the same pattern (per project notes;
not separately re-verified here). MTP heads are published pre-trained
by the model publisher; inference servers consume them.

This means H12 is also N/A by upstream consensus: **no in-tree
inference-server training recipe exists for MTP fine-tuning that kiln
could be diverging from.** A task that asks "audit kiln's MTP fine-tune
recipe against upstream" has no upstream baseline to audit against,
because no inference server in the ecosystem ships MTP fine-tuning.

## Recommended C34 (and project-level decision)

With option 1 ruled out by structural N/A, the C32 handoff narrows to:

### C34 (next agent task): option 2 — sampler contract parity

Verify that kiln's spec-decode acceptance rule matches the upstream
Qwen3.5 published evaluation harness. Specifically:

- **Verifier acceptance bound.** Greedy vs sampled, top-k/top-p
  applied or not, soft-acceptance probability ratio (Leviathan-style
  draft-target sampling vs strict argmax match), draft-token pruning
  before verifier scoring.
- **Temperature contract.** Whether the verifier uses the **same**
  temperature the draft was sampled at (or `temp=0` greedy, which is
  the typical eval setting). C23 (PR #348) cleared intra-kiln draft
  sampler drift but did not compare against the upstream eval harness.
- **MTP-position prompt distribution.** What prompts upstream
  evaluated α on (likely a code/math/text mixture from the Qwen3.5
  paper); kiln's current α=0.153 is measured on a different prompt
  set in `kiln-bench`.

This is a CPU-first audit: read kiln's
`crates/kiln-model/src/speculative.rs` accept/reject path, compare
against vLLM's `vllm/spec_decode/` and the Qwen3.5 paper's evaluation
methodology section. Likely outcome: a sampler-contract mismatch is
found and fixed (could lift α materially), OR the upstream eval
harness uses different prompts/temperatures and the gap is a
benchmark methodology gap rather than an inference-correctness gap.

### Project-level decision (escalate to Eric)

C34 may close the gap entirely or surface that 0.72 is achievable only
under a different evaluation contract. Either way, the kiln-native α
ceiling on real workloads needs an explicit project decision:

- **(a) Accept α≈0.15 as the kiln-native ceiling.** Reframe the
  decode-speed-gap-vs-vLLM goal in terms of the actual achievable
  speedup. The published 0.72 stays as a benchmark target, not a
  release gate.
- **(b) Build a kiln-side MTP fine-tune path.** Add a third training
  endpoint (`/v1/train/mtp` or extend SFT/GRPO with MTP loss
  positions) that constructs `mtp.fc`, `mtp.pre_fc_norm_*`,
  `mtp.layer.*`, and `mtp.final_layernorm` as `Var`s, computes a
  multi-position MTP loss alongside main-token CE, and ships an
  MTP-aware adapter format. This is non-trivial: candle's autograd
  needs `Var`s built into `MtpGpuWeights` from load time, and the
  PEFT format needs extending to carry MTP adapter weights. Estimate:
  multiple days of engineering, plus a real training-data pipeline
  for MTP target sequences.
- **(c) Train a new MTP head from scratch.** Replace the published
  Qwen3.5 MTP checkpoint with one trained on kiln's own decode
  distribution. Highest cost, highest potential ceiling.

Option (a) is the cheapest and aligns with the rest of the open-source
ecosystem (vLLM and SGLang also accept whatever α the published MTP
head delivers). Options (b)/(c) are project-scope changes that
materially increase kiln's surface area beyond inference-server.

## Cost

$0 pod spend. ~30 min wall-clock (CPU-only preflight + source
inspection + doc write). Same pattern as PRs #131, #163, #164, #170,
#358, #359 — doc-only redirect when the static audit conclusively
rules out the hypothesis.

## Test plan

- [x] `kiln-train` MTP search returns one cfg(test) match only
      (verified via grep on full `crates/kiln-train/src/`).
- [x] `DEFAULT_TARGET_MODULES` enumerates seven base-model projections
      (verified at `trainer.rs:35-43`).
- [x] `TrainableLoraParams::all_vars()` covers only LoRA pairs
      (verified at `trainer.rs:200-220`).
- [x] `cross_entropy_loss` and `fused_linear_cross_entropy` are the
      only loss functions invoked (verified at `trainer.rs:938-1003`
      and call sites in `checkpointed_forward_backward` /
      `standard_forward_backward`).
- [x] `model_forward_head` is the only head used for SFT/GRPO logits;
      `mtp_forward_step` is **not** invoked from `kiln-train`
      (verified by `grep "mtp_forward_step" crates/kiln-train/`
      returning empty).
- [x] `lm_head` is tied weights, no separate `Var`
      (verified at `weights.rs:228-230` and the absence of any `Var`
      construction in `kiln-model`).
- [x] LoRA is explicitly not applied to the MTP path
      (verified at `forward.rs:4440`).
- [x] vLLM upstream declares MTP "Inference-only"
      (verified by `curl https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/models/qwen3_next_mtp.py` line 3).
- [ ] No pod time spent. No new empirical artifacts.

## Pattern

Same pattern as PRs #131, #163, #164, #170, #358, #359 — doc-only
redirect when preflight static audit shows the hypothesis is
structurally inapplicable. C33's contribution to the bisect: closes
the entire training-side surface for the α-collapse investigation.
