# True on-policy OPD in kiln — what's needed

This session demonstrated kiln's OPD loss path end-to-end:

- `kiln_opd_loss_kernel` (top-K reverse KL, renormalised, CUDA-fused)
  produces correct gradients and matches the candle reference.
- `kiln_train::opd::opd_train` now has gradient-checkpointed
  forward/backward, so long-context (>700-token) prompts no longer OOM on
  a 48 GB GPU (commit `e91469ff` then `aae04d1d`).

`opd_train` now supports both explicit modes. In `on_policy` mode it samples
fresh rollouts from the current student LoRA and asks a live teacher to score
those sampled tokens. In `off_policy` mode it replays assistant actions whose
teacher rows were materialized ahead of training. The production loop is:

```text
for each step:
    1. trajectories = sample(student_with_LoRA, prompts)
    2. teacher_logprobs = teacher.compute_logprobs(trajectories)
    3. reverse_kl = student_logprobs(traj) - teacher_logprobs(traj)
    4. mean(reverse_kl).backward()
```

The executable contract is intentionally bounded: reverse-KL over teacher
top-K with K 16 or 32. Cross-entropy, full-vocabulary loss, Stable-OPD,
discounted advantages, and importance-ratio clipping are rejected rather than
accepted as inert knobs. This is on-policy sampling with direct KL
optimization, not yet the paper's importance-sampled policy-gradient form.
The remaining work is below.

## 1. Teacher implementations

Kiln has two live sources today. `RemoteTeacher` speaks strict vLLM numeric-ID
`prompt_logprobs`; `LiveLocalTeacher` scores with the model already served by
Kiln, optionally wearing a registered LoRA. What remains is an independent
in-process base-model teacher. A local teacher does not yet load the arbitrary
model named by its registry `model_id`.

The student (Qwen3.5-4B at bf16) is 8 GB. A Qwen3.6-27B teacher at bf16 is
roughly 54 GB and does not fit alongside it in 48 GB. Options are:

- **(a) Quantised teacher in kiln.** kiln-marlin-gemm already supports
  FP8 weights (per grand plan §6). Plumb FP8 weight loading into
  `kiln_model::load_model_with_options` so a 27B FP8 fits in ~27 GB.
  The marlin path covers full-attention layers; GDN linear-attention
  layers still need bf16 weights or a separate quantised kernel.
- **(b) Use a same-tokenizer teacher that fits at bf16.** Anything in
  the Qwen3 family with ≤14B parameters fits with ~22 GB of headroom.
  Useful for capability demos that don't require the full 27B.
- **(c) RemoteTeacher over HTTP.** Stand vLLM up with a quantised 27B and
  point Kiln at its numeric-ID `prompt_logprobs` endpoint. This path is
  implemented; the remaining gate is an authoritative tokenizer, model, and
  adapter identity handshake rather than trust in configured hashes.

## 2. Sampling performance

`kiln_model::sampling` has the full machinery (temperature, top-p,
penalties, on-device sampler). `kiln_model::generate` exposes it for
inference. The OPD trainer needs to call it per-step:

```rust
let student_tokens = sample_with_lora(
    &student_weights,
    &lora_params,
    prompt_tokens,
    SampleConfig { temperature: 1.0, top_p: 0.9, max_new_tokens, .. },
)?;
let full_tokens = [prompt_tokens, &student_tokens].concat();
```

The loop above is implemented. The remaining performance gap is a KV-cached
sampler; the current correctness path reruns the growing prefix.

## 3. Fixed-fixture boundaries

`distill_merge` and privileged `distill/self` route fixed action sequences to
precomputed fixtures. They require `off_policy`, explicit assistant actions,
and exact replay data. Local pump/refresh use `LiveLocalTeacher` for on-policy
runs and a fixture only when the request selects off-policy replay.

## Why this matters

Off-policy distillation (training on teacher-generated text) and
on-policy distillation (training on student-generated text scored by
teacher) are mathematically distinct (Lu §1). Off-policy suffers from
exposure bias — the student is trained in states it doesn't visit at
inference. On-policy avoids this and is the recipe DeepSeek-V4 ships
with. The kiln OPD branch was designed for on-policy from the start
(grand plan §3.1); the trainer wire-up hasn't caught up yet.

## Concrete milestones

- [x] Gradient-checkpointed OPD step.
- [x] Strict `RemoteTeacher` HTTP adapter for vLLM numeric-ID
  `prompt_logprobs`, end-to-end-validated via
  `examples/remote_teacher_smoke.rs`. SGLang uses a different echo/logprobs
  envelope and remains rejected until it has a dedicated adapter and fixture.
- [x] `opd_train` sampling step: per iter, sample N completions from
  student, score with teacher, compute reverse-KL. Sampler runs
  through `model_forward_segment` so `KILN_STREAMING_PREFILL=1`
  applies — peak transient memory scales with the GDN tile size
  (default 8192 tokens on CUDA), not the prompt length. Long
  agentic trajectories work as long as one tile's intermediates fit.
- [ ] KV-cached fast-path for the sampler. Today's sampler is O(N²)
  (re-runs prefix forward each step) so the streaming env actually
  applies — fine for short rollouts and small datasets, slow for
  long ones. A KV-aware streaming variant is the next perf win.
- [x] `LiveLocalTeacher` over the currently served weights plus an optional
  registered LoRA.
- [ ] Independent local teacher over a second model/`GpuWeights`, with an
  authoritative content-identity handshake.
- [ ] `qwen3_5_27b()` ModelConfig (for users who can fit 27B in bf16
  on larger GPUs — also unlocks LocalTeacher).
- [ ] Stable-OPD (`β·KL_ref + λ·SFT_gold`) wired to per-step decisions.
- [ ] FP8 teacher loading in kiln_model (in-process 27B teacher path).
