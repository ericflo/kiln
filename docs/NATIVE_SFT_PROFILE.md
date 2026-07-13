# Native SFT Profile

Kiln native SFT is a bounded, single-GPU online LoRA microtrainer. It is not a
general replacement for Transformers, TRL, Accelerate, or a distributed
trainer. This document is the normative contract for its only training-shape
profile:

```json
{
  "config": {
    "training_profile": "native_online_lora_v1"
  }
}
```

The field may be omitted for compatibility with older clients; omission
selects `native_online_lora_v1`. Kiln's CLI and browser UI send it explicitly,
and the effective configuration in train receipts and exact checkpoints
records it. Unknown profile names, unknown SFT config fields, unknown optimizer
fields, and invalid numeric values fail before queue publication or GPU work.

## Update contract

For the main-model LoRA phase:

- One admitted conversation is one microbatch.
- One microbatch produces one optimizer update. Gradient accumulation is 1.
- An epoch visits every admitted row exactly once. Total main-phase optimizer
  steps are `epochs * rows_kept`.
- Each epoch uses a deterministic seed-derived permutation. Exact checkpoints
  retain the epoch, permutation, and next row cursor.
- The loss is the mean next-token cross-entropy over supervised assistant
  targets in that conversation. Rows are not combined into a token-weighted or
  conversation-weighted batch reduction.
- The assistant label mask is defined separately in
  [SFT Tokenization and Assistant-Only Loss](sft-tokenization.md).
- The configured learning rate is constant from the first update through the
  last. There is no warmup, decay, scheduler transition, or per-row scaling.
- There is no gradient clipping. A non-finite loss or optimizer failure stops
  the run; Kiln does not silently skip the update.
- Gradient checkpointing may recompute model segments to reduce memory. It
  does not change the microbatch, loss, update count, or accumulation contract.

The exact-checkpoint scheduler record makes these fixed values inspectable:

```json
{
  "kind": "constant",
  "state": {
    "training_profile": "native_online_lora_v1",
    "microbatch_conversations": 1,
    "gradient_accumulation_steps": 1,
    "warmup_steps": 0,
    "gradient_clipping": "none"
  }
}
```

Requests containing general-trainer knobs such as
`per_device_train_batch_size`, `gradient_accumulation_steps`,
`lr_scheduler_type`, `warmup_steps`, or `max_grad_norm` are rejected rather
than ignored or approximately emulated.

## Backend-owned SFT loss routing

The SFT loss implementation is a backend capability, not a request field or
operator-selected mode. The current capability mapping is:

| Backend | Reported route | Gradient-checkpoint compatibility |
| --- | --- | --- |
| CUDA | `kt_tape_flce` | Uncheckpointed and checkpointed |
| ROCm | `kt_tape_flce` | Uncheckpointed and checkpointed |
| Vulkan | `vulkan_active_rows` | Uncheckpointed and checkpointed |
| Metal | `full_logits` | Uncheckpointed only; a multi-segment checkpoint plan is rejected |

This table describes the source-declared execution contract. It is not a
hardware-qualification receipt and does not establish correctness or
performance on a particular device, driver, or model.

`kt_tape_flce` makes the fused, chunked cross-entropy operation a root on the
kt autograd tape. `vulkan_active_rows` gathers supervised rows and uses the
Vulkan active-row loss shaders. `full_logits` materializes the portable
sequence-by-vocabulary logits and their cross-entropy forward/backward state.
Checkpoint tails execute outside an active kt tape, so `full_logits` cannot be
used with more than one gradient-checkpoint segment. Admission rejects that
combination with `training_invalid_request`; the trainer independently checks
the same invariant before a forward.

The route is deliberately absent from `SftConfig`, the request JSON, and the
typed `[training]` configuration. Consequently there is no TOML field and no
mechanically derived environment name for it. The former `KILN_USE_FLCE`
switch has been removed: it is not a compatibility alias, is not accepted by
the typed loader, and has no effect on current SFT routing. Changing the
process environment cannot override the resident backend's capability.

### Admission and execution binding

Before publishing a queue entry, the server reads the route from the resident
model runner and uses that exact enum in the SFT working-set estimate. The
estimate is intentionally route-specific:

- `kt_tape_flce` charges low-precision CUDA/ROCm F32 head promotion when
  required, the active-row gather, bounded vocabulary-chunk temporaries, and
  the full hidden gradient;
- `vulkan_active_rows` charges the largest legal vocabulary chunk, its F32
  weight slice and transpose, active-row buffers, and the full hidden gradient;
  and
- `full_logits` charges the dense `[T, V]` logits plus the portable
  cross-entropy forward and backward residency, including cast-back storage
  for a low-precision model.

Sequence length, maximum supervised-token count, LoRA and optimizer state,
streaming-prefill scratch, and the resolved checkpoint-boundary layout are
combined with this workspace. All size arithmetic saturates toward rejection,
so overflow cannot wrap a request into an apparently small allocation. For a
checkpoint-compatible route, automatic planning tries legal segment counts
and admits only a plan whose complete upper bound fits. `full_logits` remains
a one-segment plan and is never made admissible by silently selecting an
unsupported checkpoint route.

A request that exceeds the available training budget returns HTTP 413 before
queue publication. Its message includes estimated and available GiB plus a
breakdown containing `loss workspace ... (route=<route>)`, activation and
boundary memory, LoRA parameters and gradients, optimizer state, and residency
scratch. Operators can therefore distinguish a loss-route workspace problem
from a rank, sequence-length, or general capacity problem without changing a
hidden switch.

Admission does not merely sample a route and forget it. The selected enum is
stored in `PreparedSftAdmission`. After any queue wait, the worker compares it
with the resident runner again before governor reservation, KV replacement, or
allocator reclamation. The job-local `TrainingRuntimeContext` then carries the
pinned route into the trainer, which compares it with a freshly constructed
execution backend before resident-weight or trainable allocation. A mismatch
fails the job instead of estimating one algorithm and executing another. Every
standard and checkpointed SFT step receives the pinned enum rather than
re-reading backend state or process environment.

New SFT receipts record the executed enum at
`train_receipt.json -> runtime.sft_loss_route`. The field is optional only so
readers can consume legacy and non-SFT receipts. Exact SFT checkpoints bind the
same enum in `kiln.training-checkpoint-planning.v4`; changing it, or attempting
to resume an SFT checkpoint with the older v3 planning identity, is planning
drift. GRPO and OPD continue to use the common v3 planning identity because
this SFT route is not their loss-routing authority.

## Optimizers and learning rate

`config.optimizer` is a tagged object. Omission selects Muon with its defaults,
and kind-only AdamW or Muon objects also select their defaults. The expanded
forms below show every optional field; unknown fields are rejected:

```json
{"kind":"sgd"}
{"kind":"adam_w","beta1":0.9,"beta2":0.999,"eps":1e-8,"weight_decay":0.0}
{"kind":"muon","momentum":0.95,"nesterov":true,"ns_iters":5,"weight_decay":0.0}
```

Thus `{"kind":"adam_w"}` and `{"kind":"muon"}` are valid shorthand for
the displayed defaults. SGD has no optimizer-specific fields.

AdamW `beta1` and `beta2` must be finite and in `[0, 1)`, `eps` must be
finite and greater than zero, and `weight_decay` must be finite and
non-negative. Muon `momentum` must be finite and in `[0, 1)`, `ns_iters` must
be in `1..=20`, and `weight_decay` must be finite and non-negative.
`learning_rate`, when supplied, must be finite,
positive, and remain positive and finite after F32 conversion.

If `learning_rate` is omitted, native training resolves these constants:

| Optimizer | SFT | GRPO / OPD |
| --- | ---: | ---: |
| Muon | `1e-3` | `2e-3` |
| AdamW | `1e-4` | `1e-5` |
| SGD | `1e-4` | `1e-5` |

LoRA rank must be positive and remains subject to model-shape and live-memory
admission. Native Muon additionally requires rank `2..=48` on CUDA and ROCm and
`2..=32` on Metal and Vulkan; rank 1 would skip orthogonalization, while higher
ranks exceed those kernels' qualified shared-memory envelope. CPU reference
Muon requires rank 2+ but adds no backend-specific maximum. AdamW and SGD also
add no backend-specific rank ceiling. Their `backend_maximum` is therefore
null, while their effective `maximum` remains bounded by the resident model and
live-memory admission.
Metal does not implement a native SGD update and rejects it. The server checks
the cheap per-workload substrate and optimizer tuple before checkpoint or
corpus materialization, repeats both checks at dequeue before memory
reservation, and repeats the tuple before device residency. This ordering
applies to dedicated SFT/GRPO/OPD endpoints, the intent-tagged training front
door, recipes, judge/self-improve, the distinct DistillRefresh workload, and
every OPD-backed distillation route.
Invalid optimizer kind, rank, or hyperparameters return HTTP 400 with
structured `training_invalid_request`. An unsupported base dtype, mismatched
backend/device identity, or unavailable workload substrate returns
`training_backend_unsupported`.

## Precision and optimizer state

Kiln deliberately represents optimizer support at three static layers and one
dynamic layer:

1. The backend implementation says whether an update hook exists for a kind,
   which parameter dtypes it accepts, and whether its route is
   `native_device_hook` or `portable_reference`. CPU never claims a native
   device hook.
2. The resident optimizer tuple adds the exact backend/device identity,
   base-weight dtype, resolved LoRA dtype, fixed round-to-nearest policy, kind,
   and static LoRA-rank range.
3. Per-workload admission adds the complete SFT, GRPO, OPD, or DistillRefresh
   execution substrate. Only a supported workload's `allowed_optimizer_kinds`
   is a server-training promise.
4. Live-memory admission evaluates the concrete request after those cheap
   checks. It may reject a supported workload/tuple, but never lowers rank,
   substitutes an optimizer, or switches to a host route.

For canonical Qwen3.5-4B, whose smallest trained projection gives a
`model_maximum` of 1024, the resident optimizer-tuple matrix is:

| Backend | Supported base dtype | Resolved LoRA dtype | SGD | AdamW | Muon |
| --- | --- | --- | --- | --- | --- |
| CPU portable reference | F32 | F32 | rank 1..=1024 (backend unbounded) | rank 1..=1024 (backend unbounded) | rank 2..=1024 (backend unbounded) |
| CUDA | F32 / BF16 | same as base | rank 1..=1024 (backend unbounded) | rank 1..=1024 (backend unbounded) | rank 2..=48 |
| ROCm | F32 / BF16 | same as base | rank 1..=1024 (backend unbounded) | rank 1..=1024 (backend unbounded) | rank 2..=48 |
| Metal | BF16 | BF16 | unsupported | rank 1..=1024 (backend unbounded) | rank 2..=32 |
| Vulkan tuple | F32 / BF16 | F32 | rank 1..=1024 (backend unbounded) | rank 1..=1024 (backend unbounded) | rank 2..=32 |

This is not an executable-workload matrix. Current CPU server workloads remain
unsupported even though the portable F32 optimizer tuples are exposed for
diagnostics and direct-library testing. The normal hybrid Vulkan server has
CPU-host model weights and is rejected before data admission even though raw
Vulkan hooks and tuples can exist. A future Vulkan-resident tuple may likewise
remain unusable for one workload when its tape or loss route fails that
workload's gate. F16 remains inference-only on CUDA and ROCm.

`GET /v1/config -> training.optimizer_support` reports schema
`{"id":"kiln.training-optimizer-support","version":1}` plus the resident
`backend`, `device`, `base_weight_dtype`,
`resolved_lora_parameter_dtype`, `immutable_after_startup`, product
`rounding_modes`, and `backend_implementation_rounding_modes`. The object is
`null` for a mock runner. `optimizer_tuple_kinds` summarizes the resident tuple
kinds. Each `optimizers[]` member contains `kind`, `backend_implementation`, and
`optimizer_tuple {supported, unavailable_reason, lora_rank}`. Every rank object
has `minimum`, effective `maximum`, optional `backend_maximum`, concrete
`model_maximum`, and `live_memory_admission_required=true`. `maximum` is the
minimum of the backend and model ceilings. A null `backend_maximum` means only
that the optimizer backend is unbounded; `maximum` remains the model ceiling.
The model ceiling is the smallest input/output dimension across Kiln's uniformly
ranked trained projections, so a higher rank would no longer be a low-rank
update. Live memory can reject a lower rank without changing the static fields.

The `workloads` array contains exactly `sft`, `grpo`, `opd`, and
`distill_refresh`, each with
`supported`, `unavailable_reason`, and `allowed_optimizer_kinds`. Static
workload admission requires a real readable runner, a serving profile that
grants training GPU ownership, agreement between configured and resident
weight devices, a runtime that resolves those weights, exact native
backend/device identity, no Marlin-packed projection, and authoritative
`kt_tape_authoritative` forward/backward. SFT additionally rejects a
multi-segment checkpoint plan on the `full_logits` loss route. OPD additionally
requires its loss and phase-B backward routes. A failed workload has an empty
allowed-kind list even when `optimizer_tuple_kinds` is non-empty.

`distill_refresh` is deliberately unsupported on every backend today. It is a
distinct sequential workload, not an OPD alias: phase one applies SFT to new
knowledge and phase two uses OPD to restore behavior. Its stable
`unavailable_reason` is `distill_refresh is unavailable until admission pins
separate exact SFT and OPD phase plans, prepares the exact SFT rows, and reserves
the maximum sequential working set`. The route cannot become supported until
admission binds separate exact plans for both phases, materializes the precise
SFT rows as part of that plan, and reserves the larger phase peak for the
sequential execution. An OPD-only plan, a lazy SFT-row load, or the sum of two
non-overlapping phase estimates does not satisfy that contract.

`GET /v1/recipes` also returns
`admission {supported, unavailable_reason}` for each built-in recipe after
checking every step's workload and exact optimizer/rank tuple. This is a static
preview, not a live-memory reservation; recipe submission preflights every step
again before it loads any checkpoint, materializes a remote/local teacher,
scans a corpus, runs memory preflight, or reserves GPU capacity. Cheap teacher
alias validation and metadata pinning may happen first to preserve request-error
ordering. Any recipe with a DistillRefresh step therefore reports unsupported
with the same stable reason.

For canonical BF16 Qwen3.5-4B weights, concrete runtime storage is:

| Backend | LoRA parameters | Activations | Gradients | Resident optimizer state | Loss accumulation |
| --- | --- | --- | --- | --- | --- |
| CUDA | BF16 | BF16 | BF16 | BF16 | F32 |
| ROCm | BF16 | BF16 | BF16 | BF16 | F32 |
| Metal | BF16 | BF16 | BF16 | BF16 | F32 |
| Vulkan | F32 | F32 | F32 | F32 | F32 |
| CPU reference | F32 | F32 | F32 | F32 | F32 |

The BF16 paths do not keep a separate F32 master parameter. AdamW uses first-
and second-moment buffers; Muon uses one momentum buffer; SGD is stateless.
Resident state uses the table's runtime dtype. Exact checkpoints serialize
optimizer arrays as F32 safetensors plus per-parameter step counters, then
restore them into the declared runtime dtype. That portable serialization is
not an F32 master copy used by ordinary updates.

Product training is fixed to round-to-nearest for every supported optimizer
tuple. Stochastic rounding remains an explicit programmatic optimizer-library
policy for experiments; it is not selectable by server config, a request, or
the environment. `KILN_BF16_STOCHASTIC_ROUND` and the backend/debug optimizer
fallback variables have been removed, have no compatibility aliases, and must
be deleted from service definitions. A legacy exact checkpoint that records
stochastic rounding cannot resume into the round-to-nearest product policy;
precision-policy comparison fails closed before GPU ownership.
The concrete parameter, optimizer-state, activation, gradient, and rounding
record for a completed run is stored under
`train_receipt.json -> runtime.training_precision` and copied to the adapter
manifest.

### AdamW numerical qualification

The committed
`crates/kiln-optim/tests/fixtures/adamw_pytorch_oracle_v1.json` fixture records
every parameter, first moment, and second moment after ten ordinary-gradient
and ten epsilon-dominated low-gradient updates. It is generated by
`scripts/qualification/adamw_pytorch_oracle.py` from source-pinned PyTorch
2.13.0 using eager `torch.optim.AdamW` with `foreach=false` and `fused=false`.
The fixture covers both F32 and BF16 parameters and same-dtype moments, with no
separate master parameter.

The portable F32 reference must match every recorded lane at every step within
`2e-12 + 5e-6 * abs(expected)`. A production F32 backend uses the same bound.
Production BF16 kernels compute a fused lane in F32 and round each output once,
while eager PyTorch BF16 rounds between separate tensor operations. Their
declared per-step comparison envelope is therefore one BF16 ULP for parameters,
four for first moments, and three for second moments. These are field-specific
maximums observed and enforced across both ten-step cases, not a general
bitwise-equivalence claim.

`qualification/workloads/adamw-pytorch-oracle-v1.json` runs the fixture
contract, portable reference, and real native kernel as one fail-closed
hardware workload. A backend is qualified only by a committed passing receipt;
a compile-only check or a missing-device skip is not hardware evidence.

## Optional MTP alignment

The standalone `kiln-train` library retains an offline MTP-alignment mode. When
the base checkpoint contains native `mtp.*` weights, `train_mtp: null` uses the
library's historical automatic behavior, `false` disables it, and `true`
requests it explicitly. This separate post-SFT pass:

- trains only the MTP block's LoRA parameters;
- visits each admitted conversation at most once, independently of `epochs`;
- uses the run's optimizer and constant learning rate with fresh optimizer
  state;
- is outside the main-phase step count and exact-resume cursor; and
- is skipped when the base model has no MTP weights.

The live server does not permit this phase. Every server SFT admission
normalizes an omitted or explicit-false value to `train_mtp: false`; explicit
`true` returns `training_invalid_request` before the job is published. The
worker checks the normalized value again before corpus or GPU work. This is
fail-closed because the alignment phase does not yet participate in the
server's GPU-step coordination, memory reservation, progress cancellation, or
settlement contracts and can otherwise materialize deferred MTP weights while
inference is active. Use the offline library only in an isolated process until
that phase has passed the same accelerator qualification gates as other
server-owned training.

For offline library use, the main adapter remains usable if automatic MTP
alignment fails; the library logs the failure and omits the MTP LoRA tensors.
This auxiliary phase therefore must not be interpreted as part of the exact
main-phase continuation guarantee.

## Artifacts and resume

`train_receipt.json -> config` contains the full effective profile, resolved
learning rate, optimizer settings, seed, LoRA shape, row policy, and checkpoint
settings. `runtime.training_precision` records the concrete dtype contract and
`runtime.sft_loss_route` records the backend-owned loss implementation.
Exact `.kiln-checkpoint` manifests additionally bind the fixed scheduler state,
optimizer state, data order/cursor, RNG state, admitted-corpus identity, model
artifacts, tokenizer/template, execution provenance, and the SFT v4 planning
identity that contains the pinned loss route.

Resume first passes the current cheap SFT workload and resident optimizer-tuple
gates, then validates the checkpoint before GPU ownership. The checkpoint's
backend/device, base and LoRA dtypes, optimizer kind and state, rank, immutable
rounding mode, execution provenance, and loss/checkpoint route must remain
compatible. A legacy stochastic checkpoint, a changed native identity, a newly
Marlin-packed projection, or an unavailable authoritative tape/loss route fails
closed; Kiln does not reinterpret the artifact through another optimizer or
backend.

An ordinary PEFT adapter is a serving artifact, not an exact training
checkpoint. See [Exact Training Checkpoints](training-checkpoints.md) and
[Train Receipt Schema](TRAIN_RECEIPT_SCHEMA.md).

SFT checkpoints created before the versioned profile and fixed scheduler fields
cannot prove this contract and are not accepted for exact continuation. Their
PEFT adapter snapshots remain serving artifacts and may be selected as a new
run's `base_adapter` subject to the ordinary shape and provenance checks.

## General training boundary

Use HF/TRL directly when a run needs batching, gradient accumulation,
schedulers or warmup, clipping, packing, full-parameter training, alternate
PEFT methods, distributed execution, or broad model-family support. Kiln ships
first-party SFT and recorded-GRPO export commands, pinned HF/TRL runners, and a
resident-identity-validated PEFT import command. Use that workflow when Kiln's
row, template, split, rollout-provenance, and artifact-identity contracts must
survive external training; an arbitrary external command does not acquire
those guarantees automatically. The versioned handoff schema, commands, exact
validation boundary, and production round-trip evidence are defined in
[HF/TRL Interoperability Contract](HF_TRL_INTEROP.md).
