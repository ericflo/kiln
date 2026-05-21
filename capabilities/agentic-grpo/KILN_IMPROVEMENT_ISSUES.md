# Kiln Improvement Issues For Agentic GRPO Work

This is a concrete backlog of improvements to kiln itself, informed by the
agentic-GRPO capability experiments under `capabilities/agentic-grpo/`.

Scope is intentionally narrow. This list excludes RunPod lifecycle, B2 backup,
GitHub workflow, `ce` pod-pool ergonomics, and cap-specific training recipes.
The tasks below are for kiln-server, kiln-train, kiln-model integration,
examples, CLIs, tests, and kiln-owned docs. If all of these were completed,
future agentic-GRPO work would be much more reliable, measurable, and fast to
iterate.

Each item is written so it can become a GitHub issue or task directly.

## Actual Qwen3.5-4B RunPod Validation Checkpoint

This section records the cross-issue real-model gate that was run after the
issue 1-26 implementation pass, before continuing with issue 27. It exists so
future sessions can recover the actual-model evidence without relying on chat
history.

**Status:** Passed on 2026-05-21.

**RunPod:** Direct A6000 pod `qmfxie9izl6lc6`
(`kiln-backlog-codex-20260520`), `NVIDIA RTX A6000`, 51.5 GB VRAM.

**Source validated:** The validation driver ran with `SKIP_GIT_RESET=1` on the
pod worktree at base commit `a5b359668173ae66d070a1f02bacd5be55d86d5d` plus
the issue 1-26 code/doc changes that were later pushed. A post-run comparison
against current `origin/main` (`fe82909b60d4bab4a49001f9de8abd4cd11cbbc9`)
showed the only remaining difference was `PROFILING.md`; no Rust, server,
training, script, or config source differed.

**Model:** `/workspace/Qwen3.5-4B` with `/workspace/qwen3.5-4b` kept only as a
compatibility symlink. Server model id was `Qwen3.5-4B`.

**Sentinel and artifacts:**

- Sentinel: `/workspace/kiln-validation/actual_model_validation.done` contains
  `exit=0`.
- Driver log: `/workspace/kiln-validation/actual-model-validation/driver.log`
  ends with `ACTUAL_MODEL_VALIDATION_OK`.
- Artifact directory:
  `/workspace/kiln-validation/actual-model-validation/`.

**Validation matrix:**

| Gate | Evidence |
| --- | --- |
| CUDA release build | Driver built `kiln` and CUDA training examples before runtime checks. |
| Actual model server boot | `/health` returned `Qwen3.5-4B (32L, 16H, 4KV)`, backend `model`, `eval_mode=true`, `default_thinking_enabled=false`, and `model_loaded`/`scheduler_responsive`/`inference_prewarm_complete` checks passed. |
| Model registry | `/v1/models` returned `Qwen3.5-4B`. |
| Non-streaming chat | `/v1/chat/completions` returned model `Qwen3.5-4B` with content `Quality control`. |
| Streaming chat | SSE stream returned model `Qwen3.5-4B` with content `actual model`. |
| Server SFT training | `sft_train_receipt.json` recorded successful training on actual model data; `server_sft_verify.json` passed adapter layout, safetensors consistency, and measurable adapter effect checks with 400 tensors and 200 LoRA projection pairs. |
| Server GRPO training | `grpo_train_receipt.json` recorded 1 group / 2 completions trained, reward filter kept 1 group, adapter smoke enabled and passed, and nonzero LoRA delta norms. |
| Adapter registry and CRUD | `adapters_after_training.json` showed active/loaded `actual-model-server-grpo-smoke`; unload/delete checks cleared active and loaded adapter state without losing the GRPO adapter artifact. |
| CUDA GRPO dry run | `cuda_ablation_dry.log` and dry-run receipt validated the actual model tokenizer/data path and reward-filter bookkeeping. |
| CUDA GRPO real training | `cuda_ablation_train.log` loaded `/workspace/Qwen3.5-4B`, trained `actual-model-cuda-grpo-smoke`, installed it under `cuda-registry`, reached `peak_vram_mib=11432`, and wrote a receipt with adapter smoke enabled and passed. |
| CUDA adapter verification | `cuda_adapter_verify.json` returned `status=ok`, verified 400 safetensors tensors / 200 LoRA projection pairs, and reported measurable adapter effect (`l2_delta_proxy=1.6661485610034024`). |
| Long-context CUDA GRPO | `long_context_grpo_bench.json` completed 8K, 16K, 32K, and 64K rows with nonzero reference/policy/backward/optimizer timings. |

**Long-context benchmark results:**

| Requested seq len | Observed seq len | Total tokens | Peak VRAM MiB | Tokens/sec | Total ms | Ref fwd ms | Policy fwd ms | Backward ms | Optimizer ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8192 | 8206 | 16412 | 14344 | 62.0479 | 264505.19 | 9826.96 | 7132.21 | 247327.21 | 6.24 |
| 16384 | 16396 | 32792 | 15338 | 60.0454 | 546120.51 | 19499.15 | 14295.99 | 511910.93 | 5.51 |
| 32768 | 32776 | 65552 | 18218 | 58.1327 | 1127627.17 | 39746.51 | 31736.80 | 1055346.33 | 6.15 |
| 65536 | 65536 | 131072 | 25066 | 54.1206 | 2421852.24 | 86145.10 | 66666.31 | 2267498.04 | 5.82 |

**Remaining risk:** `kernel_launch_count` is still `null`; the benchmark does
not yet wire a kernel launch counter. Some short adapter smoke prompts produced
identical sampled text even with nonzero logit deltas, which is the motivation
for issue 27.

## P0: Make Correct Experiments Hard To Misrun

### 1. Define Explicit Adapter Semantics For Chat Completion Requests

**Area:** `crates/kiln-server`

**Problem:** Agentic evals discovered that a chat request omitting the
`adapter` field can be interpreted as "unload active adapter" in at least one
path. That makes evals silently run against base after a successful adapter
load. It invalidates results because the request body shape, not the explicit
adapter operation, controls model state.

**Task:** Define and implement one clear adapter selection policy for
`/v1/chat/completions`.

**Proposed behavior:**

- Missing `adapter`: use current server default adapter, without changing it.
- `adapter: null` or `adapter: ""`: use base for this request only, without
  changing the default.
- `adapter: "<name>"`: use that adapter for this request; fail if it is not
  loaded or available.
- `/v1/adapters/load` and `/v1/adapters/unload`: the only endpoints that mutate
  server default adapter state.

**Acceptance criteria:**

- Unit tests cover missing, null, empty, valid, and invalid `adapter` values.
- Server logs every adapter transition with old adapter, new adapter, request
  id, and reason.
- A loaded adapter remains active across requests that omit `adapter`.
- Documentation states the exact semantics.

**Status:** Completed and pushed to `main`.

**Implementation notes:** Added an explicit `ChatAdapterSelection` policy for
`/v1/chat/completions` so Serde distinguishes omitted `adapter` from explicit
`null`. Omitted adapter now resolves to the server default, explicit `null` or
`""` resolves to base for the request, and a non-empty name resolves to a
managed adapter directory. Split server default state (`active_adapter_name`)
from runtime-loaded state (`loaded_adapter_name`) so per-request overrides do
not mutate the default reported by `GET /v1/adapters`; load/unload and training
auto-load update both. Runtime adapter transitions now log old adapter, new
adapter, request id, and reason. README documents the exact chat adapter
semantics.

**Validation evidence:**

- RunPod validation passed on 2026-05-20 on RTX A6000 pod `yu0x0wt25rz8u8`,
  lease `pod-9d17510419e3a5930b68a4fc`.
- Remote command sequence: `cargo test -p kiln-server chat_adapter --lib`;
  `cargo test -p kiln-server --test adapter_path_traversal`;
  `cargo test -p kiln-server --test adapter_registry_state`;
  `cargo check -p kiln-server --tests`.
- Remote sentinel `/workspace/kiln-validation/issues-1-3.done` recorded
  `exit=0`; remote log is `/workspace/kiln-validation/issues-1-3.log`.
- The focused chat adapter test run passed 6 tests covering missing, null,
  empty, named, invalid type, and invalid name-shape adapter selection.

**Commit SHA:** `47f31b25` (`Issue 1: define chat adapter semantics`), pushed
to `origin/main` on 2026-05-20. Note: the commit contains the implementation
and initial validation notes; this status line is recorded in the follow-up
metadata commit because a commit cannot include its own final SHA.

**Remaining risk:** Runtime adapter swapping remains serialized through the
existing global runner swap path; this issue preserves default semantics but
does not redesign per-request LoRA isolation for concurrent real-backend
requests.

### 2. Make Adapter Load Fail Loudly On Wrong Directory Layout

**Area:** `crates/kiln-server`

**Problem:** `cuda_grpo_ablation --output X --adapter Y` writes weights under
`X/Y/`. One cap symlinked `X/` instead of `X/Y/`; `/v1/adapters/load` failed,
the server kept running with the previous adapter, and several evals were
misread as model regressions.

**Task:** Harden `/v1/adapters/load` so it validates the resolved adapter
directory before mutating state.

**Acceptance criteria:**

- Loading an adapter directory without `adapter_model.safetensors` returns a
  structured 4xx error and does not change active adapter state.
- Loading an adapter directory without `adapter_config.json` returns a
  structured 4xx error and does not change active adapter state.
- The error message includes the resolved absolute path and missing files.
- A test covers the common nested-output mistake.
- The server logs load failures at warn level with path and reason.

**Status:** Completed and pushed to `main`.

**Implementation notes:** Added `validate_loadable_adapter_dir` before backend
selection or model mutation in `/v1/adapters/load`. The load endpoint now
requires the resolved adapter directory to contain both `adapter_config.json`
and `adapter_model.safetensors`; missing required files return structured
`400 adapter_layout_invalid` errors with the canonical absolute path and the
missing file list. The validator detects the common `output/adapter/` nested
directory mistake and includes the nested adapter path in the error. Rejections
log at warn level with adapter name, path, operation, and reason. Existing
active/default and runtime-loaded adapter state is left unchanged on all
validation failures.

**Validation evidence:**

- RunPod validation passed on 2026-05-20 on RTX A6000 pod `yu0x0wt25rz8u8`,
  lease `pod-9d17510419e3a5930b68a4fc`.
- Remote command sequence: `cargo test -p kiln-server chat_adapter --lib`;
  `cargo test -p kiln-server --test adapter_path_traversal`;
  `cargo test -p kiln-server --test adapter_registry_state`;
  `cargo check -p kiln-server --tests`.
- Remote sentinel `/workspace/kiln-validation/issues-1-3.done` recorded
  `exit=0`; remote log is `/workspace/kiln-validation/issues-1-3.log`.
- The adapter path traversal test run passed 7 integration tests including
  missing config, missing weights, nested-output mistake, active-state
  preservation, and existing traversal protections.

**Commit SHA:** `54f8c689` (`Issue 2: validate adapter load layout`),
pushed to `origin/main` on 2026-05-20. Note: the commit contains the
implementation and initial validation notes; this status line is recorded in
the follow-up metadata commit because a commit cannot include its own final
SHA.

**Remaining risk:** Tests exercise the validation layer in mock backend mode
so they prove bad layouts fail before backend mutation. They intentionally do
not perform a real LoRA tensor load or GPU validation locally.

### 3. Expose Complete Adapter Registry State

**Area:** `crates/kiln-server`

**Problem:** Cap scripts had to infer whether an adapter was loaded by probing
loosely documented endpoints. `GET /v1/adapters` needs enough information to
prove what will actually run.

**Task:** Expand the adapter registry response.

**Response should include:**

- `active_adapter`
- `loaded_adapters[]`
- `available_adapters[]`
- adapter directory path
- `adapter_model.safetensors` sha256
- file size
- rank, alpha, alpha/rank
- target modules
- parent/base adapter metadata if present
- last load error per adapter name, if any

**Acceptance criteria:**

- `GET /v1/adapters` is sufficient for a script to verify a load.
- Invalid adapter directories appear with an error status instead of being
  silently omitted.
- Tests cover active, loaded-but-inactive, available-unloaded, and invalid
  adapters.

**Status:** Completed and pushed to `main`.

**Implementation notes:** Expanded `GET /v1/adapters` with explicit
`active_adapter`, `loaded_adapter`, `loaded_adapters`, `adapter_dir`, and
`available_adapters` fields while preserving the legacy `active` and
`available` fields. Registry entries now include resolved path, status,
config/weights presence, total directory size, safetensors file size, SHA-256,
rank, alpha, alpha/rank, target modules, base model metadata, optional lineage
or receipt metadata, last load error, and invalid-layout error text. Directory
scan includes invalid adapter directories rather than omitting them, excluding
only internal transient/cache directories. Adapter load failures are retained
in server state and cleared on successful load.

**Validation evidence:**

- RunPod validation passed on 2026-05-20 on RTX A6000 pod `yu0x0wt25rz8u8`,
  lease `pod-9d17510419e3a5930b68a4fc`.
- Remote command sequence: `cargo test -p kiln-server chat_adapter --lib`;
  `cargo test -p kiln-server --test adapter_path_traversal`;
  `cargo test -p kiln-server --test adapter_registry_state`;
  `cargo check -p kiln-server --tests`.
- Remote sentinel `/workspace/kiln-validation/issues-1-3.done` recorded
  `exit=0`; remote log is `/workspace/kiln-validation/issues-1-3.log`.
- The adapter registry state test run passed and covers active default,
  loaded-but-inactive runtime adapter, available-unloaded adapter, invalid
  directory, hashes, sizes, LoRA config fields, paths, and last load error.

**Commit SHA:** `a8d284d6` (`Issue 3: expose adapter registry state`),
pushed to `origin/main` on 2026-05-20. Note: the commit contains the
implementation and initial validation notes; this status line is recorded in
the follow-up metadata commit because a commit cannot include its own final
SHA.

**Remaining risk:** The registry records the single runtime-loaded adapter
known to kiln's current one-runner architecture. It does not attempt to prove
that a real backend's in-memory LoRA tensors still match disk beyond the state
updated by the load/unload paths.

### 4. Add `kiln adapter verify`

**Area:** kiln CLI / `crates/kiln-server`

**Problem:** Future cap authors need one command that proves an adapter is
installed, loadable, selected, and behaviorally active.

**Task:** Implement `kiln adapter verify <name-or-path>`.

**Command should:**

- Validate adapter directory shape.
- Validate config and safetensors consistency.
- Load the adapter through kiln-server or an offline loader.
- Confirm registry state after load.
- Run one fixed prompt with base and adapter.
- Compare logits or token probabilities to prove the adapter changes behavior.
- Print a machine-readable JSON receipt.

**Acceptance criteria:**

- Command exits nonzero for wrong nested path, missing file, rank mismatch, or
  no measurable adapter effect.
- Command exits zero for a known-good tiny test adapter.
- Receipt includes hashes, rank, alpha, target modules, and logit-delta summary.

**Status:** Completed and pushed to `main`.

**Implementation notes:** Added `kiln adapter verify <name-or-path>` as the
singular alias for the existing adapter command group, with
`kiln adapters verify` also available. The verifier resolves bare names
through `--adapter-dir`, `model.adapter_dir`, or `<model.path>/adapters`, and
validates adapter layout, config JSON, safetensors readability, rank/target
module consistency, A/B tensor pairing, hashes, sizes, and nonzero LoRA
effect. It prints a machine-readable JSON receipt with rank, alpha,
alpha/rank, target modules, hashes, tensor summary, and an offline
`logit_delta_summary` proxy based on LoRA delta norm. Optional `--url` server
verification loads the named adapter through `/v1/adapters/load`, confirms
`/v1/adapters` registry state, and compares a fixed base-vs-adapter chat
prompt.

**Validation evidence:**

- RunPod validation passed on 2026-05-20 on RTX A6000 pod `yu0x0wt25rz8u8`,
  lease `pod-9d17510419e3a5930b68a4fc`.
- Remote command sequence: `cargo test -p kiln-server adapter_verify --lib`;
  `cargo test -p kiln-server --test adapter_verify`;
  `cargo check -p kiln-server --tests`.
- Remote sentinel `/workspace/kiln-validation/issue4-tests.done` recorded
  `exit=0`; remote log is `/workspace/kiln-validation/issue4-tests.log`.
- Focused tests passed: 1 CLI parse test for `kiln adapter verify`, plus 5
  integration tests covering known-good tiny adapters, nested parent paths,
  missing weights, rank mismatch, and zero-effect adapters.
- `cargo fmt --all --check` could not run on the same RunPod image because
  `cargo-fmt` is not installed for `stable-x86_64-unknown-linux-gnu`; this is
  recorded in `/workspace/kiln-validation/issue4.log` with sentinel
  `/workspace/kiln-validation/issue4.done` reporting `exit=1`.

**Commit SHA:** `706b099a` (`Issue 4: add adapter verifier`), pushed to
`origin/main` on 2026-05-20. Note: the commit contains the implementation and
initial validation notes; this status line is recorded in the follow-up
metadata commit because a commit cannot include its own final SHA.

**Remaining risk:** Offline verification proves layout, tensor consistency,
and a nonzero LoRA delta norm proxy. Exact logits/token probabilities are not
available through the current chat API; the optional server path proves
registry load and observable fixed-prompt behavior, not raw token-probability
delta.

### 5. Canonicalize Trainer Output And Optional Adapter Installation

**Area:** `crates/kiln-train`, `examples/cuda_grpo_ablation.rs`

**Problem:** Every cap script had to know the trainer's nested output
convention. That produced the adapter symlink bug.

**Task:** Make training output self-describing and optionally installable.

**Changes:**

- At successful train end, print `ADAPTER_DIR=<absolute path>` on its own line.
- Write `adapter_receipt.json` containing `adapter_dir`.
- Add `--install-adapter-dir <dir>` to atomically copy or symlink the actual
  adapter directory into a serve-ready adapter registry.
- Add `--install-adapter-name <name>` to override install name.

**Acceptance criteria:**

- Cap scripts no longer need to compute `output/adapter` paths themselves.
- Install operation validates the target before replacing existing symlink.
- Receipt and stdout agree on the adapter path.

**Status:** Completed and pushed to `main`.

**Implementation notes:** Added `adapter_receipt.json` emission for PEFT and
CUDA adapter saves. The receipt records the canonical adapter directory plus
the config and safetensors paths, and can include the installed registry path
when the CUDA GRPO ablation example installs the adapter. The example now
prints `ADAPTER_DIR=<absolute path>` on successful completion while preserving
the legacy `adapter=...` line. Added `--install-adapter-dir <dir>` and
`--install-adapter-name <name>`; installs validate the produced adapter first,
create the registry when needed, replace existing symlinks atomically, and
refuse to replace a real adapter directory.

**Validation evidence:**

- RunPod validation passed on 2026-05-20 on RTX A6000 pod `yu0x0wt25rz8u8`,
  lease `pod-9d17510419e3a5930b68a4fc`.
- Remote command sequence: `cargo test -p kiln-train adapter_output --lib`;
  `cargo check -p kiln-train --tests`;
  `KILN_CUDA_ARCHS=86 cargo check --release -p kiln-train --features cuda --example cuda_grpo_ablation`.
- Remote sentinel `/workspace/kiln-validation/issue5-release.done` recorded
  `exit=0`; remote log is `/workspace/kiln-validation/issue5-release.log`.
- Focused tests passed 4 adapter-output tests covering canonical receipts,
  nested adapter diagnostics, symlink replacement after validation, and
  refusal to replace a real existing adapter directory.
- The CUDA example was checked in release profile because the first debug
  CUDA check attempted `nvcc -G` and was killed by the pod; release profile is
  the documented kiln CUDA validation path.

**Commit SHA:** `c6693c9e` (`Issue 5: canonicalize adapter outputs`),
pushed to `origin/main` on 2026-05-20. Note: the commit contains the
implementation and initial validation notes; this status line is recorded in
the follow-up metadata commit because a commit cannot include its own final
SHA.

**Remaining risk:** Validation proves the receipt/install helper behavior and
type-checks the CUDA GRPO example. It does not run a full GPU training job to
produce a real learned adapter end to end.

### 6. Enforce Base-Adapter Shape Compatibility Before Training

**Area:** `crates/kiln-train`

**Problem:** Chain-training with a base adapter of rank R1 and a new training
rank R2 can silently corrupt behavior. `pi-faithful-completion` saw
rank-mismatch adapters collapse to near-floor composite.

**Task:** Validate `--base-adapter` shape before optimizer setup.

**Acceptance criteria:**

- If `--base-adapter` rank differs from `--rank`, training fails before GPU
  work starts.
- If target module sets differ, training fails before GPU work starts.
- If tensor shapes differ, training fails with the exact offending tensor name.
- A deliberate `--allow-adapter-shape-conversion` flag is required for any
  future conversion behavior.
- Tests cover rank mismatch, missing tensor, extra tensor, and valid match.

**Status:** Completed and pushed to `main`.

**Implementation notes:** Added strict base-adapter compatibility validation
before training optimizer setup. The validator reads `adapter_config.json` and
safetensors metadata, then checks requested rank, the normalized target-module
set, the exact expected PEFT tensor key set, and every tensor shape against the
current model config. Generic SFT, in-memory GRPO, and streamed JSONL GRPO now
resolve `base_adapter` as either a direct path or a name under the adapter
parent, validate it, and only then copy weights into the seeded LoRA vars.
`cuda_grpo_ablation` and `cuda_sft_file` accept
`--allow-adapter-shape-conversion`; the flag is deliberately wired as an
opt-in marker, but conversion is not implemented and incompatible adapters
still fail. CUDA-native SFT now rejects `base_adapter` instead of silently
ignoring it.

**Validation evidence:**

- RunPod validation passed on 2026-05-20 on RTX A6000 pod `yu0x0wt25rz8u8`,
  lease `pod-9d17510419e3a5930b68a4fc`.
- Remote command sequence: `cargo test -p kiln-train adapter_shape --lib`;
  `cargo check -p kiln-train --tests`; `cargo check -p kiln-server --tests`;
  `KILN_CUDA_ARCHS=86 cargo check --release -p kiln-train --features cuda --example cuda_grpo_ablation`;
  `KILN_CUDA_ARCHS=86 cargo check --release -p kiln-train --features cuda --example cuda_sft_file`.
- Remote sentinel `/workspace/kiln-validation/issue6-rerun.done` recorded
  `exit=0`; remote log is `/workspace/kiln-validation/issue6-rerun.log`.
- Focused tests passed 6 adapter-shape tests covering valid adapters, rank
  mismatch, target-module mismatch, missing tensor, extra tensor, and exact
  tensor shape mismatch with the offending tensor name.

**Commit SHA:** `67b1e6b4` (`Issue 6: validate base adapter shapes`),
pushed to `origin/main` on 2026-05-20. Note: the commit contains the
implementation and initial validation notes; this status line is recorded in
the follow-up metadata commit because a commit cannot include its own final
SHA.

**Remaining risk:** Validation type-checks the training paths and CUDA
examples but does not run a full GPU fine-tuning job. Base-adapter shape
conversion remains intentionally unimplemented; CUDA-native SFT requires the
generic trainer for base-adapter continuation.

### 7. Guard Unsafe LoRA Scaling

**Area:** `crates/kiln-train`

**Problem:** Alpha/rank ratio above 2 caused a corrupted adapter in practice.
The trainer currently allows dangerous settings without warning.

**Task:** Add LoRA scaling validation.

**Acceptance criteria:**

- If `alpha / rank > 2.0`, training fails by default with a clear explanation.
- `--allow-high-lora-scale` can override for deliberate experiments.
- Receipt records `rank`, `alpha`, and `alpha_over_rank`.
- Tests cover accepted and rejected configurations.

**Status:** Completed and pushed to `main`.

**Implementation notes:** Added a shared LoRA scaling guard with the default
limit `alpha / rank <= 2.0`. SFT, GRPO, streamed JSONL GRPO, OPD, and
CUDA-native SFT now validate rank/alpha before optimizer setup; the CUDA
token-sequence adapter helpers also reject unsafe scaling by default.
`cuda_grpo_ablation`, `cuda_sft_file`, and `cuda_opd_remote` expose
`--allow-high-lora-scale` for deliberate experiments. `adapter_receipt.json`
now records `rank`, `alpha`, and `alpha_over_rank` from the adapter config.

**Validation evidence:**

- RunPod validation passed on 2026-05-20 on RTX A6000 pod `yu0x0wt25rz8u8`,
  lease `pod-9d17510419e3a5930b68a4fc`.
- Remote command sequence: `cargo test -p kiln-train lora_scaling --lib`;
  `cargo test -p kiln-train adapter_output --lib`;
  `cargo check -p kiln-train --tests`; `cargo check -p kiln-server --tests`;
  `KILN_CUDA_ARCHS=86 cargo check --release -p kiln-train --features cuda --example cuda_grpo_ablation`;
  `KILN_CUDA_ARCHS=86 cargo check --release -p kiln-train --features cuda --example cuda_sft_file`;
  `KILN_CUDA_ARCHS=86 cargo check --release -p kiln-train --features cuda --example cuda_opd_remote`.
- Remote sentinel `/workspace/kiln-validation/issue7.done` recorded
  `exit=0`; remote log is `/workspace/kiln-validation/issue7.log`.
- Focused tests passed 3 scaling tests covering accepted default-limit
  scaling, rejected high scaling, and explicit override. Adapter-output tests
  also passed and verify receipt `rank`, `alpha`, and `alpha_over_rank`.

**Commit SHA:** `30fac60d` (`Issue 7: guard unsafe lora scaling`), pushed
to `origin/main` on 2026-05-20. Note: the commit contains the implementation
and initial validation notes; this status line is recorded in the follow-up
metadata commit because a commit cannot include its own final SHA.

**Remaining risk:** Validation covers config checks and CUDA example
type-checks, but it does not run a full training job with an overridden high
LoRA scale.

### 8. Emit A Structured Training Receipt For Every GRPO/SFT Run

**Area:** `crates/kiln-train`

**Problem:** Several experiments required forensic reconstruction from logs.
There was no single artifact proving what data, masks, loss settings, adapter,
and model commit were actually used.

**Task:** Write `train_receipt.json` next to every adapter.

**Minimum fields:**

- kiln git commit and dirty flag
- model path and model config hash
- tokenizer config hash
- base adapter path/hash, if any
- output adapter path/hash
- training data path and sha256
- rank, alpha, lr, epochs, seed, mode
- KL coeff, clip epsilon, dynamic sampling settings
- ECHO enabled, lambda, env mask mode, warning filter
- `no_policy_loss`
- groups read, groups filtered, groups trained
- reward mean/stdev and group variance histogram
- action token count, env token count, context token count
- train wall-clock, peak VRAM if available
- LoRA delta norm summary by module

**Acceptance criteria:**

- Receipt is always written on successful training.
- Receipt is written with `"status": "failed"` on known validation failures.
- JSON schema is documented and stable.
- Existing cap scripts can parse it without scraping stdout.

**Status:** Completed and pushed to `main`.

**Implementation notes:** Added `train_receipt.json` as a stable trainer-owned
artifact with schema version 1 and documented it in
`docs/TRAIN_RECEIPT_SCHEMA.md`. Generic SFT, in-memory GRPO, streamed GRPO,
CUDA-native SFT, Vulkan-native SFT/GRPO, and OPD now write receipts next to
the produced adapter. Receipts include kiln source revision/dirty state,
model/tokenizer config hashes, base/output adapter hashes, training data
hashes, core hyperparameters, GRPO/ECHO/no-policy-loss settings, data and
reward stats, action/env/context token counts, wall-clock time, and LoRA
delta norm summaries. Known validation failures in the generic trainer,
CUDA-native SFT, and OPD write `"status": "failed"` receipts at the intended
adapter path before returning the validation error.

**Validation evidence:**

- RunPod validation passed on 2026-05-20 on RTX A6000 pod `yu0x0wt25rz8u8`,
  lease `pod-9d17510419e3a5930b68a4fc`.
- Remote command sequence: `cargo test -p kiln-train train_receipt --lib`;
  `cargo check -p kiln-train --tests`; `cargo check -p kiln-server --tests`;
  `KILN_CUDA_ARCHS=86 cargo check --release -p kiln-train --features cuda
  --example cuda_grpo_ablation`; `KILN_CUDA_ARCHS=86 cargo check --release
  -p kiln-train --features cuda --example cuda_sft_file`;
  `KILN_CUDA_ARCHS=86 cargo check --release -p kiln-train --features cuda
  --example cuda_opd_remote`.
- Remote sentinel `/workspace/kiln-validation/issue8-final.done` recorded
  `exit=0`; remote log is `/workspace/kiln-validation/issue8-final.log`.
- Focused receipt tests passed 4 tests covering success/failure round trips,
  SHA-256 stability, and reward variance histogram shape.

**Commit SHA:** `aea5db9f` (`Issue 8: emit structured training receipts`),
pushed to `origin/main` on 2026-05-20. Note: the commit contains the
implementation and schema docs; this status line is recorded in the follow-up
metadata commit because a commit cannot include its own final SHA.

**Remaining risk:** Full-workspace `cargo fmt --check` is still blocked by
pre-existing formatting drift in unrelated Vulkan files and tests. This issue
validated whitespace with `git diff --check` and avoided sweeping unrelated
formatting changes into the receipt commit.

### 9. Add GRPO Dry-Run Validation Mode

**Area:** `examples/cuda_grpo_ablation.rs`, `crates/kiln-train`

**Problem:** Many mistakes can be detected before GPU training: empty groups,
all-zero masks, wrong schema, no env tokens with ECHO enabled, zero reward
variance, missing base adapter.

**Task:** Add `--dry-run`.

**Dry run should:**

- Load and validate the JSONL.
- Build trajectory masks.
- Apply group filters.
- Resolve and validate adapters.
- Print effective config.
- Print action/env token counts.
- Print group reward variance summary.
- Exit before model forward/backward.

**Acceptance criteria:**

- Dry run catches malformed trajectory roles.
- Dry run catches empty `action_mask`.
- Dry run catches ECHO enabled with empty `env_mask`.
- Dry run catches zero groups after filtering unless explicitly allowed.
- Dry run writes a receipt.

**Status:** Completed and validated on RunPod.

**Implementation notes:** Added `grpo_dry_run_jsonl` in `kiln-train` so dry
run uses the same JSONL parser, trajectory-mask builder, dynamic-sampling
filter, base-adapter validator, LoRA scaling guard, reward statistics, token
counting, and `train_receipt.json` schema as streamed GRPO training. Added
strict dry-run trajectory role validation for Action=`assistant` and
Observation=`tool`, explicit empty `action_mask` detection, ECHO-with-empty
`env_mask` detection, and zero-valid-groups rejection with an
`--allow-empty-dry-run` escape hatch. Wired `--dry-run` and
`--allow-empty-dry-run` into `cuda_grpo_ablation`; dry run now loads the
tokenizer, applies ECHO env overrides, prints the effective config plus
action/env/context counts and reward variance histogram, writes the intended
adapter receipt, and exits before CUDA availability checks or model
forward/backward setup.

**Validation evidence:**

- RunPod validation passed on 2026-05-20 on RTX A6000 pod `yu0x0wt25rz8u8`,
  lease `pod-9d17510419e3a5930b68a4fc`.
- Remote command sequence: `cargo test -p kiln-train grpo_dry_run --lib`;
  `cargo check -p kiln-train --tests`; `cargo check -p kiln-server --tests`;
  `KILN_CUDA_ARCHS=86 cargo check --release -p kiln-train --features cuda
  --example cuda_grpo_ablation`.
- Remote sentinel `/workspace/kiln-validation/issue9.done` recorded `exit=0`;
  remote log is `/workspace/kiln-validation/issue9.log`.
- The focused dry-run test run passed 5 tests covering malformed trajectory
  roles, empty action masks, ECHO enabled with no env tokens plus failed
  receipt writing, zero groups after filtering with and without the explicit
  allow flag, and a successful dry run with action/env token counts and a
  success receipt.

**Commit SHA:** `04449b56` (`Issue 9: add GRPO dry-run validation`). This is
the implementation commit; these final metadata notes are recorded in the
follow-up backlog commit.

**Remaining risk:** Dry run intentionally mirrors the current streamed GRPO
tokenization path, including its default mask configuration. Broader Pi session
normalization and a standalone trajectory-inspection UX are left to issues 10
and 11.

### 10. Add A Trajectory Inspector CLI

**Area:** `crates/kiln-train` or kiln CLI

**Problem:** Agentic GRPO depends on correct action and env masks. Today,
debugging masks requires reading Python helper code and Rust internals.

**Task:** Implement `kiln trajectory inspect <session-or-rollout.jsonl>`.

**Output should show:**

- rendered messages
- segment role and kind
- token counts by segment
- action token count
- env token count
- warning-prefix stripped bytes
- decoded first N action target tokens
- decoded first N env target tokens
- schema warnings

**Acceptance criteria:**

- Works on Pi 0.75.1-style and 0.75.3-style session JSONL.
- Works on kiln ScoredRollout JSONL.
- Emits JSON with `--json`.
- Fails nonzero when no trainable action tokens are present.

**Status:** Completed and validated on RunPod.

**Implementation notes:** Added `kiln_train::trajectory_inspect`, a Rust
inspector that loads Pi session JSONL events, kiln `ScoredRollout` JSONL, and
kiln `AgenticGroup` JSONL, then renders each rollout through `KilnTokenizer`
and the canonical `build_masks_from_trajectory` path. The Pi session loader
handles observed 0.75.1-style `role="tool"` / `toolResult` content blocks and
0.75.3-style `role="toolResult"` text blocks, normalizes tool results to
`role="tool"`, records schema warnings, detects harness warning prefixes, and
fails when no action-mask tokens are trainable. Wired `kiln trajectory inspect`
into the server CLI with pretty output, `--json`, `--include-context`,
`--preview-tokens`, explicit `--tokenizer`, explicit `--chat-template`, and
`--model-path` tokenizer discovery.

**Validation evidence:**

- RunPod validation passed on 2026-05-20 on RTX A6000 pod `yu0x0wt25rz8u8`,
  lease `pod-9d17510419e3a5930b68a4fc`.
- Remote command sequence: `cargo test -p kiln-train trajectory_inspect
  --lib`; `cargo test -p kiln-server trajectory --lib`; `cargo check -p
  kiln-server --tests`; `cargo check -p kiln-train --tests`; `cargo run -p
  kiln-server --bin kiln -- trajectory inspect ... --json` against a generated
  Pi 0.75.3-style fixture; and a negative `kiln trajectory inspect --json`
  smoke confirming no-action input exits nonzero.
- Remote sentinel `/workspace/kiln-validation/issue10-v8.done` recorded
  `exit=0`; remote log is `/workspace/kiln-validation/issue10-v8.log`.
- Focused Rust tests cover Pi 0.75.1 action/env masks, Pi 0.75.3
  `toolResult` normalization, kiln `ScoredRollout` JSONL, no-action failure,
  warning-prefix reporting, CLI parsing, and the human-readable formatter's
  required fields.

**Commit SHA:** `81363388` (`Issue 10: add trajectory inspector CLI`). This is
the implementation commit; these final metadata notes are recorded in the
follow-up backlog commit.

**Remaining risk:** The inspector deliberately reports masks produced by the
current canonical trajectory-mask builder. Issue 11 still owns moving broader
Pi session normalization into kiln-owned ingestion code beyond this standalone
inspection command.

### 11. Move Pi Session Normalization Into Kiln-Owned Code

**Area:** `crates/kiln-train`, `crates/kiln-server` agent trace parser

**Problem:** The shared Python parser learned several Pi schema facts:
`toolResult` role normalization, `message` singular shape, tool args under
`input`. Kiln's Rust trace parser should know these too.

**Task:** Update kiln's agent trace parsing to accept observed Pi session
formats and normalize them to the canonical trajectory schema.

**Acceptance criteria:**

- Rust parser accepts role `tool` and `toolResult`.
- Rust parser accepts content blocks with `toolCall`, `toolResult`, `thinking`,
  and `text`.
- Tool-call args are read from `input` and rendered to canonical arguments.
- Unit tests use fixtures for both Pi 0.75.1 and Pi 0.75.3.
- Python and Rust parsers produce equivalent segment sequences on shared
  fixtures.

**Status:** Completed and validated on RunPod.

**Implementation notes:** Added `kiln_train::pi_trajectory` as the shared
Rust-owned Pi session normalizer. It parses Pi `{"type":"message",
"message":...}` JSONL events, accepts `tool` and `toolResult` roles, renders
`text`, `thinking`, `toolCall`, and `toolResult` blocks into canonical kiln
`TurnSegment`s, reads tool-call arguments from `input`, normalizes tool results
to `role="tool"`, records warning-prefix lengths, and preserves the legacy
`ScoredRollout.text` flattening contract. Refactored `trajectory_inspect` to
reuse this shared parser. Updated `agent_traces` so discovered Pi 0.75.x
session files fall back to the filename as id when no top-level id exists,
count user/assistant turns and assistant tool calls from message events, and
persist the normalized action/observation trajectory on each `AgentTrace`.

**Validation evidence:**

- RunPod validation passed on 2026-05-20 on RTX A6000 pod `yu0x0wt25rz8u8`,
  lease `pod-9d17510419e3a5930b68a4fc`.
- Remote command sequence: `cargo test -p kiln-train pi_trajectory --lib`;
  `cargo test -p kiln-train trajectory_inspect --lib`; `cargo test -p
  kiln-server agent_traces --lib`; `cargo check -p kiln-train --tests`;
  `cargo check -p kiln-server --tests`; and `python3
  capabilities/agentic-grpo/lib/test_pi_trajectory.py`.
- Remote sentinel `/workspace/kiln-validation/issue11.done` recorded `exit=0`;
  remote log is `/workspace/kiln-validation/issue11.log`.
- Focused Rust tests cover Pi 0.75.1 fixtures, Pi 0.75.3 `toolResult`
  normalization, `input`-sourced tool-call arguments, warning-prefix handling,
  context inclusion, action-text flattening, inspector regression coverage, and
  agent-trace persistence of normalized trajectories. The existing Python
  parser suite passed 15/15 on the same pod.

**Commit SHA:** `3fd33909` (`Issue 11: normalize Pi session trajectories`).
This is the implementation commit; these final metadata notes are recorded in
the follow-up backlog commit.

**Remaining risk:** The Rust parser now matches the observed Pi 0.75.x session
event shapes covered by the Python parser fixtures. Future upstream Pi schema
changes may still require adding new block renderers or metadata extraction
rules.

### 12. Make ECHO Activity Observable In Every Training Path

**Area:** `crates/kiln-train`

**Problem:** ECHO initially required adapter diffs and custom log greps to prove
it was firing. One path had debug logs, another did not, and tracing was not
initialized in the example.

**Task:** Standardize ECHO metrics and logs across checkpointed and
uncheckpointed paths.

**Acceptance criteria:**

- Every training run reports env token count and action token count.
- If ECHO is enabled and env tokens exist, logs show per-group env CE.
- If ECHO is enabled and env tokens are zero, training warns loudly.
- Receipt includes initial/final env CE when measurable.
- Checkpointed and uncheckpointed paths expose the same metric names.
- Tests cover ECHO on, ECHO off, and no-policy-loss.

**Status:** Completed and validated on RunPod.

**Implementation notes:** Added first/final ECHO env-CE measurements to
`train_receipt.json` when a GRPO path can measure them. Standardized
`training token counts` logs for SFT/GRPO receipt-writing paths, including
generic, Vulkan-native, and CUDA-native training. Generic GRPO now returns a
per-group step report with `echo_env_ce`, logs per-group ECHO metrics with
the same field names for checkpointed and uncheckpointed paths, and warns
when ECHO is enabled but no env tokens were observed. Streamed GRPO records
the same metrics incrementally. Vulkan-native GRPO now returns a step report,
records weighted per-group ECHO env CE, and carries those measurements into
the receipt.

**Validation evidence:**

- RunPod validation passed on 2026-05-21 on direct fallback RTX A6000 pod
  `qmfxie9izl6lc6`.
- Remote command sequence: `git diff --check`; `cargo test -p kiln-train
  train_receipt --lib`; `cargo test -p kiln-train
  test_echo_end_to_end_grpo_with_trajectory_rollouts --lib`; `cargo test -p
  kiln-train test_echo_no_policy_loss_verifier_free_e2e --lib`; `cargo test
  -p kiln-train test_echo_checkpointed_matches_uncheckpointed_loss --lib`;
  `cargo test -p kiln-train
  test_echo_checkpointed_forward_backward_threads_echo_params --lib`; and
  `cargo check -p kiln-train --tests`.
- Remote sentinel `/workspace/kiln-validation/issue12.done` recorded
  `exit=0`; remote logs are `/workspace/kiln-validation/issue12.log` and
  `/workspace/kiln-validation/issue12.inner.log`.
- Focused tests cover ECHO enabled, ECHO disabled, lambda-zero off semantics,
  no-policy-loss verifier-free mode, checkpointed/uncheckpointed env-CE
  reporting, analytic-tail env-CE reporting, and receipt env-CE bounds.

**Commit SHA:** `c0cc1b57` (`Issue 12: expose ECHO training metrics`).
This is the implementation commit; these final metadata notes are recorded in
the follow-up backlog commit.

**Remaining risk:** The receipt records the first and final measured
per-group env CE observed during the training pass, not an extra full-dataset
post-training evaluation sweep. That keeps the metric cheap and available in
streaming paths, but it should not be interpreted as an eval-set CE.

## P0: Make Evaluation Trustworthy

### 13. Add Server Health Metrics For Eval Stability

**Area:** `crates/kiln-server`

**Problem:** `pi-code-search` found kiln-server state drift: requests slowed
from milliseconds to 80-120 seconds and evals falsely looked like adapter
regressions.

**Task:** Add `/v1/health` or expand existing health output with eval-relevant
server metrics.

**Fields should include:**

- uptime
- active adapter
- loaded adapter count
- request count
- recent p50/p95/p99 latency
- recent tokens/sec
- recent timeout/error count
- VRAM allocated/reserved if available
- prefix cache size
- CUDA graph/cache state summary if available
- last error summary

**Acceptance criteria:**

- Health response is machine-readable JSON.
- Health response is cheap enough to call between eval rollouts.
- Tests cover at least active adapter and request count.

**Status:** Completed and validated on RunPod.

**Implementation notes:** Expanded `/health` and `/v1/health` with
eval-stability JSON fields for active and loaded adapter state, aggregate
request counts, recent request p50/p95/p99 latency, recent completion
tokens/sec, recent timeout/error counts, last error summary, prefix-cache
occupancy, rendered prompt and prompt-token cache counts, decode batcher and
batching engine state, nonblocking CUDA graph state, and VRAM
allocated/reserved budget fields. The handler reuses atomics, bounded rings,
and existing snapshots so it remains cheap enough to call between eval
rollouts.

**Validation evidence:**

- RunPod validation passed on 2026-05-21 on direct fallback RTX A6000 pod
  `qmfxie9izl6lc6`.
- Remote command sequence: `git diff --check`; `cargo test -p kiln-server
  api::health --lib`; and `cargo check -p kiln-server --tests`.
- Remote sentinel `/workspace/kiln-validation/issue13.done` recorded
  `exit=0`; remote log is `/workspace/kiln-validation/issue13.log`.
- Focused health tests cover the `/v1/health` alias, active adapter reporting,
  request count reporting, scheduler stats, and prewarm degraded-check
  reporting.

**Commit SHA:** `25b01d53` (`Issue 13: add eval health metrics`).
This is the implementation commit; these final metadata notes are recorded in
the follow-up backlog commit.

**Remaining risk:** Recent timeout/error counts are derived from completed
records retained in the bounded recent-request ring, while total request
counts come from process-lifetime atomics. Non-streaming failures that return
before a recent-request record is written are still visible in aggregate
request counters, but may not have a last-error summary until Issue 14 adds
structured slow/error diagnostics.

### 14. Add A Slow-Request Watchdog

**Area:** `crates/kiln-server`

**Problem:** When server requests become unexpectedly slow, cap authors need a
diagnostic record tied to the request.

**Task:** Log structured diagnostics when a chat completion exceeds a threshold.

**Log should include:**

- request id
- adapter
- prompt tokens
- max output tokens
- generated tokens
- elapsed time
- batching engine state
- thinking mode
- CUDA graph state if available
- prefix cache hit/miss
- error or finish reason

**Acceptance criteria:**

- Threshold is configurable.
- Slow requests emit one structured warning.
- Logs do not include full prompt text by default.

**Status:** Completed 2026-05-21.

**Implementation notes:**

- Added `server.slow_request_warn_secs` and `KILN_SLOW_REQUEST_WARN_SECS`
  with a default of 30 seconds; `0` disables the watchdog.
- Slow chat-completion records now emit a single structured warning on target
  `kiln_server::slow_request` when elapsed time meets the configured threshold.
- Warning fields include request id, adapter, prompt token count, max output
  token count, generated tokens, elapsed/threshold milliseconds, batching
  engine state, thinking mode, CUDA graph state, prefix-cache diagnostic,
  finish reason, error string, and streaming flag.
- The structured log path is derived from recent-request metadata and does not
  include prompt or completion body fields. Non-streaming failure/timeout paths
  also record an error-bearing request record so slow failures are eligible for
  the same watchdog warning.

**Validation evidence:**

- RunPod validation passed on 2026-05-21 on direct fallback RTX A6000 pod
  `qmfxie9izl6lc6`.
- Remote command sequence: `git diff --check`; `cargo test -p kiln-server
  slow_request_log_values --lib`; `cargo test -p kiln-server
  test_env_var_overrides --lib`; `cargo test -p kiln-server
  test_parse_full_toml --lib`; and `cargo check -p kiln-server --tests`.
- Remote sentinel `/workspace/kiln-validation/issue14.done` recorded
  `exit=0`; remote log is `/workspace/kiln-validation/issue14.log`.
- Focused unit coverage verifies threshold behavior and that the slow-log value
  object excludes prompt text even when the recent request record has prompt
  preview/full fields.

**Commit SHA:** `10452c6f` (`Issue 14: add slow request watchdog`).
This is the implementation commit; these final metadata notes are recorded in
the follow-up backlog commit.

**Remaining risk:** The prefix-cache diagnostic is precise on the direct real
generation paths (`hit`, `miss`, `disabled`, `skipped`, or
`not_used_speculative`) and labels batched/cached/mock paths separately. Error
records that fail outside generation retain `prefix_cache = "unknown"` because
no prefix-cache decision may have happened yet.

### 15. Add Eval Mode To `kiln serve`

**Area:** `crates/kiln-server`

**Problem:** Agentic evals need deterministic, stable, debuggable serving more
than maximum throughput.

**Task:** Add `kiln serve --eval-mode`.

**Eval mode should:**

- expose active adapter in every response metadata block or header
- use deterministic defaults where possible
- disable thinking by default for Qwen tool-agent eval unless overridden
- bound or reset caches that grow across requests
- warn on adapter switching during an eval session
- optionally reset per-request transient state after each completion

**Acceptance criteria:**

- `--eval-mode` is documented.
- Health endpoint reports `eval_mode: true`.
- Direct completion tests work in eval mode.
- Repeated short completions do not show latency creep in a stress test.

**Status:** Completed 2026-05-21.

**Implementation notes:**

- Added `kiln serve --eval-mode`, `server.eval_mode`, and
  `KILN_EVAL_MODE`.
- Eval mode applies deterministic omitted-request defaults
  (`temperature=0`, `top_p=1`, `top_k=0`, neutral penalties, `seed=0`) while
  preserving explicit caller overrides.
- Eval mode defaults `chat_template_kwargs.enable_thinking=false` unless the
  caller explicitly sets `enable_thinking`.
- Chat and batch completion responses now include
  `x-kiln-eval-mode`, `x-kiln-active-adapter`, and
  `x-kiln-loaded-adapter` headers.
- Non-streaming eval requests clear completed deterministic caches,
  rendered-prompt/token caches, and real prefix-cache state after completion.
- Adapter transitions emit a warning while eval mode is active, including
  composed-adapter transitions.
- `/health` and `/v1/health` report `eval_mode`.

**Validation evidence:**

- RunPod validation passed on 2026-05-21 on direct fallback RTX A6000 pod
  `qmfxie9izl6lc6`.
- Remote command sequence: `git diff --check`; `cargo test -p kiln-server
  eval_mode --lib`; `cargo test -p kiln-server
  test_health_reports_active_adapter_and_request_count --lib`; `cargo test -p
  kiln-server parses_serve_eval_mode --lib`; `cargo test -p kiln-server
  test_env_var_overrides --lib`; `cargo test -p kiln-server
  test_parse_full_toml --lib`; and `cargo check -p kiln-server --tests`.
- Remote sentinel `/workspace/kiln-validation/issue15.done` recorded
  `exit=0`; remote log is `/workspace/kiln-validation/issue15.log`.
- Focused tests cover CLI parsing/documentation surface, health reporting,
  eval-mode deterministic/no-think defaults with override preservation,
  direct chat completion headers, and repeated short eval-mode completions with
  transient caches reset after each request.

**Commit SHA:** `83d202b9` (`Issue 15: add eval serve mode`).
This is the implementation commit; these final metadata notes are recorded in
the follow-up backlog commit.

**Remaining risk:** Streaming eval-mode responses get adapter/eval headers and
deterministic/no-think defaults, but transient cache cleanup is intentionally
limited to non-streaming requests so the server does not clear prefix-cache
blocks while a streaming decode worker may still be consuming them.

### 16. Add Adapter Load/Unload Stress Test

**Area:** `crates/kiln-server` tests or integration tests

**Problem:** Repeated adapter load/unload and eval cycles appear to be a risk
factor for server drift.

**Task:** Add an integration test that repeatedly loads and unloads one or more
small test adapters while running short completions.

**Acceptance criteria:**

- Test runs at least 100 load/eval/unload cycles in a lightweight mode.
- It checks active adapter state every cycle.
- It tracks latency and fails on severe drift.
- It checks memory growth if allocator stats are available.

**Status:** Completed 2026-05-21.

**Implementation notes:**

- Added an HTTP-level adapter stress test that runs 100 load/eval/unload cycles
  across two tiny fixture adapters.
- Each cycle checks `GET /v1/adapters`, direct shared state, and eval-mode chat
  completion runtime headers for active/loaded adapter consistency.
- The stress test records per-cycle latency and fails if the final-window p95
  shows severe drift from the initial-window p95.
- Memory growth is checked when the server exposes memory-budget counters; the
  lightweight mock fixture documents that allocator counters are unavailable.
- Refactored adapter load/unload state recording into shared helpers so the
  production real-backend path and test-only lightweight stress harness exercise
  the same state transition logic. Production mock-mode adapter load/unload
  behavior remains unchanged outside `cfg(test)`.

**Validation evidence:**

- RunPod validation passed on 2026-05-21 on direct fallback RTX A6000 pod
  `qmfxie9izl6lc6`.
- Remote command sequence: `git diff --check`; `cargo test -p kiln-server
  adapter_load_eval_unload_stress_tracks_state_latency_and_memory --lib`;
  `cargo test -p kiln-server --test adapter_path_traversal
  test_load_rejects_missing_adapter_config_without_changing_active`; `cargo
  test -p kiln-server --test adapter_path_traversal
  test_load_rejects_missing_adapter_weights_without_changing_active`; and
  `cargo check -p kiln-server --tests`.
- Remote sentinel `/workspace/kiln-validation/issue16.done` recorded `exit=0`;
  remote log is `/workspace/kiln-validation/issue16.log`.

**Commit SHA:** `b7bf0c43` (`Issue 16: add adapter stress test`).
This is the implementation commit; these final metadata notes are recorded in
the follow-up backlog commit.

**Remaining risk:** The 100-cycle stress loop intentionally uses the mock
backend to stay lightweight in CI, so it catches API/state/cache/header drift
but does not parse real LoRA safetensors or exercise real runner adapter tensor
allocation churn.

### 17. Make Qwen Thinking Mode A First-Class Config

**Area:** `crates/kiln-server`, config docs

**Problem:** Qwen thinking defaults caused empty `content`, long reasoning
traces, Pi timeouts, and 40x slowdowns. The control exists through
`chat_template_kwargs.enable_thinking`, but it was not obvious.

**Task:** Promote thinking-mode behavior into kiln config and docs.

**Acceptance criteria:**

- Add a server config/env var for default thinking mode.
- Health endpoint reports the default.
- Each request can still override via `chat_template_kwargs`.
- Response metadata indicates whether thinking was enabled.
- Docs explain behavior for Qwen3.5-4B and tool-agent usage.
- Tests cover content output with thinking on and off.

**Status:** Completed and pushed to `main`.

**Implementation notes:** Added `server.default_thinking_enabled` plus the
`KILN_DEFAULT_THINKING_ENABLED` env var to configure the default
`chat_template_kwargs.enable_thinking` value when requests omit it. The legacy
`KILN_DEFAULT_NO_THINK` env var remains supported as a compatibility alias for
`false`, while explicit per-request `chat_template_kwargs.enable_thinking`
continues to win over the server default. The configured default is copied into
`AppState`, included in `/health`, and used by prompt rendering/cache keys so
Qwen3.5-4B can default to non-thinking mode for tool-agent evals. Chat
completion responses now include metadata with `thinking_enabled`,
`thinking_mode`, `thinking_source`, and the configured default.

**Validation evidence:**

- RunPod validation passed on 2026-05-20 on RTX A6000 pod `qmfxie9izl6lc6`.
- Remote command sequence: `git diff --check`; `cargo test -p kiln-server
  qwen35_server_default_thinking_can_be_disabled_and_overridden --lib`;
  `cargo test -p kiln-server
  chat_response_metadata_reports_server_thinking_default --lib`; `cargo test
  -p kiln-server test_env_var_overrides --lib`; `cargo test -p kiln-server
  test_legacy_default_no_think_env_override --lib`; `cargo test -p
  kiln-server test_health_reports_active_adapter_and_request_count --lib`;
  `cargo check -p kiln-server --tests`.
- Remote sentinel `/workspace/kiln-validation/issue17.done` recorded `exit=0`;
  remote log is `/workspace/kiln-validation/issue17.log`.
- `cargo fmt --check` was attempted on the RunPod after installing `rustfmt`,
  but it reports unrelated pre-existing formatting diffs under
  `crates/kiln-vulkan-kernel`; the Issue 17 patch itself passes
  `git diff --check`.

**Commit SHA:** `e07e9395` (`Issue 17: configure Qwen thinking default`),
pushed to `origin/main` on 2026-05-20. Note: the commit contains the
implementation and initial validation notes; this status line is recorded in
the follow-up metadata commit because a commit cannot include its own final
SHA.

**Remaining risk:** Response metadata is reported for the OpenAI-compatible
chat response body. Streaming chunks still use the existing SSE shape and do
not repeat the top-level response metadata.

### 18. Surface Reasoning Content Separately And Safely

**Area:** `crates/kiln-server` OpenAI-compatible API

**Problem:** Some Qwen responses can put reasoning in `reasoning_content` and
leave `content` empty. Pi treated empty content as no response and re-prompted.

**Task:** Make response handling explicit for models/templates that separate
reasoning and final content.

**Acceptance criteria:**

- If reasoning text exists separately, response JSON exposes it in a documented
  field.
- If final content is empty, response metadata says why.
- There is an option to fold reasoning into content for compatibility, default
  off.
- Tests cover empty content with nonempty reasoning.

**Implementation status:** Complete. Chat responses now keep separated
reasoning in `choices[].message.reasoning_content`, report
`metadata.final_content_empty`, `metadata.content_empty_reason`, and
`metadata.reasoning_folded_into_content`, and support per-request
`fold_reasoning_into_content` plus server-level
`server.fold_reasoning_into_content` / `KILN_FOLD_REASONING_INTO_CONTENT`.
Batch responses now serialize `reasoning_content` when present. Deterministic
chat, completion, and batch cache keys account for the effective folding mode,
while cache values are stored in unfolded form before response-shaping.

**Validation evidence:**

- RunPod validation passed on 2026-05-21 on RTX A6000 pod `qmfxie9izl6lc6`.
- Remote command sequence: `git diff --check`; `cargo test -p kiln-server
  reasoning --lib`; `cargo test -p kiln-server
  deterministic_cache_keys_include_reasoning_fold_mode --lib`; `cargo test -p
  kiln-server batch_items_serialize_reasoning_content_when_present --lib`;
  `cargo test -p kiln-server env_overrides --lib`; `cargo check -p
  kiln-server --tests`.
- Remote sentinel `/workspace/kiln-validation/issue18b.done` recorded `exit=0`;
  remote log is `/workspace/kiln-validation/issue18b.log`.

**Commit SHA:** `c4512584` (`Issue 18: surface reasoning content safely`),
pushed to `origin/main` on 2026-05-21. Note: the commit contains the
implementation and docs; this status line is recorded in the follow-up
metadata commit because a commit cannot include its own final SHA.

**Remaining risk:** Streaming chunks continue to use the existing SSE delta
shape; the new empty-final-content metadata is reported on non-streaming
OpenAI-compatible response bodies.

## P1: Make Training Diagnostics Strong Enough To Debug Science

### 19. Add Adapter-Effect Smoke Test After Training

**Area:** `crates/kiln-train`

**Problem:** `pi-compaction` produced byte-identical eval outputs even after
training. The trainer should detect likely no-op adapters before a full eval.

**Task:** After successful training, optionally run a small adapter-effect
smoke test.

**Smoke test should compare base vs adapter on fixed prompts using:**

- logit delta norm
- generated text difference
- adapter output length
- finite logits check

**Acceptance criteria:**

- Enabled by `--adapter-smoke-test`.
- Receipt records pass/fail and metrics.
- Warns if adapter appears to have no measurable effect.
- Warns if adapter produces empty outputs on canary prompt.

**Implementation status:** Complete. SFT and GRPO configs now accept
`adapter_smoke_test`, and `kiln train sft|grpo --adapter-smoke-test` sets it
in the submitted training payload, including streamed GRPO JSONL payloads. On
successful generic SFT/GRPO training, the trainer runs fixed canary prompts
through the base model and trained LoRA, records finite-logit status,
base-vs-adapter logit delta L2, short greedy output differences, and adapter
output length in `train_receipt.json`, and emits warnings when logits are
non-finite, the adapter has no measurable effect, or the canary output is
empty. The receipt schema remains backward-compatible for older receipts.

**Validation evidence:**

- RunPod validation passed on 2026-05-21 on RTX A6000 pod `qmfxie9izl6lc6`.
- Remote command sequence: `git diff --check`; `cargo test -p kiln-train
  adapter_smoke --lib`; `cargo test -p kiln-server adapter_smoke --lib`;
  `cargo check -p kiln-train --tests`; `cargo check -p kiln-server --tests`;
  plus `cargo test -p kiln-server build_grpo --lib`.
- Remote sentinel `/workspace/kiln-validation/issue19d.done` recorded
  `exit=0`; remote log is `/workspace/kiln-validation/issue19d.log`.
- Additional GRPO payload validation sentinel
  `/workspace/kiln-validation/issue19e.done` recorded `exit=0`; remote log is
  `/workspace/kiln-validation/issue19e.log`.

**Commit SHA:** `b9fed4c7` (`Issue 19: add adapter effect smoke test`),
pushed to `origin/main` on 2026-05-21. Note: the commit contains the
implementation and docs; this status line is recorded in the follow-up
metadata commit because a commit cannot include its own final SHA.

**Remaining risk:** Native CUDA/Vulkan training paths continue to write normal
training receipts. The CUDA GRPO path now has actual-model coverage for the
adapter canary: RunPod validation on 2026-05-21 ran
`cuda_grpo_ablation --adapter-smoke-test` against `/workspace/Qwen3.5-4B`,
produced `cuda_grpo_receipt_ok`, installed the trained adapter, and recorded
nonzero train timings. Vulkan-native training still needs equivalent
actual-model coverage.

### 20. Report LoRA Delta And Gradient Norms

**Area:** `crates/kiln-train`

**Problem:** Several hypotheses depended on whether training moved weights
enough to affect BF16 inference. We need direct norms rather than guesswork.

**Task:** Record LoRA gradient norms and post-step delta norms by module.

**Acceptance criteria:**

- Receipt includes per-target-module delta norm summary.
- Receipt includes min/mean/max grad norm over training.
- Logs warn if all LoRA deltas are near zero.
- Logs warn if any delta is extreme relative to initialized scale.

**Implementation status:** Complete. `train_receipt.json` now includes
`lora_grad_norms`, a per-target-module min/mean/max summary collected before
each generic SFT/GRPO optimizer step. Existing per-module
`lora_delta_norms` remain populated from the final adapter safetensors, and
receipt writing now emits LoRA delta warnings when all deltas are effectively
zero or when any module delta is extreme relative to `alpha / rank`. The GRPO
collector covers inline and streamed JSONL training, including token-level
group accumulation.

**Validation evidence:**

- RunPod validation passed on 2026-05-21 on RTX A6000 pod `qmfxie9izl6lc6`.
- Remote command sequence: `git diff --check`; `cargo test -p kiln-train
  lora_grad_norm --lib`; `cargo test -p kiln-train lora_delta_norm --lib`;
  `cargo check -p kiln-train --tests`; `cargo check -p kiln-server --tests`.
- Remote sentinel `/workspace/kiln-validation/issue20c.done` recorded
  `exit=0`; remote log is `/workspace/kiln-validation/issue20c.log`.

**Commit SHA:** `7ac32f74` (`Issue 20: report LoRA grad and delta norms`).
This status line is recorded in the follow-up metadata commit because a
commit cannot include its own final SHA.

**Remaining risk:** Native CUDA/Vulkan/OPD receipt paths now warn on saved
adapter delta norms, but gradient norm collection is currently limited to the
generic Candle SFT/GRPO trainers where gradients are available as `GradStore`
or accumulated gradient maps.

### 21. Add Saturated-Reward Diagnostics

**Area:** `crates/kiln-train`

**Problem:** Saturated baselines made GRPO policy gradients harmful. The trainer
can detect low variance and high reward means in the training data.

**Task:** Add diagnostics for saturated or degenerate reward distributions.

**Acceptance criteria:**

- Dry run and training receipt report reward mean, stdev, min, max, and group
  variance histogram.
- Warn if most groups are all-pass or all-fail.
- Warn if reward mean is above a configurable saturation threshold and variance
  is low.
- Suggest `--no-policy-loss` or harder data in the warning text.

**Implementation status:** Complete. GRPO reward receipts now include
`min`, `max`, `group_count`, all-pass/all-fail group counts, degenerate group
count, and the existing mean/stdev plus group variance histogram. Inline,
streamed JSONL, dry-run, and Vulkan-native GRPO receipt paths use the new
reward stats helper. `GrpoConfig` now exposes configurable
`reward_saturation_threshold` and `reward_low_variance_threshold` values, and
receipt writing logs warnings for mostly all-pass/all-fail groups or
high-mean/low-variance reward distributions. Warning text suggests
`--no-policy-loss` and harder data.

**Validation evidence:**

- RunPod validation passed on 2026-05-21 on RTX A6000 pod `qmfxie9izl6lc6`.
- Remote command sequence: `git diff --check`; `cargo test -p kiln-train
  reward_diagnostic --lib`; `cargo test -p kiln-train
  reward_stats_include_variance_histogram --lib`; `cargo test -p kiln-train
  grpo_dry_run_success_records_counts_and_receipt --lib`; `cargo check -p
  kiln-train --tests`; `cargo check -p kiln-server --tests`.
- Remote sentinel `/workspace/kiln-validation/issue21.done` recorded
  `exit=0`; remote log is `/workspace/kiln-validation/issue21.log`.

**Commit SHA:** `aabab213` (`Issue 21: add saturated reward diagnostics`).
This status line is recorded in the follow-up metadata commit because a
commit cannot include its own final SHA.

**Remaining risk:** The `--no-policy-loss` suggestion is emitted as operator
guidance; API callers can set `config.loss.no_policy_loss=true`, while the
thin CLI still passes arbitrary GRPO JSON config through unchanged.

### 22. Officialize Strong-Signal Filtering

**Area:** `examples/cuda_grpo_ablation.rs`, `crates/kiln-train`

**Problem:** Strong-signal filtering was repeatedly useful, but each cap
implemented or tuned it outside kiln.

**Task:** Add first-class reward variance filtering.

**CLI options:**

- `--filter-var-min <float>`
- `--filter-var-max <float>` if useful
- `--min-groups <n>`
- `--on-empty-filter {fail,train-all,skip}`

**Acceptance criteria:**

- Filtered group count is printed and written to receipt.
- Exact kept/dropped group ids are written to a sidecar JSON.
- Empty-filter behavior is explicit, not accidental.
- Tests cover each `--on-empty-filter` mode.

**Implementation status:** Complete. `GrpoConfig` now has first-class reward
variance filter knobs (`reward_filter_var_min`, `reward_filter_var_max`,
`reward_filter_min_groups`, and `reward_filter_on_empty`). Inline GRPO,
streamed JSONL GRPO, and dry-run JSONL apply the filter before dynamic
sampling, record reward-filter kept/dropped counts in `train_receipt.json`,
and write `reward_filter_groups.json` with exact source ids (`group:N` for
inline groups and `line:N` for JSONL), variances, final decisions, and
empty-filter action. The CUDA ablation CLI exposes `--filter-var-min`,
`--filter-var-max`, `--min-groups`, and `--on-empty-filter
fail|train-all|skip`, then prints reward-filter counts and sidecar paths.

**Validation evidence:**

- RunPod validation passed on 2026-05-21 on RTX A6000 pod `qmfxie9izl6lc6`.
- Remote command sequence: `git diff --check`; `cargo test -p kiln-train
  grpo_dry_run_reward_filter_on_empty_modes --lib`; `cargo test -p
  kiln-train test_grpo_reward_diagnostic_thresholds_deserialize --lib`;
  `cargo check --release -p kiln-train --features cuda --example
  cuda_grpo_ablation`; `cargo check -p kiln-train --tests`; `cargo check -p
  kiln-server --tests`.
- Remote sentinel `/workspace/kiln-validation/issue22f.done` recorded
  `exit=0`; remote log is `/workspace/kiln-validation/issue22f.log`.

**Commit SHA:** `8834e27c` (`Issue 22: add reward variance filtering`).
This status line is recorded in the follow-up metadata commit because a
commit cannot include its own final SHA.

**Remaining risk:** Vulkan-native GRPO keeps its existing dynamic-sampling
behavior and does not yet apply the reward variance filter; Issue 22's CLI
and generic Candle GRPO paths are covered.

### 23. Print Effective Training Config

**Area:** `examples/cuda_grpo_ablation.rs`

**Problem:** Several cap scripts assumed flags existed or misunderstood baked
defaults such as KL coeff, clip epsilon, or advantage mode.

**Task:** Add `--print-effective-config` and print config at train start by
default.

**Acceptance criteria:**

- Output includes all defaults and CLI overrides.
- JSON output mode is available.
- Help text explains that advantage formulation is selected by `--mode`.
- If `--no-echo` and `--echo-lambda` conflict, error text explains the correct
  single-argument pattern.

**Implementation status:** Complete. `cuda_grpo_ablation` now keeps effective
config printing enabled by default, exposes `--print-effective-config`,
`--no-print-effective-config`, and `--print-effective-config-json` (with
`--effective-config-json` as an alias), and emits a resolved config record that
includes CLI values, ECHO environment overrides, paths, mode, and the full
serialized `GrpoConfig`. The existing `config mode=...` compatibility line is
preserved for the ablation analyzer. Help text now explains that `--mode`
selects the effective advantage formulation and related GRPO knobs, and the
`--no-echo`/`--echo-lambda` conflict error now states the one-argument pattern
to use.

**Validation evidence:**

- RunPod validation passed on 2026-05-21 on RTX A6000 pod `qmfxie9izl6lc6`.
- Remote command sequence: `git diff --check`; `cargo build --release -p
  kiln-train --features cuda --example cuda_grpo_ablation`; CLI help smoke
  checks for `--print-effective-config-json` and `--mode` advantage text; CLI
  conflict smoke check for `--no-echo --echo-lambda`; `cargo check -p
  kiln-train --tests`; `cargo check -p kiln-server --tests`.
- Remote sentinel `/workspace/kiln-validation/issue23.done` recorded `exit=0`;
  remote log is `/workspace/kiln-validation/issue23.log`.

**Commit SHA:** `3acbdea7` (`Issue 23: print effective GRPO config`).
This status line is recorded in the follow-up metadata commit because a
commit cannot include its own final SHA.

### 24. Add Standard Failure Reasons To Training Output

**Area:** `crates/kiln-train`

**Problem:** Iter logs used many ad hoc statuses. Kiln should classify common
training and data failures.

**Task:** Standardize failure reasons in receipts and errors.

**Suggested reasons:**

- `data_schema_error`
- `adapter_load_failed`
- `zero_groups`
- `zero_action_tokens`
- `zero_env_tokens`
- `nan_loss`
- `oom`
- `shape_mismatch`
- `unsafe_lora_scale`
- `base_adapter_missing`

**Acceptance criteria:**

- Receipts include `status` and `failure_reason`.
- CLI exits nonzero for all failure reasons.
- Error messages include enough context to fix the issue.

**Implementation status:** Complete. Training receipts now keep
`failure_reason` as a stable machine-readable code and add
`failure_message` for the human-readable context. The standard classifier
covers `data_schema_error`, `adapter_load_failed`, `zero_groups`,
`zero_action_tokens`, `zero_env_tokens`, `nan_loss`, `oom`,
`shape_mismatch`, `unsafe_lora_scale`, and `base_adapter_missing`, with a
`training_error` fallback for unexpected failures. Generic SFT/GRPO,
streamed GRPO, GRPO dry-run, OPD early validation, and CUDA-native SFT early
validation now annotate returned errors with `failure_reason=<code>` while
preserving the original diagnostic text. Existing failed receipts from all
receipt writers now store the standard reason plus the detailed message.

**Validation evidence:**

- RunPod validation passed on 2026-05-21 on RTX A6000 pod `qmfxie9izl6lc6`.
- Remote command sequence: `git diff --check`; `cargo test -p kiln-train
  training_failure_reason_classifier_covers_standard_reasons --lib`; `cargo
  test -p kiln-train train_receipt_failed_status_is_stable --lib`; `cargo
  test -p kiln-train
  grpo_dry_run_rejects_echo_without_env_tokens_and_writes_receipt --lib`;
  `cargo test -p kiln-train
  grpo_dry_run_rejects_zero_groups_after_filter_unless_allowed --lib`;
  `cargo test -p kiln-train grpo_dry_run_reward_filter_on_empty_modes --lib`;
  `cargo check --release -p kiln-train --features cuda --example
  cuda_grpo_ablation`; `cargo check -p kiln-train --tests`; `cargo check -p
  kiln-server --tests`.
- Remote sentinel `/workspace/kiln-validation/issue24b.done` recorded
  `exit=0`; remote log is `/workspace/kiln-validation/issue24b.log`.

**Commit SHA:** `7110f67a` (`Issue 24: standardize training failure reasons`).
This status line is recorded in the follow-up metadata commit because a
commit cannot include its own final SHA.

**Remaining risk:** Deep native CUDA/Vulkan/OPD failures that occur after
early validation can still propagate through lower-level anyhow context before
a path-specific failed receipt is written. The shared receipt machinery and
the generic/CLI-facing GRPO/SFT validation paths now produce standardized
failure codes for the common operator-facing failures listed above.

### 25. Add Long-Context GRPO Benchmark Suite

**Area:** `crates/kiln-train`, benchmarks

**Problem:** `pi-compaction` exposed long-context training as a major kiln
bottleneck. We need a stable benchmark to prevent regressions and guide fixes.

**Task:** Add synthetic GRPO benchmarks at 8K, 16K, 32K, and 64K sequence
lengths.

**Metrics:**

- tokenize time
- mask build time
- reference forward time
- policy forward time
- backward time
- optimizer time
- peak VRAM
- kernel launch count if available
- tokens/sec

**Acceptance criteria:**

- Benchmark can run a small CPU/dry version in normal dev environments.
- CUDA benchmark works with configurable model/task size.
- Results are emitted as JSON.
- Docs explain expected use for compaction-like workloads.

**Status:** Implemented.

**Implementation notes:**

- Added `crates/kiln-train/examples/long_context_grpo_bench.rs`, a synthetic
  compaction-shaped GRPO benchmark with default length sweep
  `8192,16384,32768,65536`.
- Added trainer timing hooks for tokenization, mask build, reference forward,
  policy forward, backward, and optimizer phases.
- Dry mode emits JSON with a built-in byte tokenizer when `--model` is omitted,
  so CPU-only development environments can run a smoke without model assets.
- CUDA mode requires `--model <Qwen3.5-4B-dir>`, supports configurable
  `--lengths`, `--completions`, and `--segments`, and records peak VRAM when
  `nvidia-smi` is available. `kernel_launch_count` is emitted as `null` until a
  launch counter is wired.
- Added `docs/LONG_CONTEXT_GRPO_BENCH.md` with dry/CUDA examples and expected
  compaction-workload usage.

**Validation evidence:**

- RunPod validation passed on 2026-05-21 on RTX A6000 pod `qmfxie9izl6lc6`.
- Remote command sequence: `git diff --check`; `cargo check -p kiln-train
  --example long_context_grpo_bench`; `cargo check -p kiln-train --tests`;
  `cargo check --release -p kiln-train --features cuda --example
  long_context_grpo_bench`; `cargo run -p kiln-train --example
  long_context_grpo_bench -- --dry-run --lengths 256 --output
  /workspace/kiln-validation/issue25-dry.json`.
- Remote sentinel `/workspace/kiln-validation/issue25.done` recorded `exit=0`;
  remote log is `/workspace/kiln-validation/issue25.log`.
- Actual-model CUDA validation later passed on the same RTX A6000 pod
  `qmfxie9izl6lc6` with `KILN_MODEL_PATH=/workspace/Qwen3.5-4B`.
  `/workspace/kiln-validation/actual_model_validation.done` recorded `exit=0`
  and `ACTUAL_MODEL_VALIDATION_OK`; artifacts are under
  `/workspace/kiln-validation/actual-model-validation/`.
- `long_context_grpo_bench --cuda --lengths 8192,16384,32768,65536` completed
  with model path `/workspace/Qwen3.5-4B`, nonzero reference/policy/backward/
  optimizer timings at every length, and peak VRAM of 14344, 15338, 18218, and
  25066 MiB respectively.

**Commit SHA:** `7b62d451` (`Issue 25: add long-context GRPO benchmark`).
This status line is recorded in the follow-up metadata commit because a
commit cannot include its own final SHA.

**Remaining risk:** `kernel_launch_count` is still emitted as `null` until a
kernel launch counter is wired. The full 8K-64K CUDA runtime path has now been
validated against actual Qwen3.5-4B assets on RTX A6000.

### 26. Add Long-Context Progress Logging

**Area:** `crates/kiln-train`

**Problem:** Long-context training could appear hung for many minutes with no
progress lines.

**Task:** Emit progress markers around major long-context phases.

**Markers:**

- data loaded
- tokenize start/end
- mask build start/end
- ref forward start/end
- policy forward start/end
- backward start/end per checkpoint segment
- optimizer start/end

**Acceptance criteria:**

- Logs appear before any phase expected to take more than 30 seconds.
- Logs include token count and segment/tile config.
- Receipt includes per-phase timings.

**Status:** Implemented.

**Implementation notes:**

- Added GRPO progress logs for data load, tokenization, trajectory mask build,
  reference forward, policy forward, backward, checkpoint backward segments, and
  optimizer phases.
- Included token counts, sequence length, segment count, streaming prefill, and
  tile/reverse-mode details in long-context phase logs where applicable.
- Added `phase_timings` to GRPO train receipts and schema docs, with aggregate
  tokenization, mask build, reference forward, policy forward, backward, and
  optimizer durations.
- Updated the GRPO dry-run receipt test to assert populated tokenization and
  mask-build phase timings.

**Validation evidence:**

- RunPod validation passed on 2026-05-21 on RTX A6000 pod `qmfxie9izl6lc6`.
- Remote command sequence: `git diff --check`; `cargo check -p kiln-train
  --tests`; `cargo test -p kiln-train
  grpo_dry_run_success_records_counts_and_receipt --lib`; `cargo check
  --release -p kiln-train --features cuda --example long_context_grpo_bench`.
- Remote sentinel `/workspace/kiln-validation/issue26.done` recorded `exit=0`;
  remote log is `/workspace/kiln-validation/issue26.log`.
- Additional RunPod locked test after the `Cargo.lock` repair passed:
  `cargo test --locked -p kiln-train --lib -- --skip test_health_with_real_backend`
  (241 tests passed).
- Remote sentinel `/workspace/kiln-validation/issue26-fulltest.done` recorded
  `exit=0`; remote log is `/workspace/kiln-validation/issue26-fulltest.log`.
- Actual-model CUDA validation later passed on RTX A6000 pod `qmfxie9izl6lc6`
  with `KILN_MODEL_PATH=/workspace/Qwen3.5-4B`. The 8K, 16K, 32K, and 64K
  long-context GRPO bench rows were written to
  `/workspace/kiln-validation/actual-model-validation/long_context_grpo_bench.json`
  with nonzero phase timings and `ACTUAL_MODEL_VALIDATION_OK`.

**Commit SHA:** `1ce37ca4` (`Issue 26: add long-context progress logging`).
This status line is recorded in the follow-up metadata commit because a commit
cannot include its own final SHA.

**Remaining risk:** Validation covered compilation, the focused receipt test, the
full `kiln-train` library test suite, CUDA release example compilation, and the
full actual-model 8K-64K CUDA training runtime on A6000. Progress logging still
does not include per-kernel launch counts because that counter is not wired.

### 27. Debug Why Long-Context Adapters Can Be Byte-Identical

**Area:** `crates/kiln-train`, possibly `crates/kiln-server`

**Problem:** `pi-compaction` trained adapters that produced byte-identical eval
responses to base, even after long-context training became tractable.

**Task:** Add and use diagnostics to identify whether the no-op is caused by
adapter loading, zero gradients, KL dominance, clipping, masking, or inference
scale.

**Acceptance criteria:**

- A reproducible small long-context fixture exists.
- Training receipt shows nonzero or zero gradient/delta norms.
- Adapter verify shows whether logits change on the fixture.
- If logits change but text is identical, report sampling/determinism reason.
- If logits do not change, file a root-cause note in the doc or test.

**Status:** Completed and pending commit on 2026-05-21.

**Implementation notes:** Added explicit adapter-smoke prompt diagnostics to
`train_receipt.json`: each prompt now records whether logits changed, text
changed, both changed, neither changed, logits were non-finite, or adapter
output was empty. The receipt records non-failing notes for the important
`logits_changed_text_identical` case, explaining deterministic greedy argmax,
and the no-effect warning now names the concrete root-cause surfaces to inspect
(`adapter` load path, `lora_grad_norms`, `lora_delta_norms`, KL/clipping,
masks, and LoRA scale). Added a reusable `kiln-train` synthetic long-context
fixture module and taught `long_context_grpo_bench` to write fixture JSON with
an adapter-verify prompt carrying the synthetic trace. Added
`KILN_ADAPTER_SMOKE_PROMPT_FILE` so actual training can run the adapter-smoke
logit comparison on that fixture prompt. `kiln adapter verify --url` now reports
the fixed prompt, whether generated text differs, and a deterministic-greedy
diagnosis/note when the offline LoRA delta proxy is measurable but text remains
byte-identical.

**Validation evidence:**

- Focused RunPod checks passed on RTX A6000 pod `qmfxie9izl6lc6`; sentinel
  `/workspace/kiln-validation/issue27/focused-tests.done` recorded `exit=0`.
  Remote log: `/workspace/kiln-validation/issue27/focused-tests.log`.
- Focused checks run: `cargo test --locked -p kiln-train adapter_smoke_receipt
  --lib`, `cargo test --locked -p kiln-train long_context_fixture --lib`,
  `cargo test --locked -p kiln-server --test adapter_verify`, and
  `cargo run --locked -p kiln-train --example long_context_grpo_bench --
  --dry-run --lengths 512 --output ... --fixture-output ...`. The dry fixture
  observed 551 tokens and wrote a trace-bearing `adapter_verify_prompt`.
- Actual Qwen3.5-4B RunPod validation passed on the same A6000 with canonical
  model path `/workspace/Qwen3.5-4B`; sentinel
  `/workspace/kiln-validation/issue27/actual-model.done` recorded `exit=0` and
  driver log `/workspace/kiln-validation/issue27/actual-model/driver.log`
  ended with `ISSUE27_ACTUAL_MODEL_OK`.
- Actual-model CUDA fixture bench wrote
  `/workspace/kiln-validation/issue27/actual-model/long_context_cuda.json`:
  requested 1024 tokens, observed 1030, total tokens 2060, peak VRAM 26538 MiB,
  53.1482 tok/s, reference forward 1095.19 ms, policy forward 614.54 ms,
  backward 37009.56 ms, optimizer 9.25 ms.
- Actual-model CUDA GRPO trained `issue27-cuda-grpo` from the fixture JSONL
  with `KILN_ADAPTER_SMOKE_PROMPT_FILE` set to the fixture prompt.
  `train_receipt.json` recorded status `success`, 10 gradient-norm modules,
  10 delta-norm modules, adapter smoke passed, `logit_delta_l2=48.0585737549`,
  and prompt outcome `logits_changed_text_identical` with the deterministic
  argmax note.
- `kiln adapter verify issue27-cuda-grpo --adapter-dir ...` returned
  `status=ok`, `measurable=true`, and `l2_delta_proxy=1.6667806786`.
- `kiln adapter verify issue27-cuda-grpo --adapter-dir ... --url
  http://127.0.0.1:18427 --prompt <fixture prompt>` returned `status=ok`,
  `generated_text_different=false`, and
  `behavior_diagnosis=measurable_adapter_delta_with_identical_greedy_text`.

**Commit SHA:** Pending.

**Remaining risk:** This issue now distinguishes “logits moved but greedy text
matched” from a no-op adapter on a real Qwen3.5-4B fixture. It does not claim
that the trained fixture adapter improves pi-compaction eval quality; that
requires the cap-level eval loop after the diagnostics are available.

### 28. Make Warning-Prefix Masking Testable

**Area:** `crates/kiln-train`

**Problem:** ECHO should not learn harness warning boilerplate. The Python
parser tracks `warning_prefix_len`, but kiln needs explicit tests and metrics.

**Task:** Add tests and receipt metrics for warning-prefix filtering.

**Acceptance criteria:**

- Test trajectory with `WARNINGS:` prefix masks warning tokens out of env CE.
- Receipt reports env tokens before and after warning filter.
- Config can disable warning filter for ablation.
- Logs warn if most env tokens are stripped as warnings.

## P1: Improve Server And API Observability

### 29. Add Per-Request Performance Counters To Chat Responses

**Area:** `crates/kiln-server`

**Problem:** Cap evals need to know whether an adapter is improving behavior or
just making sessions slower and chattier.

**Task:** Add optional performance metadata to chat-completion responses.

**Fields:**

- prompt tokens
- completion tokens
- TTFT
- total latency
- decode tokens/sec
- adapter used
- thinking mode
- finish reason

**Acceptance criteria:**

- Enabled by request flag or server config.
- Does not break OpenAI compatibility by default.
- Eval scripts can parse it from a stable JSON field.

### 30. Add Train-Time And Serve-Time Config Hashes

**Area:** `crates/kiln-server`, `crates/kiln-train`

**Problem:** Results changed after tokenizer/chat-template changes. Runs need
to record which template/config was actually used.

**Task:** Hash relevant model-side config and include it in responses and
receipts.

**Fields:**

- tokenizer config hash
- chat template hash
- model config hash
- kiln env/config hash

**Acceptance criteria:**

- Training receipt includes all hashes.
- Server health includes all hashes.
- Chat response metadata can include all hashes in debug mode.

### 31. Add Model-Specific Defaults Profile For Qwen3.5-4B

**Area:** `crates/kiln-server`, config

**Problem:** Kiln deliberately supports Qwen3.5-4B. It should encode safe
defaults learned from real usage, especially around thinking mode and agentic
tool calls.

**Task:** Add an explicit Qwen3.5-4B defaults profile.

**Acceptance criteria:**

- Profile documents default thinking behavior.
- Profile documents adapter directory expectations.
- Profile documents supported chat template behavior.
- Server logs which profile is active.
- Tests assert profile defaults.

### 32. Add `/v1/debug/model-state`

**Area:** `crates/kiln-server`

**Problem:** During eval, authors need a compact answer to "what model state am
I actually hitting?"

**Task:** Add a debug endpoint for local/trusted use.

**Response should include:**

- model path
- active adapter
- loaded adapters and hashes
- config hashes
- key env flags
- batching engine status
- thinking default
- cache summaries

**Acceptance criteria:**

- Endpoint is disabled or protected in non-debug mode if needed.
- Response contains no prompt/user data.
- Tests cover active adapter and config fields.

## P2: Add Higher-Level Kiln Capabilities For This Workflow

### 33. Add `kiln eval-adapter`

**Area:** kiln CLI

**Problem:** Multi-seed eval was repeatedly needed to avoid overclaiming.
Every cap implemented its own evaluation driver and summary format.

**Task:** Add a generic adapter eval command that runs a task JSONL through a
configurable request/scorer command.

**Command shape:**

```text
kiln eval-adapter --adapter NAME --tasks eval.tasks.jsonl --seeds 3 \
  --request-template request.json --scorer ./score_one.py
```

**Acceptance criteria:**

- Runs base and adapter in paired mode.
- Emits mean, stdev, zero count, and wall-clock stats.
- Writes `eval_summary.json`.
- Warns if sigma is comparable to lift.
- Records adapter and config hashes.

### 34. Add Direct HTTP Rollout Generation Utility

**Area:** kiln CLI or `crates/kiln-server` client tooling

**Problem:** Some useful capabilities do not need the Pi tool loop. Direct
single-turn generation would avoid Pi schema and orchestration overhead.

**Task:** Add a kiln-owned utility for task JSONL -> chat-completion responses
-> scored rollout JSONL.

**Acceptance criteria:**

- Sets adapter explicitly per request.
- Sets thinking mode explicitly.
- Records latency and token counts.
- Supports deterministic seeds if server supports them.
- Emits ScoredRollout-compatible JSONL for trainer input.

### 35. Add Adapter Canary And Quarantine Status

**Area:** `crates/kiln-train`, `crates/kiln-server`

**Problem:** Some trained adapters produced empty outputs or huge latency
regressions. Kiln should mark these as unsafe before users accidentally eval or
serve them.

**Task:** Add optional canary evaluation after training and expose status in
adapter registry.

**Canaries:**

- simple short completion
- simple tool-call-shaped prompt
- output length sanity
- finite logits
- latency sanity
- nonzero logit delta from base

**Acceptance criteria:**

- Failed canaries mark adapter status as `quarantined`.
- `/v1/adapters/load` refuses quarantined adapters unless override flag is set.
- Registry shows canary status and failure reason.

### 36. Add Standard Adapter Manifest And Restore Command

**Area:** kiln CLI, adapter format docs

**Problem:** Every cap wrote its own artifact backup manifest. Kiln should own
adapter provenance regardless of storage backend.

**Task:** Define `adapter_manifest.json` and implement `kiln adapter restore`.

**Manifest fields:**

- adapter name
- safetensors hash
- config hash
- receipt hash
- parent adapter
- model config hash
- kiln commit
- training data hash

**Acceptance criteria:**

- Training writes manifest.
- Server registry reads manifest if present.
- Restore command verifies hashes after copy.
- Manifest schema is documented.

### 37. Add Off-Policy Distillation Training Mode

**Area:** `crates/kiln-train`

**Problem:** Saturated GRPO caps need a better tool than policy gradient over
noisy rewards. Several negative results point toward teacher distillation or
OPD for high-baseline capabilities.

**Task:** Add or formalize an OPD/teacher-distillation path for agentic tasks.

**Acceptance criteria:**

- Accepts teacher response/logprob data in a documented JSONL schema.
- Supports KL or cross-entropy against teacher action tokens.
- Can combine with ECHO on env tokens.
- Receipt distinguishes OPD loss from GRPO loss.
- Includes a small synthetic test.

### 38. Add Reward-Saturation-Aware Training Recommendation

**Area:** `crates/kiln-train`

**Problem:** Cap authors wasted time applying GRPO to near-ceiling baselines.
Kiln can surface when the data distribution is unlikely to benefit.

**Task:** In dry-run and training startup, print a recommendation when reward
distribution is saturated.

**Acceptance criteria:**

- If mean reward > threshold and group variance < threshold, warning says
  policy-gradient may be harmful.
- Warning suggests harder tasks, stronger rubric gates, OPD, or
  `--no-policy-loss`.
- Thresholds are configurable.

### 39. Add End-To-End Agentic-GRPO Plumbing Test

**Area:** tests / examples

**Problem:** The terminal-bench-lite synthetic runs proved the stack manually.
That should become a maintained regression test.

**Task:** Add a small test that builds synthetic action/observation
trajectories, trains with ECHO on/off, and verifies a measurable adapter
difference.

**Acceptance criteria:**

- Test can run in a reduced CPU or tiny-model mode in CI if possible.
- CUDA version can run locally for full validation.
- Verifies ECHO on/off produce different LoRA deltas.
- Verifies `--no-policy-loss` still trains env CE.
- Verifies `--base-adapter` changes step-1 loss after loading.

### 40. Add Regression Tests For Lessons Already Learned

**Area:** tests across server/train

**Problem:** The agentic-GRPO work found repeatable bugs. They should become
tests so they do not recur.

**Task:** Add regression tests for the following cases:

- wrong nested adapter directory fails clearly
- missing adapter request field does not silently unload active adapter
- `--base-adapter` loads weights, not just lineage
- base-adapter rank mismatch fails before training
- alpha/rank > 2 fails or requires override
- Pi `toolResult` role normalizes to `tool`
- ECHO enabled with env tokens records nonzero env count
- no-policy-loss zeroes policy loss while preserving ECHO training
- Qwen thinking off produces normal content on a short prompt
- repeated load/eval/unload does not cause severe latency drift

**Acceptance criteria:**

- Each test links to the bug class in its test name or comment.
- Tests are included in the normal Rust test suite where feasible.
- CUDA-only tests are marked and documented separately.

## Suggested Execution Order

1. Adapter semantics and load validation: issues 1-7.
2. Training receipts, dry-run, and ECHO/mask visibility: issues 8-12.
3. Eval stability and Qwen thinking controls: issues 13-18.
4. Training diagnostics and long-context support: issues 19-28.
5. API observability and config hashes: issues 29-32.
6. Higher-level workflow helpers and regression tests: issues 33-40.

Completing just the first three groups would remove most of the ambiguity that
made the first agentic-GRPO runs hard to interpret. Completing all forty would
turn kiln from "can train agentic adapters" into "can run agentic learning
experiments with trustworthy receipts, safe adapter state, stable evals, and
actionable diagnostics."
