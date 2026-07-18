# Kiln Server Changelog

## Unreleased — bounded thinking by tokens or decode time

- CUDA backend configuration: replaced fourteen backend hot-path environment
  reads with immutable typed `accelerator.cuda_kernel_profile` policy.
  `native_default` preserves existing accelerated dispatch without making a
  hardware-qualification claim; `portable_fallback` declines all fourteen
  owned routes. The v12 policy is source-tracked in CLI, API, health, trusted
  debug, dashboard, qualification fixtures, schemas, examples, and website.
  CUDA-specific retired per-kernel spellings are not aliases.
- ROCm kernel policy consolidation: replaced the independent production
  model-kernel environment gates with the immutable typed
  `accelerator.rocm_kernel_profile`. `qualified` installs the complete
  Strix Halo-qualified route set, `portable_fallback` declines all fifteen
  profile-governed routes for reference comparison, and experimental-only
  `experimental_multiblock` adds the unqualified multi-block GDN prefill
  route. The server installs one complete policy before backend construction,
  backend hot paths no longer read these environment variables, and the
  resolved v9 policy is reported by CLI, API, health, trusted debug, dashboard,
  qualification attestations, schemas, and the website. Retired per-kernel
  names are not compatibility aliases.
- training configuration boundary: removed the remaining process-global ECHO,
  adapter-smoke, GRPO shared-reference, and OPD sampler/rendering controls from
  `kiln-train`. Their typed request-local replacements are carried through API
  schemas, CLI, dashboard resume, receipts, and exact checkpoints. Custom
  adapter smoke prompts are submitted as request data; OPD prompt rendering is
  now an explicit algorithm enum; malformed or inert combinations fail before
  GPU work.
- workload and optimizer admission: SFT, GRPO, OPD, DistillRefresh, recipes,
  judge/self-improve, and OPD-backed distillation now run cheap static workload
  and optimizer-tuple guards after request/teacher-alias validation and metadata
  pinning but before checkpoint loading, remote/local teacher materialization,
  corpus scanning, memory preflight, or GPU reservation; repeat them
  before queue memory reservation; and revalidate the tuple before residency.
  Workload guards bind serving-profile training ownership, resident weight and
  exact native backend/device identity, non-Marlin weights, authoritative tape,
  and workload-specific loss/backward routes. CUDA/ROCm Muon is bounded to
  ranks 2..=48, Metal/Vulkan to 2..=32, Metal SGD is rejected, and CUDA/ROCm
  F16 remains inference-only. CPU portable-reference Muon requires rank 2+, but
  current CPU server workloads remain unsupported. Invalid kind, rank, or
  hyperparameters return `training_invalid_request`; an unavailable resident
  tuple or workload returns `training_backend_unsupported` without changing
  optimizer, rank, or execution route.
- DistillRefresh admission: refresh is now a distinct `distill_refresh`
  workload instead of being admitted as OPD. After cheap teacher-alias
  validation and metadata pinning, it fails closed before checkpoint loading,
  remote/local teacher materialization, corpus scanning, memory preflight, or
  GPU reservation with
  `distill_refresh is unavailable until admission pins separate exact SFT and
  OPD phase plans, prepares the exact SFT rows, and reserves the maximum
  sequential working set`. Re-enabling the sequential SFT knowledge phase and
  OPD behavior-recovery phase requires one admission record that binds both
  exact plans and the phase-one rows and reserves the larger phase peak.
- optimizer diagnostics and ergonomics: additive `GET /v1/config` field
  `training.optimizer_support` uses schema
  `{"id":"kiln.training-optimizer-support","version":1}`. It separates raw
  `backend_implementation`, exact resident `optimizer_tuple` and
  `optimizer_tuple_kinds`, per-workload `supported`, `unavailable_reason`, and
  `allowed_optimizer_kinds` for `sft`, `grpo`, `opd`, and `distill_refresh`,
  plus request-time live-memory admission. This keeps
  CPU portable-reference tuples and hybrid Vulkan hooks visible without
  claiming server execution. The dashboard reads the workload authority,
  disables unsupported training surfaces, preserves entered optimizer/rank
  instead of silently clamping or substituting them, and exposes the failed
  gate. `GET /v1/recipes` now gives every built-in descriptor
  `admission {supported, unavailable_reason}` after evaluating every step; run
  submission repeats the preflight before preparing any step.
- optimizer rank diagnostics: every `lora_rank` now separates
  `backend_maximum`, model-derived `model_maximum`, and effective `maximum`,
  which is their minimum. An unbounded backend reports
  `backend_maximum: null` while `maximum` remains the concrete model ceiling;
  canonical Qwen3.5-4B reports 1024. Live memory remains a later admission gate
  and never rewrites the static range.
- optimizer configuration cleanup: product training is now unconditionally
  round-to-nearest. Removed `KILN_BF16_STOCHASTIC_ROUND`,
  `KILN_TRAINING_HOT_PATH_DEBUG_FALLBACK`, and all four backend
  `KILN_*_TRAINING_OPTIMIZER_FALLBACK` switches without aliases. Stochastic
  rounding remains an explicit optimizer-library policy; legacy stochastic
  exact checkpoints fail closed on precision mismatch rather than changing
  their update rule during resume.
- training durability: native SFT, inline and streamed-JSONL GRPO, and OPD
  publish immutable exact checkpoints at a configured cadence and on
  cooperative cancellation. Resume restores policy/adapter, optimizer,
  reference/EMA where applicable, distinct optimizer/data cursors, RNG, loss
  history, and diagnostic state only after complete artifact, checksum, route,
  data, configuration, model, tokenizer, precision, backend, and OPD teacher
  identity validation. The API, CLI, job detail, and browser forms expose the
  same checkpoint basename contract; serving-only PEFT snapshots remain
  deliberately separate and are never accepted as resumable state.
- training correctness: GRPO is now locked to a source-pinned TRL 1.8.0 and
  PyTorch 2.13.0 oracle for loss, ratios, clipping, K3, policy-logprob
  gradients, and one AdamW step. The oracle exposed and fixed a second
  completion-length division in GSPO and a two-sided PPO clamp in CISPO.
  CISPO now has its own absolute upper-only `cispo_max_weight` control
  (default 5.0), with matching receipts, API validation, audit semantics,
  ROCm device-coefficient coverage, and Vulkan shader coverage.
- typed SFT checkpoint boundaries: `[training]` now owns sparse-boundary replay
  mode (`auto`, default), its automatic sequence crossover (8192 tokens),
  anchor stride (`auto`, default), and automatic-stride cache target (6.0 GiB).
  Canonical overrides derive mechanically as `KILN_TRAINING_<FIELD>`; the four
  old unsectioned spellings remain strict warning aliases. Malformed,
  non-Unicode, or conflicting values stop startup, and every change requires a
  restart.
- checkpoint-boundary authority and identity: startup resolves one immutable
  integral policy shared by SFT memory admission and execution, eliminating
  trainer/preflight parsing drift and runtime environment rereads. GRPO and OPD
  do not execute the SFT sparse-boundary route, but every mode records the
  policy in training checkpoint planning identity. GRPO and OPD remain v3;
  SFT extends the object to v4 with its pinned backend loss route. Policy or
  schema drift rejects exact resume. `/v1/config`, health, trusted debug, and
  the dashboard expose the same checkpoint-boundary policy.
- training cleanup: removed `KILN_EXACT_GDN_TILE_BACKWARD` and
  `KILN_EXACT_GDN_BACKWARD_TILE_TOKENS`. Their selector had no production
  caller after the tiled-reverse implementation was removed, so the variables
  could only mutate dead code. The corresponding backend precision field,
  generated capability-report column, source guard, and env-mutating test are
  gone; qualification now rejects reintroducing the two spellings.
- training authority: removed `KILN_USE_TAPE_AUTHORITATIVE`. SFT and OPD were
  already unconditionally tape-authoritative, while setting this switch false
  made GRPO fail only after reference work because its candle fallback no
  longer exists. GRPO now validates the backend tape route and precision policy
  through one shared guard before resident setup and again before each direct
  group step. Nine unscoped test mutations, including two leaked false values,
  are removed.
- tape scope authority and inference isolation: the kt tape is now a required
  workload substrate, activated solely by Kiln's internal training scope.
  Removed `KILN_USE_TAPE_FORWARD`, `KILN_USE_TAPE_FLASH_ATTN`,
  `KILN_USE_TAPE_SDPA`, `KILN_USE_TAPE_LORA_ADD`, `KILN_USE_TAPE_GDN`,
  `KILN_USE_TAPE_GDN_CONV`, `KILN_USE_TAPE_GDN_QK_NORM`, and
  `KILN_USE_TAPE_GDN_GATED_NORM` without aliases or replacement configuration.
  A global default-on switch could make ordinary inference decline forward-only
  fast paths even though no tape existed; GDN chunkwise recurrence and CUDA's
  weight-aware embedding route now defer only while a training scope is actually
  active. Debug and tuning controls may select only graph-preserving
  implementations inside that scope; they cannot disable a required recorder
  or silently sever the gradient graph.
- exact LoRA gradient consumption: every optimizer entry now requires the
  observed gradient IDs to equal the configured trainable leaf set and checks
  each tensor's shape, backward dtype, master device, and finite values before
  mutating parameters or optimizer state. Missing and unknown leaves fail
  closed; legitimate all-zero gradients remain valid. The tape bridge sums all
  recorded input contributions into one accumulated entry per leaf, but first
  requires every registered differentiable input to have a gradient so a
  connected clone cannot hide a disconnected one. Split output-projection
  chunks deposit zero-padded gradients under the original full LoRA leaf rather
  than temporary slice IDs. Checkpoint segments require the exact leaves for
  their layer range; an empty configured range accepts only an empty result,
  and the merge rejects duplicate leaf IDs across disjoint segments.
  Accumulated GRPO values incur only one finite scan at the final optimizer
  boundary. CUDA,
  ROCm, and Vulkan use backend reducers; Metal uses a
  synchronized host-scan correctness fallback instead of rejecting every step
  while its native finite reducer remains pending.
- frozen-parameter tape ownership: native training now records LoRA A/B as the
  only trainable model leaves. Embedding tables, base projection matrices,
  RMSNorm and gated-RMSNorm weights, GDN gate parameters, MTP projections, and
  FLCE/OPD/GRPO heads are saved constants. Their backward paths return only the
  activation and LoRA gradients required upstream, omit frozen dWeight GEMMs,
  and do not allocate, retain, or deposit frozen-weight gradients. CUDA/ROCm
  fused RMSNorm routes use dx-only kernels that skip dWeight atomics; the
  portable route has the same ownership. Split Q/gate projections preserve the
  original full LoRA A/B IDs, zero-pad and accumulate each B slice, and use
  tape-aware reshapes so temporary slices cannot become optimizer leaves or
  sever the activation graph. The shared final-norm helper and portable GDN
  gated-norm fallback are likewise dx-only. Four unused prefill helpers now
  reject active training scopes until their reshape/narrow chains are recorded,
  and ROCm's W8 full-attention output leaf is explicitly inference-only.
- SFT loss authority and admission: removed the process-global
  `KILN_USE_FLCE` route override. SFT now consumes the selected backend's typed
  loss route for every step. There is no replacement TOML/request field,
  derived environment name, or compatibility alias. Admission pins that route
  together with the active-token and checkpoint-boundary estimates. The queue
  revalidates it against the resident runner before memory reservation, and the
  trainer revalidates it against its execution backend before allocation.
  Route-specific upper bounds charge CUDA/ROCm F32 head promotion, Vulkan's
  maximum legal chunk workspace, or full-logits CE forward/backward residency.
  Full-logits checkpoint plans are rejected before queue publication and again
  before any trainer forward because checkpoint tails do not run inside an
  active tape.
  HTTP 413 details identify the selected loss workspace; new SFT receipts
  record `runtime.sft_loss_route`, and exact SFT checkpoints bind the route in
  planning identity v4. The obsolete Phase A validation example and live
  script override are removed.

- inference: chat, streaming chat, multi-choice chat, and batch generation now
  accept `thinking_budget_tokens` and `thinking_budget_ms`. Kiln closes an open
  thinking block at the first active limit, feeds the close-tag tokens through
  model context, and continues into the final answer. Forced tokens count toward
  completion usage and `max_tokens`; natural closes still win.
- configuration: `server.default_thinking_budget_tokens` /
  `server.default_thinking_budget_ms` and
  `KILN_DEFAULT_THINKING_BUDGET_TOKENS` /
  `KILN_DEFAULT_THINKING_BUDGET_MS` set inheritable defaults. Both default to
  unlimited; request `null` opts out of a configured dimension and `0` closes
  thinking immediately.
- typed batching configuration: `[batching]` now owns actor selection, true
  batched versus rowwise decode, strict-prefix-aware admission, and the prompt
  admission quantum. Canonical overrides derive mechanically as
  `KILN_BATCHING_<FIELD>`; the four older primary-actor spellings remain strict,
  warning compatibility aliases. Values resolve once after backend and
  effective decode-width selection, require restart, and are injected into the
  actor without production runtime environment rereads.
- direct decode rendezvous configuration: the actor-absent direct streaming
  effectively-greedy compatibility worker now consumes four additional typed
  `[batching]` startup fields for mode, maximum batch, wait microseconds, and
  mixed sequence lengths. Canonical names derive as
  `KILN_BATCHING_DIRECT_DECODE_RENDEZVOUS_<FIELD>`; the four old
  `KILN_DECODE_BATCH*` names are strict warning aliases. Conflicts and malformed
  values fail startup, including malformed legacy wait values that previously
  became zero. Backend defaults remain CPU `(8,0,false)`, CUDA `(1,0,false)`,
  ROCm `(8,0,false)`, Metal `(8,100,true)`, and Vulkan `(64,5000,true)`, with
  every auto mode enabled and maximum batch clamped to effective decode width.
- batching diagnostics: `/v1/config.batching` separates immutable configured
  intent, value provenance, backend defaults, effective actor selection,
  decode-width quantum clamping, and `actor_active`. Health repeats the same
  policy at `decode_runtime.batching_configuration`, and trusted debug adds it
  at `batching_engine.configuration`. CUDA/Vulkan preserve width-filling
  automatic admission, ROCm/Metal/CPU preserve the latency-oriented quantum of
  four, CUDA preserves backend-owned burst admission, and all are constrained
  by the final effective decode width.
- direct rendezvous diagnostics: immutable configured/backend/effective policy
  is at `batching.configuration.direct_decode_rendezvous`; sibling status at
  `batching.direct_decode_rendezvous` reports the exact scope, backend
  availability/reason, actor activity, worker activity, and route availability.
  Health and trusted debug expose the same facts. A worker may be active while
  unroutable because the actor is active; sampled, non-streaming, and
  actor-routed requests bypass this narrow fallback.
- streaming-prefill configuration: `[streaming_prefill]` now owns automatic,
  forced, and disabled dispatch; the prompt threshold; base, tape, and detached
  full-attention tiles; and final-tile LM-head behavior. Canonical environment
  names derive mechanically from those six fields. Six old spellings remain
  strict warning aliases, and legacy TOML `enabled` is accepted only when it
  agrees with any explicit `mode`. Invalid, non-Unicode, or conflicting values
  stop startup.
- streaming-prefill authority and diagnostics: startup resolves one immutable
  backend-qualified policy and injects it into generation, prompt-logprob
  scoring, native training, local OPD teachers, MTP alignment, checkpoint
  planning, and the benchmark without lower model/trainer environment rereads.
  Server teacher construction fails closed without the startup value. Explicit
  base tiles feed specialized `auto` routes; detached overrides cover ordinary,
  boundary, and tape-replay full-attention routes. `/v1/config` exposes the
  complete configured/backend/effective/source object at `streaming_prefill`,
  health at `prefill_runtime.streaming_prefill`, and trusted debug at top-level
  `streaming_prefill`. Kiln prompt-logprob teacher identity now uses
  inference-contract v2 and hashes every resolved field, so policy changes
  invalidate teacher revisions, logit caches, and OPD resume bindings. Every
  change requires restart.
- speculative serving: all speculative request and Desktop routes are now
  fail-closed pending local accelerator qualification. `kiln config` and
  startup reject every effective non-off policy before model loading,
  including the legacy `enabled=true`, `method="off"` skip-layer fallback.
  Ordinary streaming, non-streaming, and batched generation remain on the
  settled single-token/batching paths, and production model loading uses
  `load_mtp=false`.
- speculative bounds and diagnostics: the draft-window default and hard ceiling
  are K=4 pending planned local K=1/2/4 qualification. `/v1/config` separates
  configured intent from the immutable serving policy (`off`, not routable),
  reports the backend MTP capability as a diagnostic only, and exposes no live
  speculative routing, admission, scope, or loaded-weight state.
- speculative and training containment: public high-level `ModelRunner`
  speculative methods reject before tokenization or allocation while the
  low-level research steps have no serving call site; the repository's direct
  caller is the isolated qualification bench.
  Server SFT normalizes omitted `train_mtp` to false, rejects explicit true
  before queue publication, and rechecks the invariant in the worker so a live
  server cannot lazily upload and train deferred MTP weights outside GPU
  coordination, memory admission, cancellation, and settlement.
- correctness: the time clock begins with the first decode candidate, excluding
  queue and prefill, and is checked at token boundaries. Timed requests bypass
  deterministic completion caches; token budgets participate in cache keys.
- observability: non-streaming choices, batch items, and final SSE chunks report
  the budget trigger, closure state, thinking-token count, and elapsed thinking
  time. Chat metadata also reports effective limits and request/default source;
  deterministic token-budget cache hits retain the original outcome.
- product and eval: the Playground, desktop settings, rollout generator, and
  `kiln-eval` CLI expose inherited, unlimited, zero, and custom limits. Eval
  runs preflight active budgets before decode and retain raw output, effective
  limit provenance, runtime outcomes, and a generation-configuration hash.
- streaming lifecycle: direct threaded prefill and decode are supervised by
  one remaining request deadline and a cooperative cancellation handle.
  Cancellation is observed between prefill tiles and at token boundaries, and
  the server retains request ownership
  until model cleanup explicitly settles; timeout/model failures end with a
  structured `generation_error` SSE event followed by `[DONE]`.
- streaming delivery: finish, usage, and `[DONE]` are committed out of band
  after cache/accounting work, so a full ordinary-delta queue cannot hold model
  cleanup or suppress the terminal events. All request modes use the ordinary
  decode route; MTP and skip-layer serving fail before model loading until they
  provide the same explicit settlement contract and pass local qualification.
- legacy paged streaming: the synchronous mutable-`BlockManager` compatibility
  API now holds pages, recurrent state, logits, and graph coordination through
  one backend-settlement epilogue on success, error, receiver drop, or panic.
  Its already-populated receiver semantics are documented and the API is
  deprecated in favor of live threaded streaming with explicit settlement.
- prompt logprobs: token-level display decoding now rejects tokenizer/model
  vocabulary drift and propagates decoder failures as a structured HTTP 500
  `tokenization_error` with the token ID. Known special tokens retain their
  literal vocabulary text; neither mock nor real responses can turn a failed
  display decode into an empty successful field.
- prompt-logprob integrity: real forward rows must exactly match the configured
  vocabulary and contain only finite values before ranking or JSON
  serialization. Selection has deterministic token-ID tie breaks and an exact
  requested top-K set. CUDA and ROCm now use a fused, max-subtracted log-softmax kernel
  that prevents softmax underflow from corrupting representable finite tails,
  with one same-dtype output allocation instead of low-precision
  softmax-then-log or full-vocabulary F32 temporaries.
- prompt-logprob compatibility: position zero is null and each later position
  uses the preceding logits row, always includes the observed prompt token,
  returns K or K+1 distinct IDs, and reports the full-vocabulary observed rank
  when it falls outside top K. K=0 is observed-only. Context-aware display
  decoding completes split UTF-8 tokens without sharing context between
  alternatives. Local scores now use direct mixed-input-to-F32 log-softmax;
  ranks and top-K come from original logits so F32 tail collapse cannot change
  them. Real inference reuses the runner-owned backend and inference recurrent
  policy, omits the unused final row, and projects bounded chunks under a 64
  MiB vocabulary-tensor target instead of materializing full-sequence logits.
  Exclusive GPU admission prevents concurrent scratch amplification, timed-out
  workers are cooperatively cancelled and drained, and dropped HTTP futures
  signal the scorer. Projection chunks and final state cross an explicit
  backend-settlement boundary before device owners or exclusive admission are
  released; failed or panicked settlement quarantines the backend and retains
  ownership. Prompts respect both the 4096-token endpoint cap and the served
  model context window, and responses are capped at 65,536 candidates. CPU,
  Vulkan fallback, and Metal log-softmax now preserve large common-offset
  accuracy by subtracting the row maximum before subtracting log-sum-exp.
  Vulkan keeps its host-fallback result on CPU rather than performing a
  redundant round trip. Scoring is base-model only until adapter revision
  identity can be pinned; mismatched model IDs and active LoRAs are rejected.
  The remaining full-vocabulary host readback is documented as a
  correctness-first O(TV) path pending a selected-only device kernel.

## kiln-v0.4.1 — 2026-06-12 — multi-turn prefix caching actually caches

Every pi / terminal turn was silently re-prefilling its entire conversation
history on ROCm — 40-55s per turn at 16-18K tokens on Strix Halo, with
`lookup_hits=0` across whole sessions while the cache faithfully stored
entries nothing could ever match. Multi-turn agent traffic now resumes from
the prefill-split prefix entry and only prefills the new suffix. (#1573)

- prefix-cache: ROCm re-enables the prefill-split linear-state snapshot —
  the only producer of the block-aligned entries that strict-prefix lookups
  can serve. `002af558` had set `allow_prefix_cache_split_snapshot: false`
  for ROCm while optimizing long-context prefill, which turned off
  multi-turn reuse wholesale; a new backend-capability contract test pins
  the flag `true` for every backend arm so this class of regression cannot
  land silently again.
- prefix-cache: the non-batched serve path (Metal's default) captures the
  split snapshot in its non-streaming and greedy prefill branches too;
  previously only streaming prefill captured it, so prompts under the
  streaming threshold never registered a multi-turn-reusable entry.
- prefix-cache: `register()` bails before evicting when an entry can never
  fit (`needed blocks > max_blocks`) instead of destroying every resident
  entry and still registering nothing.
- tests: two real-`ModelRunner` CPU integration tests prove the full loop —
  a batching-engine turn-2 strict-prefix HIT, and a non-batched-path replay
  from the split entry on a hybrid GDN model that asserts token-identical
  output (exact linear-attention state snapshot/restore). Live-validated on
  Strix Halo ROCm: turn-2/3 each hit with ~3.3K of ~3.3K prompt tokens
  cached, 1.85s vs 6.4s predicted full re-prefill.

## kiln-v0.4.0 — 2026-06-11 — embedded pi agent: the server drives its own rollouts

The flywheel closes by itself: kiln now spawns and drives pi as a managed
child process — submit a task over HTTP, watch the trajectory stream, and the
finished session lands in the agent-trace layer the self-improvement loop
trains on. Plus Muon becomes the default optimizer across every training mode.

- agent: **embedded pi runs** (`POST /v1/agent/runs`) — the server spawns
  `pi --mode rpc` against its own OpenAI-compatible endpoint (same
  non-destructive `pi-setup` config merge as the embedded terminal, now
  serialized + atomic + skip-if-unchanged), streams the full agent event
  trajectory, and auto-merges the finished session JSONL into
  `agent_traces.json` under a cross-process file lock. Runs queue FIFO
  (`[agent].max_concurrent_runs`, default 2; `run_timeout_secs`, default 900;
  32-active backstop enforced atomically), persist to
  `<adapter_dir>/agent_runs/runs.json` across restarts (in-flight runs come
  back `interrupted`), and write sessions to per-run directories so a run
  that dies before flushing can never adopt a sibling's session.
- agent: mid-run control — `POST /v1/agent/runs/{id}/steer`, `/follow_up`,
  and `/abort`. Messages sent while a run is still queued are buffered and
  delivered the moment it starts. The live feed is
  `GET /v1/agent/runs/{id}/events?after=<seq>` (inclusive cursor; pass the
  response's `next_after` back; `truncated`/`first_available_seq` flag replay
  gaps after ring-buffer prunes or restarts).
- agent: honest outcomes — a run whose final assistant turn ended in an error
  reports `failed` with the error text, never a hollow `completed`; sessions
  from aborted/timed-out runs still index (partial trajectories are data).
- agent: security posture matches the embedded terminal — enabled on loopback
  binds only, `KILN_AGENT_RUNS=1` opts in on network binds, `=0`
  force-disables — and the gate covers the read endpoints too, since run
  records and event feeds carry task prompts, server paths, and raw tool
  output. `KILN_PI_BIN` overrides pi discovery for non-PATH installs.
- dashboard: new **Distill → Agent runs** tab — launch form (task / cwd /
  label), live run list, and a drill modal with a 1s-polled event feed
  (assistant text, tool calls and results, errors), steer/follow-up/abort,
  and `#distill/runs/{id}` deep links on the shared modal conventions.
- traces: `POST /v1/agent/traces/discover` also sweeps
  `<adapter_dir>/agent_runs/sessions/`, so rebuilding the index from
  `~/.pi` never drops the rollouts kiln generated itself.
- training: **learning rates now resolve per optimizer**. `learning_rate` is
  optional in the SFT/GRPO/OPD configs; when omitted, the trainer picks the
  selected optimizer's band — Muon (the default): SFT 2e-2, GRPO/OPD 2e-3;
  AdamW/SGD keep the legacy defaults (SFT 1e-4, GRPO/OPD 1e-5). This fixes a
  silent foot-gun from the Muon flip: the old AdamW-era defaults trained Muon
  100–400x too cold. Explicit values still deserialize and are used verbatim
  (full wire back-compat), train receipts record the *resolved* value, and an
  explicit lr more than 50x outside the optimizer's band logs a warning at
  run start. The dashboard's SFT/GRPO/OPD learning-rate fields now default to
  blank ("auto (per optimizer)") and omit the field from the request when
  blank; `kiln train sft --lr` is likewise optional. The Muon GRPO/OPD band
  scales the legacy AdamW SFT:GRPO ratio and is an initial heuristic pending
  an empirical sweep.
- training: **Muon is now the default optimizer** for every training mode
  (SFT, GRPO, on-policy distillation / OPD, and the judge-LoRA flywheel that
  rides the OPD path). Muon is momentum-orthogonalized SGD — it keeps one
  per-parameter heavy-ball momentum buffer (vs AdamW's two moments), takes a
  Nesterov look-ahead, and projects the LoRA A/B weight-matrix updates onto
  the nearest semi-orthogonal matrix via a Newton-Schulz iteration before
  stepping, rescaling by `sqrt(max(rows, cols))` so the update magnitude is
  shape-independent. It converges LoRA fine-tunes in fewer steps than AdamW at
  roughly half the optimizer state.
- training: fused on-device Muon kernels for **every backend** — CUDA, ROCm,
  Vulkan, and Metal — implementing the whole step (heavy-ball momentum +
  Newton-Schulz orthogonalization + decoupled-weight-decay descent) in a
  single fused per-matrix launch. Newton-Schulz is computed in gram space via
  a `k×k` P-accumulator (`k` = LoRA rank), so the cost is dominated by two
  skinny GEMMs over the large matrix dimension. The portable CPU reference in
  `kiln-optim` is the parity oracle the GPU kernels are validated against.
- API: the optimizer is still selected per training request. Omit the
  `optimizer` field to get Muon; opt back into the old behaviour with
  `{"optimizer": {"kind": "adam_w"}}` or `{"optimizer": {"kind": "sgd"}}`.
  Muon accepts `momentum` (0.95), `nesterov` (true), `ns_iters` (5), and
  `weight_decay` (0.0). Note Muon wants a larger learning rate than AdamW
  (~2e-2 vs ~1e-4 for LoRA), since its update is orthogonalized and
  RMS-matched to unit scale.

## kiln-v0.3.5 — 2026-06-09 — task-first dashboard interactions + embedded pi terminal

UX release that rebuilds the dashboard's interactions around the pi-user's
actual tasks — every core flow now completes where it starts, with the UI
narrating what happens next — and embeds a real pi session in the browser.

- training: drop a `.jsonl`/`.json` straight onto the SFT/GRPO forms — parsed
  and validated in place (per-row skip reasons), adapter auto-named from the
  file, hyperparameters collapsed behind an honest "Advanced" summary, and an
  epochs-vs-dataset-size overfit nudge.
- training: train an uploaded dataset by NAME — `/v1/train/{sft,grpo}` accept
  a `dataset` field resolved from the dataset store server-side (GRPO streams
  via the existing `dataset_path` path), so rows never round-trip through the
  browser and nothing is truncated. Dataset rows gain "Train SFT/GRPO →"
  actions; uploads offer training as the next step; an empty queue lands on
  the training form.
- training: "Prove it after training" wires `post_eval` (suite +
  include_baseline) so train → eval-vs-base → verdict is one submission.
- evals: "Run eval…" on an adapter card now evals THAT adapter via a scoped
  compare-vs-base modal; suite Run/A/B buttons name the adapter they target
  and disable with directions when none is active.
- notifications: training and eval completions raise persistent action-toasts
  ("finished — Prove it vs base", "+N pts vs base — View result"); completed
  job cards keep a "Prove it vs base" button; failed cards show the error.
- corrections: one-click capture on every recent-request row, preview-only
  captures flagged, and the training receipt gains "Verify the fix" with a
  pending-selection handoff that picks the adapter in Playground compare the
  moment it finishes training.
- terminal: new "pi Terminal" page — the server spawns `pi` in a PTY
  (portable-pty) over a WebSocket after running the same non-destructive
  config merge as `kiln pi-setup` against its own URL, so the embedded pi is
  the user's pi, already connected. Vendored xterm.js compiled into the
  binary; enabled for loopback binds (KILN_TERMINAL=1 opts in elsewhere,
  KILN_TERMINAL=0 disables).
- cold start: a prominent "Can't reach Kiln" banner with retry; a first-run
  journey strip (server → agent → adapter) that offers "try pi right here";
  toasts moved top-right so they never cover the button they name.
- fixes: dataset-picker format filter matched a value the server never emits
  (`sft_chat` is the serde name); adapter swaps sync `/health` so the header
  and flywheel agree; failed swaps repaint; the demo's load/unload actually
  moves the active adapter.

## kiln-v0.3.4 — 2026-06-09 — agent-backend dashboard overhaul

UI release that bends the built-in server dashboard (served at `/ui`) toward the
pi/opencode use case and makes the live-training flywheel legible end to end:
watch agent traffic → spot what the model got wrong → correct it → train an
adapter → verify it beat base → hot-swap → repeat.

- dashboard: a Corrections basket captures a bad request, lets you write the
  ideal answer, and trains one SFT adapter in a click — only edited corrections
  train, so the model is never fine-tuned on its own mistake — with a persistent
  "training started" receipt instead of a silent page jump.
- dashboard: a "Needs attention" filter narrows Recent requests to the errored,
  truncated, and base-served rows, each with a one-click "Correct" capture button.
- dashboard: the request inspect modal gains a "Latency pi felt" time-to-first-
  token vs decode breakdown, prev/next triage navigation, and a Verify A/B action
  that re-runs the prompt against the serving adapter vs the active one.
- dashboard: the "did it win?" verdict ("beats base by +N pts") is surfaced
  consistently on the flywheel ribbon, the active adapter card, and eval job cards
  from one shared computation; the adapters list shows per-adapter eval scores;
  running, queued, and failed eval jobs show a state figure instead of a
  misleading 0 score.
- server: capture the per-request `User-Agent` so recent traffic is attributable
  to pi / opencode / curl / OpenAI clients.
- desktop + docs: replace hardcoded blue accents and cool greys with the warm
  design-system tokens (amber reserved for live/active state); no blue or
  cool-grey leakage remains across the dashboard, desktop app, or docs site.

## kiln-v0.3.3 — 2026-06-09 — paged KV row-scatter contiguity hardening

Patch release for the 0.3 line after live ROCm streaming prefill exposed a
non-contiguous paged-KV row-scatter write into the shared kt cache.

- paged-kv: make every multi-token row-scatter fallback materialize a
  zero-offset contiguous row before calling `Tensor::slice_set`, covering the
  token-major BF16, head-major native BF16, and FP8 write paths.
- paged-kv: add a CPU regression with a non-contiguous block table so the
  row-scatter fallback is exercised directly instead of silently taking the
  contiguous physical-slot run path.
- tests: extend the resource/concurrency invariant suite to require the shared
  row materialization helper at all paged-KV row-scatter call sites, so the
  `slice_set` contiguity contract cannot be edited away accidentally.

## kiln-v0.3.2 — 2026-06-08 — decode graph ownership and ROCm sampled batch hardening

Patch release for the 0.3 line after live ROCm streaming stress exposed a
batched hidden-decode failure in sampled generation.

- cuda/rocm: make single-row decode graph caches owner-keyed by batching row,
  so captured graph state and decode-contiguity timelines cannot be reused
  across concurrent request rows.
- rocm: add a native sampled hidden-decode batch path for multi-row batches,
  preventing `NativeRequired` fallback failures when sampled streams coalesce
  under the batching engine.
- rocm: turn graph replay failure into a runner-local circuit breaker that
  disables further graph replay/capture for that runner and continues through
  eager decode instead of repeatedly recapturing invalid driver state.
- rocm: restore linear graph capture state on hidden-graph capture failure and
  clear sticky HIP runtime errors at the wrapper boundary so stale driver errors
  cannot contaminate later kernel checks.
- tests: extend resource/concurrency invariants to lock in owner-keyed graph
  state, ROCm sampled native batching, replay circuit-breaker behavior, and HIP
  error-slot clearing.

## kiln-v0.3.1 — 2026-06-08 — structural GPU resource concurrency hardening

Patch release for the 0.3 line focused on making the ROCm startup crash class
and related cross-platform resource races structurally unavailable.

- resource-concurrency: add the `kiln-resource` crate for process-shared,
  lock-owned atomic file updates. Shared persistence now writes through one
  API that uses sibling lock files, process-unique temp files, fsync, atomic
  rename, and owner-liveness stale-lock recovery instead of uncoordinated
  direct writes.
- cuda/rocm: make cuBLASLt and hipBLASLt cache persistence merge under the
  process-shared lock, so concurrent writers cannot truncate, interleave, or
  lose each other's successful autotune entries.
- cuda/rocm: change BLASLt workspaces from handle-global buffers to stream-owned
  buffers keyed by the active typed stream. Matmul call sites now pass the
  stream owner through the API, removing the default-stream aliasing hazard from
  concurrent GPU work.
- rocm: remove ROCm hipBLASLt disk cache restore/flush from the server runtime
  and from the public tensor API. Legacy `~/.cache/kiln/autotune` files are no
  longer read during startup or prewarm.
- server: move teacher registry and agent trace index persistence onto locked
  atomic writes.
- tests: add invariant coverage that prevents ROCm disk cache call sites from
  being reintroduced, enforces stream-owned BLASLt workspace shape, and checks
  that shared persistence routes through `kiln-resource`.

## kiln-v0.3.0 — 2026-06-08 — backend runtime unification + ROCm release artifact

Major server release for the backend-engine unification line.

- backend-runtime: complete the A-migration from `BackendRuntime` as a
  monolithic dispatch surface to focused per-backend authority traits. The
  unification report now derives every "genuine" result from concrete signals
  instead of hardcoded coverage, and `scripts/check_unification_gates.sh` is the
  runner-agnostic gate for the report, contract test, grep guards, CPU
  conformance, and latency-fixture coverage.
- cuda/rocm/metal/vulkan: finish the residency, matmul/linear, replay,
  fallback, training, decomposition, and conformance gates so the release
  scoreboard is 9/9 genuine. The remaining compatibility shims were deleted
  rather than relabeled, and call sites now route through backend-owned traits.
- rocm: promote the HIP/ROCm backend into the server release matrix with a
  Linux x86_64 ROCm 7.2.4 tarball. The tagged artifact builds CDNA, RDNA3, and
  Strix Halo targets (`gfx90a`, `gfx942`, `gfx1100`, `gfx1151`) instead of the
  cheaper single-arch CI probe.
- release: harden the server release workflow by requiring the ROCm artifact
  before publishing the draft release and using the runner-provided Rust toolchain
  setup in release jobs, matching the macOS CI fix.

## Unreleased — perf-regression CI ladder for SFT/GRPO/OPD (#1077)

Three-tier coverage ladder protects step-time + workload-shape auto-tune
decisions from silent regressions across `cuda_train.rs`, `vk_train.rs`,
`trainer.rs`, `opd.rs`, `forward.rs`, the `BackendRuntime` trait, and the
`kiln_core::vram` heuristics:

| Tier | Trigger     | Coverage                                                                                                  | Cost           |
|------|-------------|----------------------------------------------------------------------------------------------------------|----------------|
| 1a   | per-PR      | Exhaustive `(GPU class × max_seq_len)` matrix tests in `kiln_core::vram::tests` + `CheckpointConfig::auto_for_workload` wrapper test in `crates/kiln-train/src/trainer.rs` | $0             |
| 1b   | per-PR      | `perf_regression_sft_train_cpu_smoke_completes_under_30s` — end-to-end CPU SFT smoke catches 50× regressions (#1063 class) | $0             |
| 1c   | per-PR      | `perf_regression_sft_train_emits_auto_tune_log_line` — tracing-capture confirms auto-tune wire stays connected            | $0             |
| 2    | nightly cron | `.github/workflows/perf-regression-nightly.yml` → self-hosted A6000 runs `kiln-bench --training-steps 5`, gates `secs_per_step ±10%` + `peak_vram_mb ±15%` against `bench-results/regression/sft_<trainer>_a6000_baseline.json` | ~$0.30/month   |
| 3    | manual `workflow_dispatch` | Same workflow with `ref`/`trainer`/`write_baseline_if_null` inputs | as-used        |

New artefacts:

- `crates/kiln-core/src/vram.rs` — three `perf_regression_*` matrix tests
  (Qwen3.5-4B + Llama-3-8B shapes + activation-tape sanity check).
- `crates/kiln-train/src/trainer.rs` — Tier 1b/1c/auto_for_workload wrapper
  tests at the end of `mod tests`.
- `.github/workflows/perf-regression-nightly.yml` — cron + workflow_dispatch
  with `gate-self-test` (free GHA) and `cuda-bench` (self-hosted A6000) jobs.
- `bench-results/check_sft_train_regression.py` — mirror of the existing
  `check_opd_regression.py`, gates the kiln-bench JSON output.
- `bench-results/regression/{sft_native,sft_generic}_a6000_baseline.json` —
  placeholder baselines (seed via `--write-baseline-if-null` on the first
  nightly run on a fresh workload row).
- `bench-results/regression/README.md` — schema + how to update a baseline
  after intentional perf changes.

The `gate-self-test` job exercises `check_sft_train_regression.py` on
synthetic stdout (success, +29% step-time regression, +25% VRAM
regression, and null-baseline seeding) so the gate's logic is verified
on every dispatch even when the self-hosted runner is offline.

`docs/skills/kiln/SKILL.md` (auto-synced) documents the per-PR vs nightly
split, the baseline-update workflow, and the cost model.

## Unreleased — vk-native GRPO+OPD non-recompute hybrid + workload-shape auto-tune (#1076)

Closes #1076. Wires the existing `vk_grpo_train_step_with_state` (the
non-recompute hybrid GRPO step kernel) into the `vk_native_grpo_train`
and `vk_native_grpo_train_jsonl` trainers, and extends
`vk_native_opd_train` to use `vk_opd_train_step_with_state` with a real
`VkLinearAttentionState` on hybrid GDN models (not just FullAttn-only).

Until this change, hybrid GDN GRPO and hybrid GDN OPD were *always*
forced to layerwise reverse-recompute regardless of available VRAM, so
users on A6000 / H100 paid an extra ~2× per-layer forward cost even
when they had plenty of memory to hold the full activation tape.

### Dispatch decision

A new helper `vk_recommended_recompute_for_grpo_opd` consults the
same `kiln_core::vram::recommended_checkpoint_plan` that the SFT side
uses (#1073 / #1074) and resolves to:

- `Plan::Disabled` → **non-recompute** (full activation tape — fastest).
- `Plan::Enabled` or `Plan::UserOverride` → **recompute** (memory-saving).
- `None` (VRAM unknown) → conservative **recompute**.

Per-mode env overrides:

| Override                       | Effect                                      |
|--------------------------------|---------------------------------------------|
| `KILN_VK_RECOMPUTE_GRPO=1`     | Pin recompute on GRPO (overrides everything)|
| `KILN_VK_RECOMPUTE_OPD=1`      | Pin recompute on OPD                        |
| `KILN_NO_GRAD_CHECKPOINT=1`    | Pin non-recompute (no per-mode override set)|

ECHO env-CE GRPO steps always force recompute — only that path has
the ECHO term wired in.

### Tests

Four new parity / smoke tests cover both work items:

- `vk_grpo_train_step_full_attn_loss_and_backward_parity_with_recompute`
  — same loss within 1e-3 + same LoRA.B within 1e-4 between
  non-recompute and recompute on FullAttn-only after one optimizer step.
- `vk_grpo_train_step_gdn_loss_parity_with_recompute` — same loss
  within 1e-3 on a hybrid GDN model.
- `vk_opd_train_step_gdn_loss_parity_with_recompute` — same loss
  within 1e-3 on a hybrid GDN model.
- `vk_opd_train_step_gdn_state_training_loss_decreases` — multi-step
  training trajectory on hybrid GDN, loss drops monotonically.

Plus a unit test for the new env-override dispatch helper:
`vk_recommended_recompute_grpo_opd_env_overrides_are_honored`.

## Unreleased — vk-native OPD training entrypoint (#1075)

Add `vk_native_opd_train`, the off-policy distillation (OPD) analogue
of the existing `vk_native_sft_train` and `vk_native_grpo_train`. The
trainer takes pre-masked `OpdPrompt`s + an `Arc<dyn LogitSource>`
teacher, fetches top-K logprobs at the action positions, and drives
either `vk_opd_train_step_with_state` (FullAttn-only) or
`vk_recompute_opd_train_step_with_state` (hybrid Qwen3.5-4B) per step
through Vulkan kernels and the existing AdamW state book. Adapter
artifacts (`adapter_model.safetensors`, `adapter_config.json`),
checkpoint cadence, and the training receipt all match the
candle-path `opd_train`'s contract.

The server's `run_opd` now picks between the candle path and the
new VK path via `KILN_VK_NATIVE_OPD` (falls back to
`KILN_VK_NATIVE_TRAINING` when unset, mirroring how
`KILN_VK_NATIVE_GRPO` was wired up). V1 envelope is intentionally
narrow: `training_mode = off_policy`, `objective = reverse_kl`,
`loss = teacher_top_k`, `top_k ∈ {16, 32}`, no `base_adapter`, no
ECHO env-CE — anything outside that envelope returns an explicit
"use the candle path" error so misconfiguration is loud instead of
silently wrong.

The hybrid-GDN fast path (no recompute, single-pass forward+backward)
is tracked as #1076 — the kernel for the OPD step exists but the
single-pass plumbing matches the GRPO follow-up. For now hybrid
models always go through layerwise reverse-recompute, which is the
same speed/memory trade as `vk_recompute_train_step_with_state`.

## Unreleased — cuda_native: route SFT *and* GRPO through BackendRuntime (closes #1063)

Closes #1063 (cuda_native_sft_train ~50x slower than `--trainer
generic`). This PR ships the **root-cause fix** from the issue body's
"Proposed fix" section 3: route the inner step through
`BackendRuntime` + candle's autograd so the cuda_native path inherits
the fused-kernel paths that make `sft_train` fast.

PR #1070 was a thin wrapper for the same idea — it flipped the
`--trainer` default to `generic` so users *bypass* the slow native
step, but the function itself (and direct callers of it) was
unchanged. This PR fixes the function itself, so every caller of
`cuda_native_sft_train` — `cuda_sft_file`, the server's
`KILN_CUDA_NATIVE_TRAINING=1` path, the bench harness, future
recipes that don't fit `sft_train`'s interface — gets the fast path
automatically.

### The structural fix

`cuda_native_sft_train` now delegates to `sft_train` by default:

```rust
if !force_legacy && !force_recompute {
    return crate::trainer::sft_train(
        examples, config, model_config, weights, tokenizer,
        adapter_dir, adapter_name, progress_cb, None,
    );
}
```

`sft_train` dispatches each layer through `BackendRuntime` (FlashAttn,
fused RMSNorm, fused GDN kernels) and runs candle's autograd. That's
the production-tuned path; it's what `--trainer generic` already used.

Two opt-in escape hatches preserve the legacy machinery for debug /
parity testing / memory-constrained jobs:

- `KILN_CUDA_LEGACY_NATIVE_STEP=1` — run the legacy monolithic-graph
  step (the original `cuda_native` behavior). Slow; use only when
  reproducing #1063 numbers or testing the new recompute path
  against the original.
- `KILN_CUDA_RECOMPUTE_SFT=1` — run the new layerwise reverse-
  recompute step (also in this PR, see below). Same speed as legacy
  but ~30% less peak VRAM. Useful for long-context jobs where VRAM
  matters more than speed.

The default — neither env var set — is the BackendRuntime route.

### Bonus: layerwise reverse-recompute step (opt-in)

This PR also includes a separate `cuda_recompute_train_step_with_state_masked`
that mirrors `vk_recompute_train_step_with_state_masked` from the
Vulkan path. It's the structural backport called out by the issue
body's "Proposed fix" sections 1+2. Bit-for-bit backward parity with
the legacy step (verified by `cuda_recompute_step_backward_parity_with_legacy`)
and ~30% less peak VRAM at the cost of double-forward per step.

Empirically (A6000 / Qwen3.5-4B / rank-8 LoRA, 4 short steps via
`cuda_sft_file`):

| Path                                | wall  | Peak VRAM |
|-------------------------------------|-------|-----------|
| `--trainer generic` (BackendRuntime)| 13 s  | 12.6 GiB  |
| native legacy step (slow path)      | 79 s  | 15.7 GiB  |
| native + recompute (this PR opt-in) | 74 s  | 10.7 GiB  |
| native default (this PR)            | 13 s  | 12.6 GiB  |

The recompute path is bit-for-bit equivalent to legacy on the
backward and saves real VRAM, but it doesn't close the 50× gap on its
own — the per-op CPU overhead in `cuda_train`'s hand-rolled autograd
scales with kernel-launch count, and recompute *adds* launches. Hence
recompute is opt-in for memory-constrained jobs; the BackendRuntime
route is the default for everyone else.

### GRPO: same fix on the parallel path

The server's `run_grpo` path previously rejected
`KILN_CUDA_NATIVE_TRAINING=1 + GRPO` outright with `"does not yet
support GRPO - unset it for GRPO jobs"`. That was because
`cuda_train` never had its own GRPO step kernel — only the SFT step
kernel. With the root-cause fix above, the symmetric thing for GRPO
is just to route the cuda_native GRPO request through `grpo_train`
(which already uses `BackendRuntime` + candle autograd, same as the
fixed `cuda_native_sft_train`).

This PR adds:

- `kiln_train::cuda_train::cuda_native_grpo_train` — thin wrapper that
  delegates to `trainer::grpo_train`. Matches `cuda_native_sft_train`'s
  shape so the server's cuda-native flag can be uniform across SFT
  and GRPO.
- `kiln_train::cuda_train::cuda_native_grpo_train_jsonl` — streaming
  variant, delegates to `trainer::grpo_train_jsonl`. For the server's
  `dataset_path` GRPO path.
- `kiln-server::training_queue::run_grpo` now calls
  `cuda_native_grpo_train{,_jsonl}` when `KILN_CUDA_NATIVE_TRAINING=1`
  instead of returning the legacy rejection error. The vk_native path
  is unchanged.

After this PR, `KILN_CUDA_NATIVE_TRAINING=1` is a uniform flag for
the server — both SFT and GRPO get the same `BackendRuntime` route,
both run at the production-tuned step time.

### Added

- `kiln_train::cuda_train::cuda_recompute_train_step_with_state_masked`
  — exact layerwise reverse-recompute step matching
  `vk_recompute_train_step_with_state_masked`. Forwards every layer
  once with *detached* LoRA (no autograd graph) caching layer-input
  boundaries plus per-GDN-layer entry states; wraps the final hidden
  as a fresh parameter for the RMSNorm+FLCE backward; then replays
  one layer at a time with the live LoRA so each `cuda_backward`
  walks a per-layer graph. Drops AdamW grads into a shared
  `TensorId`-keyed map and applies the optimizer once at the end via
  the existing `cuda_adamw_step_from_store`.
- `cuda_forward_to_layer_input` / `cuda_forward_layer_boundaries`
  helpers (private, in cuda_train.rs) that handle the detached-LoRA
  forward and the per-GDN-layer state snapshots.
- `cuda_gdn_lora_layer_with_entry_state` helper that wraps the
  reshape/transpose plumbing the legacy step kernel inlined, so the
  forward, boundary-cache, and reverse-replay paths agree byte-for-
  byte on the recurrence formulation.
- New env knobs:
  - `KILN_CUDA_LEGACY_NATIVE_STEP` (bool, default off) — set `=1` to
    bypass the BackendRuntime route and run the legacy monolithic-
    graph step. Slow; debug / parity testing only.
  - `KILN_CUDA_RECOMPUTE_SFT` (tristate, default off) — set `=1` to
    bypass the route and run the layerwise reverse-recompute step.
    Trades a few percent step time for ~30% peak VRAM.
  - `KILN_CUDA_RECOMPUTE_BOUNDARY_CACHE` (tristate, default auto) —
    cache layer-input boundaries in GPU memory; auto-engages when
    the cache fits within the per-host budget. Falls back to exact
    per-layer replay when memory is tight.
  - `KILN_CUDA_RECOMPUTE_BOUNDARY_CACHE_GB` (float) — explicit memory
    budget override for the boundary cache.
  - `KILN_PROFILE_CUDA_RECOMPUTE` (bool) — per-layer
    forward/reverse begin/end tracing, matching
    `KILN_PROFILE_VK_RECOMPUTE`.

### Changed

- `cuda_native_sft_train` now routes through `BackendRuntime` via
  `sft_train` by default. The legacy step is reachable via
  `KILN_CUDA_LEGACY_NATIVE_STEP=1`; the new recompute step is
  reachable via `KILN_CUDA_RECOMPUTE_SFT=1`.
- `cuda_native_sft_train` now supports `base_adapter` (inherited from
  `sft_train`) when on the default route. The previous "not
  supported" bail is preserved on the legacy / recompute opt-in
  paths because their inner step kernels don't load base adapters.
- Per-step `info!` lines on the opt-in paths now include `seq_len`
  and `step_ms` so the step path is diagnosable from a single log
  tail.

### Tests

- `cuda_recompute_step_loss_parity_with_legacy_step` — forward
  parity. One legacy step and one recompute step on identical LoRA
  init produce loss values within 1e-3 relative.
- `cuda_recompute_step_backward_parity_with_legacy` — backward
  parity. After one step, every LoRA A/B element produced by
  recompute matches the legacy step's output within 1e-5 absolute.
  This is the structural correctness guarantee for the recompute
  backward.
- `cuda_recompute_step_boundary_cache_vs_no_cache_parity` — flipping
  `KILN_CUDA_RECOMPUTE_BOUNDARY_CACHE` between on and off must not
  change the returned loss.

All three are gated on `feature = "cuda"` and skip with a message
when CUDA is unavailable.

## Unreleased — ECHO: agentic multi-turn GRPO loss term

Wires the ECHO technique (Shrivastava, Awadallah, Papailiopoulos — MSR
AI Frontiers, 2026) into kiln's GRPO training stack. ECHO adds a
length-normalized cross-entropy loss on environment-observation tokens
to the standard policy-gradient loss on action tokens — same forward
pass, different mask. Headline paper result: doubles TerminalBench-2.0
pass@1 at 8B (2.7% → 5.2%) and 14B (5.2% → 10.8%).

See `docs/papers/echo/` (paper + blog conversion) and
`docs/plans/echo-integration-plan.md` for the full design.

### Added
- Canonical trajectory schema in `kiln-train::trajectory`: `TurnKind`,
  `TurnSegment`, `ScoredRollout`, `AgenticGroup`. `ScoredCompletion`
  and `GrpoGroup` remain valid as `pub type` aliases for the new
  types; their existing field names (`text`, `reward`, `completions`,
  `messages`) are preserved so legacy callers compile unchanged.
- Masking primitive in `kiln-train::trajectory_mask`:
  `build_masks_from_trajectory(trajectory, prompt_messages, tokenizer,
  &MaskConfig) -> MaskedRollout`. Emits separate `action_mask`
  (policy-gradient targets), `env_mask` (ECHO env-CE targets), and
  per-segment token spans. Handles Qwen's `<tool_response>` wrapper
  for tool-role segments. Paper §3.2 warning-prefix exclusion via
  `MaskConfig::warning_filter` (on by default).
- `kiln-train::echo::echo_step_loss` — the env-CE loss term. Mirrors
  `opd_step_loss`'s signature so `LossConfig { policy, echo, opd }`
  composes structurally. Calls `kiln-flce-kernel`'s
  `fused_linear_cross_entropy_dispatch` under the hood, with the
  paper §3.1 `|O'|/|O|` rescale for length normalization.
- `LossConfig` on `GrpoConfig`: `echo: Some(EchoConfig::default())`
  is on by default at λ=0.05. `opd: None` reserved for OPD branch
  rebase. `no_policy_loss: bool` field implements paper §5.5
  verifier-free env-only adaptation. Set `loss.echo = None` (or
  `--no-echo` on CLI) to opt out.
- ECHO is wired into **all three** loss paths:
  - Uncheckpointed candle path (CUDA / CPU / Metal).
  - Checkpointed analytic-tail path via `EchoTailParams` threaded
    through `analytic_grpo_tail_loss_grad_pre_final_norm` — env-CE
    folded into the same vocab-chunk loop as GRPO with zero
    additional intermediates.
  - Vulkan-native path via `VkEchoStepParams` threaded through
    `vk_recompute_grpo_train_step_with_state`.
- `cuda_grpo_ablation` CLI flags: `--echo-lambda <f64>`, `--no-echo`,
  `--no-policy-loss` (verifier-free per paper §5.5), `--opd-lambda
  <f64>` (placeholder for OPD branch rebase).
- Env-var overrides: `KILN_ECHO_ENABLED`, `KILN_ECHO_LAMBDA`,
  `KILN_ECHO_ENV_MASK_MODE`, `KILN_ECHO_WARNING_FILTER`. Env > CLI
  precedence.
- HTTP route `POST /v1/train/agentic` — canonical alias of
  `/v1/train/grpo`, semantically matched to the multi-turn shape.
  `GrpoRequest::groups` accepts the `agentic_groups` JSON alias for
  the same reason. Legacy `/v1/train/grpo` + `groups` clients are
  unaffected.
- HTTP route `POST /v1/completions` for vLLM-compatible
  `prompt_logprobs` queries. `kiln-train::RemoteTeacher` can now use a
  kiln-served model as its OPD teacher via the same token-id prompt
  request shape it already uses for vLLM/sglang.
- Receipt schema additions in `kiln-train::receipt`:
  `DiagnosticSummary.echo: Option<EchoDiagnosticSummary>` with
  `lambda`, `env_ce_initial`, `env_ce_final`, `env_ce_drop_pct`,
  `lambda_effective_final`, `env_tokens_supervised`, plus the
  paper §5.2 dynamics-test fields `dynamics_holdout_ce_initial` and
  `dynamics_holdout_ce_final` (populated by
  `pi-terminal-bench-lite/calibration/dynamics_holdout.py`).
- Shared Python lib `capabilities/agentic-grpo/lib/pi_trajectory.py`
  that converts pi session JSONL into the canonical trajectory schema
  (with warning-prefix detection).
- `pi-doctest/rollout.py` migrated to emit the new trajectory shape
  via `pi_trajectory.build_scored_rollout`.
- `capabilities/agentic-grpo/pi-terminal-bench-lite/` — Phase 2 cap
  for paper-reproduction; ECHO defaults pre-wired, dynamics-holdout
  calibration script, `ECHO_MODE=on|off` paired-run recipe.
- `capabilities/agentic-grpo/pi-script-fixup/` — Phase 3 cap for
  paper §5.5 verifier-free env-only adaptation; `--no-policy-loss`
  CLI flag, run_verifier_free.sh recipe.
- `docs/ECHO_GUIDE.md` — operational guide (CLI, HTTP, env vars,
  diagnostics, capability-author checklist, OPD composition story).
- `docs/plans/grand-plan-for-extraordinarily-great-echo-for-everyone.md`
  — long-form user-facing companion to the integration plan.
- `kiln-polish-prerequisites.md` §1 — multi-turn assistant-token
  masking — now ✅ RESOLVED. The gap that blocked agentic-GRPO
  multi-turn training since pi-doctest v0 is closed.

### Tests
- ~50 ECHO-related Rust tests across kernel, masking, serde wire
  format, `LossConfig`, env-var overrides, end-to-end trainer paths,
  and Appendix C acceptance gates of the integration plan.
- Appendix C.1 #1: `lambda=0.0` is bit-equivalent to `echo=None`.
- Appendix C.1 #3: paper §3.1 normalization `mean_ce ∝ |O'|/|O|`.
- Appendix C.1 #4: checkpointed analytic-tail ECHO loss matches
  uncheckpointed within 1e-3 on a tiny-model fixture.
- Phase 3 e2e: `no_policy_loss=true` upholds the linearity invariant
  `loss_full ≈ loss_grpo_only + loss_vf` (the GRPO term genuinely
  zeroed when verifier-free).
- 16 legacy GRPO/SFT tests still pass; 7 kiln-server training-API
  tests still pass; 15 Python `pi_trajectory.py` unit tests pass.

### Behaviour for legacy single-turn rollouts
- When a rollout has no `trajectory` field (only the legacy `text`
  string), behaviour is bit-identical to the pre-ECHO loss math:
  `env_mask` is all-false, `total_obs_len` is 0, the ECHO branch
  short-circuits at zero cost. The default `loss.echo =
  Some(EchoConfig)` is safe for legacy callers.

### Deferred (Phase 4)
- OPD composition: the `LossConfig.opd: Option<OpdAuxConfig>` slot is
  reserved with `None` default. When the OPD branch rebases on top of
  this work the composition is mechanical — the analytic tail already
  handles heterogeneous active-position kinds (`PosKind::Action` and
  `PosKind::Env`); OPD slots in as a third arm. `--opd-lambda` on the
  CLI parses but warns-and-ignores until that lands.

## kiln-v0.2.16 — 2026-05-15 — Phase 11 evals as a first-class peer of training

Adds an end-to-end eval system: pluggable scorers, dataset → suite synthesis, post-training auto-eval, head-to-head adapter comparison, and the local A/B judgment flywheel. The whole loop runs on the same single-GPU binary; no frontier-LLM call is ever required.

### Added
- `kiln-eval` crate (pure CPU): `EvalSuite`, `EvalExample`, `EvalGenerationParams`,
  and a serde-tagged `Scorer` enum with kind labels `exact_match` / `contains` /
  `regex` / `json_validity` / `multiple_choice` / `numeric_tolerance` /
  `tool_call` / `code` (with `bash` introspection for inlined `python -c` /
  `node -e` / `uv run`) / `llm_judge` / `all` / `any` composites, plus dataset →
  suite synthesis (`final_assistant`, `first_assistant_turn`,
  `every_assistant_turn`, `tool_call_predict`) with reservoir-sampled
  `max_examples` and an `auto_detect` scorer choice that picks per example from
  the target shape.
- Server eval module (`crates/kiln-server/src/eval/`): `SuiteRegistry` on disk,
  `DatasetRegistry` with manifest + sampled stats, `JudgmentStore` + judgment
  compile + validate, `EvalQueue` + `spawn_eval_worker` (FIFO, terminal-job GC,
  concurrent with training and serving), `generator_from_state`,
  `synthesis_driver` (preview + commit), and re-run-only-failures.
- HTTP surface: `GET/POST /v1/eval/suites`, `GET/DELETE /v1/eval/suites/{name}`,
  `POST /v1/eval/run`, `POST /v1/eval/compare`, `GET /v1/eval/jobs`,
  `GET /v1/eval/jobs/{id}`, `POST /v1/eval/jobs/{id}/rerun`,
  `POST /v1/eval/datasets/upload`, `POST /v1/eval/datasets/{name}/synthesize`,
  plus `GET/POST /v1/judgments`, `POST /v1/judgments/{name}/rows`,
  `POST /v1/judgments/{name}/compile`, `POST /v1/judgments/{name}/validate`.
- Post-training auto-eval: `SftRequest`/`GrpoRequest` accept
  `post_eval: { suite, include_baseline, generation }`. `enqueue_post_training_eval`
  pushes one (or two, with baseline) eval jobs the moment training succeeds;
  IDs are linked on `TrainingJobInfo.linked_eval_job_ids` at queue time.
- `kiln-eval` CLI binary (`crates/kiln-server/src/bin/kiln_eval_cli.rs`):
  list/run/compare suites and trigger synthesis from the shell.
- `/ui` overhaul (Phase 11): Cmd-K command palette, drill-in modals for eval /
  training / adapter detail with live progress + content-key change-detection,
  loss curves with downsampled history, A/B compare playground with keyboard
  shortcuts that writes directly to `JudgmentStore`, VRAM donut, tok/s sparkline.
- `[eval]` config section (`root`, `max_tracked_eval_jobs`, default
  `generation` block).
- Documentation: `docs/EVAL_GUIDE.md` (full scorer reference + request shapes),
  `docs/site/evals.html`, README "The Eval Loop" section, QUICKSTART §10
  walkthrough, ARCHITECTURE "Evaluation Pipeline" deep-dive.

### Notes
- Single bundled commit — this is the first shipping cut of the eval system,
  not a series of incremental PRs.
- The eval worker rides the GPU read-lock, so evals run concurrently with both
  serving and training. The training queue keeps its exclusive write-lock model.
- Suites, datasets, and judgments persist under `<eval-root>/{suites,datasets,judgments}/`
  (defaults to `<state_dir>/eval`).

## kiln-v0.2.16 — 2026-05-15 — Phase 2 Vulkan training hardening

Active branch since v0.2.15. Targets the Strix Halo / unified-memory APU
training path that previously OOM-killed `kiln serve` and (after the first
Vulkan training-route changes) twice hard-hung the host on the
`/tmp/sft-data.jsonl` SFT repro.

### Added
- training: lazy candle-CPU-storage sync. Eliminates the per-step
  `var.set` GPU→CPU readback that previously kept candle storage of
  every LoRA Var (and its `(m, v)` moment Vars) in lock-step with
  the registry buffer.
  - `apply_sgd_update` and `apply_adamw_update` no longer call
    `var.set` on the on-device success path — the registry buffer
    becomes the canonical source of truth between training steps.
  - `VulkanLoraOp::bwd` now reads `A` and `B` values directly from
    the registry buffers (`buffer_to_tensor` helper in
    `kiln-vulkan-kernel`) instead of candle CPU storage. The
    backward math itself still runs on candle CPU (rank≤64 matmuls
    are negligible vs. the readback sync cost), but candle storage
    of LoRA Vars no longer needs to be current during backward.
  - `TrainableLoraParams::sync_to_candle` and
    `OptimizerState::sync_to_candle` pull registry values back into
    candle Var storage on demand. Trainer calls
    `params.sync_to_candle(&*backend)` right before `save_peft`
    (final + checkpoint intervals) so the serialized adapter
    reflects the post-step state.
  - Net effect on a Qwen3.5-4B SFT run with rank=8 (~210 LoRA
    Vars, ~9 MB total): removes ~27 MB GPU→CPU readback per step
    (param + m + v) plus ~210 candle CPU broadcast_matmul reads of
    `to_dtype(F32)` per backward pass. Candle CPU storage of LoRA
    Vars is touched only at save_peft time.
  - Tests: `lazy_sync_keeps_candle_stale_until_explicit_sync` in
    the Vulkan backend (asserts the 3-step contract: post-dispatch
    candle is stale, registry is current, post-sync candle matches),
    plus `sync_to_candle_is_noop_on_cpu_backend` in the trainer for
    the fallback branch. The existing `vulkan_lora_op_backward_parity_small`
    test continues to pass with the registry-driven bwd, confirming
    the math is unchanged.
- training: on-device AdamW (decoupled weight decay) via Vulkan.
  - New shaders `adamw_step_f32.comp` and `adamw_step_bf16.comp`
    update param + first moment + second moment in-place in a single
    dispatch. Bias-correction terms are precomputed host-side and
    shipped via push constants; BF16 variant uses bf16↔f32 lane
    bit-expansion (no `VK_KHR_shader_bfloat16` requirement) and
    runs internal moment math in f32 to avoid underflow on `v`.
  - `dispatch_adamw_step_{f32,bf16}` kernel helpers + Vulkan backend
    `dispatch_adamw_step` impl that resolves param/grad/m/v from
    `RESIDENT_ACTIVATION_REGISTRY` by `TensorId` and dispatches the
    dtype-appropriate kernel; falls back when any operand isn't
    resident (mirrors `dispatch_sgd_step`).
  - `TrainableLoraParams::allocate_adamw_state` allocates a zero-init
    `(m, v)` Var pair per LoRA Var; `OptimizerState::{register,evict}_with_backend`
    register the moment buffers in the residency registry. Both the
    SFT and GRPO loops own the optimizer state for the lifetime of
    the training run.
  - `Optimizer` enum on `SftConfig` + `GrpoConfig`
    (`Sgd` / `AdamW { beta1, beta2, eps, weight_decay }`); default
    is **AdamW** per LoRA fine-tuning best practice. `SftConfig`
    deserializes as `{ "optimizer": { "kind": "adamw" } }` (the
    default if omitted) or `{ "optimizer": { "kind": "sgd" } }`.
  - Trainer routes via new `optimizer_step` / `optimizer_step_from_map`
    that dispatch to `apply_sgd_update` or `apply_adamw_update`. The
    AdamW path prefers on-device dispatch when the backend supports
    residency; CPU fallback runs the same math via candle `affine` ops
    in f32 then quantizes back to the param dtype. Step counter is
    1-indexed at the kernel level and incremented inside the
    optimizer routine before iterating Vars.
  - Tests: `dispatch_adamw_step_resident_round_trip_f32` (1e-6
    tolerance vs. scalar reference), `dispatch_adamw_step_resident_round_trip_bf16_two_step`
    (two-step trajectory to catch bias-correction precompute bugs),
    `dispatch_adamw_step_falls_back_when_not_resident`,
    `adamw_cpu_fallback_updates_params_and_moments`, and
    `adamw_first_step_matches_unbiased_reference` (sign + bias-correction
    sanity at step=1, where `update ≈ lr * sign(g)`).
- vulkan: in-op chunking in `VulkanLinearOp` — oversized BF16-packed
  matmuls split along the output dim (forward) or batch dim (backward)
  via the existing offset/transposed kernels. Each per-chunk submit
  calls `queue_wait_idle()` for compositor preemption. Per-chunk FLOP
  ceiling tunable via `KILN_VULKAN_LINEAR_MAX_GFLOP` (default 20 GFLOP,
  ≈800 ms per submit at 25 TFLOPS — comparable to FLCE's empirically
  safe per-chunk cadence).
- vulkan: `linear_prefill_apply_offset` sub-chunks internally when a
  single chunk_len would exceed the FLOP ceiling, instead of bailing
  to FLCE's CPU fallback. Same kernel, same buffer, finer-grained
  submits.
- vulkan: new `sdpa_prefill_f32.comp` shader replaces the buggy
  `flash_attn.comp` placeholder. F32 in/out, online-softmax, optional
  causal mask, head_dim ≤ 128. Wired into `flash_attn_prefill_vulkan`
  behind opt-in env var `KILN_VULKAN_SDPA=1`.
- vulkan: Phase 4.2 — `sgd_step_f32.comp` shader,
  `dispatch_sgd_step_f32` helper, and `BackendRuntime::dispatch_sgd_step`
  trait method. Vulkan impl looks up both operands in
  `RESIDENT_ACTIVATION_REGISTRY` and falls back to the candle CPU
  path if either is not resident. Drops in for the trainer once
  Phase 4.1 lands and `TrainableLoraParams` are registry-resident.
- vulkan: Phase 4.1 partial (LoRA Vars on device, forward inference
  path).
  - `TrainableLoraParams::register_with_backend` uploads each LoRA
    Var to the resident activation registry at init time. Production
    callers (sft_train, grpo_train) wire it.
  - `BackendRuntime::lora_delta_resident` hook computes
    `(x @ A.T @ B.T) * scale` against registry-resident A and B
    via two dispatches of the existing transposed bf16 kernel.
    Plumbed into `add_lora_delta_to_base` in forward.rs.
  - **Gated to inference-only**: the on-device dispatch returns a
    candle-leaf Tensor with no autograd back-link, so it's only
    invoked when neither x, A, nor B is autograd-tracked. Training
    still uses the candle CPU `compute_lora_delta` path. A future
    CustomOp3 wrapper (with analytic LoRA backward) will make the
    on-device path training-safe.
  - Registry encoding bifurcated by dtype:
    BF16 tensors store packed BF16 bytes (2 bytes/elem) so every
    `load_weight(idx) = data_w[idx >> 1]` kernel reads them
    correctly; F32 tensors store F32 bytes (preserves the original
    boundary-state resolve path). `resolve_resident_activation`
    knows about both encodings.
  - `BackendRuntime::update_resident_activation` keeps the registry
    buffer in sync with candle CPU `var.set(...)` after the SGD
    step. Wired into `sgd_step` and `sgd_step_from_map`. Without
    this, `lora_delta_resident` would forever read the init bytes.
- training: Phase 3.2 partial drop also covers the tiled checkpoint
  paths — boundary_states[seg_idx]'s candle CPU mirror is replaced
  with a 1-element BF16 stub right after the tiled subroutine
  returns (it took `&boundary_states` so couldn't mutate; the
  outer loop does the drop). Frees ~9 MB per boundary on Vulkan
  in tiled mode too.
- inference: Phase 4.1 step 4 — `Model::load_adapter` now registers
  the adapter's A/B tensors in the resident activation registry,
  and `Model::unload_adapter` (and the load-replace path) evicts
  the previous adapter. New `LoraWeights::register_with_backend`
  mirrors the trainer-side TrainableLoraParams API. Effect: an
  inference request that uses an adapter now dispatches the LoRA
  delta on the GPU (via `lora_delta_resident`) instead of going
  through candle CPU `compute_lora_delta`.
- core: tests — `kiln_core::env_flag::TEST_ENV_LOCK` is now `pub`
  (cfg(test)) so sibling test modules (vram, etc.) can hold the
  same mutex when mutating env vars. Deflakes
  `test_unified_memory_apu_corrects_oversized_carveout` which
  used to race with `test_unified_memory_reserve_env_override`.
- vulkan: `BackendRuntime::resolve_resident_activation` read-back
  hook + Vulkan impl using `VulkanBuffer::read_back` +
  `kernels::create_tensor_from_data`. API surface for Phase 3.2's
  actual memory-saving wiring (drop the candle CPU mirror after
  registering, re-materialise from the device buffer when the
  recompute pass needs it).
- core: `kiln_core::env_flag::env_flag(name, default) -> bool` and
  `env_tristate(name) -> Option<bool>` helpers. Centralised the
  truthy/falsy spellings (`1`/`true`/`yes` and `0`/`false`/`no`,
  case-insensitive, whitespace-trimmed) so every `KILN_*` flag
  parses the same way and unrecognised values fall back to the
  default rather than silently flipping behaviour. Refactored
  4 call sites (KILN_VULKAN_LINEAR, KILN_VULKAN_SDPA, KILN_W4A16,
  KILN_VULKAN_FLCE, KILN_VULKAN_RMSNORM_TRAINING).
- vulkan: `register_resident_activation` bails silently on a
  zero-byte tensor (some drivers reject `vkCreateBuffer(size=0)`);
  caller falls through to its CPU path.
- telemetry: one-shot trace for first offset sub-chunked dispatch
  (`linear_prefill_apply_offset` engages internal sub-chunking when
  a FLCE chunk_len exceeds the FLOP ceiling).
- training: Phase 3.2 partial — both `checkpointed_forward_backward`
  and `checkpointed_grpo_forward_backward` now prefer
  `resolve_resident_activation` over `clone()` for the per-segment
  recompute boundary input. Doesn't free the candle CPU mirror yet
  (pending Phase 4.x storage interception) but exercises the
  resolve path inside real training; any registry bytes-divergence
  surfaces as a loss difference vs the non-resident path.
- server: `resident_activation` and `sgd_step_on_device` fields
  added to the "Vulkan training acceleration profile" startup log.
- vulkan: Phase 3.1 hooks — `supports_resident_activation`,
  `register_resident_activation`, `evict_resident_activation`,
  `has_resident_activation` on `BackendRuntime` plus a
  `RESIDENT_ACTIVATION_REGISTRY` impl on `VulkanBackend`. Trainer
  call sites in `checkpointed_forward_backward` and
  `checkpointed_grpo_forward_backward` invoke the lifecycle hooks
  at boundary state push/eviction; gated on
  `supports_resident_activation()` so non-Vulkan backends pay zero
  overhead.
- server: "Vulkan training acceleration profile" startup log surfacing
  the on/off state of every `KILN_VULKAN_*` training-path env var.
- server: HTTP 413 rejection messages from `/v1/training/*` include
  a `vram_source=...` clause when the corrected budget came from the
  unified-memory path (makes `KILN_TRAINING_MEMORY_RESERVE_GB` the
  obvious knob).
- core: `EffectiveBudget` accessor + `vram_source` field on the "GPU
  memory budget" log so the corrected unified-memory budget is honest
  about its provenance (`linux-drm-sysfs-unified` vs the raw DRM
  carveout it overrides).
- telemetry: one-shot tracing for first chunked
  `VulkanLinearOp::cpu_fwd` and `::bwd` dispatch (logs chunk count,
  per-chunk GFLOP) and first
  `VulkanBackend::register_resident_activation` call.
- docs: Phase 2.1 CPU-fallback matmul leak audit, Phase 2 hardware
  validation runbook, and canonical env-var reference in
  `docs/audits/`.
- docs: env-var documentation for `KILN_GPU_MEMORY_GB`,
  `KILN_TRAINING_MEMORY_RESERVE_GB`, and the unified-memory detection
  heuristic in `kiln.example.toml`.

### Changed
- training: FLCE provider auto-engages at `active_count ≥ 16` (was
  `active_count × num_chunks ≥ 50_000`). The 50K threshold was tuned
  against the pre-Phase-2 baseline; post-host-crash the comparison is
  FLCE-chunked-Vulkan vs the unfused lm_head dispatch that hung the
  host. Any non-trivial supervised batch now routes through the
  chunked FLCE path.
- vulkan: **`KILN_VULKAN_LINEAR` defaulted ON** (after the chunking +
  CustomOp3 LoRA + BF16 SGD wiring made the full training stack
  correct + safe by construction). Set to `0` to opt out for parity
  comparisons.
- vulkan: **`KILN_VULKAN_SDPA` defaulted ON** (after the
  sdpa_prefill_f32 kernel was parity-tested at 4 shapes including
  Qwen3.5-4B head_dim=128). Set to `0` to opt out.
- vulkan: `BackendRuntime::dispatch_sgd_step` now handles BF16
  param + grad pairs via the new `dispatch_sgd_step_bf16` kernel
  (one thread per u32 word, bf16↔f32 lane bit-expansion). Mixed
  dtypes still fall back to the candle CPU path.
- training: `sgd_step` and `sgd_step_from_map` now prefer the
  on-device Vulkan SGD path when the Var is registry-resident.
  Per-step flow: register grad → dispatch_sgd_step (writes new
  bytes to param buffer in-place) → resolve buffer → var.set to
  keep candle storage in sync for save_peft and other readers
  → evict grad. Falls back to candle CPU SGD when not resident
  or when the dispatch declines.
- vulkan: `lora_delta_resident` Vulkan impl now wraps the
  on-device dispatch in `VulkanLoraOp` (CustomOp3) with analytic
  backward returning gradients for x, A, and B. The
  inference-only gate in `add_lora_delta_to_base` was removed —
  training-time forward now uses the on-device LoRA delta path.

### Fixed (continued)
- vulkan: dispatch helpers now query the actual device per-axis
  workgroup-count limit (`maxComputeWorkGroupCount[i]`, typically
  ≈ 2^31 - 1 on AMD/Strix Halo) instead of the conservative Vulkan
  spec minimum (65535). Without this, the LoRA second matmul
  `hidden @ B.T` for Qwen3.5-4B's gate_proj/up_proj at T=918
  needs ~294K workgroups on axis 0, which the conservative guard
  would falsely reject.
- training: `apply_sgd_update` propagates `dispatch_sgd_step`
  errors instead of swallowing them via `.ok().unwrap_or(false)`.
  Shape-mismatch and similar bugs surface immediately rather than
  silently degrading to CPU SGD with potentially-wrong results.

### Trait additions
- `BackendRuntime::dispatch_adamw_step` is now fully implemented in
  the Vulkan backend (was a forward-looking stub before this entry).
  See the AdamW bullet under "Added" for kernel + trainer wiring.

### Fixed
- vulkan: lm_head training-time forward queuing ~4.36M workgroups in
  a single submit on a 40-CU APU (the root cause of two host
  hard-hangs on `/tmp/sft-data.jsonl`). Mitigation lands as a
  combination of the FLCE auto-engagement (so the lm_head matmul
  goes through chunked FLCE for the SFT loss path) and the in-op
  chunking (so any oversized matmul that does reach `VulkanLinearOp`
  is split into TDR-safe per-submit pieces).
- transposed weight cache writer: replace blocking
  `thread::sleep(initial_delay)` with `recv_timeout`-based wait so
  a queued message wakes the writer immediately even mid-deferral.
  Fixes a flaky-under-`cargo test`, fine-under-`cargo nextest`
  unit test (`queue_cache_write_persists_payload_on_background_writer`)
  whose 2-second deadline timed out when an earlier test had
  initialized the OnceLock'd writer with the production-default
  120 s initial delay.

## kiln-v0.2.15 — 2026-05-10

A short follow-up cut covering 15 commits since v0.2.14, focused on CUDA
decode-path fusion and a macOS Metal build fix. No desktop source changes;
the aligned desktop release simply re-pins to the new server payload.

### Changed
- cuda: combine full-attention QKV projections into a single GEMM at decode
  time, mirroring the GDN AB combine in v0.2.14.
- cuda: combine GDN AB projections at prefill (extending the decode-time
  fusion) and use the combined projection on short prompts.
- cuda: enable GDN prefill gates by default and route mixed GDN batches
  through the row loop, with the row-loop forced path generalized for batched
  GDN.
- cuda: fast-path paged KV cache writes and route CUDA decode through paged
  attention end-to-end.
- cuda: fuse decode QKV prep, fuse GDN decode RMSNorm, and fuse the LoRA
  decode add (new `kiln-rmsnorm-kernel/csrc/fused_lora_add.cu`) so single-token
  decode dispatches fewer kernels per layer.
- cuda: use empty CUDA outputs for overwrite kernels and align the default
  ModelRunner CUDA graph behavior.
- training: default CUDA projection drop on for the training fit path so
  training memory residency tracks the inference path.

### Fixed
- macos: pass `backend=None` to `add_lora_delta_to_base` from the
  metal-feature-gated decode path so `cargo build --features metal` compiles
  again. The CUDA LoRA decode add fuse landed in this cut added a `backend`
  parameter and missed the Metal call site (#1019).

## kiln-v0.2.14 — 2026-05-10

This is a large release covering ~325 commits since v0.2.13: a new Vulkan
backend, an embedded server-UI overhaul, replayable LoRA storage, the Phase 12
batched-decode hot path, and continued CUDA / Metal kernel fusion.

### Added
- vulkan: bring up a Vulkan inference backend covering GDN (recurrent, conv1d,
  in-projection, gated norm, QK norm), MLP (gate/up/down), batched gemv, and
  paged decode. Ships with f32 + bf16 paths, rowpair / rowquad batched gemv
  variants, batched chunk transfers, fused-submit conv1d prefill (now on by
  default), and chained MLP dispatches. BF16 GDN state and device-local
  recurrent batch state cut readback overhead and keep state resident on-device.
- training: deterministic replayable LoRA storage with parent lineage so a
  trained adapter records the exact replay set + parent chain it descended from.
  Storage is content-addressed and reproducible across machines (#1015).
- server UI: total visual overhaul of the embedded server dashboard — clearer
  status, request feed, training, and adapter panels; live regions for
  accessibility; mobile-friendly forms; sample training payloads; CLI/REST
  cross-links from every panel (#1009, #1013).
- windows: Windows CUDA release zips now bundle the CUDA 12.4 redist DLLs
  (cudart, cublas, cublasLt, cuRAND, nvrtc) so users without the CUDA Toolkit
  installed no longer hit `cublas64_12.dll was not found` on launch (#726).

### Performance
- phase 12 batched paged decode: per-stream graph replay, per-request
  `KvWriteSlot` to eliminate the prefill mutex contention that bottlenecked
  multi-stream decode, FA-2 varlen paged decode for batched (c>1) decode, and
  batched paged decode API surface (#762, #968, #995, #996, #998, #1000, #1002,
  #1008).
- cuda fusion: fold GDN QK-norm into the recurrent kernel, fuse RoPE with QK,
  fuse MLP SiLU and multiply, fuse attention-gate sigmoid+multiply, BF16 GDN
  state, inference-only RMSNorm fast path, projection-original memory
  compaction, rate-limited graph cache-full warnings.
- metal: full SDPA prefill, GDN in-proj rowpair / rowquad / serial-x2 fast
  paths, MLP gate/up rowquad gemv, batched gemv rowpair / rowquad / row-triple
  variants for batch sizes ≥3 / ≥4 / ≥5, LoRA delta paths covering rank-4 /
  rank-8 / serial-decode and all positive ranks, plus high-rank LoRA bench
  coverage and qkv LoRA-linear bench coverage.
- vulkan perf: enable GDN in-proj rowpair-batch3, GDN rowpair-batch3 chunk
  transfers, fused-submit conv1d prefill (now default), chained MLP dispatches,
  device-local recurrent batch state, BF16 inference state, skip the final GDN
  state readback. f32 GDN recurrent step routing, bf16 hybrid MLP gate/up/down,
  and gdn-recurrent kernel-stage profiling support.
- scheduler / decode batcher: reduce batched linear-state scatter copies, route
  through unexpanded GDN decode QK on both CUDA and Vulkan.

### Fixed
- compatibility: guard GDN fast paths from autograd tensors so training and
  inference share the same kernel surface without aliasing autograd graph
  state.
- compatibility: disable CUDA graphs by default on WSL CUDA and guard the CUDA
  decode batching default (and revert PR #1008's LM-head guard regression).
- ui smoke: `docs/site` + server UI v1 overhaul follow-up — fix CI smoke checks
  that broke against the new dashboard markup (#1013).
- adapters: fix CLI adapter API routes (#774); fix adapter API docs examples
  (#947); align training submission payload shapes (#739).
- quickstart / cli: friendly error when the kiln server is unreachable (#972);
  fix README quickstart timeout anchor (#896); fix quickstart training-status
  command (#765); fix GRPO CLI file help (#738); device init now uses `?`
  instead of a missing `anyhow::Context` import (#734).

### Documentation
- onboarding: cold-reader pass through README, QUICKSTART, and CLI help —
  surface the kiln health probe before the chat curl, surface model-weights
  download before the Docker block, deduplicate Desktop App install tables
  between README and QUICKSTART, surface BENCHMARKS.md from README nav,
  cross-link architecture and GRPO guides, drop legacy prefill notes, and
  collapse the QUICKSTART reader map / prerequisites / KV-cache OOM section
  (#964–#989, plus #969–#977).
- ui: explain decode and scheduler metrics for cold readers via tooltips
  (#971); persistent UI header help links (#798); accessible training tabs
  (#782) and live regions for dashboard panels (#783).
- governance: scrub stale publicity changelog entries and remove launch
  announcement surfaces (#718, #719).

## kiln-v0.2.13 — 2026-05-02

### Observability
- throughput: explain the round-10 workers=2 evidence as request mix and concurrency effects rather than a scheduler regression, and add request, prefill, decode histograms plus an active-request peak metric so future release-gate runs can separate queue shape from decode-path regressions (#712).

### Documentation
- docs: scrub changelog references to removed Phase 11 docs artifacts after the tracked files and page were removed (#717, #718).

## kiln-v0.2.12 — 2026-05-02

### Fixed
- reliability: complete the production-shaped prefill OOM hardening series from #694/#697/#699. KV auto-sizing now uses post-load CUDA residency instead of only the static model estimate, CUDA streaming prefill starts at 8k tokens for production-shaped prompts below the old 32k threshold, and new VRAM metrics expose the static estimate, post-load snapshot, and observed prefill peak (#701).

## kiln-v0.2.11 — 2026-05-02

### Fixed
- memory: bound retained `RealPrefixCache` GDN `LinearAttentionState` snapshots via `prefix_cache.max_entries` / `KILN_PREFIX_CACHE_MAX_ENTRIES`, and expose new prefix-cache state metrics so sustained workers=2 traffic cannot accumulate hidden ~49 MiB state entries into post-v0.2.10 OOM cascades (#697).

## kiln-v0.2.10 — 2026-05-02

### Fixed
- server: route prefix-cache prefill through the tiled/streaming long-prefill dispatcher so first-touch 43k-token prompts no longer bypass the GDN activation-bounded path (#686).

### Observability
- metrics: add `kiln_request_prefill_tokens_completed` to show live long-prefill progress instead of making operators infer whether a request is wedged from latency alone (#686).

### Configuration
- server: raise the default `request_timeout_secs` from 300 to 600 seconds so the production default is honest for long-prefill workloads; explicit TOML/env overrides remain unchanged (#686).

## kiln-v0.2.9 — 2026-05-01

Phase 11 reliability, throughput, and onboarding release. Cancel-drain on
prefill-timeout (#666) closes the v0.1.0 reliability blocker (#664); a
prefix-cache double-counting bug (#674) is root-caused and fixed, allowing the
workers=2 emergency serialize shim from #672 to be reverted (#675) — restoring
+23% throughput / −75% p99 ITL on the default config. Also lands the
tools-bearing chat-template render fix (#653), the Phase 11 demo and onboarding
docs (#669–#671), and the post-v0.2.8 Phase 10 audits (closure, FleCE T=16384
OOM probe, GDN training-streaming reachability) and Phase A FLCE A6000
validation accumulated since the v0.2.8 cut.

### Fixed
- server: cancel-drain `spawn_blocking` generation on prefill timeout — closes the v0.1.0 reliability blocker (#664) where a stuck prefill would orphan the generation thread (#666).
- prefix-cache: never return retained blocks in `evicted_blocks` — root cause of the workers=2 prefill cascade (#673) that motivated the #672 emergency serialize shim (#674).
- server: revert the workers=2 emergency serialize shim from #672 now that #674 fixes the underlying prefix-cache double-decrement (#675).
- template: tools-bearing chat-template render — register the `tojson` minijinja filter and pass `tool_calls.arguments` as a dict so Qwen3.5-4B's chat_template stops 4xx-ing on `unknown filter: tojson` and on string-typed arguments (#653).

### Performance
- Reverting the #672 serialize shim (via #675, enabled by #674's prefix-cache fix) restores +23% throughput and −75% p99 ITL on the default workers=2 config relative to the shim's exclusive `GpuCoordinationLock` path.

### Memory
- memory: relax auto KV cache cap on CUDA — let memory-aware sizing drive instead of clipping at `max_position_embeddings.div_ceil(block_size)`, which had hard-capped Qwen3.5-4B at ~8.6 GiB / 16384 blocks (one full-context window) regardless of available VRAM (#655).

### Documentation
- docs(demo): asciicast script + asciinema player scaffolding for the 60s online-learning demo (#669).
- docs(audits): Phase 11 ops checklist verification (#670).
- docs/demo: record canonical 60s online-learning asciicast and update `SCRIPT.md` to match shipping API (#671).
- docs(quickstart): OpenAI tools/function-calling example (#654).
- docs: workers=1 workaround for tools-bearing long prompts (refs #664, #656) — operator guidance pending the #666 cancel-drain fix (#665).

### Tests
- ci(chat-template): vendor Llama 3.1 + Qwen2.5 fixtures with stress-render tests (closes #657) (#658).
- ci(chat-template): vendor Mistral v0.3 [INST] tools-bearing fixture with stress-render test (#660).
- ci(chat-template): vendor DeepSeek V3 fixture with tool-calls render test (#661).
- ci(chat-template): vendor Hermes-3-Llama-3.1 fixture with tool-calls render test (#662).
- ci(api): integration smoke test for tools-bearing chat-completions render path (closes #659) (#663).

### Audits
- Phase 9 v0.1.0 release-readiness audit. New `docs/audits/PHASE9_V0_1_0_READINESS.md` walks each Phase 9 checklist item from the kiln project description and reports concrete evidence (PR/release/file) per item. **Verdict: Phase 9 is shipped — the "v0.1.0 release with semantic versioning" goal is stale.** kiln-v0.1.0 was tagged on 2026-04-19; the production line is now at kiln-v0.2.8 (Sigstore-signed build provenance, GHCR `kiln-server` Docker image auto-publishing on every `kiln-v*` tag, the `docs/site/` landing page live at https://ericflo.github.io/kiln/, and all 12 findings of `docs/audits/security-audit-v0.1.md` resolved). PR #651's framing ("what remains is an audit pass + the v0.1.0 semver cut") was written without rechecking the release index — the v0.1.0 tag predated that doc by 10 days; this audit is the audit pass and there is no remaining v0.1.0 cut to do.
- Phase 10 (Liger Kernel Integration) closure consolidating §1–§3 verdicts (PR #650 et al). New `docs/audits/PHASE10_CLOSURE.md` is the single state-of-play pointer for the Phase 10 chapter ledger (§1 RMSNorm fusion, §1.5 FLCE Phase B, §2 Mode B trace, §3 next-kernel candidate audit + FleCE T=16384 OOM probe), reproduces #649's math-ceiling table, and ranks three post-Phase-10 pivot directions: Phase 9 v0.1.0 release prep, GDN training-time streaming follow-ups (active in-flight: #635/#636/#637), and a LoRA precision study (FP32→BF16 with FP32 accumulate) targeting the ~42% FP32 SGEMM hotspot surfaced in #649. Records the explicit non-goal that no further Liger kernel ports are planned without a fresh re-profile that surfaces a candidate ≥1.05× ceiling. `PROFILING.md` gains a top banner pointing readers at the closure doc.

### Phase 10 — training-time streaming GDN prefill (impl)
- Implemented the audit-recommended remediation §1 from PR #634: added a streaming branch inside `model_forward_segment` that tiles GDN layers along T while threading `LinearAttentionState` per tile (full-attention layers always run monolithic — training has no KV cache to thread). New helper `gated_deltanet_forward_streaming` slices `[B,T,hidden]` into tiles of `KILN_STREAMING_TILE_TOKENS` (default 8192, must be a multiple of `GDN_CHUNK_SIZE=64`), calls `gated_deltanet_forward` per tile, and `Tensor::cat`s the results back into `[B,T,hidden]`. Bit-exact with the monolithic call by construction (the state hand-off matches `model_forward_paged_streaming_with`). Two new CPU parity tests (`test_gated_deltanet_forward_streaming_matches_monolithic_cpu`, `test_model_forward_segment_streaming_matches_monolithic_cpu`) pass. **A6000 SFT bench result: AMBER — dispatch is reachable now (CPU parity passes; T=8192 STREAMING ON tile=4096 takes 1.3 s of GPU work before OOM vs 0.7 s for monolithic; T=2048 STREAMING-ON loss matches STREAMING-unset bit-for-bit) but T=8192 SFT still OOMs at the 48 GB A6000 ceiling because the autograd recompute saves all per-tile activations simultaneously.** Net: necessary infrastructure for any future training-time GDN streaming work, but on its own this PR doesn't unblock T=8192 SFT — the next slice is the audit's remediation §2 (per-tile forward+backward inside `checkpointed_forward_backward`). See `docs/audits/PHASE10_GDN_TRAINING_STREAMING_IMPL.md` for the full validation, dispatch-reachability evidence, and root-cause analysis. Raw log: `docs/flce_phase_a_streaming_impl_raw_2026-04-29.log`.

### Audits
- Audit-recommended FleCE T=16384 OOM probe on A40 (sm_86, 46 GiB ceiling — A6000 capacity-blocked at task start, A40 is the audit's named arch-equivalent fallback). Two arms: (1) default A40 path (`fused_path=OFF`, A40 below the PR #644 ≥47 GiB gate) → status=OOM, peak=45,447 MiB / 46,068 MiB, delta=29,104 MiB above post-load baseline, step=9.44 s; (2) force-fused (`KILN_FORCE_RMSNORM_KERNEL=1`, simulates A6000's gate-ON dispatch on A40) → status=OOM, peak≥45,095 MiB / 46,068 MiB, delta=28,752 MiB, step=0.89 s (poller likely missed the actual peak in the early-failure case). **§3 closes for FleCE — zero ROI even though T=16384 OOMs.** The ~29 GiB delta is GDN/MLP saved activations across the 8 grad-checkpoint segments, *not* lm_head logits (Phase B's chunked-vocab CustomOp1 already avoids `[T_active, V]` materialization). FleCE Phase C — Liger-style vocab-axis chunking on top of Phase B's existing vocab-axis chunking — addresses the head, which is already <1 GiB at T=16384 with Phase B; it does not move the GDN/MLP-dominated bottleneck. Path forward for long-context SFT is GDN training-time streaming (PR #635/#636/#637 follow-ups), not a head-side fusion. Both audit branches resolve to "no kernel" under the actual mechanism. See the `Addendum 2026-04-29 — FleCE T=16384 OOM probe (closure)` section in `docs/audits/PHASE10_S3_CANDIDATE_PREFLIGHT.md` for full results, mechanism, A6000 implications, and reproduction. Raw logs: `docs/flce_phase_b_t16384_oom_probe_a40_raw_2026-04-29.log` (arm 1) and `docs/flce_phase_b_t16384_oom_probe_a40_fused_raw_2026-04-29.log` (arm 2). Also lands the `kiln-server` cuda → `kiln-train/cuda` Cargo.toml propagation fix that PR #647 and PR #649 each had to apply locally on the pod (see agent note `kiln-server-cuda-doesnt-propagate-to-flce`).

- Audited whether the existing Phase 7 streaming GDN prefill (`KILN_STREAMING_PREFILL`) is reachable from the SFT training path as a remediation for Finding 2 of the Phase A validation (T=8192/T=16384 SFT OOM on A6000). **Result: RED — streaming dispatch lives only in `model_forward_paged_streaming*`, none of which the trainer calls; the env flag is parsed but has no effect on training.** Empirical confirmation on A6000: T=2048 STREAMING ON peak VRAM matches STREAMING unset to 0 MiB and loss is bit-identical; T=8192 STREAMING ON at three tile sizes (default/4096/2048) all OOM identically at the 48 GB ceiling in 0.7 s. Closing Finding 2 requires net-new implementation work (e.g., a streaming branch inside `model_forward_segment`), not configuration. See `docs/audits/PHASE10_GDN_TRAINING_STREAMING.md` for source/empirical evidence and the remediation menu. Raw log: `docs/flce_phase_a_streaming_raw_2026-04-29.log`.

### Validation
- Validated Phase 10 Phase A FLCE on A6000 (48 GB) at T ∈ {2048, 8192, 16384} with rank-8 LoRA and gradient checkpointing ON. **Result: RED — Phase A is insufficient on A6000.** Two findings: (1) `KILN_USE_FLCE=1` fails before completing a step at T=2048 with `matmul is only supported for contiguous tensors` — the V-axis chunk of `embed_tokens_t` is non-contiguous and the chunked-vocab matmul rejects it; (2) even with that bug fixed, T=8192 and T=16384 still pin peak VRAM at the 48 GB A6000 ceiling, indicating GDN-side activations (not the head) dominate at long T. See `docs/audits/PHASE10_FLCE_PREFLIGHT.md` (Phase A validation section, 2026-04-29) for the table, raw log, and required follow-ups. Bench: `crates/kiln-server/examples/flce_phase_a_validation_bench.rs`.

## kiln-v0.2.8 — 2026-04-29

Patch release: post-v0.2.7 dep bumps and CI hygiene. First release that ships
[Sigstore-signed build provenance attestations](SECURITY.md#supply-chain-provenance)
for `kiln-v*` artifacts and the GHCR image.

### Reproducibility / release
- ci: attach build provenance attestations to release artifacts via `actions/attest-build-provenance`. `server-release.yml` and `docker-server-release.yml` now mint Sigstore-signed attestations bound to each artifact's sha256 and the workflow run; verify with `gh attestation verify <artifact> -R ericflo/kiln` (#627).

### CI / release
- ci: gate Apple codesign secrets (`APPLE_*`) to tagged releases so non-tag CI runs (PRs, branches) skip macOS code signing instead of failing on missing secrets (#625).
- ci: also gate Tauri signing secrets (`TAURI_SIGNING_*`) to tagged releases for the same reason as #625 (#626).
- chore(actions): bump `actions/upload-artifact` from 4 to 7 (#614).
- chore(actions): bump `actions/checkout` from 4 to 6 (#615).
- chore(actions): bump `docker/metadata-action` from 5 to 6 (#613).

### Dependencies
- chore(deps): bump `rand` from 0.9.4 to 0.10.1 (#616).
- chore(deps): bump `safetensors` from 0.5.3 to 0.7.0 (#617).
- chore(deps): bump `toml` from 0.8.23 to 1.1.2+spec-1.1.0 (#618).
- chore(deps): bump `tokenizers` from 0.22.2 to 0.23.1 (#619).

### Documentation
- docs(site): scaffold landing page + GitHub Pages workflow under `docs/site/` so https://ericflo.github.io/kiln/ has a published presence (Phase 9) (#624).

### Tests
- fix(test): serialize env-mutating config tests to eliminate flakiness when run in parallel (#623).

## kiln-v0.2.7 — 2026-04-28

Security release: closes all 12 findings in `docs/audits/security-audit-v0.1.md`.
Adds path-traversal hardening on adapter routes, per-route body-size limits,
webhook redirect-disable, training queue + tracked-jobs caps, adapter-dir
disk caps, LRU eviction for the composed-adapter cache, loopback default bind,
and reproducible Docker builds. No new features and no GPU/runtime breaking
changes.

### Security
- Fix path traversal in `DELETE /v1/adapters/:name` and `POST /v1/adapters/load` (Phase 9 audit §2b/§2c, HIGH).
- Validate source adapter names in `POST /v1/adapters/merge` (Phase 9 audit §2d, LOW).
- Per-route body-size limits on training + completions endpoints: 64 MiB for `/v1/train/sft` and `/v1/train/grpo`, 8 MiB for `/v1/chat/completions` and `/v1/completions/batch` (Phase 9 audit §1, LOW).
- Disable HTTP redirects on the training-completion webhook client to prevent server-side redirect chasing into internal infra (Phase 9 audit §7, item 10, LOW).
- Cap training queue depth (audit MEDIUM §4 part 1) — bounded queue with explicit reject when full (#607).
- Cap tracked-jobs map size and TTL Completed/Failed entries (audit MEDIUM §4 part 2) — prevents unbounded memory growth (#608).
- Cap total adapter_dir disk usage on upload (audit LOW §8 / item 9) — `KILN_ADAPTERS_DIR_MAX_BYTES` rejects uploads that would exceed the cap (#620).
- LRU eviction for `.composed/` adapter cache (audit LOW §8 / item 8) — bounded byte-size + entry-count caps with oldest-mtime-first eviction (#621). Closes the last open finding in security-audit-v0.1; all 12 findings now resolved.

### Changed
- Default server listen host changed from `0.0.0.0` to `127.0.0.1` (loopback). Set `server.host = "0.0.0.0"` or `KILN_HOST=0.0.0.0` to accept remote connections; pair with a trusted reverse proxy. Closes security-audit-v0.1 MEDIUM §9.

### Reproducibility / release
- docker: use `--locked` in `deploy/Dockerfile` cargo builds so the published `ghcr.io/ericflo/kiln-server` image matches the exact Cargo.lock dependency set (#601)
- Pin cargo-deny-action to a specific commit SHA and document deny.toml schema (audit LOW §11) (#612).

### Documentation
- docs(security): document training-data trust invariants (security audit §3 / item 12) in README + QUICKSTART. Calls out that `/v1/train/sft` and `/v1/train/grpo` apply a faithful gradient update to anything POSTed — kiln validates structure, not semantics — so the operator's training corpus must be treated as security-sensitive.

## kiln-v0.2.6 — 2026-04-26

Patch release: 3 server bug fixes + first release with bundled
THIRD_PARTY_LICENSES.md asset (cargo-about, MIT/Apache/BSD-only).

### Bug fixes
- metal: disable candle SDPA full path entirely to eliminate intermittent NaN on Apple Silicon (ff84800)
- server: re-emit prefilled `<think>\n` opener in chat-completion responses so streaming clients see it (7548e5a)
- server: split `<think>...</think>` content into llama.cpp-shaped `reasoning_content` field on chat completions (b1ae711)

### CI / release
- First release to ship THIRD_PARTY_LICENSES.md alongside binaries (#598, #599)

## kiln-v0.2.5 — 2026-04-26

Patch release: 4 server bug fixes since v0.2.4.

### Bug fixes
- server: close use-after-free race in prefix-cache streaming path (fad7c6b)
- server: make `stream: true` actually stream tokens in real time (11062f8)
- server: load model chat template so Qwen3.5 gets the `<think>\n` prefix in the rendered chat (e1fcc16)
- metal: bypass candle SDPA full kernel for `8 < q_seq < bq` to avoid a kernel crash (c7cf1ab)

## kiln-v0.2.4 — 2026-04-26

CI / release-prep release. No user-facing API or behavior changes since
v0.2.3; this cut exists to publish the kiln-server Docker image to GHCR
and to validate the auto-publish-on-platforms-green workflow end-to-end.

### CI / release
- Auto-publish the GitHub Release once all 3 platform jobs succeed, instead of leaving each tag in Draft (#592)
- Publish prebuilt server Docker image to `ghcr.io/ericflo/kiln-server` on every `kiln-v*` tag (#593)

## kiln-v0.2.3 — 2026-04-26

Phase 8 advanced features release: batch generation, adapter upload/download,
TIES + concatenation merge modes, per-request adapter composition, and webhook
notifications on training completion. Also lands the Phase 7 `/ui` adapter
controls, refreshed Phase 8 documentation (QUICKSTART, README, ARCHITECTURE,
plus a new docs/GRPO_GUIDE.md), and governance hygiene marking all workspace
crates `publish=false` and tightening cargo-deny wildcards to `deny`.

### Phase 8 advanced features
- POST /v1/completions/batch — efficient multi-prompt batch generation API for GRPO (#583)
- POST /v1/adapters/upload — multipart tar.gz import (#577)
- GET /v1/adapters/{name}/download — streaming tar.gz export (#575)
- TIES merge mode for /v1/adapters/merge (#578)
- Concatenation merge mode for /v1/adapters/merge (#579)
- Per-request adapter composition: stack multiple LoRAs with scaling on /v1/chat/completions (#581)
- Webhook notifications on training completion (#582)

### Phase 7 UI
- Add adapter download / upload / merge controls to `/ui` dashboard (#586)

### Docs
- Document Phase 8 API surface in QUICKSTART.md (#584)
- Refresh README + CHANGELOG for Phase 8 (upload/download/merge modes/composition/batch/webhooks) (#585)
- Refresh ARCHITECTURE.md for Phase 8 (upload/download, TIES/concat merge, composition, webhooks, batch generation) (#587)
- Add docs/GRPO_GUIDE.md with worked verifiable-rewards examples (math, JSON, code) (#588)

### Cleanup
- Move audit/preflight docs into docs/audits/, drop runtime log (#574)

### Governance / hygiene
- Mark workspace crates `publish=false`; tighten cargo-deny wildcards from warn to deny (#589)

### Test fixes
- Rewrite test_upload_rejects_path_escape_in_archive to actually emit a traversal tarball (#580)

## kiln-v0.2.2 — 2026-04-25

Coordinated release aligned with desktop-v0.2.2. Supersedes the unpublished
kiln-v0.2.1 draft; all v0.2.1 changes are included here. Highlights since
v0.2.0 are the radix prefix cache reuse path, more Metal/CUDA decode
fusions, governance docs, and dependency hygiene.

### Phase 7 prefix cache + decode reuse
- Implement radix prefix cache core (#512)
- Wire real append prefix cache (#515) and streaming real prefix cache reuse (#520)
- Use prefix cache with CUDA graphs and warn when bypassed (#518, #521)
- Expose prefix cache metrics (#513)
- Speed up greedy paged prefill defaults (#519)
- Default CUDA streaming prefill for long prompts and lower Metal threshold (#511)
- Refresh post-#521 profiling artifacts (#522)

### Phase 6 / Phase 7 Metal + CUDA fusions
- Fuse Metal attention output gate (#514)
- Fuse Metal GDN gates with recurrent decode
- Fuse Metal contiguous paged decode attention (#501)
- Fuse Metal GDN prefill conv split (#499) and Metal paged KV slot writes (#497)
- Fuse GDN recurrent RMSNorm decode (#496)
- Add CUDA GDN qk norm GQA fast path (#500)
- Add opt-in CUDA GDN decode fuse hook (#498)
- Route shared Metal greedy decode through argmax (#510)
- Defer transposed cache writer (#508); make transposed cache writes reliable (#506)
- Precompile Metal kernels before prewarm lock (#505)
- Batch MTP verifier argmax (#493)

### MTP audits and α-stability work
- H15c stratified C29 v2 reject-row probe (#529)
- H17 SGLang and H15c/H17b/H15a vLLM α microbenches (#530, #532, #533)
- H18 hand-rolled HF transformers MTP α reference (#534)
- H16 external-α reference options audit (#531)
- MTP acceptance-rate state-of-play audit (#527)
- End-to-end native-MTP self-spec decode bench post-#535 (#536)

### Phase 7 CLI / UX
- Recent requests panel on `/ui` dashboard (last 100) (#551)
- Live decode tok/s + p50/p99 ITL on `/ui` dashboard (#550)
- `kiln health` pretty-printed tree output + `--json` escape hatch (#549)
- `kiln train status` CLI subcommand + fix post-submit hint (#548)
- Surface structured server error hints in CLI (#545)
- `KILN_LOG_FORMAT=auto` — TTY-detect pretty vs JSON default (#544)
- GPU name + VRAM in startup banner (#543)
- ProgressBars for model load, SFT, GRPO (#540, #541, #542)

### Server runtime
- Move health adapter scan off runtime
- Document phase 7 prefix cache reuse benchmark
- Audit kiln radix prefix cache vs SGLang RadixAttention (#526)
- Audit vLLM fused_recurrent_gated_delta_rule against kiln-gdn-kernel (#525)
- Kill-switch bisection ruled out a single fused-kernel owner of the post-#166 decode gap (#524)
- Prefix-cache A/B ruled out cache hooks as bench regression source (#523)
- Fast-guard disabled MTP debug taps; reduce safetensors loader map allocations
- RunPod task tasks no longer pin `KILN_CUDA_ARCHS` (#494)

### Governance + CI
- Add Dependabot config for cargo + GitHub Actions (#558)
- Add `cargo-deny` license/source/bans policy and CI check job (#555, #556)
- Add CONTRIBUTING.md, SECURITY.md, CODE_OF_CONDUCT.md (#552, #553, #554)
- Add GitHub issue + PR templates (#557)

### Dependencies
- Bump tokenizers 0.21.4 → 0.22.2 (#565)
- Bump indicatif 0.17.11 → 0.18.4 (#564)
- Bump console 0.15.11 → 0.16.3 (#563)
- Migrate to rand 0.9 (#567)
- Bump cc in cargo-minor-and-patch group (#562)
- Bump docker/login-action 3 → 4 (#561) and docker/build-push-action 5 → 7 (#560)
- Bump Jimver/cuda-toolkit (#559)

### Docs / repo cleanup
- Refresh ARCHITECTURE.md for post-Phase-6 outcomes (#539)
- Refresh BENCHMARKS.md with post-#536 numbers + add vLLM/SGLang comparison (#537)
- A6000 llama.cpp re-bench at 512 → 128 (#538)
- README + QUICKSTART refresh (`/ui`, banner GPU/VRAM, logging defaults) (#546, #547)
- Archive 71 phase-cXX docs subdirs into `docs/archive/phase-c/` (#570)
- Archive frozen profiling/bench MD reports into `docs/archive/` (#568)
- Purge profiling artifact dirs from working tree (#569)

### CI fixes carried over from the unpublished v0.2.1
- Bump Jimver/cuda-toolkit from v0.2.19 to v0.2.35 to handle NVIDIA's renamed installer URLs (#469)
- Install MSVC dev env on Windows before CUDA build; fixes `M_LOG2E` undefined in `flash_api_c.cu` under MSVC (#472)
- Force static MSVC CRT on Windows CUDA build; fixes CRT mismatch between `esaxx-rs` and `kiln-marlin-gemm` (#477)

## kiln-v0.2.1 — 2026-04-24

(unpublished — superseded by kiln-v0.2.2)


Server re-cut to include CI fixes that were missing from kiln-v0.2.0. No
user-facing behavior changes from v0.2.0 in the core server; this cut also
picks up phase-6 Metal and CUDA kernel work landed on main between v0.2.0 and
v0.2.1.

### CI fixes shipped for the full platform matrix
- Bump Jimver/cuda-toolkit from v0.2.19 to v0.2.35 to handle NVIDIA's renamed
  installer URLs (#469)
- Install MSVC dev env on Windows before CUDA build; fixes M_LOG2E undefined
  in flash_api_c.cu under MSVC (#472)
- Force static MSVC CRT on Windows CUDA build; fixes CRT mismatch between
  esaxx-rs and kiln-marlin-gemm (#477)

### Phase 6 CUDA decode work
- Fuse CUDA GDN gated RMSNorm (#466)
- Add CUDA conv1d prefill fast path and fix conv1d prefill launch bounds
  (#481)
- Document post-466 and post-468 MTP decode profiles and post-476 MTP profile
  failure (#468, #480)
- Audit GDN conv decode hotspot and refresh post-#481 current-main profile
  (#473, #483)

### Phase 6 Metal decode work
- Fuse Metal LM-head argmax for greedy decode and reduce Metal LM-head argmax
  on GPU (#471)
- Speed up Metal decode GEMV and route GDN out-proj through Metal decode GEMV
- Fuse Metal full-attention QKV projections and fuse Metal GDN decode QKV
  conv norm
- Persist transposed weight cache asynchronously

### Infrastructure
- Cap cargo and nvcc parallelism and add OOM postmortem helper for RunPod
  builds (#474)

## kiln-v0.2.0 — 2026-04-24

Coordinated release aligned with desktop-v0.2.0. Headline work is the Metal
decode path for Apple Silicon: a new fused GDN kernel family, MTP speculative
decoding improvements, and a batch of macOS startup and prefill reductions.

### Metal GDN kernel fusion
- Fuse Metal GDN decode input projections
- Fuse Metal GDN chunk prep
- Fuse Metal RoPE for prefill (#418)
- Fuse Metal GQA qk norm expansion (#393)
- Default Metal MLP gate-up fusion (#447)
- Add Metal full-chunk GDN prefill (#394), head-last layout (#395)
- Speed up Metal GDN recurrent prefill (#398) and avoid zeroing recurrent outputs (#419)
- Use direct GDN chunk slices on Metal (#391); read full chunks from strided views (#455)
- Use unexpanded GDN QK for Metal decode (#456)
- Use head-major KV read for Metal decode (#452) and head-major SDPA for paged decode (#342)
- Use uninitialized Metal outputs for full-write kernels (#449)
- Parallelize Metal conv1d prefill

### MTP (multi-token prediction) decode
- Route MTP prefill through Metal streaming (#400)
- Speed up macOS default MTP decode
- Mirror desktop speculative routing in bench (#404)
- Align skip-layer bench draft state (#410)
- Defer MTP upload and trim draft state (#408)
- Avoid native MTP during Metal prewarm (#402)
- Guard non-streaming MTP final window (#454)
- Raise Metal skip-layer crossover to 4096 (#442)
- Route long macOS decode through paged skip-layer

### macOS startup and prefill
- Reduce macOS startup and KV prefill overhead
- Speed up macOS startup and skip-layer prefill
- Improve macOS startup and short-prompt routing (#440) and speculative routing (#437)
- Defer Metal precompile until background prewarm (#453); precompile Metal kernels during startup
- Move tokenizer warmup after listen (#460)
- Prewarm macOS speculative path (#386) and make prewarm opportunistic
- Tune macOS Metal hot paths (#385) and streaming prefill defaults (#377)
- Enable tiled prefill by default on Metal (#367)
- Route Metal prefix attention through head-major SDPA (#366)
- Optimize Metal paged KV prefill reads (#416)
- Speed up Metal LM head decode
- Gate Metal readiness prewarm (#332)
- Harden Metal prewarm and KV auto-sizing
- Drop redundant Metal embedding upload (#335)
- Batch Metal auxiliary weight uploads (#443); stream transposed weight cache reads (#445); mmap transposed weight cache hits (#461)

### Server runtime
- Keep default GPU sampling on device (#336); speed up default sampling and fix speculative KV advance (#328)
- Avoid zero-filling server KV pools (#337)
- Hoist paged decode debug gates (#333)

### Profiling and phase 6 kernel work
- Extensive phase 6 decode profiling work (C35–C50) and MTP α-stability re-benches documented in PROFILING.md and PROFILING-MTP-*.md

## kiln-v0.1.2 — 2026-04-20
- See the GitHub release for details.

## kiln-v0.1.1 — 2026-04-20
- See the GitHub release for details.

## kiln-v0.1.0 — 2026-04-18
- Initial public release.
