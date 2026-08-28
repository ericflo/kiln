# Native Training Checkpoints

Kiln can resume native SFT, GRPO, and OPD from immutable, checksummed
checkpoints. A resume continues the same optimizer run; it is not a convenient
way to reuse weights under a different configuration.

The generated [Training and Agent Control Plane API
Schema](../../contracts/kiln-control-plane-v1.schema.json) is the source of truth
for request and response fields. This guide explains the operational contract:
what Kiln saves, when it saves it, what must match, and how to recover safely.

## Checkpoint or adapter?

Kiln publishes two different artifacts:

| Artifact | Use it for | Do not use it for |
|---|---|---|
| Resumable checkpoint | Continue an interrupted native SFT, GRPO, or OPD run | Serving, PEFT import, or warm-starting a different run |
| PEFT adapter snapshot | Serve the trained LoRA, import it into Hugging Face PEFT, or use it as a weights-only base adapter | Restoring optimizer, cursor, scheduler, or RNG state |

A resumable checkpoint is a directory whose name ends in
`.kiln-checkpoint`. It contains `checkpoint_manifest.json`, adapter state,
optimizer state when the optimizer has state, loop and scheduler state, and
checksums for every declared file.

A PEFT snapshot contains `adapter_config.json` and
`adapter_model.safetensors`. It does not contain enough state for an exact
resume. Kiln rejects a PEFT directory passed as `resume_checkpoint` instead of
silently restarting the optimizer.

## API, CLI, and browser workflow

### Start a checkpointed run

Set a positive checkpoint interval when you submit SFT or GRPO. OPD defaults to
25 committed optimizer steps, although spelling out the interval makes the
durability policy visible.

```bash
kiln train sft \
  --file corrections.jsonl \
  --adapter support-bot \
  --epochs 3 \
  --checkpoint-interval 25

kiln train grpo \
  --file scored-groups.jsonl \
  --adapter reward-bot \
  --checkpoint-interval 25

kiln train opd \
  --file opd-request.json \
  --adapter distilled-bot \
  --teacher qwen35@vllm \
  --checkpoint-interval 25
```

SFT and GRPO have no built-in request default. The server-level
`training.checkpoint_interval` supplies one when their request value is absent.
OPD's request default is 25. A non-null request value takes precedence, and zero
is invalid.

Checkpoints are direct children of the configured adapter registry:

```text
support-bot-checkpoint-step-00000025.kiln-checkpoint/
```

They are independent of the temporary directory used to publish the final
adapter.

### Find the checkpoint to resume

`GET /v1/train/jobs/{job_id}` reports `latest_checkpoint`. The summary includes
the resume basename, training kind, data-source kind, committed step, total
steps, next cursor, and completion state. SFT reports an epoch/example cursor;
GRPO reports a group cursor; OPD reports a candidate cursor.

The CLI prints the same basename:

```bash
kiln train status --job-id JOB_ID
```

Job detail also retains the admitted effective seed and validated effective
configuration. OPD detail includes the teacher alias plus separate teacher
identity and content revisions.

Status reads and validates bounded checkpoint metadata without rehashing large
tensor files on every poll. Resume admission performs the full file-set, size,
and SHA-256 validation.

### Resume the run

Submit the same training kind, output adapter, source, route, and effective
configuration with the reported basename:

```bash
kiln train sft \
  --file corrections.jsonl \
  --adapter support-bot \
  --epochs 3 \
  --checkpoint-interval 25 \
  --resume-checkpoint support-bot-checkpoint-step-00000025.kiln-checkpoint

kiln train grpo \
  --file scored-groups.jsonl \
  --adapter reward-bot \
  --checkpoint-interval 25 \
  --resume-checkpoint reward-bot-checkpoint-step-00000025.kiln-checkpoint

kiln train opd \
  --file opd-request.json \
  --adapter distilled-bot \
  --teacher qwen35@vllm \
  --checkpoint-interval 25 \
  --resume-checkpoint distilled-bot-checkpoint-step-00000025.kiln-checkpoint
```

The HTTP form places the same value under `config`:

```json
{
  "dataset_path": "/absolute/path/scored-groups.jsonl",
  "config": {
    "output_name": "reward-bot",
    "checkpoint_interval": 25,
    "resume_checkpoint":
      "reward-bot-checkpoint-step-00000025.kiln-checkpoint"
  }
}
```

The server accepts a single basename beneath the adapter registry. It also
accepts an absolute path only when that path is directly beneath the same
registry. Traversal, nested paths, the wrong training kind, and an
adapter-name mismatch fail before GPU work.

The dashboard can prepare a resume request from job detail. It deliberately
clears inline SFT or GRPO data and OPD prompts that the checkpoint summary
cannot reconstruct. Restore the identical source before submitting. For OPD,
the currently registered teacher alias must still have the checkpoint's exact
identity revision; admission then reconstructs and verifies its content
revision.

## What “exact resume” requires

Resume is continuation, not warm-starting. Kiln compares the checkpoint with
the new request and the running server before taking GPU ownership.

The following must still match:

- training kind, output adapter, data route, source bytes, and source order;
- effective training configuration, LoRA shape and scaling, optimizer,
  scheduler, and resolved learning rate;
- effective seed and every named RNG stream;
- model configuration and the byte content of every base-weight shard;
- tokenizer, inference template, and SFT training template;
- backend, device, driver/runtime evidence, executable, precision, and kernel
  contract;
- gradient-checkpoint and streaming-prefill planning;
- the objective-specific loss, tape, backward, reference, and sampling state.

Relocating or renaming unchanged base-weight shards is allowed because resume
compares their byte-content identity. Changing a shard digest, size, or
multiplicity is not.

Use `base_adapter` for a weights-only warm start. Do not edit a checkpoint
manifest to force compatibility: the manifest and its artifacts are
checksummed, and a manual edit destroys the identity Kiln is designed to
prove.

### Checkpoint planning identity

The outer manifest remains `kiln.training-checkpoint.v1`. Its auxiliary
planning identity is versioned independently:

- GRPO and OPD use schema `kiln.training-checkpoint-planning.v3`.
- SFT uses schema `kiln.training-checkpoint-planning.v4`, which also binds
  `sft_loss_route`.

The current SFT route values are `kt_tape_flce`, `vulkan_active_rows`, and
`full_logits`. These are capability-derived backend routes, not device-name,
vendor-ID, request, CLI, or environment allowlists. Admission selects the
supported route, the queue rechecks it against the resident runner, and the
trainer rechecks it before allocation.

Any change to the complete planning identity fails closed. In particular, a
prior SFT v3 checkpoint cannot resume as v4 because it does not prove which
loss route admission budgeted and execution used. Older artifacts remain
inspectable and checksum-valid; they are not exact-resume authority under a
new planning contract.

SFT additionally binds the checkpoint-boundary replay policy. Depending on that
policy and the sequence length, it either retains every boundary or replays
between sparse anchors. GRPO and OPD currently retain all planned boundaries,
but they still bind the common v3 policy so future execution changes cannot
reinterpret an older checkpoint.

## Queue-time revalidation

Admission fully validates the checkpoint and retains a compact identity: the
checkpoint ID, a digest of the validated manifest, and the effective seed. The
queue does not retain all checkpoint tensor bytes in memory.

At dequeue, before memory reservation, Kiln reloads the checkpoint, revalidates
every declared artifact, recomputes the manifest identity, and derives the seed
again. A replaced manifest, changed artifact, different checkpoint ID, or
different seed rejects the queued job.

This is revalidation, not a filesystem snapshot. The queue cannot prevent a
separate process from changing files after the dequeue check. Keep checkpoint
directories immutable and access-controlled for their entire lifetime.

## Crash and cancellation behavior

Checkpoint publication uses a hidden sibling staging directory:

1. Kiln creates the staging directory and an `.incomplete` sentinel.
2. It writes and synchronizes all declared state files.
3. It records each file's size and SHA-256 in the manifest and synchronizes the
   manifest.
4. It removes the sentinel and atomically renames the directory to its
   canonical `.kiln-checkpoint` name.
5. It synchronizes the parent directory.

The canonical name is therefore absent before publication and complete after
publication. A process crash can leave a hidden staging directory, but that
directory is neither discoverable nor loadable as a checkpoint. A later writer
can safely publish the intended canonical name beside the orphan.

On a shared serving GPU, Kiln holds the serving write lock only while copying
authoritative adapter and optimizer state into CPU-owned tensors. Encoding,
hashing, file writes, synchronization, and rename happen after the lock is
released. Logs separate GPU wait, device snapshot, and publication time.

Periodic checkpoints are published after committed boundaries:

| Training mode | Safe boundary |
|---|---|
| SFT | Optimizer step |
| GRPO | Optimizer group |
| OPD | Settled source/sample candidate |

When checkpointing is enabled, cooperative cancellation publishes at the next
safe boundary. A hard process loss may discard the in-flight unit, but not the
last published checkpoint.

## What the checkpoint restores

All three training modes restore:

- adapter parameters and stateful optimizer tensors by stable parameter name;
- optimizer and scheduler step;
- exact data cursor and ordering;
- loss history and objective diagnostics;
- effective configuration, precision, model, base weights, tokenizer,
  templates, backend, and runtime planning;
- all named RNG streams.

The objective-specific state differs by mode.

### SFT

SFT restores the epoch/example cursor, divergence and gradient diagnostics,
per-example checkpoint plan, and backend loss route.

### GRPO

GRPO restores frozen or EMA reference tensors and their refresh cadence,
policy-audit accumulators, and exact group identity. The streamed JSONL route
also restores its physical line and byte cursor, consumed-line hashes, token
counts, and gradient plans.

### OPD

OPD restores its collapse guardrails, rollout RNG, base adapter, and exact
teacher identity and numeric-content revision. Its candidate cursor is
intentionally separate from its optimizer-step counter: a deterministic
candidate can produce no update and must still be consumed exactly once.

## Checkpoint validation and security

The strict loader rejects:

- a noncanonical basename or missing `checkpoint_manifest.json`;
- an `.incomplete` sentinel;
- unsupported schema, checkpoint type, or unknown manifest fields;
- absolute, escaping, non-normalized, or untracked artifact paths;
- symlinks, and hard links on platforms where link counts are available;
- missing or extra files, size drift, and checksum drift;
- inconsistent progress, optimizer, scheduler, RNG, or auxiliary state.

Validation finishes before any checkpoint state is restored into a trainer.
Checkpoint names are immutable and are never overwritten.

## Promotion is a separate decision

A checkpoint protects training progress. It does not activate a model.
Training completion publishes a final PEFT adapter, and `auto_load` controls
whether Kiln may make that adapter active.

With `auto_load: true` and no promotion gate, Kiln can activate the completed
adapter after its serving canary passes. When the request includes a held-out
post-eval accuracy gate, activation is deferred: the previous adapter remains
active until the evaluation passes. A failed, inconclusive, or unavailable
gate leaves the candidate unpromoted. Training can still be successful even
when post-training evaluation cannot be enqueued.

Use the training job's linked evaluation IDs and promotion outcome to
distinguish “adapter was trained” from “adapter was promoted.” See [Evaluation
Guide](../guides/EVAL_GUIDE.md) for suite and comparison semantics.

## Recovery checklist

After an interruption:

1. Inspect the job and copy `latest_checkpoint.resume_checkpoint`.
2. Confirm that the original dataset, route, teacher, model, and server
   execution environment are still available.
3. Submit the same request with `resume_checkpoint`.
4. Treat any compatibility error as evidence of drift; do not edit the
   checkpoint or weaken validation.
5. If exact continuation is no longer possible, start a new run. Use a PEFT
   snapshot or `base_adapter` only when a weights-only warm start is intended.

Prefer the newest checkpoint. If you deliberately resume an older checkpoint,
first archive later same-adapter checkpoints outside the adapter registry.
Otherwise the resumed run will stop when it reaches an immutable name that
already exists.

Kiln does not prune checkpoints on a timer. Deleting an idle adapter through
the adapter API also removes checkpoint directories with that adapter's
checkpoint prefix. Active or physically loaded adapters must be unloaded before
deletion. Archive any checkpoint you must retain before deleting its adapter.

The older `replay.jsonl` and `lineage.json` audit trail is not a checkpoint.
`kiln-replay verify` checks request-lineage hash integrity; it does not rerun
training or compare losses, tensors, or outputs. See [Request-Lineage
Integrity](../contracts/REPLAY_INTEGRITY.md).

## Current support and qualification

Native SFT, inline and streamed-JSONL GRPO, and OPD support exact resume.
Capability-distillation routes that execute through OPD use the same OPD
checkpoint contract. DistillRefresh is a separate two-phase workload and does
not currently have an admitted exact-resume contract.

Opt-in hardware qualification covers fresh-repeat and
cancel-then-resume equivalence for the public SFT, inline GRPO, streamed GRPO,
and OPD routes on real ROCm and Vulkan devices. It compares loss history,
checkpoint state, final adapter bytes, receipts, manifests, and
objective-specific loop state.

That evidence establishes repeatability only inside the recorded model,
executable, backend, device, driver/runtime, precision, kernel,
tokenizer/template, configuration, data, and seed envelope. It does not claim
byte-identical training across different devices, backends, builds, drivers, or
machines.

For the complete identity schemas, see [Base-Weight
Provenance](../contracts/BASE_WEIGHT_PROVENANCE.md), [Execution
Provenance](../contracts/EXECUTION_PROVENANCE.md), and the [Native SFT
Profile](NATIVE_SFT_PROFILE.md).
