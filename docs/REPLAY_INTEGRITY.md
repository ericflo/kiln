# Request-Lineage Integrity

Kiln retains `replay.jsonl` and `lineage.json` for compatibility with the
original adapter audit format. The `kiln-replay` command verifies the integrity
of those records. It is not a replay-to-output or retraining command.

## What is bound

For each adapter in a root-to-leaf lineage, `replay_hash` covers:

- the parent adapter's recorded `replay_hash`;
- the recorded base-model ID, optional revision, and optional config digest;
- every serialized training request record in chronological order, including
  its request body, effective seed, Kiln version/commit string, and timestamp.

`kiln-replay verify ADAPTER_DIR` walks the named parent directories, recomputes
those hashes, and rejects a changed request, base identity, lineage pointer, or
record order. `kiln-replay show ADAPTER_DIR` prints the recorded chain without
executing it.

```bash
kiln-replay verify ./adapters/my-lora
kiln-replay show ./adapters/my-lora
```

A successful verification means only that the request-lineage records are
internally consistent with their stored hashes. The success message explicitly
states that no training or output replay was performed.

## What is not bound

The request-lineage hash does not prove or reproduce:

- complete base-weight shard bytes;
- tokenizer, chat-template, executable, source-tree, driver/runtime, device,
  precision, kernel, or effective-environment identity;
- optimizer, scheduler, RNG, data cursor/order, reference/EMA, or loop state;
- bytes behind a dataset path, external teacher responses, or other mutable
  dependencies;
- final loss, adapter tensors, logits, generated tokens, or eval outcomes.

Outcome records are deliberately excluded from `replay_hash`, so the hash
cannot attest that an outcome was reproduced. The name `replay_hash` and the
filenames remain stable on disk for schema compatibility; they describe the
historical request log, not an output-replay guarantee.

The JSONL writer synchronizes a request before its optimizer step begins and
Kiln enforces one training writer per adapter. It does not claim regular-file
append atomicity: an interrupted partial line is invalid and verification fails
closed. Crash-atomic resumable state belongs in `.kiln-checkpoint`.

## Exact continuation

Use a validated immutable `.kiln-checkpoint` for exact SFT, GRPO, or OPD
continuation. That format restores optimizer, scheduler, RNG, cursor/order,
reference/EMA where applicable, concrete precision, effective configuration,
complete base-weight shard identity, and execution provenance. Its guarantee is
still scoped to continuation inside the declared deterministic envelope. It is
not a promise of cross-build, cross-driver, cross-device, or cross-backend
byte-identical outputs.

Train receipts, adapter manifests, eval results, health, and the gated debug
endpoint provide complementary artifact and execution evidence. See
[Native Training Checkpoints](training-checkpoints.md),
[Execution Provenance](EXECUTION_PROVENANCE.md), and
[Base-Weight Provenance](BASE_WEIGHT_PROVENANCE.md).
