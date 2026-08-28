# Request-lineage verification

`kiln-replay` verifies the historical request-lineage files attached to some
SFT and GRPO adapters. Despite its name, the command does not rerun training,
load a model, regenerate output, compare tensors, or prove reproducibility.

Use it when you need to know whether the **hashed request chain** still agrees
with the files on disk. Use checkpoints, receipts, manifests, and evaluation
artifacts for stronger continuation or outcome claims.

## Verify or inspect a chain

Pass the leaf adapter directory:

```bash
kiln-replay verify ./adapters/my-lora
kiln-replay show ./adapters/my-lora
```

`verify` walks from the root adapter to the named leaf, recomputes each
`replay_hash`, and exits nonzero on a parse, chain, or hash error. Success is
deliberately explicit:

```text
OK: request-lineage integrity at ./adapters/my-lora verifies; no training or output replay was performed
```

`show` only prints the recorded root-to-leaf chain. It does **not** verify the
hashes, so do not treat successful display as an integrity result.

The command infers the adapter root from the leaf directory's parent.
Every `parent_lora.name` is resolved as a sibling beneath that root:

```text
adapters/
├── base-tune/
│   ├── lineage.json
│   └── replay.jsonl
└── my-lora/
    ├── lineage.json
    └── replay.jsonl
```

Keep every referenced parent directory available under the same root when you
move or verify a chain.

## The two files

| File | Contents | Role in verification |
| --- | --- | --- |
| `replay.jsonl` | `request` records written before an optimizer step and `outcome` records written after completion or failure. | Request records are hashed in file order. Outcome records must parse but are not hashed. |
| `lineage.json` | Base-model fields, an optional parent name and hash, display metadata, and this adapter's `replay_hash`. | Supplies the claimed hash and the next parent pointer. |

Kiln keeps these names and the `replay_hash` field for on-disk compatibility
with the original adapter audit format. In schema version 1, the hash is a bare
64-character lowercase hexadecimal SHA-256 digest; it does not use the
`sha256:` prefix used by newer provenance contracts.

## Exactly what `replay_hash` covers

Kiln hashes these inputs in order, separated as defined by the v1
implementation:

1. the parent adapter's recorded `replay_hash`, or an empty root value;
2. `base_model.id`;
3. the optional `base_model.revision`, or an empty value;
4. the optional `base_model.config_digest`, or an empty value; and
5. every deserialized `request` record in `replay.jsonl`, re-serialized by
   Kiln in file order.

Each request record contributes all of its fields:

- `request_id`;
- training `kind` (`sft` or `grpo`);
- the accepted, deserialized `request_body`;
- the resolved `seed`;
- that request's `kiln_commit` value; and
- its `submitted_at` timestamp.

This binds the stored JSON values, not the original HTTP bytes. Differences
such as transport whitespace are gone before the request record is written.
The order is the order of request records in the file; Kiln does not sort them
by timestamp.

For every link in the chain, `verify` also:

- resolves `parent_lora.name` to a sibling directory;
- detects directory cycles;
- checks that the child's recorded parent hash equals the verified parent's
  recorded hash; and
- checks that the recomputed current hash equals `lineage.replay_hash`.

## Fields that v1 does not bind

The v1 digest does not cover these `lineage.json` fields:

- `schema_version`;
- `adapter_name`;
- the lineage-level `kiln_commit`; or
- the lineage-level `created_at`.

The parent name selects a directory, but the hash binds the selected parent's
recorded hash rather than the spelling of that name. Renaming a parent
directory and updating the name can therefore preserve a valid chain.

Outcome records are also excluded. Changing a parseable outcome's status,
loss, elapsed time, or error does not change `replay_hash`; malformed JSON
still makes verification fail while reading the file. Use
`train_receipt.json` and adapter content hashes when those facts matter.

## What successful verification proves

A passing `verify` result establishes only that:

- each available parent points to the recorded parent hash;
- the base-model descriptor and stored request records recompute to the
  claimed v1 hashes; and
- the request chain is internally consistent at verification time.

It does not authenticate who wrote the files. Anyone who can rewrite all
records and recompute all hashes can create a new internally consistent chain.

## What it cannot reproduce or prove

Request-lineage verification does not bind:

- complete base-weight shard bytes;
- adapter tensor bytes or final content identity;
- tokenizer or chat-template identity;
- executable, source tree, driver, runtime, device, precision, kernel, or
  effective environment;
- optimizer, scheduler, RNG-stream, data-cursor, batch-order, reference-model,
  EMA, or loop state;
- bytes behind a dataset path, external teacher responses, or other mutable
  dependencies; or
- final loss, logits, generated tokens, or evaluation outcomes.

The request append is synchronized before its optimizer step begins, and Kiln
enforces one training writer per adapter. Regular-file append is not
crash-atomic: an interrupted partial JSON line is invalid and verification
fails closed.

## Use exact checkpoints for continuation

Use a validated immutable `.kiln-checkpoint` to continue supported SFT, GRPO,
or OPD training from an exact saved boundary. The checkpoint contract restores
optimizer, scheduler, RNG, cursor and ordering state, reference or EMA state
where applicable, effective configuration, precision, complete base-weight
identity, and execution provenance.

That guarantee is still scoped to continuation inside the checkpoint's
declared deterministic envelope. It does not promise byte-identical output
across builds, drivers, devices, or backends.

Use the surrounding evidence according to the claim:

| Claim | Primary evidence |
| --- | --- |
| “These request-lineage files were not changed without recomputing the chain.” | `kiln-replay verify` |
| “These are the exact base-weight bytes.” | [Base-weight identity](BASE_WEIGHT_PROVENANCE.md) |
| “This process declared the same execution envelope.” | [Execution identity](EXECUTION_PROVENANCE.md) |
| “These are the adapter files and hashes that were published.” | [Adapter manifest](ADAPTER_MANIFEST.md) |
| “Continue from this exact training boundary.” | [Native training checkpoints](../training/training-checkpoints.md) |

No single one of these artifacts proves full end-to-end reproduction.
