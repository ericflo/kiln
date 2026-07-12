# HF/TRL Interoperability Contract

Kiln's native trainer is the bounded `native_online_lora_v1` single-GPU LoRA
profile described in [NATIVE_SFT_PROFILE.md](NATIVE_SFT_PROFILE.md). General,
distributed, and broadly configurable training belongs in Hugging Face
Transformers, TRL, and PEFT. The interoperability route moves data and
adapters between those systems without discarding the identities needed to
audit the handoff.

The version-1 manifest and validation library are implemented. The public
export/import API and reference training scripts are not yet available; until
those surfaces ship, Kiln does not claim that manually assembled PEFT output
has passed this contract.

## Bundle Model

One handoff directory contains two independently self-verifying documents:

- `kiln_hf_export.json` (`kiln.hf-trl-export.v1`) is written by Kiln before
  external training.
- `kiln_hf_result.json` (`kiln.hf-trl-result.v1`) is written after external
  training and refers to the exact export digest.

The export binds all of the following:

- the served model ID and complete base-weight shard manifest;
- exact model configuration, tokenizer, and tokenizer-vocabulary identity;
- inference, native-training, and TRL-training template bytes;
- exact exported dataset bytes, ordered corpus identity, and row count;
- SFT kept/rejected-row evidence or the exact rollout-provenance schema for
  GRPO;
- an optional evaluation split manifest and optional input PEFT adapter; and
- Kiln's exact reference script.

The result binds the export digest, task, package versions, trainer kind,
effective configuration, preserved executed-script bytes, and exact PEFT
configuration and safetensors output. A custom script is permitted and remains
distinguishable from Kiln's reference script by content hash.

Every artifact identity contains a normalized relative path, nonzero byte
length, and a complete lowercase `sha256:<64 hex>` digest. Bundle paths must
not be absolute, contain `..`, contain backslashes, or traverse symlinks.
Readers reject unsupported schema versions, unknown or duplicate fields,
missing files, non-regular files, and byte or length mismatches.

## Template Identities

The three template artifacts are intentionally distinct:

- `chat_template.jinja` is the inference template loaded by Kiln.
- `kiln_training_chat_template.jinja` is the exact effective template identity
  retained in native checkpoints, receipts, and adapters.
- `training_chat_template.jinja` is the template supplied to HF/TRL.

For the source-pinned Qwen3.5 route, the last file retains TRL's
`{% generation %}` blocks for assistant-only labels. Kiln's native minijinja
renderer cannot parse that extension, so its effective identity uses the same
template with those blocks converted through Kiln's internal span path. The
two hashes must not be substituted for one another.

## Stable Digests

Manifest self-digests exclude only their own digest field. Kiln recursively
sorts JSON object keys, emits compact UTF-8 JSON, hashes the resulting bytes
with SHA-256, and prefixes the lowercase digest with `sha256:`. Arrays retain
their order.

Effective trainer settings use tagged values: `boolean`, `integer`,
`unsigned`, `decimal`, `text`, and `text_list`. Decimal values are finite
numbers carried as exact strings. This avoids cross-language digest drift from
valid but different Rust and Python floating-point JSON spellings.

The result's executed script is preserved as `executed_train.py`. Import can
therefore report whether it is byte-identical to the exported `train.py`
without preventing deliberate custom training code.

## Validation Boundary

This contract proves byte and identity continuity across an explicitly
provided handoff. It does not prove that an external process executed the
reported script, that package version strings are trustworthy, or that the
trainer produced a correct optimization result. The pinned reference scripts,
oracle fixtures, and round-trip tests provide behavioral evidence separately.

Import must validate both self-digests, the result-to-export link, every
referenced file, trainer/task consistency, and current resident
base/tokenizer/template identities before publishing an adapter. Validation is
performed on a Kiln-owned, quiescent bundle directory; concurrently mutating a
bundle during validation is outside the contract.
