# Immutable vLLM teachers

Kiln accepts remote prompt logprobs only when the response carries an
authoritative `TeacherIdentityV1`. For vLLM 0.25 or newer, launch the teacher
through [`scripts/vllm_teacher.py`](../scripts/vllm_teacher.py). The launcher
fingerprints local inputs, puts the canonical identity in vLLM's native custom
`system_fingerprint`, disables runtime adapter updates, and starts vLLM without
a shell. See vLLM's official
[fingerprint API](https://docs.vllm.ai/en/latest/api/vllm/entrypoints/openai/fingerprint/)
for the custom mode this launcher uses.

A stock vLLM fingerprint is not sufficient. Its default fingerprint describes
vLLM's runtime configuration; a served model name is an alias. Neither proves
the exact tokenizer ID mapping, selected weight bytes, static adapter content,
or the limits Kiln relies on. Kiln therefore rejects stock `vllm-*` fingerprints
for remote teachers.

## Requirements

- vLLM 0.25.0 or newer and Transformers in the same Python environment.
- A local Hugging Face model directory. Hub IDs, revisions, custom tokenizers,
  `trust_remote_code`, alternate model arguments, and shell commands are not
  accepted by the launcher.
- Safetensors base weights discoverable by
  `scripts/qualification/model_fingerprint.py`.
- For an adapter process, one local PEFT adapter containing exactly one of
  `adapter_model.safetensors` or `adapter_model.bin`, plus
  `adapter_config.json`. Prefer safetensors; treat a pickle-based `.bin` file as
  executable input and use it only from a trusted source.
- Immutable model and adapter storage for the lifetime of the process. The
  launcher detects symlinks and changes during individual reads, but it cannot
  prevent a privileged operator from rewriting files after the process starts.

Use one process for a base teacher or one static adapter. Do not configure
multiple static adapters, resolver plugins, or runtime load/unload. The
launcher owns all LoRA, model, tokenizer, logprob, fingerprint, generation
config, and middleware arguments. It forces
`VLLM_ALLOW_RUNTIME_LORA_UPDATING=0`, rejects `VLLM_PLUGINS`, and rejects every
caller-supplied option containing `lora`.

vLLM documents the runtime mutation boundary in its
[security guide](https://docs.vllm.ai/en/stable/usage/security/).

## Base teacher

Run from the Kiln repository. Additional vLLM arguments come after `--` and use
one unambiguous `--key=value` token each. The launcher allows a small set of
documented valueless boolean switches such as `--enforce-eager`.

```bash
python3 scripts/vllm_teacher.py \
  --model-path=/models/Qwen3.5-4B \
  --served-model-id=qwen35-4b-teacher \
  --max-top-k=20 \
  --max-model-len=32768 \
  -- \
  --host=127.0.0.1 \
  --port=8000 \
  --dtype=bfloat16 \
  --api-key=replace-with-a-random-secret
```

The executed command uses the same Python interpreter as the launcher:

```text
python3 -m vllm.entrypoints.cli.main serve /models/Qwen3.5-4B \
  --served-model-name=qwen35-4b-teacher \
  --max-model-len=32768 \
  --max-logprobs=20 \
  --logprobs-mode=raw_logprobs \
  --generation-config=vllm \
  --fingerprint-mode=custom \
  --fingerprint-value=kiln-teacher-v1.<base64url-json>.<sha256>
```

The launcher prints the complete system fingerprint immediately before
`execve`. Kiln must receive that same value in every scoring response.

## Static adapter teacher

Add one adapter path. The API model ID is also the static adapter name:

```bash
python3 scripts/vllm_teacher.py \
  --model-path=/models/Qwen3.5-4B \
  --adapter-path=/adapters/math-v7 \
  --served-model-id=qwen35-4b-math-v7 \
  --max-top-k=20 \
  --max-model-len=32768 \
  -- \
  --host=127.0.0.1 \
  --port=8000 \
  --dtype=bfloat16 \
  --api-key=replace-with-a-random-secret
```

The launcher adds exactly one vLLM 0.25 JSON module argument:

```text
--enable-lora --max-loras=1 --max-cpu-loras=1 --max-lora-rank=<config-derived-cap> \
--lora-modules={"name":"qwen35-4b-math-v7","path":"/adapters/math-v7","base_model_name":"kiln-base-<base-hash-prefix>"}
```

This is vLLM's documented [JSON `--lora-modules`
form](https://docs.vllm.ai/en/stable/features/lora/#new-format-for---lora-modules).

vLLM lists the internal base alias as well as the adapter. Configure Kiln only
with `qwen35-4b-math-v7`; the identity and every Kiln request name that adapter.
The process fingerprint is not a valid identity for requests made directly to
the internal base alias.

## Identity contract

The compact JSON field order is normative:

```text
schema, protocol, served_model_id, base_model_sha256,
tokenizer_vocab_sha256, tokenizer_config_sha256, adapter, vocab_size,
max_top_k, max_model_len, logprobs_mode, implementation,
inference_config_sha256
```

The constants are:

```text
schema:         kiln.teacher-identity.v1
protocol:       vllm.prompt-logprobs.numeric-token-ids.causal.v1
logprobs_mode:  raw_logprobs
fingerprint:    kiln-teacher-v1.<base64url without padding>.<sha256 of JSON>
```

All SHA fields inside the identity are exactly 64 lowercase hexadecimal
characters without a `sha256:` prefix.

`base_model_sha256` matches Kiln's Rust loader. For each selected shard, hash
the raw bytes and record `(digest, byte_length)`. Sort records by digest and
then length. Hash the domain `kiln.base-model-content.v1\0`, a little-endian
`u64` record count, then a little-endian `u64` length and the raw 32-byte digest
for each record. Paths, mtimes, `config.json`, and the shard index are excluded.

`tokenizer_vocab_sha256` hashes the complete Transformers `get_vocab()` map.
After the domain `kiln.tokenizer-vocab.v1\0` and little-endian `u64` pair
count, pairs are sorted by `(u32 ID, raw UTF-8 token bytes)` and encoded as
little-endian ID, little-endian `u64` byte length, and token bytes. This is the
same golden byte contract as `KilnTokenizer::vocab_identity_sha256()`.

`tokenizer_config_sha256` hashes the exact UTF-8 string returned by the fast
tokenizer's `backend_tokenizer.to_str()`. This corresponds to Rust
`Tokenizer::to_string(false)`. It is deliberately not a hash of
`tokenizer_config.json`.

For a static adapter, `weights_sha256` is a domain-separated digest over the
selected filename, byte length, and raw weight-file digest. `config_sha256` is
the raw `adapter_config.json` digest. The adapter field is `null` for a base
teacher. The launcher reads `r` and every `rank_pattern` value, then selects the
smallest vLLM-supported maximum rank that fits. That derived cap is folded into
`inference_config_sha256`; it is not an additional identity field.

`inference_config_sha256` binds the exact model `config.json` digest, owned
logprob and context settings, adapter mode, non-transport vLLM arguments, and
relevant `VLLM_`, CUDA, ROCm/HIP, NCCL, Torch, and PyTorch environment values.
Host, port, API key, and TLS arguments are excluded because they do not change
logits. The API key itself is never embedded in the identity.

## Inspect without vLLM

Tests and tooling can use a strict input manifest without importing vLLM,
Transformers, or touching model weights. This mode can never launch a server.

```json
{
  "schema": "kiln.vllm-teacher-input.v1",
  "base_model_sha256": "<64 lowercase hex>",
  "model_config_sha256": "<64 lowercase hex>",
  "tokenizer_vocab_sha256": "<64 lowercase hex>",
  "tokenizer_config_sha256": "<64 lowercase hex>",
  "adapter": null,
  "adapter_max_rank": null,
  "vocab_size": 248320,
  "implementation": "vllm:0.25.0"
}
```

Emit only the identity:

```bash
python3 scripts/vllm_teacher.py \
  --identity-input=/tmp/teacher-input.json \
  --served-model-id=qwen35-4b-teacher \
  --max-top-k=20 \
  --max-model-len=32768 \
  --manifest-only
```

Add `--model-path` and use `--dry-run` to emit the redacted argv as JSON. A
manifest supplied through `--identity-input` is never trusted for a real
launch; a real launch always fingerprints local content itself.

## Network trust boundary

The fingerprint makes the identity document self-consistent. It does not prove
which server sent it. A malicious endpoint can fabricate or replay a valid
document. Use verified TLS to authenticate a remote server, keep certificate
verification enabled, and pin the expected identity when registering it with
Kiln. An API key authorizes a client but does not authenticate the server.

Bind plain HTTP to loopback only. For remote access, terminate TLS at a trusted
reverse proxy or pass vLLM's `--ssl-keyfile`, `--ssl-certfile`, and CA options.
Restrict the process and model directories to trusted operators. The vLLM API
key is present in process argv; the launcher's dry-run output redacts it, but a
reverse proxy is preferable when process-list visibility is a concern.

## CUDA handoff

Run the following on the 16 GB and 24 GB CUDA machines before treating the
vLLM path as qualified:

1. Install the intended vLLM 0.25+ build and matching Transformers/tokenizers
   versions, then run the Python unit tests below.
2. Run `--manifest-only` and `--dry-run`; archive the identity and redacted
   argv with the machine receipt.
3. Launch the base teacher. Send a non-streaming `/v1/completions` request with
   numeric prompt IDs, `prompt_logprobs`, `max_tokens: 1`, and the exact model
   ID. Confirm the response fingerprint is byte-for-byte identical to the
   printed value.
4. Register the teacher in Kiln and run the remote-teacher prompt-logprob smoke
   across first, interior, and final causal positions.
5. Restart without changes and confirm the fingerprint is stable. Change a
   copy of one shard, `config.json`, tokenizer backend JSON, and each adapter
   input in turn; confirm the appropriate identity digest changes.
6. Launch the static adapter process and prove requests named for the adapter
   use it. Confirm a request using the internal base alias is never accepted as
   that adapter by Kiln.
7. Confirm the runtime LoRA load/unload endpoints reject mutation and that a
   `VLLM_PLUGINS` resolver configuration fails before launch.
8. Record correctness and throughput at the intended batch sizes. Identity
   qualification establishes provenance; it does not substitute for numerical
   parity, pause analysis, or large-batch performance measurements.

Portable tests:

```bash
python3 -m unittest scripts/qualification/tests/test_vllm_teacher.py -v
```
