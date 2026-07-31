# Off-policy OPD teacher JSONL

This guide defines the server-local JSONL corpus used by off-policy
distillation. It is not the JSON request passed to `kiln train opd --file`.
That request names the corpus through `dataset_path`.

The production path currently supports one offline objective:
`training_mode: "off_policy"`, `objective: "reverse_kl"`, and scored
top-K teacher rows. The `cross_entropy` value remains parseable for
compatibility, but the server rejects it because cross-entropy is not wired
into the production OPD loss root.

The generated [Training and Agent Control Plane API
Schema](../contracts/kiln-control-plane-v1.schema.json) is the normative
contract for `OpdRequest`, `OpdConfig`, job responses, and post-eval gates.
The generated [Artifact Lifecycle API
Schema](../contracts/kiln-artifacts-v1.schema.json) defines teacher
registration and identity. This page covers only the offline corpus workflow.

## Prepare and submit a corpus

1. Register the exact teacher deployment that produced the scores.
2. Copy its server-generated `off_policy_manifest` as the first non-empty line
   of the corpus.
3. Append one scored example per line. Score every supervised action token
   with the same model, tokenizer, inference configuration, and optional
   adapter named by the manifest.
4. Put the corpus where the running Kiln server can read it.
5. Submit an OPD request whose `dataset_path` names that server-local file.

For example, extract the canonical manifest for a registered teacher:

```bash
curl -fsS http://localhost:8420/v1/teachers \
  | jq -er '.teachers[]
      | select(.spec.alias == "teacher-large@vllm")
      | .off_policy_manifest' \
  > teacher-data.jsonl
```

Append scored rows to `teacher-data.jsonl`, then create the separate CLI
request file:

```json
{
  "dataset_path": "/srv/kiln/data/teacher-data.jsonl",
  "teacher": "teacher-large@vllm",
  "config": {
    "training_mode": "off_policy",
    "objective": "reverse_kl",
    "loss": "teacher_top_k",
    "top_k": 32
  }
}
```

Submit that request:

```bash
kiln train opd \
  --file opd-request.json \
  --adapter offline-distilled
```

The CLI reads `opd-request.json` on the CLI host. The running server reads the
absolute `dataset_path`; the CLI does not upload the JSONL corpus.

## Identity manifest

The current reverse-KL path requires numeric top-K logprobs, so every
production corpus requires the canonical manifest. The value returned as
`off_policy_manifest` is already a complete, compact JSON object such as:

```json
{"schema":"kiln.off-policy-distillation-manifest.v1","teacher_identity":{"schema":"kiln.teacher-identity.v1","protocol":"vllm.prompt-logprobs.numeric-token-ids.causal.v1","served_model_id":"...","base_model_sha256":"...","tokenizer_vocab_sha256":"...","tokenizer_config_sha256":"...","adapter":null,"vocab_size":248320,"max_top_k":32,"max_model_len":32768,"max_prompt_logprob_candidates":1000000,"logprobs_mode":"raw_logprobs","implementation":"...","inference_config_sha256":"..."}}
```

The `...` values above illustrate the shape and are not valid input. Use the
server-produced string instead of constructing this record yourself.

The manifest rules are strict:

- It must be the first non-empty record.
- Its line must match the server-generated compact JSON byte for byte. Leading
  whitespace, trailing whitespace, pretty-printing, or key reordering makes
  the record noncanonical.
- Its complete teacher identity must equal the identity pinned by the
  registered alias at submission time. A matching alias, model name, or digest
  fragment is not enough.
- It is provenance, not authentication. Accept a scored corpus only from a
  trusted producer and storage path.

Kiln hashes the exact JSONL bytes and records that SHA-256 digest in
`train_receipt.json`. Editing whitespace, metadata, or an already-consumed row
therefore changes the dataset identity.

## Example-record shape

The following record is deliberately abbreviated to show the nested shape. It
is not submission-ready: a real row needs one `teacher_tokens` entry for every
tokenized action target, and each entry needs at least the requested 16 or 32
top-logprob candidates.

```json
{
  "id": "example-001",
  "messages": [
    {"role": "system", "content": "Answer with JSON only."},
    {"role": "user", "content": "List files under src/"}
  ],
  "teacher_response": "{\"directory\":\"src\",\"recursive\":true}",
  "teacher_tokens": [
    {
      "token": "{",
      "token_id": 90,
      "logprob": -0.02,
      "top_logprobs": [
        {"token_id": 90, "logprob": -0.02},
        {"token_id": 4913, "logprob": -4.10}
      ]
    }
  ],
  "trajectory": [
    {
      "role": "assistant",
      "content": "{\"directory\":\"src\"}",
      "kind": "action",
      "tool_call_id": "call-001"
    },
    {
      "role": "tool",
      "content": "src/lib.rs\nsrc/main.rs",
      "kind": "observation",
      "tool_call_id": "call-001"
    }
  ],
  "metadata": {"source": "trusted-teacher-export"}
}
```

### Top-level fields

| Field | Required | Contract |
| --- | --- | --- |
| `id` | no | Caller-defined label. It participates in the exact source hash but is not copied into the training receipt. |
| `messages` | yes | Non-empty student-visible prompt. Do not include the replayed teacher answer here. |
| `teacher_response` | yes | Non-empty assistant text. When `trajectory` is empty, Kiln appends this as the replayed assistant turn. When `trajectory` is present, the trajectory supplies the training sequence and this compatibility field is still required but does not replace those segments. |
| `teacher_tokens` | yes for production reverse-KL | One entry per tokenized action target, in target order. The count must exactly match the action-token count produced by Kiln's tokenizer and mask builder. |
| `trajectory` | no | Ordered action, observation, or context segments. When non-empty, it is the replayed training sequence. |
| `metadata` | no | Arbitrary caller JSON. It participates in the exact source hash but is not copied into the training receipt. |

The manifest is not an example and does not count toward the example total.

### `teacher_tokens` fields

| Field | Required | Contract |
| --- | --- | --- |
| `token_id` | no | If present, it must equal Kiln's tokenized action target at the same position. Use it to catch tokenizer or alignment drift. |
| `token` | no | Human-readable audit text. The trainer does not use it to compute the loss or validate token identity. |
| `logprob` | no | Optional audit value for the selected token. It does not drive the production reverse-KL loss; a numeric value still triggers the manifest requirement. |
| `top_logprobs` | yes | Ranked candidate list. Kiln consumes the first requested `top_k` entries. |

Every consumed `top_logprobs` entry requires a vocabulary-valid `token_id` and
a finite, non-positive `logprob`. Within each row, token IDs must be unique,
logprobs must be in non-increasing order, and their represented probability
mass must not exceed one. The production kernels resolve only top-K sizes 16
and 32; 32 is the request default.

## Action and observation supervision

Without `trajectory`, Kiln tokenizes `messages` plus `teacher_response` and
applies reverse-KL only to assistant action targets.

With `trajectory`, each segment's `kind` controls supervision:

| `kind` | Effect |
| --- | --- |
| `action` | Receives reverse-KL supervision and consumes aligned `teacher_tokens`. |
| `observation` | Can receive ECHO environment-token cross-entropy when `config.echo` is present and its `lambda` is nonzero. |
| `context` | Provides context but receives no direct loss. This is also the default when `kind` is omitted. |

`role` controls chat-template rendering; `kind` controls the loss mask. Do not
assume that an `assistant` role automatically means `action` or that a `tool`
role automatically means `observation`.

When ECHO has eligible observation tokens, Kiln adds
`lambda × mean_environment_cross_entropy` to the same OPD step. The final
receipt records action and environment token counts separately, whether the
ECHO term actually fired, its configured lambda, and the final environment
cross-entropy value. A configured ECHO block with no eligible observation
tokens contributes nothing and is reported as not enabled for that run.

## Admission and failure behavior

Kiln admits the corpus before it queues GPU work:

- `dataset_path` must resolve to a regular UTF-8 file no larger than 64 MiB.
- Blank lines are ignored. Every other line must contain one JSON object.
- The file must contain at least one example after the optional manifest.
- A malformed record, empty prompt, empty response, token-count mismatch,
  invalid token ID, invalid logprob row, identity mismatch, or conflicting
  duplicate scored row rejects the whole submission. There is no skip-invalid
  mode for OPD.
- Unknown fields in example records are currently ignored by the parser.
  Treat that as compatibility behavior, not an extension mechanism.
- Kiln materializes the admitted examples and scored fixture once, then binds
  the exact source SHA-256 to checkpoints and the final receipt.

These checks make a corpus fail before training rather than silently shifting
the teacher-token alignment.

## Request controls that are not corpus fields

Sampler, anomaly-detection, checkpoint, optimizer, and ECHO settings belong in
`OpdConfig`, not in JSONL records.

`sampler_segments` controls only the memory-bounded student sampler. Omit it
for the automatic value of 18, capped at the model's layer count. It is
separate from server-selected gradient checkpointing and is not a
device-specific tuning rule.

`rollout_prompt_rendering` selects how on-policy student rollout prefixes are
constructed:

- `legacy_action_boundary` is the default compatibility path.
- `chat_template` re-renders the prompt with thinking disabled and remains
  experimental for structured-output workloads.

`detect_anomaly: true` scans every backward operation for NaN or Inf and fails
at the producing operation. The default is `false` because the deeper scan can
synchronize the device. Ordinary training still performs mandatory finite
checks at the loss and optimizer boundaries.

## Exact checkpoint and resume

OPD defaults to an immutable checkpoint every 25 committed optimizer steps.
`GET /v1/train/jobs/{job_id}`, `kiln train status --job-id JOB_ID`, and the
dashboard report the latest checkpoint basename. Cooperative cancellation
publishes a checkpoint at the next settled source/sample candidate boundary
when work remains.

Resume by submitting the same CLI request file and overrides with the reported
basename:

```bash
kiln train opd \
  --file opd-request.json \
  --adapter offline-distilled \
  --resume-checkpoint offline-distilled-checkpoint-step-00000025.kiln-checkpoint
```

The JSONL path and exact bytes, teacher alias and content revision, output
adapter, effective configuration, model and base weights, tokenizer, backend,
precision, optimizer state, RNG streams, and candidate cursor must all still
match. Reformatting the manifest, replacing a same-name teacher, or changing
an already-consumed row is a hard error.

A `.kiln-checkpoint` directory contains exact optimizer-continuation state. A
PEFT adapter snapshot is serving state and cannot replace it. See [Native
training checkpoints](training-checkpoints.md#opd) for the complete durability
and browser handoff contract.
