# Off-Policy OPD Teacher JSONL

The normative HTTP field contract for `OpdRequest`, `OpdConfig`,
distillation variants, teacher-facing prompts, post-eval gates, and training
responses is the generated [Training and Agent Control Plane API
Schema](../contracts/kiln-control-plane-v1.schema.json). The immutable teacher
identity and registry lifecycle are defined by the generated [Artifact
Lifecycle API Schema](../contracts/kiln-artifacts-v1.schema.json). This guide
owns the offline JSONL record workflow rather than duplicating those HTTP
contracts.

Kiln accepts one JSON object per line for off-policy teacher distillation.
Each example contains the prompt seen by the student, the teacher response to
replay, and optional teacher logprobs for reverse-KL training.

## Identity manifest

If any example contains numeric `logprob` or `top_logprobs` data, the first
non-empty record must be the exact canonical manifest for the registered
teacher:

The following shows the field shape only; values containing `...` are not
valid input.

```json
{"schema":"kiln.off-policy-distillation-manifest.v1","teacher_identity":{"schema":"kiln.teacher-identity.v1","protocol":"vllm.prompt-logprobs.numeric-token-ids.causal.v1","served_model_id":"...","base_model_sha256":"...","tokenizer_vocab_sha256":"...","tokenizer_config_sha256":"...","adapter":null,"vocab_size":248320,"max_top_k":32,"max_model_len":32768,"max_prompt_logprob_candidates":1000000,"logprobs_mode":"raw_logprobs","implementation":"...","inference_config_sha256":"..."}}
```

Do not assemble or reformat this line. After registering the teacher, extract
the server-produced canonical string and write it verbatim:

```bash
curl -s http://localhost:8420/v1/teachers \
  | jq -r '.teachers[] | select(.spec.alias == "qwen35@vllm") | .off_policy_manifest' \
  > teacher-data.jsonl
```

Append example records after it. Kiln requires the full manifest identity to
equal the submit-time pinned registry identity and hashes the exact JSONL bytes
into the receipt. A served-model name, tokenizer label, or identity digest by
itself is insufficient. The manifest is provenance, not a signature: accept
pre-scored files only from a trusted producer and storage path.

For cross-entropy rows without numeric teacher scores, the manifest may be
omitted. If a manifest is present, it is still validated and bound.

## Example record

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
    {"role": "assistant", "content": "{\"directory\":\"src\"}", "kind": "action"},
    {"role": "tool", "content": "src/lib.rs\nsrc/main.rs", "kind": "observation"}
  ],
  "metadata": {"source": "teacher-qwen3.6-27b"}
}
```

Fields:

| Field | Required | Meaning |
| --- | --- | --- |
| `messages` | yes | Student-visible chat prompt. Do not include the teacher answer here. |
| `teacher_response` | yes | Assistant text to replay for off-policy distillation. |
| `teacher_tokens` | reverse-KL only | One entry per teacher action token. Each entry must include at least `top_k` `top_logprobs` with `token_id` and finite `logprob`. |
| `trajectory` | no | Agentic Action/Observation segments. When present, these replay turns are the training sequence: Action tokens receive OPD supervision and Observation tokens receive ECHO env-token supervision when `config.echo` is enabled. |
| `metadata` | no | User metadata preserved for dataset provenance. |

The manifest is not an example and is excluded from example counts.

Objectives:

- `reverse_kl`: uses `teacher_tokens[*].top_logprobs` on action tokens.
- `cross_entropy`: builds a one-hot teacher fixture from the replayed teacher
  action tokens and does not require logprobs.

When `trajectory` includes `kind: "observation"` segments and OPD config has
`echo` enabled, kiln adds ECHO env-CE to the same OPD training step and records
OPD action tokens and ECHO env tokens separately in `train_receipt.json`.

## Student sampler and prompt identity

`config.sampler_segments` selects the positive layer-segment count used only
by the memory-bounded student sampler. Omit it for the proven default of 18,
capped at the model layer count. This setting is distinct from
`grad_checkpoint_segments`: changing one does not change the other. The CLI
flag is `--sampler-segments N`; the browser control is under OPD Advanced.

`config.rollout_prompt_rendering` is an explicit algorithm choice:

- `legacy_action_boundary` (default) preserves the admitted token sequence up
  to the first supervised action and is the qualified compatibility path.
- `chat_template` re-renders the prompt through the model chat template with
  thinking disabled. It is experimental because the changed token sequence
  has produced unreliable adapters on some structured-output workloads.

Use `--rollout-prompt-rendering VALUE` or the browser selector. Both fields are
validated before GPU work and retained in effective config, checkpoints,
teacher-fixture identity, and training receipts. No process environment value
can alter them after admission.

## Backward anomaly diagnosis

Set `config.detect_anomaly` to `true`, pass `--detect-anomaly` to
`kiln train opd`, or enable **Detect gradient anomalies** in the browser OPD
form to scan every backward operation's returned gradients. The first NaN or
Inf fails with the producing operation name and tape position. The policy is
captured per full or checkpoint-segment tape and cannot be changed by process
environment or leak into inference and other jobs.

The default is `false` because each scan adds a finite reduction and may
synchronize the device. Ordinary OPD still performs mandatory loss and
optimizer-boundary finite checks; enable this deeper scan only to localize a
corrupting operation. The effective config and exact checkpoint retain the
selection, so resume requires the same value.

## Exact checkpoint and resume

Set `config.checkpoint_interval` to a positive number of committed optimizer
steps (OPD defaults to 25). Kiln publishes an immutable
`NAME-checkpoint-step-NNNNNNNN.kiln-checkpoint` directory beneath the adapter
registry and reports its basename through `GET /v1/train/jobs/{job_id}` and
`kiln train status --job-id JOB_ID`. Cooperative cancellation also publishes
at the next settled source/sample candidate boundary.

Resume by submitting the identical JSONL path and exact bytes, teacher alias,
training mode, output adapter, and effective configuration with
`config.resume_checkpoint`, or by adding `--resume-checkpoint BASENAME` to the
same `kiln train opd --file ...` command. Kiln revalidates the JSONL content
hash, manifest identity, teacher content revision, model/base weights,
tokenizer, backend, precision, adapter/optimizer tensors, RNG streams, and
candidate cursor before continuing. Reformatting the manifest, replacing a
same-name teacher, or changing even an already-consumed row is a hard error.

The `.kiln-checkpoint` directory is exact optimizer continuation state. A PEFT
adapter snapshot is serving state only and cannot replace it. See
[Native Training Checkpoints](training-checkpoints.md#opd) for the complete
durability and browser handoff contract.
