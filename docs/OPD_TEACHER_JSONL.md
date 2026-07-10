# Off-Policy OPD Teacher JSONL

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
