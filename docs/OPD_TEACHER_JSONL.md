# Off-Policy OPD Teacher JSONL

Kiln accepts one JSON object per line for off-policy teacher distillation.
Each line contains the prompt seen by the student, the teacher response to
replay, and optional teacher logprobs for reverse-KL training.

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

Objectives:

- `reverse_kl`: uses `teacher_tokens[*].top_logprobs` on action tokens.
- `cross_entropy`: builds a one-hot teacher fixture from the replayed teacher
  action tokens and does not require logprobs.

When `trajectory` includes `kind: "observation"` segments and OPD config has
`echo` enabled, kiln adds ECHO env-CE to the same OPD training step and records
OPD action tokens and ECHO env tokens separately in `train_receipt.json`.
