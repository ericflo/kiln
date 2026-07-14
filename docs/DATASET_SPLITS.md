# Dataset Splits and Train/Eval Separation

Kiln treats a registered dataset as one content-addressed corpus with three
persisted views: `train`, `validation`, and `holdout`. Native SFT and GRPO use
the `train` view by default. Dataset-to-suite synthesis uses `holdout` by
default. A post-training eval declared held out is checked against the exact
corpus admitted for training before the job is published to the queue.

This page defines that contract. The generated
[Eval and Judgment API Schema](../contracts/kiln-evals-v1.schema.json) and
[Training Control Plane Schema](../contracts/kiln-control-plane-v1.schema.json)
are the field-level references. The generated
[HTTP API Contract](../contracts/kiln-http-api-v1.openapi.json) owns route,
status, and media-type details.

## Recommended workflow

1. Upload one `sft_chat` or `grpo_groups` JSONL dataset.
2. Inspect its persisted split and counts.
3. Change the split seed or percentages before training if the defaults are
   unsuitable.
4. Train the named dataset's `train` partition.
5. Synthesize an eval suite from its `holdout` partition.
6. Attach that suite as a held-out `post_eval`.
7. Inspect `training_data` on the training job to retain the exact corpus and
   split-manifest identities used at admission.

The browser implements this workflow directly. Dataset pickers show train,
validation, and holdout counts; named training selects `train`; suite
synthesis selects `holdout`; and post-training evaluation defaults to
`Held-out evaluation`. `Train-set diagnostic` is a separate deliberate mode.

## Upload and inspect

Upload JSONL with the dataset registry API:

```bash
curl -sS http://localhost:8420/v1/eval/datasets/upload \
  -F name=support-corrections \
  -F format=sft_chat \
  -F description='Reviewed support corrections' \
  -F file=@support-corrections.jsonl
```

The returned dataset manifest includes:

| Field | Meaning |
| --- | --- |
| `corpus_sha256` | Ordered aggregate of exact canonical row identities. |
| `normalized_corpus_sha256` | Ordered aggregate after conservative string normalization. |
| `split_manifest_sha256` | SHA-256 identity of the persisted `split.json`. |
| `split_config` | Seed, train percentage, and validation percentage. |
| `split_counts` | Actual row counts in train, validation, and holdout. |
| `num_groups` / `num_sessions` | Distinct declared grouping identities found in rows. |

List all manifests or fetch one:

```bash
curl -sS http://localhost:8420/v1/eval/datasets
curl -sS http://localhost:8420/v1/eval/datasets/support-corrections
```

Fetch the complete row assignment manifest:

```bash
curl -sS \
  http://localhost:8420/v1/eval/datasets/support-corrections/split
```

The split response contains the dataset identities, config, counts, and one
entry per non-empty JSONL row with its one-based row number, exact identity,
normalized identity, partition, and optional group/session IDs.

## Split configuration

New and migrated datasets default to:

```json
{
  "seed": "0",
  "train_percent": 80,
  "validation_percent": 10
}
```

Holdout receives the remainder, so the default target is 80/10/10. To
replace the persisted assignment:

```bash
curl -sS -X PUT \
  http://localhost:8420/v1/eval/datasets/support-corrections/split \
  -H 'content-type: application/json' \
  -d '{"seed":"20260713","train_percent":70,"validation_percent":15}'
```

`train_percent` must be greater than zero. Train plus validation must be less
than 100, which reserves a nonzero holdout percentage. The seed is serialized
as decimal text in responses so JavaScript clients cannot round a full `u64`.

Percentages assign independent connected components, not individual rows.
Actual row percentages can therefore differ from the requested percentages,
especially when a large group or session must remain intact. Inspect
`split_counts`; do not assume that an 80/10/10 policy produced exact row
counts.

For small datasets, Kiln deterministically repairs the hash assignment when
the number of independent components permits the partition:

- one or more independent components always produce a train partition;
- two or more always produce distinct train and holdout partitions;
- three or more always produce distinct train, validation, and holdout
  partitions when `validation_percent` is non-zero.

A corpus whose rows all belong to one connected component cannot be honestly
split. Reading an empty materialized partition for training or synthesis
fails with a request error instead of silently falling back to another
partition.

## Row and group identity

Every non-empty JSONL row receives two SHA-256 identities:

- exact identity: recursively key-sorted canonical JSON;
- normalized identity: the same structure after strings are lowercased and
  runs of whitespace are collapsed.

JSON formatting and object-key order therefore do not change exact identity.
Case-only and whitespace-only string changes do not change normalized
identity. Array order, numbers, booleans, nulls, and structural changes remain
significant.

Kiln also reads optional `group_id` and `session_id` strings from the row
root or from its `metadata`, `provenance`, or `extra` object. It constructs
transitive connected components across:

- equal normalized row identities;
- equal non-empty `group_id` values;
- equal non-empty `session_id` values.

An entire connected component receives one partition. This prevents exact or
normalized duplicates, multiple candidates from one reward group, and turns
from one declared session from leaking across train and eval partitions.
Grouping works only when producers preserve those IDs. Add stable group and
session IDs at data creation time when row relationships matter.

Component assignment is deterministic over the split schema, seed, and the
component's stable normalized identity. Repeating the same corpus and config
produces the same assignment. Changing row content, group/session links, or
split config can change assignments and therefore changes the persisted
split-manifest hash.

## On-disk artifacts

The default root is `<adapter_dir>/.eval/datasets/<name>/`; `eval.eval_dir`
can relocate the `.eval` root. Each registered dataset contains:

| Artifact | Purpose |
| --- | --- |
| `data.jsonl` | Original registered corpus and source of truth. |
| `manifest.json` | Format, timestamps, identities, split config, counts, and statistics. |
| `identity.json` | Exact and normalized identity for every source row. |
| `split.json` | Typed, content-bound assignment manifest. |
| `train.jsonl` | Derived rows assigned to train. |
| `validation.jsonl` | Derived rows assigned to validation. |
| `holdout.jsonl` | Derived rows assigned to holdout. |

Do not edit derived artifacts. Upload, replacement, append, removal, and split
updates rebuild them from `data.jsonl`. Loading a pre-split manifest from an
older Kiln version also rebuilds the missing identity and split artifacts on
first access while preserving the original dataset content and split config
where available.

## Train a named partition

Registered SFT datasets must have format `sft_chat`; registered GRPO datasets
must have format `grpo_groups`. Both requests accept `dataset_split` only with
a registered `dataset`. It defaults to `train` when omitted.

SFT:

```bash
curl -sS -X POST http://localhost:8420/v1/train/sft \
  -H 'content-type: application/json' \
  -d '{
    "dataset": "support-corrections",
    "dataset_split": "train",
    "config": {
      "training_profile": "native_online_lora_v1",
      "output_name": "support-v2",
      "epochs": 3
    }
  }'
```

GRPO:

```bash
curl -sS -X POST http://localhost:8420/v1/train/grpo \
  -H 'content-type: application/json' \
  -d '{
    "dataset": "support-preferences",
    "dataset_split": "train",
    "config": {
      "output_name": "support-reward-v2",
      "behavior_policy": "current"
    }
  }'
```

The API permits an explicit non-empty `validation` or `holdout` partition for
deliberate experiments. The browser always selects `train`. Inline data,
`dataset_path`, and the virtual `corrections:active` source cannot set
`dataset_split`, because they are not bound to a registered split manifest.

SFT and streamed GRPO perform exact corpus preparation before queue
publication. A request does not become a queued job and then discover that
its selected partition is empty, invalid, or contaminated.

## Synthesize a held-out suite

Preview and synthesis accept `source_split`; both default to `holdout`.
State it explicitly in automation so the policy is visible in review:

```bash
curl -sS -X POST \
  http://localhost:8420/v1/eval/datasets/support-corrections/synthesize/preview \
  -H 'content-type: application/json' \
  -d '{
    "suite_name": "support-holdout",
    "source_split": "holdout",
    "strategy": "final_assistant",
    "head_n": 5
  }'

curl -sS -X POST \
  http://localhost:8420/v1/eval/datasets/support-corrections/synthesize \
  -H 'content-type: application/json' \
  -d '{
    "suite_name": "support-holdout",
    "source_split": "holdout",
    "strategy": "final_assistant",
    "scorer": {"kind":"auto_detect"},
    "force": true
  }'
```

Every synthesized example carries private `kiln_dataset_provenance` metadata
with the source dataset, selected partition, exact source-row identity,
normalized identity, and any group/session IDs. That metadata lets later
training admission detect source-level overlap even when synthesis transforms
one conversation into a different prompt/target shape.

Selecting `source_split: "train"` is allowed for diagnostics. It does not
make those examples held out. Link such a suite to training only with the
explicit diagnostic label described below.

## Post-training evaluation policy

`post_eval.data_scope` has two values:

| Value | Contract |
| --- | --- |
| `held-out` | Default. Admission rejects detected overlap with the exact training corpus. It may set `min_accuracy`. |
| `train-set-eval` | Explicit diagnostic. Overlap is allowed. It cannot set `min_accuracy`. |

A normal held-out gate looks like:

```json
{
  "dataset": "support-corrections",
  "dataset_split": "train",
  "config": {"output_name":"support-v2"},
  "post_eval": {
    "suite": "support-holdout",
    "data_scope": "held-out",
    "include_baseline": true,
    "min_accuracy": 0.80
  }
}
```

The omitted `data_scope` is also `held-out`. Kiln loads the registered suite
and rejects the training submission before queue publication when it finds:

- an exact prompt/target match;
- a prompt/target match after conservative case/whitespace normalization;
- an exact or normalized source-row identity match;
- a shared source `group_id`;
- a shared source `session_id`.

Prompt matching covers canonical assistant targets and conservative variants
with a system message removed or trailing tool-result messages removed. That
catches common differences between SFT/GRPO training rows and synthesized
eval prompts without treating arbitrary semantic similarity as equality.

For an intentional overfit or memorization diagnostic:

```json
{
  "post_eval": {
    "suite": "support-train-diagnostic",
    "data_scope": "train-set-eval",
    "include_baseline": true
  }
}
```

`train-set-eval` plus `min_accuracy` is rejected. Its results are descriptive
and cannot drive Kiln's accuracy promotion/demotion gate. Adapter `auto_load`
is a separate training setting; diagnostic labeling does not change that
setting.

## Training-data provenance

Status, archive, and rich job-detail responses expose the immutable
`training_data` admitted for a job:

```json
{
  "source": "named_dataset",
  "dataset": "support-corrections",
  "split": "train",
  "dataset_corpus_sha256": "sha256:...",
  "split_manifest_sha256": "sha256:...",
  "admitted_corpus_sha256": "sha256:...",
  "rows": 812
}
```

| Field | Interpretation |
| --- | --- |
| `source` | `inline`, `dataset_path`, `named_dataset`, or a virtual source such as `corrections`. |
| `dataset` | Registered dataset name, when applicable. |
| `split` | Selected persisted partition, when applicable. |
| `dataset_corpus_sha256` | Identity of the complete registered dataset at admission. |
| `split_manifest_sha256` | Identity of its exact split policy and row assignment at admission. |
| `admitted_corpus_sha256` | Identity of the exact examples/groups that passed admission. |
| `rows` | Number of admitted SFT examples or GRPO groups. |

Fetch it from `GET /v1/train/status/{job_id}` or the richer
`GET /v1/train/jobs/{job_id}` response. Once a full status is available, the
dashboard shows dataset, partition, and admitted count on the job card and all
hashes in the job drill-down. Later dataset edits or resplitting do not
rewrite an archived job's recorded provenance.

## Failure modes

| Error | Resolution |
| --- | --- |
| Selected split is empty | Upload enough independent components or change the split config; do not reuse another partition implicitly. |
| Dataset format does not match route | Use `sft_chat` for SFT and `grpo_groups` for GRPO. |
| `dataset_split` used with inline/path data | Register the dataset first, or omit `dataset_split`. |
| Held-out suite overlaps via prompt/target | Use a genuinely disjoint suite; use `train-set-eval` only when training-data measurement is intentional. |
| Held-out suite overlaps via row/group/session | Correct the source split or producer grouping; prompt edits do not make a related source group independent. |
| Diagnostic sets `min_accuracy` | Remove the gate or replace the suite with disjoint held-out data. |
| Resplitting changes counts unexpectedly | Inspect connected groups/sessions; requested percentages apply to components, not rows. |

## What this does and does not prove

This contract prevents known training/eval leakage under exact content,
conservative normalized content, and declared source relationships. It makes
the admitted corpus and split policy auditable and stable across UI, API,
archives, and restarts.

It does not detect arbitrary paraphrases, semantically equivalent code,
unlabeled relationships, leakage from model pretraining, or data the operator
trained through another system. A suite without dataset provenance can be
checked only through its prompt/target content. `held-out` therefore means
"no overlap detected under this declared policy against this admitted Kiln
training corpus," not a universal claim that an example has never influenced
the model.

The splitter is deterministic and group-aware, not statistically stratified.
It does not guarantee balanced labels, difficulty, domains, or reward
distributions. Inspect all three partitions and use a separate external test
corpus for high-stakes claims.

Content and split identities establish data provenance. They do not, by
themselves, make model outputs reproducible across different binaries,
weights, tokenizers, adapters, devices, kernels, precision policies, or
generation settings. Use training receipts, execution provenance, exact
seeds, and eval result identities for those additional dimensions.
