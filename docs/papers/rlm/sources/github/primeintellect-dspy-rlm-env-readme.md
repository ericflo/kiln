---
source_type: "github-readme"
title: "primeintellect-dspy-rlm-env-readme"
source_url: "https://github.com/PrimeIntellect-ai/verifiers/tree/main/environments/dspy_rlm"
raw_url: "https://raw.githubusercontent.com/PrimeIntellect-ai/verifiers/main/environments/dspy_rlm/README.md"
license: "MIT"
license_url: "https://raw.githubusercontent.com/PrimeIntellect-ai/verifiers/main/LICENSE"
generated_utc: "2026-05-27T02:09:42+00:00"
---

# primeintellect-dspy-rlm-env-readme

## Corpus Note

- Source repository/page: https://github.com/PrimeIntellect-ai/verifiers/tree/main/environments/dspy_rlm
- Raw README: https://raw.githubusercontent.com/PrimeIntellect-ai/verifiers/main/environments/dspy_rlm/README.md
- License: MIT (https://raw.githubusercontent.com/PrimeIntellect-ai/verifiers/main/LICENSE)
- RLM relevance: DSPy RLM environment example for GSM8K math problems.

## Verbatim README

# dspy-rlm

<a href="https://github.com/PrimeIntellect-ai/verifiers/tree/main/environments/dspy_rlm">
<img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="Source Code">
</a>

### Overview

- **Environment ID**: `dspy-rlm`
- **Short description**: V1 Taskset/Harness example using DSPy's RLM (Recursive Language Model) module on GSM8K math problems.
- **Tags**: v1, taskset, harness, dspy, rlm, math, gsm8k

### Datasets

- **Primary dataset(s)**: `gsm8k` train (train) and test (eval) via `load_example_dataset`
- **Source links**: Uses the example loader in `verifiers.utils.data_utils`
- **Split sizes**: Configurable via taskset config; defaults to 50 train / 20 eval

### Task

- **Type**: `vf.Env` with a GSM8K `vf.Taskset` and DSPy RLM `vf.Harness`
- **Rubric overview**: Exact numeric match on answer extracted from DSPy structured output

### How it works

The taskset owns GSM8K train/eval task loading and reward logic. The harness runs an in-process DSPy RLM program, builds its LM from `state.get_endpoint_config(api="chat")`, and routes every model call through the V1 interception endpoint.

DSPy RLM requires Deno to be available in the runtime environment.

### Quickstart

Run an evaluation with default settings:

```bash
prime eval run dspy-rlm
```

Configure model and sampling:

```bash
prime eval run dspy-rlm \
  -m gpt-4.1-mini \
  -n 10 -r 3 -t 1024 -T 0.7
```

### Taskset Config

| Field                | Type  | Default | Description                    |
| -------------------- | ----- | ------- | ------------------------------ |
| `num_train_examples` | int   | `50`    | Number of training examples    |
| `num_eval_examples`  | int   | `20`    | Number of evaluation examples  |

### Metrics

| Metric   | Meaning                                                    |
| -------- | ---------------------------------------------------------- |
| `reward` | 1.0 if agent's answer matches target numerically, else 0.0 |

## Verbatim License

```text
MIT License

Copyright (c) 2026 Prime Intellect

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
