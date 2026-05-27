---
source_type: "github-readme"
title: "primeintellect-rlm-swe-v1-env-readme"
source_url: "https://github.com/PrimeIntellect-ai/verifiers/tree/main/environments/rlm_swe_v1"
raw_url: "https://raw.githubusercontent.com/PrimeIntellect-ai/verifiers/main/environments/rlm_swe_v1/README.md"
license: "MIT"
license_url: "https://raw.githubusercontent.com/PrimeIntellect-ai/verifiers/main/LICENSE"
generated_utc: "2026-05-27T01:56:32+00:00"
---

# primeintellect-rlm-swe-v1-env-readme

## Corpus Note

- Source repository/page: https://github.com/PrimeIntellect-ai/verifiers/tree/main/environments/rlm_swe_v1
- Raw README: https://raw.githubusercontent.com/PrimeIntellect-ai/verifiers/main/environments/rlm_swe_v1/README.md
- License: MIT (https://raw.githubusercontent.com/PrimeIntellect-ai/verifiers/main/LICENSE)
- RLM relevance: RLM coding environment using the R2E-Gym SWE taskset and packaged RLM harness.

## Verbatim README

# rlm-swe-v1

v1 RLM coding environment using the R2E-Gym SWE taskset and packaged `RLM`
harness.

```python
import verifiers as vf

env = vf.load_environment("rlm-swe-v1")
```

Tune the taskset and harness through typed v1 config objects:

```python
import verifiers as vf
from verifiers.v1.packages.harnesses import RLMConfig
from rlm_swe_v1 import RlmSweTasksetConfig, load_environment

env = load_environment(
    config=vf.EnvConfig(
        taskset=RlmSweTasksetConfig(timeout_minutes=90),
        harness=RLMConfig(rlm_repo_ref="main", rlm_tools=["bash", "edit"]),
    )
)
```

The taskset is fully implemented in this environment package on the v1 stack.
It loads the full `R2E-Gym/R2E-Gym-Subset` train split by default, converts each
row into a v1 task, creates the per-instance sandbox config from the dataset
image, stages hidden tests for scoring, runs `run_tests.sh`, and parses pytest
output for reward.

`RLM` owns the CLI program, intercepted endpoint config, RLM installation, and
trajectory filtering. Harbor is not used here because the R2E setup is dataset
and image backed rather than a Harbor task directory corpus.

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
