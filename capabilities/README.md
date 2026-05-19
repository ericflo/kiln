# `capabilities/` — capability uplift workdir

One subdirectory per capability the team has run an experiment session on,
grouped by training paradigm:

```
capabilities/
├── opd/             # On-policy distillation (kiln OPD trainer + remote teacher)
│   ├── code-fence-language-fidelity/
│   ├── code-symbol-extraction/
│   ├── diff-patch-fluency/
│   ├── faithful-code-summarization/
│   ├── tool-call-arg-fidelity/
│   └── transcript-compaction/
├── sft/             # Supervised fine-tuning (kiln SFT trainer)
│   ├── json-schema-adherence/
│   ├── math-broad/
│   └── python-algo/
└── agentic-grpo/    # Multi-turn agentic GRPO (kiln + pi, ECHO-on by default)
    ├── lib/                          # shared pi-trajectory parsing
    ├── pi-compaction/                # paper-track caps
    ├── pi-doctest/
    ├── pi-script-fixup/
    ├── pi-terminal-bench-lite/
    ├── pi-precondition-check/        # coding-agent caps (10-cap suite)
    ├── pi-code-search/
    ├── pi-code-comprehension/
    ├── pi-diff-patch-apply/
    ├── pi-tool-call-efficiency/
    ├── pi-shell-hygiene/
    ├── pi-test-interpretation/
    ├── pi-failure-triage/
    ├── pi-source-mod-workflow/
    └── pi-faithful-completion/
```

Each capability dir is the output of a single skill session
(`opd-capability-creator`, `sft-capability-creator`, or agentic-GRPO
scaffolds). For agentic-GRPO, see
`capabilities/agentic-grpo/README.md` for the bucket's shared layout,
ECHO defaults, and standard workflow. For OPD/SFT, the corresponding
`SKILL.md` is authoritative.

## Reproducibility — no adapters in git

Adapters are **derived artefacts**. Given:

- the kiln commit SHA (recorded in each `capability.jsonl` row)
- the base model identifier
- the teacher model identifier + version
- the training prompts file (committed)
- the hyperparameters + seed (recorded in each row)
- the rubric (`rubric.py`, committed)

…the adapter is reproducible by re-running training. Re-checking out the
exact commit and re-running `run_iter<N>.sh` should yield a
byte-equivalent adapter (modulo teacher non-determinism — see below).

We therefore do **not** commit adapter binaries. They live under
`<cap>/adapters/` (gitignored) and `/tmp/opd-*/` for the running trainer.
If you need an adapter and don't have it locally, re-run the manifest's
`reproduce.sh`.

### Teacher reproducibility note

The remote teacher (vLLM) is the only source of non-determinism that
isn't fully captured by the manifest. Two paths:

1. **Pin the teacher and live with it.** The manifest records vLLM
   version, model identifier (e.g. `qwen3.6-27b-awq`), the quantisation,
   and the launch flags. If you spin up the same teacher with the same
   flags, you get the same top-K logprobs to within Q4 numerical noise.
   Good enough for adapter quality, not for bit-identical reproduction.

2. **Save teacher logprobs as a fixture.** Future work: dump
   `(rollout_tokens, active_positions, teacher_top_k_logprobs)` per step
   to a JSONL during training, and use `FixtureLogitSource` to replay
   offline. With the fixture in hand, training is bit-deterministic.
   Tracked in the `kiln-polish.jsonl` of capability dirs that need it.

## What's committed vs. ignored

See `capabilities/.gitignore` for the full list. In short: the *contract*
(rubric, prompts, hypotheses, log, scripts, manifest) is committed; the
*output* (adapter, responses, transient training logs, eval set) is not.

The eval set itself is gitignored because the skill's information
firewall (SKILL.md §1) requires the agent not to read it. Treating
`datasets/eval.jsonl` as a write-only artefact prevents accidental
inclusion in a diff review.
