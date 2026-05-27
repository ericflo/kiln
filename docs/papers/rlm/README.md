# Recursive Language Models Research Guide

This directory is a working corpus for understanding Recursive Language Models
(RLMs), their direct follow-ups, and the adjacent long-context methods that
frame the design space. Start here, then use [SOURCE_MANIFEST.md](SOURCE_MANIFEST.md)
to jump into the mirrored papers, implementation READMEs, and metadata-only
source notes.

## The Core Idea

An RLM is an inference scaffold that changes where the long prompt lives. A
normal LLM call pushes the prompt into the model context window. An RLM stores
the prompt as data in an external environment, usually a REPL, and gives the
model code-level ways to inspect pieces of that data, transform it, call the
model recursively on slices, and aggregate results into a final answer.

The central interface is still language-model shaped: `completion(prompt) ->
response`. Internally, though, the prompt is no longer consumed as one giant
token sequence. The root model sees metadata about the external prompt and
environment, writes code, observes bounded outputs, and decides whether to
inspect, split, recurse, or finish. The canonical paper uses a Python REPL,
stores the prompt as a variable, exposes sub-call functions, and returns the
answer when a final variable is set.

This makes RLMs different from three nearby ideas:

- RLMs are not just RAG. Retrieval selects snippets from a corpus; RLMs can
  programmatically traverse the original prompt and choose how much semantic
  work to spend over arbitrary parts of it.
- RLMs are not just summarization or compaction. The source remains externally
  addressable, so the model can revisit details instead of trusting a lossy
  summary.
- RLMs are not just generic subagents. The important move is programmatic
  recursion over environment-resident data, not merely asking another agent in
  natural language.

## Latest Direct RLM Papers

These are the direct RLM papers and RLM-labeled variants found in this pass,
current as of 2026-05-27.

| Date | Source | What It Adds |
| --- | --- | --- |
| 2026-05-13 online | [Consolidation-Expansion Operator Mechanics](sources/arxiv/2605.09968-opmech-adaptive-learning.md) | Theoretical stopping/control signal for recursive workflows via an order-gap measure. |
| 2026-05-11 latest version | [Recursive Language Models](sources/arxiv/2512.24601-recursive-language-models.md) | Canonical RLM paper; v3 of arXiv:2512.24601. |
| 2026-05-07 | [Architecture Matters: Comparing RAG Systems under Knowledge Base Poisoning](sources/arxiv/2605.05632-rag-poisoning-architectures.md) | Evaluates RLMs as a RAG-style architecture under adversarial poisoning. |
| 2026-02-14 / 2026-05 arXiv id | [LCM: Lossless Context Management](sources/arxiv/2605.04050-lossless-context-management.md) | Deterministic extension of the recursive context-management paradigm. |
| 2026-03-20 | [The Y-Combinator for LLMs](sources/arxiv/2603.20105-lambda-rlm.md) | lambda-RLM: typed functional runtime instead of open-ended REPL code generation. |
| 2026-03-07 | [Recursive Language Models Meet Uncertainty](sources/arxiv/2603.15653-self-reflective-program-search.md) | SRLM: uncertainty-aware self-reflective program search for choosing context-interaction programs. |
| 2026-03-03 | [Think, But Don't Overthink](sources/arxiv/2603.02615-reproducing-rlms.md) | Reproduction and depth ablation; metadata only because the paper is not CC BY. |
| 2026-02-18 | [RLMs for Jailbreak Detection](sources/arxiv/2602.16520-rlm-jailbreak-detection.md) | RLM-JB applies recursive chunk analysis and evidence aggregation to jailbreak detection. |

## What The Evidence Says

The original paper reports that RLMs can process prompts beyond a normal model's
context limit and can outperform vanilla long-context calls, context compaction,
CodeAct-style scaffolds, and coding-agent baselines on long-context tasks. Its
small post-training run, RLM-Qwen3-8B, is especially relevant for this repository:
it suggests that a small open model can be trained to use the scaffold rather
than merely being prompted to tolerate it.

The first follow-ups make the picture more nuanced:

- [Think, But Don't Overthink](sources/arxiv/2603.02615-reproducing-rlms.md)
  argues that recursion depth is not free. Depth 1 can help complex aggregation,
  but deeper recursion can degrade accuracy and sharply increase cost/latency.
- [SRLM](sources/arxiv/2603.15653-self-reflective-program-search.md) argues
  that program selection is a key unresolved variable. It uses self-consistency,
  reasoning length, and verbalized confidence to select programs, and reports
  gains over RLM under the same time budget.
- [lambda-RLM](sources/arxiv/2603.20105-lambda-rlm.md) turns the open-ended
  REPL loop into typed functional combinators such as split, map, filter, and
  reduce. Its main claim is that structured control can improve reliability,
  latency, and analyzability.
- [LCM](sources/arxiv/2605.04050-lossless-context-management.md) pushes even
  more control into the engine: recursive compression and task partitioning are
  deterministic mechanisms rather than model-written loops.
- [OpMech](sources/arxiv/2605.09968-opmech-adaptive-learning.md) points at a
  likely missing primitive for production RLMs: principled stopping rules.

Read together, the current consensus is not "make recursion deeper." It is:
externalize context, preserve source addressability, make decomposition cheap
and auditable, train the model to use the interface, and tightly control when
recursion is worth the cost.

## Implementation Sources

Use these before writing code:

- [Official RLM README](sources/github/alexzhang13-rlm-readme.md): full
  framework, supported environments, provider adapters, and logging hooks.
- [RLM minimal README](sources/github/alexzhang13-rlm-minimal-readme.md):
  easiest place to understand the basic REPL plus sub-call loop.
- [lambda-RLM README](sources/github/lambda-calculus-llm-lambda-rlm-readme.md):
  typed-control alternative and benchmark harness.
- [Prime Verifiers README](sources/github/primeintellect-verifiers-readme.md):
  RL/eval environment framework that includes RLM harnesses and RLMEnv work.
- [Prime RLM SWE environment](sources/github/primeintellect-rlm-swe-v1-env-readme.md):
  example of packaging an RLM harness for software-engineering tasks.

The source manifest also includes metadata-only notes for the Alex Zhang blog,
Prime Intellect blog, and alphaXiv RL post. Those pages are not mirrored
verbatim because no clear reuse license was found during this pass.

## Adjacent Work

RLM sits in a larger context-management ecosystem. The most relevant clusters:

- Context folding: [Scaling Long-Horizon LLM Agent via Context-Folding](sources/arxiv/2510.11967-context-folding.md),
  [AgentFold](sources/arxiv/2510.24699-agentfold.md), and
  [U-Fold](sources/arxiv/2601.18285-u-fold.md) study branch/return or proactive
  folding for agents. These are closest to RLM at the agent-control level.
- Compression and context engineering: [ACON](sources/arxiv/2510.00615-acon.md),
  [PAACE](sources/arxiv/2512.16970-paace.md), [Active Context Compression](sources/arxiv/2601.07190-active-context-compression.md),
  and [Agentic Context Engineering](sources/arxiv/2510.04618-agentic-context-engineering.md)
  are the strongest foils for the RLM claim that source-preserving external
  context beats repeated lossy compaction on dense tasks.
- Memory systems: [MEM1](sources/arxiv/2506.15841-mem1.md), [LightMem](sources/arxiv/2510.18866-lightmem.md),
  [Mem0](sources/arxiv/2504.19413-mem0.md), and the
  [gist-memory reading agent](sources/arxiv/2402.09727-gist-memory-reading-agent.md)
  offer persistent or staged memory alternatives.
- Architecture-side long context: [Infini-attention](sources/arxiv/2404.07143-infini-attention.md),
  [HMT](sources/arxiv/2405.06067-hierarchical-memory-transformer.md),
  [Titans](sources/arxiv/2501.00663-titans.md), [ATLAS](sources/arxiv/2505.23735-atlas.md),
  and [LM2](sources/arxiv/2502.06049-large-memory-models.md) attack the problem
  inside the model rather than through an inference scaffold.
- Evaluation: [RULER](sources/arxiv/2404.06654-ruler.md) and
  [OOLONG](sources/arxiv/2511.02817-oolong.md) are required background for
  interpreting RLM long-context claims.

## Practical Design Notes

An RLM implementation needs more than a prompt template.

- Environment: store the full prompt and intermediate artifacts outside the
  model context; expose bounded inspection, slicing, search, and persistence.
- Control loop: cap root iterations, recursion depth, stdout size, sub-call
  fanout, wall time, and total tokens.
- Sub-call API: make child calls explicit and logged; distinguish plain LLM
  subcalls from recursive RLM subcalls.
- Safety: use an isolated REPL or narrow DSL for untrusted tasks. The official
  implementation supports local, IPython, Docker, Modal, Prime, Daytona, and
  E2B-style environments.
- Training data: capture trajectories, not just final answers. Parent actions,
  child prompts, snippets inspected, code executed, stdout, costs, and final
  answer quality are all learning signals.
- Evals: include simple retrieval tasks, dense aggregation tasks, codebase QA,
  adversarial/poisoned inputs, and latency/cost budgets. RLMs can hurt simple
  retrieval, so a good eval suite must include both easy and dense tasks.

For a small-model training effort, the most promising route is to treat RLM as
an agentic interface and train the model to choose good environment operations:
when to inspect, when to split, when to recurse, when to aggregate, and when to
stop. The follow-up literature suggests that these control decisions matter as
much as the recursive machinery itself.

## Suggested Reading Order

1. Read [Recursive Language Models](sources/arxiv/2512.24601-recursive-language-models.md)
   through Sections 1-2 and skim the experiments.
2. Read [RLM minimal](sources/github/alexzhang13-rlm-minimal-readme.md) to map
   the paper idea to code.
3. Read [Think, But Don't Overthink](sources/arxiv/2603.02615-reproducing-rlms.md)
   and [SRLM](sources/arxiv/2603.15653-self-reflective-program-search.md) to
   understand failure modes and program-selection issues.
4. Read [lambda-RLM](sources/arxiv/2603.20105-lambda-rlm.md),
   [LCM](sources/arxiv/2605.04050-lossless-context-management.md), and
   [OpMech](sources/arxiv/2605.09968-opmech-adaptive-learning.md) for structured
   control, deterministic recursion, and stopping rules.
5. Read [Prime Verifiers](sources/github/primeintellect-verifiers-readme.md) and
   the Prime source notes for RL/evaluation packaging ideas.
6. Use the adjacent-work cluster only after the core RLM mechanism is clear.

## Corpus Generation

The corpus was generated with:

```bash
uv run --with requests --with beautifulsoup4 --with markdownify \
  python docs/papers/rlm/tools/fetch_rlm_sources.py
```

Full-text mirroring is intentionally conservative. CC BY 4.0 arXiv HTML and
MIT-licensed READMEs are mirrored as markdown with attribution. Non-CC-BY arXiv
records, web articles, and blogs are indexed with metadata and relevance notes
instead of full-text copies.
