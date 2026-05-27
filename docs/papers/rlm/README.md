# Recursive Language Models: Self-Contained Research Brief

This README is meant to be enough for an agent to understand Recursive Language
Models (RLMs) without reading the full corpus first. The mirrored papers,
READMEs, and metadata notes remain under `sources/` for verification and depth;
use [SOURCE_MANIFEST.md](SOURCE_MANIFEST.md) as the complete source index.

## Short Version

An RLM is an inference-time scaffold around a language model `M`. Instead of
putting a long prompt directly into `M`'s context window, an RLM stores the
prompt as an external object in an environment `E`, usually a REPL. The model
receives compact metadata about `E`, writes code to inspect and transform the
external prompt, invokes model sub-calls on slices or derived prompts, stores
intermediate results as variables/files, and returns the final value when done.

The core move is:

```text
ordinary LLM:  y = M(P)

RLM:           E.prompt = P
               y = loop(M, E, metadata(E), sub_model_call)
```

The claim is not merely "use tools" or "spawn agents." The distinctive pieces
are:

- Symbolic handle: the root model manipulates `P` by reference instead of
  ingesting all of `P`.
- Environment-resident state: intermediate strings, tables, files, and outputs
  can be larger than the model window.
- Symbolic recursion: code inside `E` can call `M` or another RLM on generated
  sub-prompts inside loops.
- Standard API: the outer interface is still `completion(prompt) -> response`.
- Inference-time scaling: effective semantic work can grow as `Omega(|P|)` or
  `Omega(|P|^2)` without stuffing that work into one context window.

The current state of the literature is more precise than "recursion is always
good." RLMs look strongest when the task requires dense access to long context
or structured decomposition. They can hurt simple retrieval tasks, over-recurse,
produce code errors, or waste compute. The follow-up papers mostly try to keep
the good part, prompt-as-environment, while controlling program selection,
termination, cost, and safety.

## Formula And Concept Index

Use this as the local cheat sheet before diving into any source paper.

| Idea | Local form | Where it is expanded |
| --- | --- | --- |
| RLM interface | `RLM_M : Sigma* -> Sigma*` with `RLM_M(P) = Y` | [Formal Model](#formal-model) |
| Prompt externalization | `E.prompt = P`; root sees metadata/snippets, not all of `P` | [Canonical Algorithm](#canonical-algorithm) |
| Root loop | `code = M(history)` then `E, stdout = REPL(E, code)` | [Canonical Algorithm](#canonical-algorithm) |
| Dense-work scaling | semantic work can be `O(N)` or `O(N^2)` over environment slices | [Task Complexity](#task-complexity) |
| SRLM self-consistency | `prob(a) = (1 / K) * sum 1[out(p^(k)) = a]` | [SRLM](#srlm-program-search-instead-of-just-recursion) |
| SRLM confidence | `VC(p^(k)) = sum log(nu_t^(k) / 100)` | [SRLM](#srlm-program-search-instead-of-just-recursion) |
| lambda-RLM depth | `d = ceil(log_{k*}(n / tau*))` | [lambda-RLM](#lambda-rlm-typed-functional-recursion) |
| lambda-RLM call count | `N(n) = (k*)^d + 1` | [lambda-RLM](#lambda-rlm-typed-functional-recursion) |
| lambda-RLM cost | `T(n) = k* T(n / k*) + C_oplus(k*)` | [lambda-RLM](#lambda-rlm-typed-functional-recursion) |
| OpMech order-gap | `Omega(theta; e) = || Q(P_e(theta)) - P_e(Q(theta)) ||` | [OpMech](#opmech-a-stopping-rule-theory) |
| Practical build | inspect/slice/search, subcall/batch-subcall, write final | [Implementation Blueprint](#implementation-blueprint) |

## Source Map

The most important local files are:

- Core paper: [Recursive Language Models](sources/arxiv/2512.24601-recursive-language-models.md)
- Official code: [alexzhang13/rlm README](sources/github/alexzhang13-rlm-readme.md)
- Minimal code: [alexzhang13/rlm-minimal README](sources/github/alexzhang13-rlm-minimal-readme.md)
- RLM-Qwen3-8B card: [metadata note](sources/web/huggingface-rlm-qwen3-8b.md)
- SRLM: [Self-Reflective Program Search](sources/arxiv/2603.15653-self-reflective-program-search.md)
- lambda-RLM: [Y-Combinator for LLMs](sources/arxiv/2603.20105-lambda-rlm.md)
- LCM: [Lossless Context Management](sources/arxiv/2605.04050-lossless-context-management.md)
- OpMech: [Consolidation-Expansion Operator Mechanics](sources/arxiv/2605.09968-opmech-adaptive-learning.md)
- RLM-JB: [jailbreak detection](sources/arxiv/2602.16520-rlm-jailbreak-detection.md)
- Reproduction: [Think, But Don't Overthink](sources/arxiv/2603.02615-reproducing-rlms.md)
- Poisoning robustness: [RAG architectures under poisoning](sources/arxiv/2605.05632-rag-poisoning-architectures.md)
- Adjacent recursion: [THREAD](sources/arxiv/2405.17402-thread-recursive-spawning.md)
- Compaction baseline: [ReSum](sources/arxiv/2509.13313-resum.md)
- Long-context evals: [RULER](sources/arxiv/2404.06654-ruler.md),
  [OOLONG](sources/arxiv/2511.02817-oolong.md),
  [LongCoT](sources/arxiv/2604.14140-longcot.md), and
  [LongMemEval](sources/arxiv/2410.10813-longmemeval.md)

## Formal Model

Let:

- `M` be a base neural language model with maximum context size `K`.
- `Sigma*` be the set of finite strings.
- `P in Sigma*` be an arbitrary-length prompt, possibly `|P| >> K`.
- `E` be a persistent external environment.
- `Y in Sigma*` be the final response.

An RLM is a scaffold around `M`:

```text
RLM_M : Sigma* -> Sigma*
RLM_M(P) = Y
```

but it computes `Y` through environment interaction:

```text
E.prompt = P
E.sub_call = sub_RLM_or_sub_LM
history = [metadata(E)]

repeat:
    code = M(history)
    E, stdout = REPL(E, code)
    history += [code, metadata(stdout)]
until E.Final is set

return E.Final
```

The important bound is that the model does not see `P` directly. It sees small
metadata, snippets, observations, and prior code/actions. If each root turn is
trimmed to `c` tokens, a root model with window `K` has roughly `K / c` root
iterations before the root history fills. This is a practical limit, not a
fundamental one; root state can itself be externalized if needed.

The target capabilities are:

- Unbounded input: handle `|P| >> K`.
- Unbounded output: build output in environment variables/files, not one model
  emission.
- Unbounded semantic horizon: perform `Omega(|P|)` or `Omega(|P|^2)` semantic
  operations by programmatic sub-calls over chunks, lines, files, or pairs.

## Canonical Algorithm

The core paper contrasts the RLM algorithm with a weaker "agent with tools"
design. The difference is where the prompt and output live, and whether subcalls
can be launched programmatically.

```text
Algorithm: Recursive Language Model around M

Input: prompt P
Output: response Y

state = InitREPL(prompt=P)
state = AddFunction(state, sub_RLM_M)
hist = [Metadata(state)]

while true:
    code = M(hist)
    state, stdout = REPL(state, code)
    hist = hist || code || Metadata(stdout)

    if state["Final"] is set:
        return state["Final"]
```

The weak alternate scaffold usually does something like:

```text
actions = {Finish, Exec, Search, sub_LLM_M}
hist = [Metadata(actions), P]

while true:
    action, value = M(hist)
    if action == Finish:
        return value
    out = RUN(action, value)
    hist = hist || (action, value, out)
    if Tok(hist) > K:
        hist = Compact(hist)
```

The weak design fails three RLM requirements:

1. It puts `P` in `hist`, so it inherits `M`'s context limit.
2. It emits the final response directly, so output length is model-window and
   max-output constrained.
3. It has a separate `sub_LLM` action but not symbolic recursion. The model can
   ask a few verbalized subquestions, but cannot write a program that loops over
   every chunk or every pair of chunks and calls a submodel.

## What Counts As An RLM

A system is recognizably RLM-like if it has these properties:

- Prompt-as-environment: source material is addressable outside the root prompt.
- Programmatic inspection: the model can search, slice, transform, and count.
- Persistent state: partial outputs survive across root turns.
- Sub-call primitive: code can call a model on generated strings.
- Recursive closure: a sub-call can be a plain LLM call at depth 1 or another
  RLM at deeper depths.
- Bounded observations: stdout/tool output is summarized or truncated before
  returning to the root model.
- Explicit finish contract: the environment has a final variable/file/tag.

Things that are adjacent but not sufficient by themselves:

- RAG with top-k retrieval.
- A coding agent with the full prompt pasted into the initial context.
- A summarization/compaction loop.
- A multi-agent system where all subquestions are manually written in natural
  language.
- Long-context architecture changes such as Infini-attention or Titans.

## Why It Works

RLMs exploit a split between structural work and semantic work.

Structural work is cheap and deterministic:

- Split files.
- Enumerate lines.
- Build tables.
- Sort, filter, deduplicate.
- Generate chunk prompts.
- Store partial outputs.
- Run aggregation code.

Semantic work is delegated to model calls:

- Classify this item.
- Extract evidence from this chunk.
- Answer this subquestion.
- Compare these two entries.
- Summarize this bounded slice.
- Verify this candidate.

This matters because dense long-context tasks fail when one model call must
internally remember and compose too many distant facts. RLMs externalize memory
and let the model spend separate inference calls on local semantic decisions.

## Task Complexity

The original paper argues that "long context" difficulty depends on how much
work the answer requires as the prompt grows.

| Task type | Complexity in prompt length | Example | RLM relevance |
| --- | --- | --- | --- |
| Constant retrieval | `O(1)` | single needle in haystack | Base long-context models can already do well; RLM overhead can hurt. |
| Sparse multi-hop | roughly `O(k)` with small `k` | BrowseComp-style evidence across docs | RLM can programmatically search and selectively recurse. |
| Dense aggregation | `O(N)` | OOLONG: label every item then aggregate | RLM can call submodels line-by-line/chunk-by-chunk. |
| Pairwise dense reasoning | `O(N^2)` | OOLONG-Pairs | RLM can generate loops over pairs and store stitched results. |
| Long reasoning graph | graph/DAG depth and width | LongCoT-mini | RLM can delegate nodes and memoize verified answers. |

Simple retrieval is not the ideal RLM benchmark. Dense aggregation and pairwise
tasks are where prompt-as-environment and symbolic recursion matter most.

## Main Benchmark Results

The original paper evaluates GPT-5, Qwen3-Coder-480B-A35B, and Claude Opus 4.1
baselines on CodeQA, BrowseComp-Plus, OOLONG, and OOLONG-Pairs.

Key GPT-5-family numbers from the main table:

| Method | CodeQA | BrowseComp+ | OOLONG | OOLONG-Pairs |
| --- | ---: | ---: | ---: | ---: |
| GPT-5 base | 24.0 | 0.0 | 44.0 | 0.1 |
| Compaction agent | 58.0 | 70.5 | 46.0 | 0.1 |
| OpenCode + context offload | 64.0 | 94.0 | 52.0 | 4.8 |
| RLM depth 0 | 58.0 | 88.0 | 36.0 | 43.9 |
| RLM depth 1 | 62.0 | 91.3 | 56.0 | 58.0 |
| RLM depth 2 | 66.0 | 92.0 | 56.5 | 65.5 |
| RLM depth 3 | 58.0 | 92.0 | 58.0 | 76.0 |

Key Qwen3-Coder numbers:

| Method | CodeQA | BrowseComp+ | OOLONG | OOLONG-Pairs |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-Coder base | 20.0 | 0.0 | 36.0 | 0.1 |
| Compaction agent | 50.0 | 38.0 | 44.1 | 0.31 |
| OpenCode + context offload | 40.0 | 58.0 | 24.0 | 2.1 |
| RLM depth 0 | 66.0 | 46.0 | 43.5 | 17.3 |
| RLM depth 1 | 56.0 | 44.7 | 48.0 | 23.1 |
| RLM depth 2 | 54.0 | 68.0 | 26.0 | 19.0 |
| RLM depth 3 | 44.0 | 68.7 | 32.0 | 21.1 |

Interpretation:

- REPL/offloading alone matters. Depth 0 RLMs can already beat many baselines
  because the prompt is outside the context window.
- Recursive sub-calls matter most for dense semantic transformations like
  OOLONG and OOLONG-Pairs.
- Higher depth is not uniformly better. GPT-5 benefits on OOLONG-Pairs; Qwen3-
  Coder often degrades at higher depth because syntax/control errors propagate.
- Costs are often comparable to or cheaper than baselines, but outlier failed
  trajectories can be expensive.

## LongCoT Result

RLMs also help outside ordinary long-context QA. On LongCoT-mini, the paper
compares GPT-5.2 with an RLM using GPT-5.2:

| Method | Overall | MATH | CHEM | CS | LOGIC | CHESS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GPT-5.2 base | 38.7 | 26.0 | 37.0 | 40.4 | 53.6 | 36.6 |
| RLM depth 1 | 50.6 | 5.6 | 50.0 | 11.0 | 86.7 | 93.0 |
| RLM depth 1 + decomposition hints | 65.6 | 32.0 | 52.0 | 46.0 | 99.0 | 99.0 |

The lesson is that the REPL is not only a long-input trick. It can also serve as
a graph/DAG workspace for decomposing long reasoning chains, memoizing verified
subanswers, and assembling the final result by lookup rather than by fragile
single-chain reasoning.

## Training Native RLMs

The RLM paper includes a small but important training result:

- Teacher/scaffold: RLM(Qwen3-Coder-480B-A35B) with Qwen3-8B subcalls.
- Collection: 750 English LongBenchPro tasks, 2,250 candidate trajectories.
- Filtering:
  - Remove zero-score trajectories.
  - Remove one-turn trajectories.
  - Keep 1,072 candidate trajectories after initial filtering.
  - Split each root RLM turn into an SFT sample.
  - Remove samples exceeding the Qwen3-8B context budget.
  - Programmatically patch common final-answer template mistakes.
- Common template mistakes found:
  - 16% of turns incorrectly used final-answer wrappers.
  - 13% incorrectly treated a `FINAL_VAR`-style variable as a direct answer.
- Training:
  - Qwen3-8B fine-tuned for 300 steps.
  - Batch size 64.
  - 48 H100-hours.
  - `prime-rl` used for fine-tuning.
- Result:
  - RLM-Qwen3-8B improved over base Qwen3-8B as an RLM by a median 28.3%
    across the four main evaluation tasks.
  - It was more than 3x faster in RLM trajectories due to better REPL decisions.

The key training hypothesis is that leaf sub-calls are mostly ordinary LLM
requests; the hard thing to train is the root model's ability to operate the
environment, choose decomposition, call children at the right moments, and stop.

A second RLVR experiment trained Qwen3-4B-Instruct-0527 as an RLM on MRCRv2:

- Train split: 32k-64k tokens, 2 needles.
- Eval split: 512k-1M tokens, 8 needles.
- 150 RL steps.
- Batch size 128.
- 4 rollouts per example.
- Max output tokens per turn 4096.
- Max RLM iterations 20.

The reported lesson is length generalization: a model trained to operate the RLM
interface at shorter lengths can generalize to longer, harder RLM settings.

## Failure Modes And Negative Results

RLMs add a control layer, and that layer can fail.

- Simple retrieval overhead: on S-NIAH-like tasks, RLMs can be slower and less
  accurate than just asking a strong base model.
- Over-recursion: some models call the submodel for everything, producing
  thousands of calls on basic tasks.
- Syntax/runtime errors: code-writing ability is now part of reasoning ability.
  Qwen3-Coder had more syntax errors than GPT-5 in the original analysis.
- Error propagation: if sub-RLMs have their own REPL loops, parent mistakes can
  become recursive child mistakes.
- Output-token failures: thinking models can exhaust output budget before they
  produce usable code or final values.
- Blocking subcalls: sequential subcalls make naive RLM implementations slow.
  Parallel/asynchronous subcalls are a major practical improvement.
- Brittle final-answer tags: `FINAL(...)` and `FINAL_VAR(...)` are easy for
  prompted models to misuse; native training or typed tool contracts should
  reduce this.
- Prompt/model mismatch: the same RLM system prompt did not transfer cleanly
  across GPT-5, Qwen3-Coder, and Qwen3-8B.
- Sandbox risk: a REPL controlled by a model needs isolation for untrusted
  inputs and production use.

## SRLM: Program Search Instead Of Just Recursion

SRLM asks whether the important thing is recursion itself, or choosing a good
context-interaction program under uncertainty.

Formal setup:

- Query: `q`
- Long context: `C = (c_1, ..., c_N)`, `N >> L`, where `L` is model window.
- Program trajectory: `p = (p_1, ..., p_T)`.
- Execution state:

```text
e_t = Exec(p_t, e_{t-1}, C),  e_0 = empty
out(p) in answer space A
```

SRLM samples `K` candidate programs:

```text
p^(k) ~ pi_theta(. | q, C),  k = 1..K
```

It then scores/filters candidates with three uncertainty signals:

1. Self-consistency over final answers:

```text
prob(a) = (1 / K) * sum_{k=1..K} 1[out(p^(k)) = a]
a_hat = argmax_a prob(a)
S = {p^(k) : out(p^(k)) = a_hat}
```

2. Verbalized confidence:

```text
VC(p^(k)) = sum_{t=1..T_k} log(nu_t^(k) / 100) <= 0
```

where each step asks the model for a confidence `nu_t in (0, 100]`.

3. Reasoning trace length:

```text
RL(p^(k)) = T_k
```

The high-level point: select a program trajectory, not merely an answer.

Reported SRLM claims:

- Up to 22% improvement over RLM under the same wall-clock budget.
- Recursion is not always the primary performance driver.
- RLM can degrade within the model's native window, while SRLM is more robust.
- Semantically intensive tasks need richer trajectory selection than heuristic
  recursive program search.

## lambda-RLM: Typed Functional Recursion

lambda-RLM keeps prompt-as-environment but removes arbitrary model-written
control code. Instead, the runtime exposes a typed functional library:

```text
Split, Map, Filter, Reduce, Concat, Cross, Peek
```

The model is used mainly at bounded leaves. A deterministic planner chooses the
task type, branching factor, threshold, and composition operator.

Important variables:

- `n = |P|`: prompt length.
- `K`: base model window.
- `k*`: chosen branching factor.
- `tau*`: leaf threshold.
- `d`: recursion depth.
- `oplus`: composition operator.
- `Phi`: registered recursive executor.

Depth formula:

```text
d = ceil(log_{k*}(n / tau*))
```

Leaf call count:

```text
N(n) = (k*)^d + 1
```

The `+1` is the task-detection call.

Cost recurrence:

```text
T(n) = k* T(n / k*) + C_oplus(k*)
```

Unrolled at depth `d`:

```text
T(n) = (k*)^d T(n / (k*)^d)
       + [((k*)^d - 1) / (k* - 1)] C_oplus(k*)
```

With `d = ceil(log_{k*}(n / tau*))`, leaves fit under `tau*`.

Core executor:

```text
Phi(P):
    if |P| <= tau*:
        q = Template[task_type].format(P)
        return sub_M(q)

    chunks = Split(P, k*)

    if plan includes Filter:
        previews = Map(lambda p: Peek(p, 0, floor(tau*/10)), chunks)
        chunks = Filter(Relevant, Zip(chunks, previews))

    results = Map(lambda p_i: Phi(p_i), chunks)
    return Reduce(oplus, results)
```

Reported lambda-RLM claims:

- Wins 29 of 36 model-task comparisons against standard RLM.
- Improves average accuracy by up to +21.9 points.
- Reduces latency by up to 4.1x.
- Provides termination by construction if recursive calls strictly reduce input
  rank.
- Makes cost and call structure auditable before execution.

The tradeoff is flexibility. Standard RLM lets the model invent arbitrary code;
lambda-RLM restricts it to trusted combinators.

## LCM: Engine-Managed Recursive Context

Lossless Context Management (LCM) moves even more control out of the model and
into the engine. It treats RLM as one end of a spectrum:

```text
RLM: model writes the loops and memory strategy.
LCM: engine provides deterministic memory and recursion operators.
```

LCM has two main mechanisms:

- Recursive context compression: a hierarchical summary DAG stores compact
  summary nodes while retaining pointers to every original message or artifact.
- Recursive task partitioning: engine-managed `LLM-Map` and `Agentic-Map`
  replace model-written loops.

Operator-level recursion:

- `LLM-Map`: apply a prompt to every item as a stateless LLM call.
- `Agentic-Map`: spawn a full tool-using sub-agent per item.
- Inputs/outputs are files, usually JSONL, outside the active context.
- Outputs are schema-validated and retried on type errors.
- The engine handles iteration, concurrency, retries, locks, and status.

Delegation termination uses a scope-reduction invariant:

- A sub-agent must declare delegated scope and retained work.
- If it delegates all responsibility and retains nothing, the engine rejects it.
- Each nested delegation must reduce responsibility, so recursion bottoms out.

LCM's conceptual claim is that RLM-style symbolic recursion is like unrestricted
`goto`: maximally expressive but hard to reason about. Engine-managed operators
are like structured control flow: less flexible, more reliable.

## OpMech: A Stopping Rule Theory

Consolidation-Expansion Operator Mechanics (OpMech) is broader than RLM, but it
adds a useful formal lens for recursive systems: stop when new evidence no
longer changes the settled state.

Core objects:

- Knowledge state: `theta in X`
- Evidence/event: `e in E`
- Expansion operator: `P_e : X -> X`
- Consolidation operator: `Q : X -> X`

The order-gap is:

```text
Omega(theta; e) = || Q(P_e(theta)) - P_e(Q(theta)) ||
```

Interpretation:

- First path: incorporate evidence, then consolidate.
- Second path: consolidate, then incorporate evidence.
- If order matters a lot, the system is still sensitive to evidence ordering.
- If the gap is small and stable, more processing is less likely to change the
  outcome.

Decision-facing form with decision map `pi`:

```text
Omega^pi(theta; e) =
    d_Z(pi(Q(P_e(theta))), pi(P_e(Q(theta))))
```

For RLMs, this suggests a replacement for fixed recursion budgets:

- Represent the aggregated RLM state over chunks as `S`.
- Embed state with `phi(G)`.
- Track whether adding/processing more chunks changes the consolidated state.
- Stop when the order-gap remains below a threshold under noise assumptions.

The practical takeaway: production RLMs need convergence signals, not only
`max_depth`, `max_iter`, and token budgets.

## Security And Robustness Results

RLM-JB applies the RLM pattern to jailbreak detection:

- Root model orchestrates a bounded analysis procedure.
- Input is normalized/de-obfuscated.
- Text is chunked to guarantee coverage and avoid context dilution.
- Worker models screen chunks in parallel.
- Root aggregates cross-chunk evidence into an auditable decision.

Reported abstract-level result:

- AutoDAN-style adversarial inputs.
- Three LLM backends.
- Attack success/recall detection effectiveness 92.5-98.0%.
- Precision 98.99-100%.
- False positive rate 0.0-2.0%.

The poisoning-architecture paper evaluates RLMs as one RAG-style architecture
under single-document knowledge-base poisoning:

- Dataset: 921 Natural Questions QA pairs.
- Architectures: vanilla RAG, agentic RAG, MADAM-RAG, RLM.
- Attack: CorruptRAG-AK, targeting credibility assessment.
- Clean accuracy for vanilla/agentic/RLM around 92%.
- Attack success under CorruptRAG-AK ranges from 81.9% for vanilla RAG to 24.4%
  for RLM, a nearly 58 point spread.

The lesson is architectural: how evidence is decomposed, cross-checked, and
aggregated can be more important than the backbone model alone.

## Relation To Adjacent Work

| Family | Examples | Relation to RLM |
| --- | --- | --- |
| Context folding | Context-Folding, AgentFold, U-Fold | Agent trajectories branch/return or fold context; closest conceptual neighbor. |
| Compaction | ReSum, ACON, PAACE, Active Context Compression | Compress histories; RLM tries to keep source addressable instead of trusting lossy summaries. |
| Memory systems | Mem0, LightMem, MEM1, LongMemEval | Persistent memory/retrieval; RLM is more like an active workspace over a provided prompt. |
| Architecture changes | Infini-attention, HMT, Titans, ATLAS, LM2 | Increase model-side memory; RLM is model-agnostic scaffold-side memory/control. |
| Recursive agents | THREAD, ReDel, Claude subagents | Recursive task decomposition, but often without prompt-as-environment. |
| Coding agents | Claude Code, OpenCode, coding-agent long-context paper | Use files/tools as context; RLM formalizes prompt-as-environment and symbolic recursion. |
| Structured recursion | lambda-RLM, LCM | Retain RLM insight but constrain control for termination and predictable cost. |

## Implementation Blueprint

A minimal RLM needs:

1. Environment
   - Store `prompt` as a string/file/object outside model context.
   - Expose safe operations: `len`, `peek`, `slice`, `search`, `read`, `write`.
   - Persist variables across turns.

2. Sub-call API
   - `llm_query(prompt) -> string` for depth 1.
   - `rlm_query(prompt, depth-1) -> string` for deeper recursion.
   - Batch version for parallel fanout.

3. Root loop
   - Send system prompt, environment metadata, and bounded observations.
   - Parse model code/actions.
   - Execute in sandbox.
   - Append only code plus bounded stdout metadata.
   - Stop on final variable/file/tool call.

4. Guardrails
   - `max_root_iters`
   - `max_depth`
   - `max_subcalls`
   - `max_wall_time`
   - `max_total_tokens`
   - stdout truncation
   - sandbox isolation
   - allowlisted imports/tools
   - structured final answer contract

5. Logging
   - Root prompts and outputs.
   - Code executed.
   - Environment diffs or variable summaries.
   - Sub-call prompts and responses.
   - Token/cost/latency per call.
   - Errors, retries, final state.

6. Training data
   - Store each root turn as an SFT sample.
   - Preserve trajectory metadata.
   - Filter zero-score, one-turn, over-window, malformed, and tool-error traces.
   - Programmatically patch trivial template errors before SFT.
   - Add RL/GRPO/RLVR later for on-policy interface use.

## Pseudocode For A Practical RLM

```python
def rlm_completion(prompt, root_model, sub_model, max_depth=1):
    env = Sandbox()
    env["prompt"] = prompt
    env["Final"] = None

    def llm_query(q):
        return sub_model.complete(q)

    def rlm_query(q):
        if max_depth <= 0:
            return sub_model.complete(q)
        return rlm_completion(q, root_model, sub_model, max_depth=max_depth - 1)

    env.register("llm_query", llm_query)
    env.register("rlm_query", rlm_query)
    env.register("peek", lambda start, n: prompt[start:start+n])
    env.register("search", lambda needle: prompt.find(needle))

    hist = [metadata(env)]

    for step in range(MAX_ROOT_ITERS):
        code = root_model.complete(render(hist))
        result = env.exec(code, timeout=STEP_TIMEOUT)
        hist.append({"code": code, "stdout": summarize(result.stdout)})

        if env["Final"] is not None:
            return env["Final"]

    raise RuntimeError("RLM did not finish")
```

In production, do not run arbitrary model code in the host process. Use Docker,
Modal, E2B, Daytona, Prime sandboxes, an IPython subprocess, or a typed DSL.

## Prompting Patterns

Useful root instructions:

- Treat the prompt as data in `prompt`, not text you have already read.
- First probe structure: length, prefix, delimiters, counts, line format.
- Decide task complexity: retrieval, search, aggregation, pairwise, code QA,
  graph reasoning, extraction.
- Use Python/string tools for mechanical work.
- Use sub-model calls only for semantic judgments.
- Batch independent subcalls where possible.
- Store all reusable results in variables/files.
- Verify intermediate outputs before composing.
- Write final answer to `Final` or a designated answer file.
- Stop when further recursion is unlikely to change the answer.

Bad root behavior:

- Reading huge chunks into stdout.
- Solving semantic subtasks with brittle keyword heuristics.
- Calling a submodel per item when a cheap filter can reduce the set.
- Recomputing child answers instead of memoizing.
- Emitting final answers before variables are populated.
- Hiding errors instead of inspecting stack traces and retrying a smaller step.

## Evaluation Checklist

A serious RLM eval should include:

- Simple retrieval, to measure overhead and regressions.
- Dense aggregation, where every item matters.
- Pairwise or combinatorial tasks, where symbolic loops help.
- Code repository QA, where filesystem-style access matters.
- Multi-hop document QA, where search and evidence composition matter.
- Long-horizon reasoning DAGs, where memoization matters.
- Poisoned/conflicting evidence, where cross-checking matters.
- Cost, latency, token count, subcall count, and failure-rate reporting.
- Ablations:
  - depth 0 vs depth 1 vs deeper RLM
  - plain LLM subcalls vs recursive RLM subcalls
  - with/without in-context decomposition examples
  - blocking vs parallel subcalls
  - local REPL vs isolated sandbox
  - model-written loop vs typed/engine-managed operators

## Design Guidance For This Repo

For a small-model RLM effort, the strongest path is not to hand-roll a massive
long-context architecture. It is to train the model to use an RLM interface well.

Concrete priorities:

1. Build a deterministic, logged RLM environment first.
2. Keep the action surface small: inspect, slice/search, execute, subcall, batch
   subcall, write final.
3. Generate trajectories with a stronger teacher RLM.
4. Filter aggressively for successful, multi-turn, syntactically valid traces.
5. SFT root turns to teach environment operation.
6. Add verifiable rewards for tasks with exact answers.
7. Penalize unnecessary subcalls, syntax errors, and oversized stdout.
8. Evaluate on both easy retrieval and dense aggregation.
9. Add structured control variants once the open-ended scaffold works.

The highest-leverage learned behaviors are:

- Probe before decomposing.
- Select the right decomposition type.
- Use code for mechanical iteration.
- Use subcalls for semantics.
- Memoize child outputs.
- Verify before aggregating.
- Stop early when confident.

## Latest Direct RLM Papers

These are the direct RLM papers and RLM-labeled variants found in this pass,
current as of 2026-05-27.

| Date | Source | What it adds |
| --- | --- | --- |
| 2026-05-13 online | [Consolidation-Expansion Operator Mechanics](sources/arxiv/2605.09968-opmech-adaptive-learning.md) | Order-gap stopping/control theory for recursive workflows. |
| 2026-05-11 latest version | [Recursive Language Models](sources/arxiv/2512.24601-recursive-language-models.md) | Canonical paper and RLM-Qwen3 training result. |
| 2026-05-07 | [Architecture Matters](sources/arxiv/2605.05632-rag-poisoning-architectures.md) | RLM robustness under RAG poisoning. |
| 2026-02-14 / 2026-05 arXiv id | [LCM](sources/arxiv/2605.04050-lossless-context-management.md) | Deterministic engine-managed recursive context. |
| 2026-03-20 | [Y-Combinator for LLMs](sources/arxiv/2603.20105-lambda-rlm.md) | Typed lambda-calculus control and formal bounds. |
| 2026-03-07 | [RLMs Meet Uncertainty](sources/arxiv/2603.15653-self-reflective-program-search.md) | Self-reflective program search and uncertainty signals. |
| 2026-03-03 | [Think, But Don't Overthink](sources/arxiv/2603.02615-reproducing-rlms.md) | Reproduction and depth ablation. |
| 2026-02-18 | [RLMs for Jailbreak Detection](sources/arxiv/2602.16520-rlm-jailbreak-detection.md) | Procedural RLM defense for tool-augmented agents. |

## Implementation Sources

- [Official RLM README](sources/github/alexzhang13-rlm-readme.md): full
  framework, environments, providers, logging, visualizer.
- [RLM minimal README](sources/github/alexzhang13-rlm-minimal-readme.md):
  smallest official implementation of the REPL plus subcall idea.
- [lambda-RLM README](sources/github/lambda-calculus-llm-lambda-rlm-readme.md):
  typed-control implementation and benchmark harness.
- [Prime Verifiers README](sources/github/primeintellect-verifiers-readme.md):
  RL/eval environment framework with RLM harnesses.
- [Prime RLM SWE environment](sources/github/primeintellect-rlm-swe-v1-env-readme.md):
  RLM harness packaged for software-engineering tasks.
- [Hampton IO RLM](sources/github/hampton-io-rlm-readme.md): TypeScript RLM
  with JavaScript REPL, CLI, tracing, streaming, and multi-provider support.
- [Aleph](sources/github/hmbown-aleph-readme.md): MCP server/skill exposing
  RLM-like external working state, search, code execution, and recursion.
- [RLM-FORGE](sources/github/q00-rlm-forge-readme.md): bounded inner calls and
  evidence-gated synthesis experiment.

## Reading Order If You Do Want The Details

1. [Recursive Language Models](sources/arxiv/2512.24601-recursive-language-models.md)
   Sections 1-2, then Table 1 and Appendix A.
2. [RLM minimal README](sources/github/alexzhang13-rlm-minimal-readme.md).
3. [Think, But Don't Overthink](sources/arxiv/2603.02615-reproducing-rlms.md)
   and [SRLM](sources/arxiv/2603.15653-self-reflective-program-search.md).
4. [lambda-RLM](sources/arxiv/2603.20105-lambda-rlm.md),
   [LCM](sources/arxiv/2605.04050-lossless-context-management.md), and
   [OpMech](sources/arxiv/2605.09968-opmech-adaptive-learning.md).
5. [Prime Verifiers](sources/github/primeintellect-verifiers-readme.md) and the
   implementation notes for practical harness ideas.

## Corpus Generation

The corpus was generated with:

```bash
uv run --with requests --with beautifulsoup4 --with markdownify \
  python docs/papers/rlm/tools/fetch_rlm_sources.py
```

Full-text mirroring is intentionally conservative. CC BY 4.0/CC0 arXiv HTML and
MIT-licensed READMEs are mirrored as markdown with attribution. Other arXiv
records, web articles, model cards, packages, and blogs are indexed with
metadata and relevance notes instead of full-text copies.
