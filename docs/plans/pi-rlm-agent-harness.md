# Pi RLM Agent Harness Plan

This plan turns Pi into a Recursive Language Model client without requiring Pi
itself to own recursion. Pi still sends a normal OpenAI-compatible chat request
and still receives one normal assistant message or tool call. The new harness
between Pi and Kiln externalizes the full Pi turn, runs a bounded internal RLM
loop, and returns the next Pi-visible action only when the loop finishes.

## Implemented Surface

This branch adds a first-pass local harness:

```bash
python3 scripts/pi_rlm_harness.py \
  --listen 127.0.0.1:8421 \
  --upstream http://127.0.0.1:8420/v1 \
  --model Qwen3.5-4B \
  --tokenizer /workspace/Qwen3.5-4B/tokenizer.json \
  --require-tokenizer \
  --state-dir .kiln/pi-rlm-harness
```

The harness exposes:

- `GET /v1/models`
- `POST /v1/chat/completions`

It accepts Pi's existing OpenAI-style payload, stores `messages` and `tools` as
external state, runs the root RLM loop against the upstream Kiln server, and
returns either:

- `{role:"assistant", content:"..."}`
- `{role:"assistant", tool_calls:[...], finish_reason:"tool_calls"}`

`kiln pi-setup` can now point Pi at the harness while preserving the direct Kiln
provider:

```bash
kiln pi-setup \
  --url http://127.0.0.1:8420 \
  --rlm-url http://127.0.0.1:8421
```

This writes two Pi providers:

- `kiln-local`: direct Kiln fallback at `:8420/v1`, model `Qwen3.5-4B`
- `kiln-rlm-local`: RLM harness at `:8421/v1`, model `Qwen3.5-4B-RLM`

The RLM provider is made Pi's default. It advertises a 16k context and 4096 max
output tokens to match the fixed-window adapter design below.

There is also a Pi payload extension prototype:

```text
capabilities/caps/pi-rlm-harness/extensions/provider-payload-rlm.ts
```

It forces provider requests to `max_tokens=4096`, `temperature=0`, `top_p=1`,
and annotates request metadata with the fixed RLM window. The harness itself is
still the component that externalizes state and runs recursion.

## RLM Contract

One Pi provider turn becomes one potentially recursive harness turn.

```text
Pi -> RLM harness:
  messages, tools, tool_choice, current request metadata

RLM harness:
  E.messages = full Pi prompt/history
  E.tools = full Pi tool catalog
  repeat:
    root_output = M(system_prefix, env_summary, bounded_observations)
    action = parse_json(root_output)
    observation = execute_internal_action(action, E)
  until action == finish

RLM harness -> Pi:
  assistant content or assistant tool_call
```

The root model never needs the whole Pi transcript in its context. It sees an
environment summary plus bounded observations from these internal actions:

```json
{"action":"inspect","target":"summary"}
{"action":"inspect","target":"message","index":0}
{"action":"inspect","target":"tools"}
{"action":"search","query":"needle","regex":false,"max_matches":8}
{"action":"slice","index":0,"start":0,"length":4096}
{"action":"subcall","prompt":"bounded prompt","system":"optional","max_tokens":1024}
{"action":"spawn_agent","task":"child task","message_refs":[0,3],"slices":[{"index":2,"start":0,"length":4096}],"max_iters":8}
{"action":"finish","content":"assistant text for Pi"}
{"action":"finish","tool_call":{"name":"Bash","arguments":{"cmd":"pytest -q"}}}
```

`subcall` is a bounded semantic call to the upstream model. `spawn_agent` is the
load-bearing recursive primitive: it creates a child RLM environment from parent
message references, exact slices, and optional artifacts, then runs the same
controller loop at `depth+1`. That child can itself inspect/search/slice,
subcall, and spawn more child agents until `--max-depth` is reached. The child
returns its final content/tool-call proposal as a parent observation; the final
top-level `finish.tool_call` is the only tool call Pi sees and executes.

The prototype keeps `subrlm` as a compatibility alias for `spawn_agent`, but the
training target should use `spawn_agent` so traces are explicit about recursive
agent creation.

The harness now treats invalid controller actions as environment observations
instead of silently passing them through. Unknown actions, malformed slices,
empty searches, invalid spawns, and final tool calls to unknown Pi tools are
fed back into the loop as structured observations. That gives SFT/GRPO a clean
negative trace: the model sees the invalid action, sees the harness rejection,
and gets another chance before max-iteration fallback.

## Fixed-Window Adapter Shape

The adapter should be trained to operate one root RLM step with a fixed prompt
shape:

```text
8192 tokens  stable prefix:
             RLM policy, internal action schema, Pi tool-call contract,
             format examples, safety/sandbox rules, final-answer rules

4096 tokens  dynamic input:
             environment summary, message inventory, latest user preview,
             recent internal observations, selected exact slices

4096 tokens  output:
             one JSON internal action or one JSON finish action
```

The important compression is not "summarize the whole Pi session into 4096
tokens." The full session remains in `E`. The 4096-token dynamic input is just
the root controller's current observation.

Window enforcement must use the Qwen3.5 tokenizer, not character counts. The
Python prototype loads `tokenizer.json` from:

- `--tokenizer`
- `KILN_TOKENIZER_PATH`
- `KILN_MODEL_PATH/tokenizer.json`
- `./Qwen3.5-4B/tokenizer.json`
- `/workspace/Qwen3.5-4B/tokenizer.json`

Production and training runs should pass `--require-tokenizer`; char fallback is
only for local script smoke tests.

This collapses Pi's visible multi-turn assistant behavior into a sequence of
single supervised root decisions:

```text
(prefix, env_summary, obs_0) -> inspect/search/slice/subcall
(prefix, env_summary, obs_1) -> next internal action
...
(prefix, env_summary, obs_t) -> finish with Pi-visible tool call
```

Each deployed controller step is now persisted with its exact fixed-window
training prompt:

```json
{
  "prompt_messages": [
    {"role":"system","content":"<8192-token RLM prefix>"},
    {"role":"user","content":"<4096-token dynamic observation>"}
  ],
  "action": {"action":"search","query":"pytest"},
  "observation": {"action":{...},"output":{...}},
  "prompt_tokens": 3170,
  "action_tokens": 12
}
```

That makes the state directory directly useful for training and audit instead
of merely useful for debugging.

## ECHO Is a Natural Fit

Yes: ECHO is probably one of the load-bearing pieces for making this a strong
RLM rather than a brittle prompt trick.

In the RLM loop there are two token classes:

```text
Action tokens:
  root controller JSON such as inspect/search/slice/spawn_agent/finish

Observation tokens:
  harness responses such as message inventories, search hits, slices,
  child-agent returns, validation errors, and max-depth notices
```

That is exactly Kiln's agentic trajectory schema:

```json
[
  {"role":"assistant","kind":"action","content":"{\"action\":\"search\",\"query\":\"...\"}"},
  {"role":"tool","kind":"observation","content":"{\"output\":[...]}"},
  {"role":"assistant","kind":"action","content":"{\"action\":\"spawn_agent\",...}"},
  {"role":"tool","kind":"observation","content":"{\"child_request_id\":\"...\"}"}
]
```

GRPO alone learns only from final reward assigned to action tokens. ECHO adds
environment cross-entropy on the observation tokens, so the adapter learns the
terminal dynamics of its own harness: what an inspect result looks like, where
search spans appear, how validation failures are phrased, how child agents
return, and how much evidence is usually needed before a final Pi tool call.
That matters because recursive agents are only useful if the model can predict
the information flow created by its own internal actions.

The harness can now export both forms:

```bash
# Per-step SFT rows that exactly match deployed fixed-window root decisions.
python3 scripts/pi_rlm_harness.py \
  --export-state-dir .kiln/pi-rlm-harness \
  --export-sft-jsonl /tmp/pi-rlm-root-actions.sft.jsonl

# Agentic GRPO/ECHO groups: one JSONL line per root/child environment.
python3 scripts/pi_rlm_harness.py \
  --export-state-dir .kiln/pi-rlm-harness \
  --export-echo-jsonl /tmp/pi-rlm-rollouts.echo.jsonl \
  --export-default-reward 1.0

# Sanity-check masks before training.
kiln trajectory inspect /tmp/pi-rlm-rollouts.echo.jsonl \
  --tokenizer /workspace/Qwen3.5-4B/tokenizer.json
```

The exported ECHO rows use `rollouts[].trajectory` with `kind:"action"` and
`kind:"observation"`, which Kiln's `/v1/train/agentic` and `/v1/train/grpo`
already accept. The `text` field is the `<TURN_BREAK>`-joined action stream for
legacy compatibility; ECHO consumes the structured `trajectory`.

## Training Data

There are three useful datasets.

1. Teacher RLM trajectories

Run a stronger upstream model through the harness on real Pi prompts. Store
every root step:

```json
{
  "messages": [
    {"role":"system","content":"<8192-token RLM prefix>"},
    {"role":"user","content":"<4096-token env observation>"}
  ],
  "assistant": "{\"action\":\"search\",\"query\":\"...\"}",
  "metadata": {
    "session_id": "...",
    "root_step": 2,
    "eventual_pi_tool": "Bash",
    "eventual_reward": 1.0
  }
}
```

Filter out malformed JSON, over-window samples, one-step lucky direct answers
when training decomposition, and trajectories that never finish.

The harness exports this directly:

```bash
python3 scripts/pi_rlm_harness.py \
  --export-state-dir .kiln/pi-rlm-harness \
  --export-sft-jsonl train.rlm_actions.sft.jsonl
```

2. Production-trace tool-call eval rows

Use PR #1383's idea: materialize every production assistant tool-call turn as:

```text
prompt = exact prefix before the assistant tool-call message
target = canonical semantic tool_calls JSON
```

Then run the RLM harness against `prompt`. It passes if the harness eventually
returns the same semantic Pi-visible tool call, even if it used any number of
internal inspect/search/slice/subcall steps first.

The harness has a local same-tool-eventually runner for this:

```bash
python3 scripts/pi_rlm_harness.py \
  --upstream http://127.0.0.1:8420/v1 \
  --model Qwen3.5-4B \
  --tokenizer /workspace/Qwen3.5-4B/tokenizer.json \
  --require-tokenizer \
  --eval-suite production_trace_suite.json \
  --eval-output production_trace.rlm_report.json
```

The Python runner intentionally scores only the harness-specific question:
"did the recursive loop eventually emit the target Pi-visible tool call?" The
full PR #1383 scorer remains the richer Rust path for bash/Python/path
equivalence, Wilson intervals, provenance slicing, and large audit reports.

3. Negative and efficiency rows

Add tasks where the right behavior is not to recurse:

- simple final answer
- obvious single Bash call
- no tool needed
- impossible/ambiguous request
- huge trace with irrelevant repeated observations

These prevent the adapter from learning "always inspect 10 times."

## SFT First, GRPO Second

Start with SFT on root actions:

```text
loss = CE(root_action_tokens)
```

The root action is short and structured, so SFT should teach the interface
quickly. Keep assistant outputs under 4096 tokens and reject any sample whose
rendered input exceeds 4096 dynamic tokens under the Qwen3.5 tokenizer.

Then add GRPO/RLVR:

```text
reward =
  +1.0 if final Pi-visible tool call semantically matches target
  +0.2 if correct tool name but imperfect args
  -0.1 per invalid JSON root action
  -0.02 per unnecessary internal step after a correct action is obvious
  -0.2 if max_iters fallback is used
```

For RLM rollouts, keep ECHO enabled. The observation tokens are not external
shell output; they are the harness dynamics themselves. That is still exactly
what ECHO is for. Use the exported `rollouts.echo.jsonl` shape and start with
Kiln's default `loss.echo.lambda = 0.05`.

Useful ablations:

```text
SFT only:
  train root-action JSON on exact fixed-window rows

SFT -> GRPO:
  reward final same-tool-eventually pass/fail

SFT -> GRPO + ECHO:
  same reward, plus env-CE on inspect/search/slice/spawn observations

SFT -> ECHO-only maintenance:
  no_policy_loss=true on strong accepted traces to improve harness dynamics
  without pushing policy away from accepted behavior
```

## Eval Protocol

Primary eval should mirror #1383:

1. Import huge production traces.
2. Materialize each assistant tool-call turn from the exact message prefix.
3. Send that prefix to the Pi RLM harness, not directly to the base model.
4. Let the harness run up to `max_iters`.
5. Score the final Pi-visible tool call semantically:
   - same tool name
   - equivalent JSON arguments
   - shell command equivalence for Bash
   - nested Python command equivalence where applicable
6. Slice metrics by production model, tool name, trace source, split, prompt
   length bucket, internal step count, and fallback usage.

Minimum acceptance gates:

- Base direct model vs RLM harness on the same trace suite.
- RLM harness with base adapter vs RLM harness with trained adapter.
- Pass rate and Wilson confidence intervals by tool.
- Mean internal steps, p95 internal steps, fallback rate.
- Latency and token/cost per successful final tool call.

## Implementation Phases

Phase 0: local harness smoke

- Run Kiln on `:8420`.
- Run `scripts/pi_rlm_harness.py` on `:8421`.
- Run `kiln pi-setup --rlm-url http://127.0.0.1:8421`.
- Confirm Pi receives normal assistant content/tool calls.
- Inspect `.kiln/pi-rlm-harness/*.json` traces.

Phase 1: production trace eval

- Bring PR #1383's `production_trace` importer into the target branch.
- Add a runner mode that hits the harness endpoint.
- Score "same tool eventually" from final harness output.
- For quick local checks before the Rust scorer lands, run
  `scripts/pi_rlm_harness.py --eval-suite ... --eval-output ...`.

Phase 2: teacher trajectory collection

- Run a frontier/large teacher through the harness.
- Persist root-step SFT examples plus final outcome metadata.
- Add strict tokenizer window checks: 8192 prefix, 4096 dynamic input, 4096
  output.
- Export both `*.sft.jsonl` and `*.echo.jsonl` from the same state directory.
- Inspect exported ECHO masks with `kiln trajectory inspect` before training.

Phase 3: adapter training

- SFT root actions.
- Evaluate on production trace suite.
- GRPO on sampled trace rows with semantic tool-call reward.
- Keep ECHO enabled for action/observation RLM traces; run one no-ECHO ablation
  and one ECHO-only maintenance ablation so the contribution is measurable.
- Keep adapters behind Pi A/B until pass rate, latency, and fallback rate beat
  the base harness.

Phase 4: native server integration

- Move the Python proxy contract into Rust if the prototype wins.
- Expose `/v1/rlm/chat/completions` from `kiln serve`.
- Store harness traces in the existing agent trace layer.
- Let training jobs consume harness traces directly.

## Risks

- Streaming tool-call deltas are provider-specific. The prototype sends one
  complete delta; some Pi versions may require more granular deltas.
- Approximate char budgets are not tokenizer budgets. Training and final eval
  must use Qwen3.5 token counts.
- A weak model may learn to emit direct final calls without useful inspection.
  Keep decomposition-heavy data and step-efficiency penalties.
- Recursion can waste latency. Track internal steps and stop/fallback rates as
  first-class eval metrics.
- The harness executes no Pi tools internally. That is intentional: only Pi
  executes final tool calls. Internal actions inspect the prompt environment or
  call models.
