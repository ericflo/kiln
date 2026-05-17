# Capability: Strict JSON-Schema Adherence

## Description
The student model must, when given a natural-language query and a JSON Schema,
emit **only** a JSON object that (a) parses as valid JSON, (b) validates against
the schema, (c) has no surrounding text — no preamble, no postamble, no markdown
fences, no commentary — and (d) has substantive content (no `"TODO"` /
`"example"` / `"lorem ipsum"` placeholders).

Concretely: the entire model response, after `.strip()`, must be `json.loads`-able
and the parsed object must `jsonschema.validate(instance, schema)` cleanly.

Failure modes the 4B exhibits (and that this capability targets):
- Wraps JSON in ``` ```json fences ```
- Emits preamble: `"Here's the JSON object:"`, `"Sure! "` etc.
- Misses `required` fields when the schema is deeply nested
- Picks values outside `enum` constraints
- Emits extra fields when `additionalProperties: false`
- Returns numbers as strings or vice versa
- Returns dates in the wrong format

This capability is the prototype for all future capability development per the
user goal (2026-05-16): "use OPD to distill from Qwen 3.6 27B".

## Base model
`Qwen/Qwen3.5-4B` (kiln-served at `:8420`)

## Teacher
`Qwen/Qwen3.6-27B` (local; same architecture, same tokenizer ⇒ clean per-token
reverse-KL with no cross-family tokenizer hacks)

## Rubric (programmatic)
For each (query, schema) → response, compute four sub-scores in `[0,1]`:

1. **`parses`** — `json.loads(strip_optional_fence(response))` succeeds.
2. **`validates`** — parsed object validates against the schema.
3. **`is_pure`** — response is exactly the JSON object with optional whitespace
   only (no preamble/postamble/fences).
4. **`is_substantive`** — heuristic: no literal placeholder strings
   (`"TODO"`, `"example"`, `"lorem ipsum"`, `"placeholder"`, `"foo"`, `"bar"`)
   AND total string-field byte length ≥ 50.

**Composite:** `0.4*parses + 0.3*validates + 0.2*is_pure + 0.1*is_substantive`

`parses` and `is_pure` are gating: if either is 0, the whole composite is heavily
penalised. The 27B should score ≥0.9 composite; the 4B baseline expected ~0.4-0.6.

## Plan
1. Build 250 (query, schema) prompts: 200 train, 50 eval.
2. Baseline 4B composite via kiln-server `/v1/chat/completions`.
3. Run 27B over the train split, save top-32 teacher logprobs + the actual
   teacher completion to `kiln_train::FixtureLogitSource`.
4. Run kiln `opd_train` against the fixture.
5. Eval LoRA on held-out split, log delta.
6. Iterate on hyperparameters (rank, lr, cold-start, Stable-OPD knobs,
   samples_per_prompt) until uplift saturates.
