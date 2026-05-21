# Results: JSON-Schema-Adherence Capability Distillation

Prototype for the OPD-from-Qwen3.6-27B pipeline. User goal:
> "use on policy distillation to distill from Qwen 3.6 27B. first, come up with a target capability that you think could be distilled, then build a rubric carefully measuring this kiln student on that capability, then try as many experiments as you need to uplift that capability significantly from Qwen 3.6 27B into a kiln lora. be relentless. this will be the prototype for all future capability development!"

## Pipeline

**Models:**
- Student: Qwen3.5-4B (kiln-served, ~8 GB VRAM)
- Teacher: Qwen3.6-27B in 4-bit NF4 via `bitsandbytes` (~18 GB VRAM)
- Same tokenizer (SHA-identical) → no cross-vocab hacks needed

**Capability:** Strict structured-output adherence — emit ONE JSON object that
parses, validates against a JSON Schema, has substantive content, and has no
preamble/postamble/markdown-fences.

**Rubric (programmatic, `rubric.py`):**
- `parses` (0/1): `json.loads()` succeeds after optional fence-strip
- `validates` (0/1): `jsonschema.validate()` passes against the schema
- `is_pure` (0/1): response starts with `{` or `[` and ends with the matching closer
- `is_substantive` (0/1): no placeholder strings + total string bytes ≥ 25
- **composite = 0.4·parses + 0.3·validates + 0.2·is_pure + 0.1·is_substantive**

**Corpus (`build_corpus.py`):** 235 (query, schema) prompts across 10 domains
(book, bug, coordinate, endpoint, event, invoice, pet, recipe, task, user).
80/20 split → 188 train / 47 eval. Domains exercise nested objects, enums,
strict regex patterns (ISBN, INV-NNNNNN, T-AAAAAAA), `additionalProperties:false`,
discriminated unions (`oneOf`), `prefixItems`/tuples.

**Teacher fixture (`teacher_inference.py`):** Loads Qwen3.6-27B (4-bit),
renders each train prompt via the kiln-canonical minijinja path + HF tokenizer
(token-identical to kiln's `tokenize_for_training`), generates an assistant
turn with `enable_thinking=False` greedy decoding, then forwards the full
conversation to extract top-32 teacher logprobs at each active position.

**Training:** `cuda_sft_file --trainer generic` (off-policy SFT distillation on
teacher-generated text). The `kiln_train::opd::opd_train` path (top-K reverse-KL
on teacher distributions) OOMs on long sequences because it lacks the
gradient-checkpointed segmented forward/backward that the SFT path has —
filed as a follow-up; the prototype uses SFT-on-teacher-text which is the
off-policy distillation baseline Lu et al. (2025) compares OPD against.

**Eval (`eval_kiln.py`):** Hits kiln-server `/v1/chat/completions` with
`chat_template_kwargs={"enable_thinking": false}` + `adapter=NAME`, scores
every response via `rubric.score_response`.

## Results

| Adapter | composite | parses | validates | is_pure | is_substantive | notes |
|---------|-----------|--------|-----------|---------|----------------|-------|
| baseline | 0.9043 | 1.000 | 0.745 | 1.000 | 0.808 | 4B base, no LoRA |
| sft-v1-r16-lr1e4-2ep (43 prompts) | **0.9447** | 1.000 | **0.894** | 1.000 | 0.766 | +4.0pt composite, +14.9pt validates |

### Per-domain (baseline → sft-v1)

| Domain | composite (base→sft) | validates (base→sft) | what changed |
|--------|---------------------|----------------------|--------------|
| book | 0.70 → **1.00** | 0.00 → **1.00** | ISBN pattern `^[0-9]{3}-[0-9]{10}$` |
| bug | 1.00 → 1.00 | 1.00 → 1.00 | (already perfect) |
| coordinate | 0.925 → 0.925 | 1.00 → 1.00 | (no change) |
| endpoint | 0.96 → 0.92 | 1.00 → 1.00 | regression: terser → fails substantive |
| event | 0.97 → 0.99 | 1.00 → 1.00 | small gain |
| invoice | 0.95 → **1.00** | 0.83 → **1.00** | INV-NNNNNN pattern |
| pet | 0.93 → 0.90 | 1.00 → 1.00 | regression: terse oneOf details |
| recipe | 1.00 → 0.80 | 1.00 → 0.33 | regression: thin recipes (training corpus only had 5 recipe examples in partial 43-row data) |
| task | 0.85 → 0.95 | 0.50 → 0.83 | T-AAAAAAA pattern |
| user | 0.82 → 0.88 | 0.40 → 0.60 | username pattern `^[a-z0-9_]{3,16}$` |

The big wins are on **strict regex patterns** (book ISBN, invoice ID, task ID,
username) where the 4B baseline emitted real-world-looking formats but the
schema required strict literals. The 27B reads patterns more carefully and
the SFT distillation transfers that.

The regressions cluster around:
1. `is_substantive` going down (student became more terse — mimics teacher)
2. `recipe` domain — under-covered in partial 43-row training data

## Open questions / follow-ups

1. **Full 188-row SFT** — running. Expected to recover the recipe regression
   since the full corpus has 25 recipe prompts vs 5 in the partial subset.
2. **Hyperparameter sweep** — rank ∈ {8, 16, 32, 64}, epochs ∈ {1, 2, 3},
   lr ∈ {5e-5, 1e-4, 2e-4}. `sweep.sh` queues a focused subset.
3. **Oracle ceiling** — `eval_teacher.py` runs Qwen3.6-27B on the eval set to
   establish the realistic uplift target.
4. **True OPD (reverse-KL, on-policy)** — needs gradient checkpointing in
   `kiln_train::opd::opd_train`. Tracked as a follow-up; the trainer's
   `model_forward_no_head` keeps all activations resident through backward,
   OOMing at ~750 tokens of context on a 48 GB GPU. The SFT path's
   `checkpointed_forward_backward` is the model to port.
5. **Substantive-rubric tuning** — for sparse-string schemas (coordinate, pet)
   the 25-byte threshold flags valid teacher outputs as "thin". Either drop
   the substantive component or scale-by-schema.

## Files

| File | Purpose |
|------|---------|
| `capability.md` | Capability description + plan |
| `rubric.py` | Scoring function + self-test |
| `build_corpus.py` | Generate train/eval JSONL |
| `eval_kiln.py` | Score an adapter via `/v1/chat/completions` |
| `eval_teacher.py` | Score the 27B teacher (oracle ceiling) |
| `teacher_inference.py` | Generate teacher fixture + completions |
| `filter_short.py` | Subset to short sequences (for OPD attempts) |
| `run_sft_exp.sh` | One end-to-end SFT experiment |
| `sweep.sh` | Priority SFT sweep |
| `sweep_opd.sh` | OPD attempts on short subset |
| `auto_after_teacher.sh` | Auto-run after teacher generation finishes |
| `summarize.py` / `diff_adapters.py` / `per_domain.py` | Result analysis |
| `capability.jsonl` | Append-only experiment log |
| `judgments/<name>.json` | Per-adapter eval output (per-prompt scores + sample responses) |

The OPD-from-fixture binary `cuda_opd_from_fixture` is built and the kernel
plumbing works end-to-end on small fixtures; only the model-side
gradient-checkpointing gap blocks full runs on the production corpus.
