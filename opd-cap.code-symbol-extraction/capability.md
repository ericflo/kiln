# Capability: code-symbol-extraction

## Description
The model is shown a code snippet (30–80 lines, Python/Rust/Go/JS/TS) and asked
to list every top-level **defined** symbol — functions, classes, structs,
enums, traits, type aliases — one symbol per line, **nothing else**. No
explanations, no markdown, no commentary.

The 4B baseline routinely (a) adds prose commentary the eval can't parse,
(b) misses nested or less-common symbol kinds (e.g. Rust `trait`, Go
struct methods on a receiver), and (c) over-recalls by listing local
variables or imported names.

A teacher (27B AWQ) does this cleanly and concisely. OPD should pull the
4B toward the teacher's compact symbol-only output.

## Base model
Qwen3.5-4B (kiln serve on http://localhost:8420)

## Teacher
`vllm/qwen3.6-27b-awq` at http://localhost:8002 (AWQ-INT4, max_logprobs=64)

## Rubric

| Sub-score | Weight | What it measures |
|-----------|--------|-------------------|
| `parses` | 0.15 | Response is non-empty text, no gibberish. |
| `format_compliance` | 0.15 | Output is symbol-name-per-line, no markdown bullets, no prose. Lines must be plain identifiers (allow `name (kind)` form). |
| `symbol_recall` | 0.35 | Fraction of ground-truth symbols the model listed. |
| `symbol_precision` | 0.35 | Fraction of listed names that are actually ground-truth symbols. |

Composite = `0.15 × parses + 0.15 × format_compliance + 0.35 × recall + 0.35 × precision`.
Direction: higher is better. Composite is in `[0, 1]`.

The two heavy weights (recall+precision = 0.70) make hallucinated symbols
hurt as much as missed symbols. This is intentional: it forces the model
to balance saying-enough against not-making-things-up.

## Baseline

| Sub-score | Weight | Baseline | Headroom (w×(1−b)) |
|-----------|--------|----------|---------------------|
| parses | 0.15 | 1.0000 | 0.0000 |
| format_compliance | 0.15 | 1.0000 | 0.0000 |
| symbol_recall | 0.35 | 1.0000 | 0.0000 |
| symbol_precision | 0.35 | 0.8039 | 0.0686 |
| **Total movable** | | | **0.0686** |

Composite: **0.9314** (headroom 0.069 above the 0.05 floor — proceed).

## Target sub-score

**`symbol_precision`** owns 100% of movable headroom. The 4B over-recalls — typical failure is listing imported names alongside defined ones (e.g. `EventEmitter` when the snippet does `extends EventEmitter`). OPD against the 27B should suppress this.

## Hypothesis log
| iter | slug | family | composite | comp Δ | target Δ | verdict |
|------|------|--------|-----------|--------|----------|---------|
| 1 | h1-r16-6ep | H1 | 0.9370 | +0.0056 | +0.0160 | ? Inconclusive (97% skip rate; bump epochs) |

## Dead ends
(none yet)

## Open questions
(none yet)

## Checkpoints
(every 3rd iter, write a brief progress summary here)
