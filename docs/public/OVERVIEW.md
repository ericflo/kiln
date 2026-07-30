# Kiln overview

Kiln is a local server for one continuous loop: **serve a model, collect
evidence, train an adapter, evaluate it, and put the winner back into service**.
It packages that loop in one Rust process instead of making you operate a
separate inference server, training service, evaluation system, and adapter
registry.

## Is Kiln for me?

Use Kiln when you want to improve Qwen3.5-4B on a machine you control and the
improvement loop matters as much as raw serving throughput.

Kiln is a good fit for:

- OpenAI-compatible local inference for agents and applications;
- SFT, GRPO, OPD, and adapter hot-swap on the same accelerator;
- local eval suites, comparisons, judgments, and replay;
- traceable artifacts with model, request, training, and execution provenance;
- a single-GPU workstation rather than a distributed serving fleet.

Kiln is not presently positioned as a replacement for vLLM in a
high-concurrency, serving-only deployment. See [Benchmarks](BENCHMARKS.md) for
the measured boundary and the metrics behind it.

## The shortest path

1. Follow the [five-minute Quickstart](https://ericflo.github.io/kiln/quickstart.html).
2. Send an OpenAI-compatible chat request.
3. Open the embedded dashboard at `http://127.0.0.1:8420/ui/`.
4. Import or collect training examples.
5. Train an adapter, run an eval, and promote the result deliberately.

The public [product tour](https://ericflo.github.io/kiln/demo/) shows the current
dashboard and explains which parts are live product UI, seeded example data, or
historical recordings.

## What runs where

Kiln supports CUDA, ROCm, Metal, and Vulkan builds. Backend availability,
supported training routes, memory limits, and optimization maturity differ.
Startup resolves the selected backend and exposes the effective configuration
through `/health`, `/v1/config`, the CLI, and the dashboard.

The project does not treat “the process started” as proof that every route is
qualified. Performance pages distinguish source revisions, hardware, workloads,
and metric definitions; qualification receipts remain separate from user
guidance.

## How the loop fits together

| Stage | What Kiln does | Where to continue |
|---|---|---|
| Serve | Runs chat, completion, streaming, batching, and adapter-aware requests | [API guide](https://ericflo.github.io/kiln/api.html) |
| Observe | Records latency phases, request lineage, outputs, and runtime state | [Latency observability](../LATENCY_OBSERVABILITY.md) |
| Teach | Accepts SFT, GRPO, OPD, and related training jobs | [GRPO guide](https://ericflo.github.io/kiln/grpo.html) |
| Evaluate | Runs local suites, comparisons, judgments, and replay | [Evals guide](https://ericflo.github.io/kiln/evals.html) |
| Promote | Saves, validates, loads, and selects LoRA adapters | [Adapter manifest](../ADAPTER_MANIFEST.md) |

## Documentation map

The site is split intentionally:

- **Product guides** answer the common “how do I do this?” questions.
- **Core documentation** explains configuration, architecture, performance,
  and supported workflows without reproducing internal engineering ledgers.
- **Reference library** contains exact schemas, contracts, qualification
  protocols, and maintainer material when you need source-level detail.

If a core page requires a 20,000-word schema dump to answer a basic question,
that page is broken. The reference should be linked, searchable, and available,
but it should not be the first thing every reader has to parse.
