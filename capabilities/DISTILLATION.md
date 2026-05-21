# `DISTILLATION.md` — Cluster → new base flywheel

This is the doc that ties everything else to the long-term goal:

> Create a cluster of capability adapters that are each incredible, then
> distill that cluster back into a single new base model. Use that new base
> for the next round of capability training. Repeat ad infinitum.

Each loop of "cluster → distill → new base → next round" is a **round**.
After each round, the base model carries more shipped capability into its
own weights; the capability cluster represents the *experimental frontier*
that hasn't been consolidated yet.

This doc is **deferred** to Phase G of the unification (see
[`README.md`](README.md) phases). It is written now so the agent has the
destination in view while building the per-stage pipelines that feed it.

Read [`METHODS.md`](METHODS.md) and [`PIPELINE.md`](PIPELINE.md) first.

---

## §1. When to distill

A round is ready for distillation when ALL hold:

1. **≥ 5 shipped pipelines** with `status: shipped` in their `pipeline.md`.
2. Each shipped pipeline has `final_composite - baseline_composite ≥ 0.05`
   with 3-seed evidence.
3. Each shipped pipeline passes `kiln adapter verify <final_adapter>` and
   has a valid `adapter_manifest.json`.
4. `integration/cross-cap-coherence/` has been run against every shipped
   pipeline's `final_adapter`. The full matrix exists in
   `rounds/round-<N>/sibling_matrix.json`.
5. The set of shipped pipelines has **at least one compatible cluster**
   (defined in §2 below).

Distilling fewer than 5 caps risks producing a new base that's worse on
the long tail than the old base. The flywheel only works when the cluster
is substantial enough to outweigh distillation noise.

---

## §2. Cluster compatibility

Not every shipped adapter is compatible with every other. Stacking conflicting
LoRAs destroys each other. The cluster compatibility check identifies which
subsets can safely be merged.

### §2.1 Sibling matrix

For each ordered pair `(A, B)` of shipped pipelines, evaluate adapter `A`
against cap B's `datasets/eval.tasks.jsonl`:

```
sibling_delta(A, B) = composite(A on cap_B.eval) - baseline(cap_B.eval)
```

Negative means A regresses B. Positive means A accidentally helped B.

`integration/cross-cap-coherence/` already computes this for the current
batch of adapters. Round-end snapshot lands in `rounds/round-<N>/sibling_matrix.json`.

### §2.2 Compatibility classes

```
A and B are COMPATIBLE if:
  sibling_delta(A, B) > -0.02
  AND sibling_delta(B, A) > -0.02

A and B are INCOMPATIBLE if either delta < -0.05.

A and B are SOFT INCOMPATIBLE if either delta ∈ [-0.05, -0.02].
```

A **cluster** is a set of pipelines where every pair is compatible.
Soft-incompatible pairs may be tolerated if the gain elsewhere is large
and the regression doesn't push the affected cap below its baseline.

### §2.3 Clustering algorithm

Greedy maximum compatible cluster:

```
1. Start with the pipeline with the highest composite_delta.
2. Iteratively add the pipeline that:
   - is compatible with every already-added pipeline
   - has the next-highest composite_delta
3. Stop when no more pipelines can be added.
4. If multiple clusters of similar size exist, distill each independently
   (A/B → pick the new base that wins more caps).
```

`lib/cluster_summary.py` runs this and writes `rounds/round-<N>/cluster_manifest.json`.

### §2.4 Single-cluster vs multi-cluster outcomes

| Outcome | What it means | What to do |
| --- | --- | --- |
| **1 cluster, all caps** | Adapters compose cleanly. | Distill the full cluster into base_(N+1). |
| **1 cluster, subset** | Some caps are off-axis. | Distill the cluster; the off-axis caps stay in round N+1 as is. |
| **2+ overlapping clusters** | Different skill subsets conflict. | Distill each independently → 2+ base candidates. Eval against vanilla benchmarks and against every cap. Pick the base that wins more caps. |
| **Many small clusters** | Caps don't compose. | Likely a rubric/eval problem — caps are training conflicting behaviors. Audit before distilling. |
| **Zero compatible pairs** | Distillation is not safe. | Re-evaluate caps. Either the eval sets overlap (causing spurious regression measurement) or rubrics genuinely conflict. |

---

## §3. Distillation methods

Three methods, listed in order of preference. Pick the first one that's
feasible for the cluster.

### §3.1 Method A: Multi-task SFT on adapter-generated rollouts

**The simplest and most predictable.**

```
For each pipeline P in the cluster:
  Generate N teacher rollouts from P.final_adapter on P's training corpus
    (use kiln rollout or pi against the kiln-served adapter)
  Keep rollouts that P.rubric.score_one > 0.8
  Write to /tmp/distill-corpus/<cap>.jsonl

Concatenate all <cap>.jsonl files into one mega-corpus.
Shuffle.

cuda_sft_file \
  --data /tmp/distill-corpus/all.jsonl \
  --model /workspace/Qwen3.5-4B \
  --output /workspace/distill-round-<N>/adapter \
  --adapter base-round-(N+1) \
  --rank 32 --alpha 64 --lr 1e-4 --epochs 2 \
  --no-anchor-eval  # the cluster IS the regression suite

# Then merge the LoRA into base weights to produce base_(N+1):
kiln adapter merge \
  --adapter base-round-(N+1) \
  --base /workspace/Qwen3.5-4B \
  --output /workspace/Qwen3.5-4B-round-(N+1)
```

**Why this is the default:** SFT on rollouts is the most predictable
training signal. The mega-corpus is the *consolidated* skill of the cluster
in dataset form. SFT loss curves are informative; you can spot bad data.
Rank 32 / alpha 64 is the smallest LoRA that reliably absorbs a multi-cap
mega-corpus on Qwen3.5-4B (round-1 anchor evidence).

**Validation:** §4 below.

### §3.2 Method B: Multi-task OPD with each adapter as teacher

**More expensive but preserves more nuance than SFT.**

```
For each pipeline P in the cluster:
  Serve P.final_adapter via kiln on a dedicated port (P.port).

Distillation trainer reads the cluster prompts, routes each prompt to the
right teacher port (based on which cap it came from), and runs OPD against
that teacher's logits.

cuda_opd_remote_multi_teacher \  # ← new trainer needed (see §6)
  --prompts /tmp/distill-corpus/all-prompts.jsonl \
  --teacher-router /tmp/distill-corpus/router.json \
  --model /workspace/Qwen3.5-4B \
  --output /workspace/distill-round-<N>/adapter \
  --adapter base-round-(N+1) \
  --rank 32 --alpha 64 --lr 1e-4 --epochs 4
```

**Why this beats SFT in some cases:** Token-by-token reverse-KL preserves
distribution shape, not just argmax. For caps where the *style* matters
(e.g. faithful summarization, tool-call shape), OPD often holds the line
better than SFT. The downside is the multi-teacher serving infra cost
(N kiln-serve processes during training).

**Prerequisite:** cuda_opd_remote_multi_teacher does not yet exist as of
this writing. Filing it would be the first kiln work for Phase G.

### §3.3 Method C: Sequential SFT with anchor checks

**Use when Method A or B caused regression in §4 validation.**

```
For each pipeline P in the cluster (ordered by composite_delta, highest first):
  cuda_sft_file --data <P-rollouts> --output adapter-step-<k>
  Run anchor.sh on all previously absorbed caps
  If any anchor regresses > 0.02:
    Lower lr or skip this cap
  Otherwise:
    Promote adapter-step-<k> as the running base for the next step
```

**Why this is the recovery method, not the default:** Sequential SFT
preserves earlier caps more strongly than concurrent SFT, because each step's
gradient only sees one cap's data and the anchor-check stops clobber. But
it's serial — N× wall clock — and tends to settle on the smallest LoRA
that satisfies all anchors, which may not be the strongest.

---

## §4. Validation gate before promoting base_(N+1)

A distilled base candidate `base-round-(N+1)` is NOT promoted to actual
base until it passes ALL of these checks. The cluster_manifest.json records
the result of each check.

### §4.1 Per-cap retention

For each cap P in the cluster:

```
Eval base-round-(N+1) ALONE (no adapter loaded) on P.eval.tasks.jsonl.

Required: composite ≥ 0.80 × P.final_composite_on_base_N
```

That is, base_(N+1) on its own must absorb at least 80% of the lift that
P's adapter gave when running on base_N. If absorption is < 80%, the cap
needs to remain in the next round; if absorption is consistently < 80%
across the cluster, the distillation method itself is the problem.

### §4.2 Cross-cap matrix on new base

For each cap P:

```
Eval base-round-(N+1) + adapter A_Q against cap_P.eval for every other Q in the cluster.

Required: max sibling_delta on the new base ≤ max sibling_delta on the old base.
```

I.e., the new base shouldn't make sibling regression *worse*. If it does,
distillation introduced cross-cap interference that the original cluster
didn't have. Investigate.

### §4.3 Vanilla benchmark check

```
Run BENCHMARKS.md suite on base-round-(N+1).

Required: each benchmark within -0.02 of base_N.
```

If a distilled base regresses HumanEval or MBPP or GSM8K or whatever vanilla
benchmark by > 2pp, the cluster's skills came at a cost. Either accept the
trade explicitly (document in `rounds/round-(N+1)/README.md`) or redo
distillation with lower rank / fewer epochs.

### §4.4 If validation fails

Try, in order:

1. Same method, lower rank/lr/epochs. Distillation may be over-fitting the cluster.
2. Different method (A → B, or A → C if B not available).
3. Smaller cluster. Drop the soft-incompatible caps; distill the strict-compatible core.
4. Accept partial distillation: promote `base-round-(N+1)` but mark some caps as **not yet consolidated**; they continue with their adapter in round N+1.

---

## §5. The rounds/ directory

```
rounds/
├── round-1/
│   ├── README.md            # narrative: what was attempted, what shipped
│   ├── base_sha256          # vanilla Qwen3.5-4B sha
│   ├── capability_summary.jsonl   # one row per shipped cap with final composite
│   ├── cluster_manifest.json      # null for round-1 (no distillation happened)
│   ├── sibling_matrix.json        # cross-cap delta matrix at round close
│   └── distillation_recipe.md     # absent for round-1
├── round-2/
│   ├── README.md
│   ├── base_sha256          # vanilla Qwen3.5-4B (still no distillation)
│   ├── capability_summary.jsonl
│   ├── cluster_manifest.json
│   ├── sibling_matrix.json
│   └── distillation_recipe.md  # planned but deferred
├── round-3/                 # current
│   ├── README.md
│   ├── base_sha256
│   └── (in progress)
└── round-4/                 # future
    └── (placeholder)
```

### §5.1 capability_summary.jsonl

One row per shipped cap. Sorted by composite_delta descending.

```json
{"cap": "pi-faithful-completion", "pipeline_status": "shipped", "stages": 3, "baseline": 0.612, "final": 0.918, "delta": 0.306, "n_stages_used": "sft,opd,agentic-grpo"}
{"cap": "pi-code-comprehension", "pipeline_status": "shipped", "stages": 1, "baseline": 0.611, "final": 0.7405, "delta": 0.1295, "n_stages_used": "agentic-grpo"}
...
```

### §5.2 cluster_manifest.json

What `lib/cluster_summary.py` produces. Describes the cluster chosen for
distillation, the method, and the validation results.

```json
{
  "round": 3,
  "cluster_members": ["pi-faithful-completion", "pi-code-comprehension", ...],
  "cluster_excluded": [{"cap": "pi-X", "reason": "soft-incompatible with pi-Y, sibling_delta=-0.03"}],
  "distillation_method": "A",
  "distill_recipe": {...},
  "validation": {
    "per_cap_retention": {"pi-faithful-completion": {"orig": 0.918, "on_new_base_alone": 0.882, "ratio": 0.961, "passed": true}, ...},
    "cross_cap_matrix": {...},
    "vanilla_benchmarks": {"humaneval": 0.71, "mbpp": 0.59, "gsm8k": 0.82, "deltas": [-0.01, 0.00, -0.02], "passed": true}
  },
  "promoted_to_base": true,
  "new_base_sha256": "<sha>",
  "ts": "..."
}
```

### §5.3 Round README

The agent-written narrative for the round. What new caps were added, which
stages worked, which got retired, what the distilled base did to the
overall capability surface. This is the artifact a human researcher reads
to understand why round N produced base_(N+1).

---

## §6. Kiln work needed to support distillation (Phase G prerequisites)

These items are NOT in the round-1 + round-2 backlog
(`KILN_IMPROVEMENT_ISSUES.md` is complete). They'd be filed as the first
new kiln issues when Phase G starts:

1. **`kiln adapter merge`** — produce a merged-weight base from
   `base_N + LoRA`. Currently no first-class CLI; the workflow is bespoke.
2. **`cuda_opd_remote_multi_teacher`** — distillation Method B requires a
   trainer that routes each prompt to a different teacher port. Needed
   only if Method A doesn't pass validation.
3. **`kiln cluster eval`** — efficient batched eval of one adapter against
   multiple caps' eval sets simultaneously. Used heavily by
   integration/cross-cap-coherence and by §4 validation.
4. **`kiln rollout --teacher-routing`** — generate teacher rollouts at scale
   for Method A's mega-corpus.
5. **`kiln serve --multi-adapter`** — serve multiple adapters on different
   slots of one server, eliminating the N-process serving cost for Method B.

When Phase G is approached, these are the first issues to file.

---

## §7. The flywheel as a whole

```
Round 1: vanilla Qwen3.5-4B base
  → run capability experiments
  → ship 4 winners (pi-doctest, pi-code-comprehension, pi-faithful-completion, pi-code-search)
  → no distillation (too few caps)
  → snapshot rounds/round-1/

Round 2: same vanilla base (round 1 too thin to distill)
  → add OPD caps (6) + reshaped agentic caps + new agentic caps
  → ship ~10 pipelines
  → all infrastructure landed (40 kiln issues)
  → no distillation (still single-method)
  → snapshot rounds/round-2/

Round 3 (current, after unification): vanilla base
  → run multi-stage pipelines per METHODS.md routing
  → ship 8-12 pipelines with chain composite >> single-method composite
  → first distillation produces Qwen3.5-4B-cluster-round3 if validation passes
  → snapshot rounds/round-3/

Round 4: base = Qwen3.5-4B-cluster-round3
  → re-baseline all caps against new base
  → 2-4 caps "consolidate" (composite_delta on new base alone is now < 0.05)
    → mark retired in pipeline.md
  → 4-8 caps still have lift on new base, possibly with shorter pipelines
  → 2-4 NEW caps become tractable (harder caps that previously had no headroom)
  → ship pipelines; second distillation
  → snapshot rounds/round-4/

Round N: base = Qwen3.5-4B-cluster-round(N-1)
  → diminishing returns: each round consolidates a few caps, adds a few harder ones
  → eventually no shipped cap has composite_delta > 0.05 → no new distillation
  → declare round N base = the round's actual release base
```

The terminal condition is *capability cluster goes to zero new shipped
pipelines on round N*. At that point you've extracted everything a 4B
LoRA cluster can extract from the cap suite. The shipped base is the
final artifact of the project.

---

## §8. Why this is the right level of formalism

A capability cluster + distillation flywheel could be three lines in a
research notebook. We need this much formalism because:

1. **Distillation can clobber.** Without §4 validation, base_(N+1) can be
   worse than base_N on un-distilled tasks. Vanilla benchmarks catch this.
2. **Cluster choice is non-trivial.** Greedy by composite_delta is right
   *most* of the time but soft-incompatibility forces explicit decisions.
3. **Reproducibility matters across rounds.** Re-running round-3 on a new
   base means each pipeline's pipeline.md must hold all the recipe info,
   not chat history.
4. **The skill must know when to stop.** Round-N+1 is wasted GPU if no caps
   improved on the new base. The flywheel has a natural terminal condition
   and we have to recognize it.

---

## §9. What to do *right now* in Phase A

Phase G (distillation) is deferred until ≥ 5 multi-stage round-3 pipelines
ship. But the supporting structure should exist now:

- `rounds/round-1/` and `rounds/round-2/` snapshots are populated from
  existing `CONSOLIDATED_REPORT.md` and round-2 capability rows.
- `rounds/round-3/` skeleton is empty; populated as pipelines ship.
- `lib/cluster_summary.py` exists and parses `pipeline.md` headers but is
  not yet exercised at scale.
- This doc (`DISTILLATION.md`) is the contract for when Phase G fires.

That's it. The Phase G work itself happens later. Today's job is to make
the per-cap pipelines that will eventually feed Phase G.

---

## §10. Open questions for Phase G (revisit when starting)

These are deliberately deferred. They get answered with data, not now:

1. **Optimal distillation rank?** Method A uses rank 32 / alpha 64 as a
   starting point, but the round's cluster size and skill diversity may
   change this. Sweep at Phase G start.
2. **Should distillation happen on the LoRA or on merged weights?**
   Merging into base weights (`kiln adapter merge`) is irreversible. Keeping
   the distilled LoRA as the new "base" allows easier rollback. The right
   default is probably "merge after a round of validation" but TBD.
3. **How much capability data should the distillation corpus contain?**
   Round 2 shipped pipelines have a few thousand training rollouts each;
   the full cluster mega-corpus could be 30-50K rows. Whether to subsample
   per cap to balance the cluster is unknown.
4. **Anchor-suite for the distilled base.** What non-target capabilities
   should distillation NOT regress? The vanilla benchmarks in
   `BENCHMARKS.md` are the floor; whether we need cap-specific anchors is
   TBD.
5. **Round-over-round drift on rubrics.** If a round-2 rubric was too lax
   and we tightened it in round 3, does the round-2 distillation evidence
   still apply? Probably not — re-run the cap pipeline on the new rubric
   before including in a cluster.

These get answered when Phase G actually fires.
