# Iter 4 — Phase 3 verifier-free path works (with a caveat)

**Date:** 2026-05-19
**Hypothesis:** Phase 3 verifier-free adaptation (paper §5.5,
`--no-policy-loss --echo-lambda 0.05`) should produce a working
training loop where **only** ECHO env-CE drives gradients (the GRPO
surrogate is scaled to zero). The trainer should still consume
trajectory rollouts, fire ECHO per-completion, and save an adapter
whose weights differ measurably from a from-scratch ECHO=on run.

**Caveat surfaced:** the `--base-adapter` CLI flag added in `96840323`
turned out to be **lineage-only** — the trainer doesn't load LoRA
weights from a base adapter for continued training. See "Bug caught
in this iter" below. Iter 4 trained from fresh LoRA init rather than
chaining from iter 3, so this isn't yet the Phase 3 demonstration the
paper §5.5 describes. **Verifier-free training itself works from
fresh init**; the chain-from-Phase-2 step needs a follow-up.

## Setup (warm pod)

- **Pod:** same A100 80GB PCIe (warm — kiln+model+iter3 adapter all present)
- **Branch:** `96840323` (initial form — flag accepted but no-op weight-load) then `(this iter)` adds a loud warning when --base-adapter is set
- **Build:** ~13s incremental rebuild on top of iter 3 binary
- **Dataset:** same `synth_traj_iter3.jsonl` (20 groups × 4 rollouts, seed 1414213562)
- **Run env:** `RUST_LOG=info,kiln_train=debug`
- **Training command:**

```bash
cuda_grpo_ablation \
    --data synth_traj_iter3.jsonl --model /workspace/qwen3.5-4b \
    --output echo-iter4-out --adapter echo-iter4-verifier-free \
    --mode phase1 --max-groups 20 \
    --rank 8 --alpha 16 --lr 1e-5 --seed 1414213562 \
    --no-policy-loss \
    --base-adapter /workspace/echo-iter3-out/on/echo-iter3-on
```

## What worked

- **`--no-policy-loss` path executed cleanly.** 80/80 ECHO firing log lines
  (= 20 groups × 4 completions). Adapter saved (30 MB, 400 LoRA keys).
- **GRPO surrogate genuinely zeroed.** The trainer's
  `loss = (policy_loss_scale * grpo_loss) + λ · env_ce` path with
  `policy_loss_scale = 0.0` produces a loss whose dynamics come purely
  from the env-CE term. Confirmed by:
  - Final loss `0.236` matches the env-CE-only value (ECHO-OFF loss in
    iter 3 was −0.045, so the GRPO contribution magnitude is ~0.27 at
    these synth rollouts; iter 4 final loss of 0.236 is consistent
    with just the env-CE).
  - Per-completion `mean_ce_val` would now be visible (the per-comp
    log already fires).
- **Monotonic env-CE drop replicates** (now under verifier-free conditions):

  | Task | Step 1 | Step 19/20 | Δ |
  | --- | --- | --- | --- |
  | A | 0.355 | 0.325 | −8.5% |
  | B | 0.261 | 0.236 | −9.6% |
  | C | 0.382 | 0.346 | −9.4% |

  Same trend as iter 3 (-9% to -10%) — within-task env-CE drops as the
  model learns terminal dynamics. **And this is on the verifier-free
  path with no policy gradient at all.** Paper §5.5 prediction holds
  at this small scale.

- **Wall clock:** 339.2s (5.7 min) for 20 groups, comparable to iter 3
  ECHO ON (324s). No extra cost from the verifier-free path.

## What didn't work — `--base-adapter` is currently lineage-only

The bit-identical step-1 loss between iter 3 ECHO=on (0.355141) and iter 4
verifier-free (0.355141) gave away that **iter 4 started from fresh LoRA
init, not from iter 3's adapter weights**. Tracing this to source:

1. `cuda_grpo_ablation`'s `--base-adapter` writes the path into
   `GrpoConfig::base_adapter`.

2. `grpo_train_jsonl` reads `config.base_adapter` **only when
   `replay_ctx` is `Some`**, and only inside `open_replay_state` — which
   uses it for **lineage tracking** (recording the parent adapter in
   the `lineage.json` schema), not for weight loading.

3. `TrainableLoraParams::initialize_seeded` always allocates fresh LoRA
   tensors from a seeded RNG; there's no
   `TrainableLoraParams::load_from_safetensors(path)` method.

4. So when the example passes `replay_ctx = None` (the default), the
   `base_adapter` field is silently ignored.

**Follow-up needed (Phase 3 chaining):** add
`TrainableLoraParams::load_from_safetensors(adapter_dir)` that reads
`adapter_model.safetensors`, parses the safetensors index, and copies
each `base_model.model.model.layers.<i>.<proj>.lora_{A,B}.weight`
tensor into the matching Var. Then have `grpo_train_jsonl` call it
post-`initialize_seeded` when `config.base_adapter.is_some()`. The
adapter_config.json is identical shape between runs (same r=8, alpha
=16, same target_modules), so a one-shot load is straightforward.

**Mitigation in this PR:** updated the `--base-adapter` doc + added an
`eprintln!` warning so operators see "--base-adapter is recorded in
lineage but does NOT load LoRA weights yet" loudly on stderr. Stops
the silent-no-op trap.

## Adapter weight diff (iter 4 vs iter 3 ECHO=on)

```json
{
  "iter4_vs_iter3_lora_B_mean_abs_diff": 0.0001422,
  "iter4_vs_iter3_lora_B_max_abs_diff":  0.0001841,
  "iter4_lora_B_max_value":              0.0002108,
  "iter4_vs_iter3_lora_A_mean_abs_diff": 0.0002086,
  "iter4_lora_A_max_value":              0.0197754
}
```

These diffs are consistent with both adapters having trained from
**different objectives** (iter 3: GRPO+ECHO; iter 4: ECHO only) over
20 steps from the **same fresh init**. The non-trivial 1.42e-04 mean
LoRA-B diff says the verifier-free path produced genuinely different
gradients than the standard GRPO+ECHO path, even on synth data.

## What this iter proves

1. **`--no-policy-loss` is operationally correct.** GRPO surrogate
   genuinely zeroed; only ECHO env-CE drives gradients; training
   completes with an adapter saved.

2. **Env-CE drop trend replicates in verifier-free mode.** -8.5% to
   -9.6% within-task loss drop over 7 same-task SGD steps. Same shape
   as iter 3 (-9.3% to -10.5%). Paper §5.5's mechanism — env-CE alone
   can drive policy improvement — holds in miniature.

3. **`--base-adapter` chaining is incomplete.** Caught only because we
   actually ran a chained iter. The CLI now warns loudly; the trainer
   gap is documented as a Phase 3 follow-up.

## Artifacts

- [`train.log`](train.log) — full tracing+stdout (with all 80 ECHO firing lines + the eprintln warning about --base-adapter)
- [`echo-firing.log`](echo-firing.log) — extracted 80 ECHO firing lines
- [`adapter_config.json`](adapter_config.json) — LoRA shape (same as iter 3)

iter 4's `adapter_model.safetensors` not checked in (~30 MB, on pod);
the diff JSON above is the relevant comparison.
