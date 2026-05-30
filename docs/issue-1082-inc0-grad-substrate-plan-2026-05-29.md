# Issue #1082 — Increment 0 (kt-native training substrate) mergeable decomposition (2026-05-29)

> Map+synthesis of the kiln-train grad-delivery/optimizer-keying surface. 6-PR sequence to move off candle GradStore/Var/TensorId. PR1 (moments re-key) landed first.

All anchor points confirmed exactly as the maps describe. The `moments` re-keying is provably isolated: `OptimizerState.moments` is read at exactly two key-bearing sites (`7007`, `7324`), written at one (`785`), and iterated key-agnostically at three (`827`, `841`, `859`). The `sgd_step` and `sgd_step_from_map` paths don't touch moments. The synthesis is complete.

# Increment-0 Synthesis: kt-native grad delivery + optimizer keying

## Resolving the four-map disagreement

The maps split on what the *first* PR should be:

- **Maps A/B** push toward replacing the SFT/GRPO GradStore-harvest hack (`trainer.rs:10756-10785`, `11267-11296`) — the headline Increment-0 target.
- **Map D** identifies a strictly smaller, provably-isolated first step: re-keying `OptimizerState.moments` from `candle TensorId` to `KtTensorId`.

**Map D wins for PR #1.** The GradStore-harvest replacement (A/B) is the *destination*, but it is large and risky: it changes a function's return type (`GradStore` → kt map), relocates the `kt_tensor_to_candle_cuda_copy` boundary, touches three call sites + telemetry, and forces a decision on the candle CPU/opt-out paths *in one PR*. Map D's moments re-keying advances the *same keying migration* the harvest needs (it forces `cd_tensor_id_to_kt` into the live optimizer path and exercises the kt key under the CP-4 gates), is ~5 edits in one file, crosses zero API boundaries, and is the exact migration the in-tree doc sketch (`cd_types.rs:73-88`) prescribes as "the next call site."

I verified the isolation directly: `OptimizerState.moments` is key-read only at `trainer.rs:7007` and `7324`, key-written only at `785`, and iterated key-agnostically at `827`/`841`/`859`. opd.rs and the SFT/GRPO loops thread `OptimizerState` opaquely. This is genuinely standalone.

---

## 1. Increment-0 mergeable decomposition

Ordered by dependency. Every PR keeps `main` building under both `--features cuda` and default, and keeps the three CP-4 gates green.

### PR 1 — Re-key `OptimizerState.moments` to `KtTensorId`
- **Scope:** Flip the AdamW moment map's key type from candle `TensorId` to `KtTensorId`, routing every key through `cd_tensor_id_to_kt`. No grad-container or `Var`-storage change.
- **Files:** `crates/kiln-train/src/trainer.rs` only.
- **Changes:** `OptimizerState.moments` field type (`814`); the alloc local (`777`); the insert (`785`); the two key reads (`7007`, `7324`). 5 edits, 0 new imports (`use crate::cd_types::*;` at `411` already brings `KtTensorId` + `cd_tensor_id_to_kt` into scope since both are `pub(crate)`).
- **Why mergeable:** The moments map never crosses an API boundary; `cd_tensor_id_to_kt` is equality-preserving (`cd_types.rs:113-120`, tested `213-255`). The candle `GradStore`/`GradMap` (`grads`) stays candle — it's a *separate* map. AdamW math and `Var` storage untouched. Removes the first real consumer of the candle `TensorId` alias.
- **Validation:** `tape_authoritative_sft_converges_bf16` (exercises 100-step AdamW with the re-keyed moments), `tape_authoritative_grads_match_candle_baseline_bf16`, `tape_grad_matches_finite_difference_bf16`, full `cargo nextest run -p kiln-train --features cuda`.

### PR 2 — Add `kiln_autograd::GradStore`-typed SFT producer (parallel, behind the existing tape gate)
- **Scope:** Add a new function `standard_forward_backward_tape_authoritative_kt` returning `(f64, kiln_autograd::GradStore)` (kt-keyed) that builds the kt grad map directly from `grads_by_candle_raw` — no `loss.backward()` harvest, no `kt_tensor_to_candle_cuda_copy`. **Do not yet repoint call sites.** The old `GradStore`-harvesting version stays so `main` builds.
- **Files:** `crates/kiln-train/src/trainer.rs`.
- **Changes:** New fn cloned from `standard_forward_backward_tape_authoritative` (`10715-10787`); body deletes the harvest (`10752-10758`) and per-grad copy (`10781`); for each `(candle_raw, kt_grad)` in `grads_by_candle_raw`, if `var_by_raw` contains it, `store.insert(KtTensorId::from_raw(candle_raw as u64), kt_grad)`. Uses `kiln_autograd::GradStore::new()` (public — `grad_store.rs:14-67`).
- **Why mergeable:** Dead code (no caller) → no behavior change, both profiles build. `kiln_autograd` is already a dep.
- **Validation:** Same 3 gates + suite (proves it compiles; behavior validated in PR 4).

### PR 3 — Add kt-keyed optimizer consumer `optimizer_step_from_kt_map`
- **Scope:** Add an `optimizer_step` variant reading `kiln_autograd::GradStore` (kt-keyed), bridging the kt grad to candle at the per-Var boundary via `kt_tensor_to_candle_cuda_copy` so AdamW math + `Var` storage stay candle. Reuses the PR-1 kt-keyed `moments`.
- **Files:** `crates/kiln-train/src/trainer.rs`.
- **Changes:** New fn cloned from `optimizer_step_from_map` (`7294-7348`): `let kt_id = cd_tensor_id_to_kt(var.as_tensor().id())`; `grads.get(kt_id)` → kt tensor → `kt_tensor_to_candle_cuda_copy(...)` → existing `apply_adamw_update`; `state.moments.get(&kt_id)` (already kt-keyed from PR 1). No caller yet.
- **Why mergeable:** Dead code; `apply_adamw_update`/`apply_sgd_update` unchanged (still take candle `&Tensor` grad).
- **Validation:** Same gates + suite.

### PR 4 — Repoint SFT call site + grad-norm telemetry to the kt path
- **Scope:** Switch the SFT non-checkpointed CUDA path to PR-2 producer → PR-3 consumer. Migrate `observe_lora_grad_norms_from_grad_store` (`5220`) usage at `2300` to the kt-map sibling (`5207`). This is the first behavior change.
- **Files:** `crates/kiln-train/src/trainer.rs` (`2289-2308`).
- **Why mergeable:** Only the tape-authoritative CUDA branch flips; the candle CPU / `KILN_USE_TAPE_AUTHORITATIVE=0` fallbacks (`10835-10874`) are untouched, so the function's overall contract is preserved by keeping `standard_forward_backward` returning a unified shape (lift residual candle backward through the trivial `GradStore→kt-map` conversion already demonstrated in-tree at `16528-16533`).
- **Validation:** `tape_authoritative_sft_converges_bf16` is now the real gate — it runs the kt producer→consumer end-to-end. Plus the other two gates + full suite.

### PR 5 — Same flip for GRPO
- **Scope:** Apply PR-2/PR-3 pattern to GRPO: kt-typed producer for `grpo_step_forward_backward_tape_authoritative` (`11191`), delete the GRPO harvest (`11267-11296`), repoint the token-level `merge_grad_maps`+`optimizer_step_from_map` (`4813`/`4825`) and per-completion path (`5069-5092`). Leave `grpo_candle_shim.rs:225` loss-head oracle in place (separate, localized).
- **Files:** `trainer.rs`, possibly `grpo_candle_shim.rs` (no edit if oracle stays).
- **Why mergeable:** GRPO checkpointed/ECHO paths stay candle. CP-4 SFT gates unaffected; GRPO has its own coverage in the cuda suite.
- **Validation:** Full kiln-train cuda suite (GRPO tape tests).

### PR 6 — Decide CPU/opt-out residual-candle fate
- **Scope:** Either CUDA-gate the candle CPU fallback (`10806` comment says it's exercised only by `perf_regression_sft_train_cpu_smoke`) or give it a kt-CPU route. Resolve `KILN_USE_TAPE_AUTHORITATIVE=0` opt-out path. Delete the now-orphaned `GradStore`-harvest helpers.
- **Files:** `trainer.rs`.
- **Why last:** It's the only PR that can remove the candle GradStore alias entirely; must follow all consumers being kt.
- **Validation:** Full suite + `perf_regression_sft_train_cpu_smoke_completes_under_30s` (Tier-1b).

---

## 2. FIRST PR — exact edits

PR 1: Re-key `OptimizerState.moments` to `KtTensorId`. All edits in `crates/kiln-train/src/trainer.rs`. Both profiles build (this is pure kiln-train, CUDA-agnostic — the `Var` storage and candle GradStore are untouched).

### Edit 1 — field type (line 814)

```rust
pub struct OptimizerState {
    pub moments: HashMap<TensorId, AdamWMoments>,
    pub step: u32,
}
```
→
```rust
pub struct OptimizerState {
    pub moments: HashMap<KtTensorId, AdamWMoments>,
    pub step: u32,
}
```

### Edit 2 — alloc local + insert (lines 777, 785)

```rust
    pub fn allocate_adamw_state(&self, device: &CdDevice) -> Result<OptimizerState> {
        let mut moments: HashMap<TensorId, AdamWMoments> = HashMap::new();
        for var in self.all_vars() {
            let shape = var.as_tensor().shape().clone();
            let dtype = var.as_tensor().dtype();
            let m = var_zeros(shape.clone(), dtype, device)
                .with_context(|| "allocating AdamW first-moment Var")?;
            let v = var_zeros(shape, dtype, device)
                .with_context(|| "allocating AdamW second-moment Var")?;
            moments.insert(var.as_tensor().id(), AdamWMoments { m, v });
        }
        Ok(OptimizerState { moments, step: 0 })
    }
```
→
```rust
    pub fn allocate_adamw_state(&self, device: &CdDevice) -> Result<OptimizerState> {
        let mut moments: HashMap<KtTensorId, AdamWMoments> = HashMap::new();
        for var in self.all_vars() {
            let shape = var.as_tensor().shape().clone();
            let dtype = var.as_tensor().dtype();
            let m = var_zeros(shape.clone(), dtype, device)
                .with_context(|| "allocating AdamW first-moment Var")?;
            let v = var_zeros(shape, dtype, device)
                .with_context(|| "allocating AdamW second-moment Var")?;
            moments.insert(
                cd_tensor_id_to_kt(var.as_tensor().id()),
                AdamWMoments { m, v },
            );
        }
        Ok(OptimizerState { moments, step: 0 })
    }
```

### Edit 3 — GradStore-path read (line 7007)

```rust
            for var in params.all_vars() {
                if let Some(grad) = grads.get(var.as_tensor()) {
                    let moments = state.moments.get(&var.as_tensor().id()).ok_or_else(|| {
                        anyhow::anyhow!(
                            "optimizer_step: missing AdamW moments for Var id {:?}",
                            var.as_tensor().id()
                        )
                    })?;
```
→
```rust
            for var in params.all_vars() {
                if let Some(grad) = grads.get(var.as_tensor()) {
                    let moments = state
                        .moments
                        .get(&cd_tensor_id_to_kt(var.as_tensor().id()))
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "optimizer_step: missing AdamW moments for Var id {:?}",
                                var.as_tensor().id()
                            )
                        })?;
```

### Edit 4 — map-path read (line 7324)

Context: `let id = var.as_tensor().id();` is at `7317`; `grads.get(&id)` at `7318` reads the **candle `GradMap`** and MUST stay candle-keyed. Only the moments lookup re-keys.

```rust
                let id = var.as_tensor().id();
                if let Some(grad) = grads.get(&id) {
                    let grad = if grad.device().same_device(var.as_tensor().device()) {
                        grad.clone()
                    } else {
                        grad.to_device(var.as_tensor().device())?
                    };
                    let moments = state.moments.get(&id).ok_or_else(|| {
                        anyhow::anyhow!(
                            "optimizer_step_from_map: missing AdamW moments for Var id {:?}",
                            id
                        )
                    })?;
```
→
```rust
                let id = var.as_tensor().id();
                if let Some(grad) = grads.get(&id) {
                    let grad = if grad.device().same_device(var.as_tensor().device()) {
                        grad.clone()
                    } else {
                        grad.to_device(var.as_tensor().device())?
                    };
                    let moments =
                        state.moments.get(&cd_tensor_id_to_kt(id)).ok_or_else(|| {
                            anyhow::anyhow!(
                                "optimizer_step_from_map: missing AdamW moments for Var id {:?}",
                                id
                            )
                        })?;
```

### Imports / new types
**None.** `KtTensorId` (`cd_types.rs:99`) and `cd_tensor_id_to_kt` (`cd_types.rs:122`) are both `pub(crate)` and already in scope in trainer.rs via the wildcard `use crate::cd_types::*;` at `trainer.rs:411`. No new helper needed.

### Sites verified to need NO change
- `register_with_backend` (`827`), `evict_from_backend` (`841`), `sync_to_candle` (`859`) — all iterate `self.moments.values()`, key-agnostic.
- `grads.get(var.as_tensor())` (`7006`) / `grads.get(&id)` (`7318`) — read the candle `GradStore`/`GradMap`, NOT moments; stay candle.
- `sgd_step` (`6973`) / `sgd_step_from_map` — read `grads` only, never `moments`.
- opd.rs / SFT / GRPO loops — thread `OptimizerState` opaquely (`opt_state.as_mut()`), never key into `.moments`.

---

## 3. Risks + validation

**Riskiest assumption — `cd_tensor_id_to_kt` is collision-free within a single moments map.** The map is keyed exclusively by bridged candle `Var` ids (`785` insert, `7007`/`7324` reads), and `usize → u64` widening is injective on 64-bit hosts (`cd_types.rs:113-120`, tested at `213-255`). No `KtTensorId::next()`-minted ids ever enter this map, so cross-space collision is impossible. **Confirmed sound** — but the pod test is what proves no production `Var` id exceeds the value space or hits an unexpected path.

**Hazard — `optimizer_step` AdamW vs SGD both consume `OptimizerState`?** No: SGD (`sgd_step` `6993`) never touches `moments`. Only the AdamW arm reads moments. Re-keying is AdamW-only by construction; SGD runs are unaffected. The CP-4 SFT convergence gate uses AdamW, so it exercises the changed path.

**Hazard — shared `apply_adamw_update` between tape + candle-auth.** PR 1 does NOT touch `apply_adamw_update`/`apply_sgd_update` (`7111`/`7051`) — both still receive a candle `&Tensor` grad. The moments map a moment Var is looked up *from* is the only thing re-keyed; the moment Vars themselves and the update math are byte-identical. Both the GradStore consumer (`optimizer_step`, used by tape non-ckpt SFT + GRPO per-completion) and the GradMap consumer (`optimizer_step_from_map`, used by OPD + GRPO token-level) read the same re-keyed map, so they stay consistent.

**Hazard — CPU-path candle requirement / ECHO.** Irrelevant to PR 1: the moments map is allocated and keyed identically on CPU and CUDA (`allocate_adamw_state` takes any `device`); ECHO-active steps still go through the same `OptimizerState`. No path is gained or lost. (These hazards bite PRs 4-6, not PR 1.)

**What the pod build/test MUST confirm:**
1. `main` builds under `--features cuda` AND default (no-cuda) — PR 1 is CUDA-agnostic so both should be trivially green; this catches any stray `cd_types::TensorId` moments reference the grep-survey missed.
2. **`tape_authoritative_sft_converges_bf16`** — the load-bearing gate: 100-step AdamW with the re-keyed moments must still converge identically (moment state must resolve every step, no "missing AdamW moments" error).
3. **`tape_authoritative_grads_match_candle_baseline_bf16`** + **`tape_grad_matches_finite_difference_bf16`** — confirm grad math unchanged (these don't touch moments keying directly but gate the whole optimizer path).
4. Full `cargo nextest run -p kiln-train --features cuda` — catches the GradMap consumer (`optimizer_step_from_map`, used by OPD/GRPO) reading the re-keyed moments correctly, since OPD/GRPO tests hit `7324`.

**sccache caveat (from memory):** preface the RunPod build with `cargo clean -p kiln-gdn-kernel && SCCACHE_RECACHE=1` to avoid the stale-CUDA-dlink `cudaErrorSymbolNotFound` flake — though PR 1 touches no kernel code, the gdn_gates SASS-drop hazard is build-cache-wide.

**kt substrate thread-safety (from memory):** the CP-4 cuda suite must run **single-threaded** (`--test-threads=1` or structural gating) — concurrent multi-thread GPU kt ops corrupt memory and can false-pass/flake. This is a test-harness requirement independent of PR 1, but applies to every validation run in this decomposition.
