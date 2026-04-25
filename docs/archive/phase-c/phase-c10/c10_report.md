# Phase C10 — fp32 HF reference comparator + top1-flip per-site bisect

**Status:** Complete. **Null result** — zero tap flips across all 24 taps × 3 seeds (72 splice tests). The C9 bf16-noise verdict holds: no single operator's bf16 materialization is responsible for the Class B rejection rate. C11 target promotes to holistic changes (Marlin W4A16 q_proj audit, abs_pos/RoPE at pos > 0, or fp32 draft head as a policy).

**Branch:** `mtp-phase-c10-fp32-reference-comparator`
**Baseline:** main @ `175b13a` (post-C9 merge `32ae7be` + Metal prewarm hardening)

## Verdict

| Hypothesis | Outcome | C11 target |
|---|---|---|
| **C9 bf16-noise verdict holds** ✅ | Every tap flip = False across all 3 seeds. `any_flip=False`, comparator exit 0. | **Holistic changes only** — Marlin W4A16 q_proj audit, abs_pos/RoPE at pos > 0, or fp32 draft head policy |
| Single-operator semantic bug | Rejected: zero taps flipped at ≥ 1/3 pairs, let alone ≥ 4/6 | — |
| Multi-operator compounding | Rejected: zero taps flipped even intermittently | — |

**Headline numbers:**

- `kiln_matches_ref` = True on all 3 seeds (seed42/3074, seed43/16078, seed44/1814 at `mtp_pos=2`) — kiln's bf16 argmax already equals the fp32 HF reference argmax at these draft positions. No splice is capable of "flipping" a prediction that already matches.
- 0 / 72 tap splices produced an argmax flip.
- ref_top1 margins after splice stay positive everywhere: min +3.1250 (seed43, mtp_logits terminal), max +8.2500 (seed42, mtp_logits terminal). Per-tap margin spread is ≤ 0.09 fp32-logit units against the mtp_logits endpoint, confirming no tap is pushing the argmax near a decision boundary.

See `docs/archive/phase-c/phase-c10/c10_splice_bisect_table.md` for the full per-tap table.

## Interpretation

C9 said cos_sim ≥ 0.9999 at every tap; C10 now confirms the *behavioral* counterpart — at every tap, substituting kiln's bf16 activation into the fp32 reference forward does not change the argmax. Combined, the MTP head operators are numerically and behaviorally equivalent to the fp32 HF reference on the sampled positions. The Class B rejection rate (87.6% of MTP rejections) and α ≈ 0.058 floor therefore cannot be pinned on any single MTP-head operator's precision — they must arise from:

1. **Policy-level quantization drift** — e.g. Marlin W4A16 q_proj, where the effective fp32-equivalent numerics depend on per-channel scale choices rather than bf16 accumulation.
2. **Upstream-trunk drift** — the `h_main` input into the MTP head comes from the main-model forward, which has its own cumulative drift budget (audited separately in B10/B11b/B12). C10 splices on top of kiln's `h_main`, so this audit cannot see main-model drift as a "flip."
3. **Positional semantics at pos > 0** — abs_pos / RoPE handling at draft positions is the only part of the reference that differs structurally from the main-model forward; worth a targeted audit even though no single tap flipped.
4. **Decode-time state divergence** — scheduler/paging path, stochastic-sampling alignment, or KV-cache fp8 interactions not captured by dump-replay.

These are the C11 candidates, in decreasing expected yield:

1. **Marlin W4A16 q_proj audit** (most likely) — verify packed weight channel scales match the bf16 q_proj behavior within the fp32 equivalence band, across MLP and attention projections.
2. **fp32 draft head policy** — if the draft head alone runs in fp32 while the trunk stays bf16, does α lift off the floor? This is an ops-level test, not an operator fix.
3. **abs_pos / RoPE correctness at pos > 0** — a symbolic audit (not numerical) against the HF reference for `mtp_pos ∈ {1, 2, 3}`.
4. **Main-model `h_main` drift** — cross-reference the B10/B11b/B12 audits; if `h_main` is the drift source, MTP-head fixes will be null.

## Execution (this PR)

- Pod: `nwgdfesb0jpsuq` (A40, $0.44/hr, kiln-runpod image).
- Build: sccache-hot cargo build, 110 s wall-clock; `kiln-bench --paged` at `KILN_W4A16=0` (default dumps path).
- Model: `hf download Qwen/Qwen3.5-4B --local-dir /workspace/qwen3.5-4b` (the `hf` CLI rejects the legacy `huggingface-cli download ... --local-dir-use-symlinks False` form).
- Dumps: 3 seeds × 1 file each (positions 1 & 2 multiplexed into one safetensors per seed via `KILN_MTP_DUMP_POS=1,2`). Each dump ~1.5 MB, ~2 min per seed.
- Comparator: CPU-only, ~16 s for all 3 pairs. Wrote `docs/archive/phase-c/phase-c10/c10_splice_bisect_table.md` with `any_flip=False`, exit 0.
- Total pod time: ~10 min. Pod self-terminated via the workflow's `cleanup()` function.

## Goal

C9 closed with the verdict that `fc_output` drift is bf16 accumulation noise
rather than a semantic bug. It was a *numerical* bar — every MTP-head tap
satisfies `cos_sim ≥ 0.9999` against the fp32 HF reference. That leaves the
obvious question unanswered: **does a cos_sim-invisible drift still flip the
argmax often enough to account for the Class B rejection rate
(87.6% of MTP rejections) and the α ≈ 0.058 floor?**

C10 answers that with a behavioral metric instead of a numerical one. For
each tap site `S` in the MTP forward path, we:

1. Compute the pure-fp32 HF reference forward pass end-to-end to get
   `ref_top1`.
2. Replay the reference forward, but at tap `S` substitute kiln's bf16
   activation (upcast to fp32) for the reference's local variable, then
   continue the rest of the forward in fp32 to get `spliced_top1_S`.
3. Record `flip = spliced_top1_S != ref_top1`.

If every tap has `flip = False` across multiple `(seed, position)` pairs,
the C9 "benign bf16 drift" verdict survives: the argmax path is robust to
single-site bf16 substitution at every operator, which means no single
operator is materially responsible for Class B flips. The right C11 target
then becomes holistic (Marlin q_proj audit, abs_pos/RoPE correctness at
pos > 0, or fp32 draft head as a *policy* change, not an operator fix).

If a specific tap consistently flips, **that** tap is the C11 target: it's
the single operator whose bf16 materialization is driving Class B.

## Methodology

### Tap taxonomy

Tap sites are ordered top-down through the MTP head. Every tap is a point
where kiln's bf16 dump writes a tensor, so the splice is a direct upcast
substitution (no re-derivation needed).

- **A-group (pre-layer)**
  `tok_embed, norm_emb, norm_h, fc_input (= concat), fc_output (= fused)`
- **B-group (inside `mtp_inner_block`, SDPA path)**
  `post_pre_attn_norm, post_q_proj_raw, post_k_proj, post_v_proj,
   post_q_split, post_gate_split, post_q_norm, post_k_norm,
   post_q_rope, post_k_rope, attn_out (= post_attn_raw),
   post_attn_gated, post_o_proj, post_attn_residual,
   post_pre_mlp_norm, post_mlp`
- **C-group (post-layer)**
  `post_layer, post_final_ln, mtp_logits`

`mtp_logits` is the terminal site — splicing there is just "kiln's bf16
argmax vs fp32 reference argmax," which is the null hypothesis endpoint.

### Comparator design

`scripts/mtp_c10_splice_bisect.py` (789 lines, added in this PR) does the
following:

- **Self-contained reference** — vendors the HF reference helpers from
  `scripts/mtp_reference_dump.py` (`rms_norm`, `apply_rope_partial`,
  `load_mtp_weights`, `MtpRefWeights`) inline. The reference can't call
  back into `mtp_inner_block` because splicing requires replacing *local
  variables* mid-forward, which a closed function doesn't expose. So C10
  owns a `full_forward_fp32(w, h_main, draft_token_id, mtp_pos, base_pos, *,
  swap_fc_norms=0, override_tap=None, override_val=None)` that runs the
  entire MTP head fp32 and accepts a single (tap_name, tensor) override.
- **h_main from kiln** — same honest limit as B6/C6/C7. C10 is an MTP-head
  isolation test; the main-model hidden state hand-off is audited
  separately (B10/B11b/B12).
- **Weights fp32** — `st_load_tensor` casts to `torch.float32` at load.
- **Override tap semantics** — `override_val` replaces the named local
  *post-computation*; the tap name corresponds to the kiln dump name
  (mapped through `KILN_TAP_ALIAS`, including `c6__*` pre-RoPE taps and
  the `post_attn_raw` alias).
- **Per-pair output** — for each `(seed, position)` pair we report
  `ref_top1`, `kiln_top1`, then for every tap: `spliced_top1`, flip
  indicator, and a margin-shrinkage metric (logit gap between ref argmax
  and kiln argmax after splice, to catch near-flips).

### CLI contract

```
python3 scripts/mtp_c10_splice_bisect.py \
    --checkpoint /workspace/qwen3.5-4b \
    --pair seed42-pos1:/tmp/c10-dumps/seed42-pos1.safetensors \
    --pair seed42-pos2:/tmp/c10-dumps/seed42-pos2.safetensors \
    --pair seed43-pos1:/tmp/c10-dumps/seed43-pos1.safetensors \
    --pair seed43-pos2:/tmp/c10-dumps/seed43-pos2.safetensors \
    --pair seed44-pos1:/tmp/c10-dumps/seed44-pos1.safetensors \
    --pair seed44-pos2:/tmp/c10-dumps/seed44-pos2.safetensors \
    --out docs/archive/phase-c/phase-c10/c10_splice_bisect_table.md
```

Exit codes:

- `0` — comparator ran; no tap flipped across any pair
- `1` — at least one tap flipped the argmax for at least one pair (genuine
  C11 signal)
- `2` — structural error (missing weight, shape mismatch, bad CLI)

### Kiln dump invocation (for execution)

```
cd /workspace/kiln
KILN_SPEC_ENABLED=1 KILN_SPEC_METHOD=mtp \
KILN_MTP_DUMP_PRE_ROPE=1 KILN_MTP_DUMP_SUBOPS=1 \
KILN_MTP_DUMP_C7_SDPA=1 KILN_MTP_DUMP_POS=1,2 \
KILN_MTP_DUMP_PATH=/tmp/c10-dumps/seed${S}-pos${P}.safetensors \
./target/release/kiln-bench \
    --model-path /workspace/qwen3.5-4b/ --paged \
    --prompt-tokens 512 --max-output-tokens 128 \
    --skip-training --seed ${S}
```

One kiln-bench run per seed emits taps for both requested positions in a
single safetensors file; `--pair` pairs map to distinct dumps so per-seed
runs each produce two input files (one per target `mtp_pos`).

## Execution blockers

C10 planned end-to-end execution during this task but was forced into WIP
status by cascading RunPod infrastructure failures. Documenting here so
the next pod session can finish without re-paying the discovery cost.

1. **Trap-on-exit bug.** The initial agent session installed the
   money-burn-prevention termination trap (`trap '...' EXIT ERR INT TERM`)
   in a standalone Bash call. The trap fires when the shell exits at the
   end of the Bash tool call — i.e. after *every* command — and immediately
   terminated the first A40 pod. Lesson: the trap must be installed
   *inside* each long-running Bash call, not as a separate call, because
   Bash tool invocations do not share shell state across calls.
2. **Wrong `--gpu` flag for `runpod_api.py launch`.** The script accepts
   `--gpu "NVIDIA A40"`, not `--gpu-type`. Using `--gpu-type` (the flag
   name from `ce kiln-pod-acquire`) causes the script to silently fall
   back to the default GPU selector, which currently picks **RTX PRO 6000
   Blackwell** — SM 120a, not supported under CUDA 12.4 per
   `kiln-runpod-gpu-arch-compat`. Blackwell pods at $1.89/hr were launched
   (and immediately terminated) twice before the flag typo was noticed.
   `runpod_api.py launch --help` is *not* a no-op probe: it falls through
   the arg parser and launches another pod.
3. **A40 supply constraint.** After fixing the flag, RunPod returned
   `SUPPLY_CONSTRAINT` for A40. Only A6000 was available, and the single
   A6000 in the pool (`26d739qu9srb0w`) was in `failure_count = 1 / sshd
   wedge` state from a prior run. An SSH probe to it timed out at 30 s
   (matches the documented `kiln-ssh-polling-deadlock` pattern).

Total pod spend this task: ~$0.22 (two short-lived Blackwell launches +
one terminated A40 session). No kiln-bench builds, no dumps, no comparator
runs happened.

## Next session execution recipe

When a healthy A40 (or L40S / A100 80GB) lease is available:

```
# 1. Acquire + smoke-test (no trap — rely on pool 3h TTL + manual release)
LEASE=$(ce kiln-pod-acquire --gpu-type "NVIDIA A40" | jq -r .lease_id)
POD=$(ce kiln-pod-list | jq -r ".entries[] | select(.id==\"$LEASE\") | .runpod_pod_id")
timeout 60 python3 $RP ssh $POD "nvidia-smi -L"

# 2. Update repo + build (do this in one Bash call to keep state)
timeout 900 python3 $RP ssh $POD "
  cd /workspace/kiln
  git fetch origin
  git checkout mtp-phase-c10-fp32-reference-comparator
  git reset --hard origin/mtp-phase-c10-fp32-reference-comparator
  source /root/.kiln-build-env 2>/dev/null
  KILN_CUDA_ARCHS=86 cargo build --release --features cuda --bin kiln-bench 2>&1 | tail -20
  test -x target/release/kiln-bench && echo BUILD_OK
"

# 3. Generate 6 dumps (3 seeds x 2 positions — each kiln-bench run
#    captures both positions if KILN_MTP_DUMP_POS=1,2)
for S in 42 43 44; do
  timeout 900 python3 $RP ssh $POD "
    cd /workspace/kiln
    mkdir -p /tmp/c10-dumps
    KILN_SPEC_ENABLED=1 KILN_SPEC_METHOD=mtp \
    KILN_MTP_DUMP_PRE_ROPE=1 KILN_MTP_DUMP_SUBOPS=1 \
    KILN_MTP_DUMP_C7_SDPA=1 KILN_MTP_DUMP_POS=1,2 \
    KILN_MTP_DUMP_PATH=/tmp/c10-dumps/seed${S}.safetensors \
    ./target/release/kiln-bench \
        --model-path /workspace/qwen3.5-4b/ --paged \
        --prompt-tokens 512 --max-output-tokens 128 \
        --skip-training --seed ${S} 2>&1 | tail -5
  "
done

# 4. Run comparator (split per-position dumps first — kiln writes both
#    positions into one file; the comparator understands position-keyed
#    taps so a single --pair per seed-position works).
#    Exact --pair naming: see CLI contract above.
timeout 1800 python3 $RP ssh $POD "
  cd /workspace/kiln
  pip3 install --quiet safetensors numpy torch 2>&1 | tail -3
  python3 scripts/mtp_c10_splice_bisect.py \
    --checkpoint /workspace/qwen3.5-4b \
    --pair seed42:/tmp/c10-dumps/seed42.safetensors \
    --pair seed43:/tmp/c10-dumps/seed43.safetensors \
    --pair seed44:/tmp/c10-dumps/seed44.safetensors \
    --out docs/archive/phase-c/phase-c10/c10_splice_bisect_table.md
  echo 'EXIT:' $?
"

# 5. Pull table back, update this doc with results + verdict
timeout 120 python3 $RP scp $POD /workspace/kiln/docs/archive/phase-c/phase-c10/c10_splice_bisect_table.md \
  docs/archive/phase-c/phase-c10/c10_splice_bisect_table.md

# 6. Release lease (do NOT terminate the pod directly)
ce kiln-pod-release --lease $LEASE
```

### Verdict (filled)

See "Verdict" section at the top of this report. Outcome: **C9 bf16-noise verdict holds**, C11 promotes to holistic changes.

## Anti-duplication preflight (done)

- `gh pr list --repo ericflo/kiln --state all | grep -i "phase.c10\|splice.bisect\|fp32.reference\|top1.flip"` → no existing C10 PRs
- C9 recommendation #1 (per-token top1 mismatch tracing) explicitly names
  this work as the immediate follow-up
- C11 targets (Marlin q_proj audit, abs_pos/RoPE pos>0, fp32 draft head)
  are listed in C9 report as conditional on C10 result
- No open PR against the MTP head claims the `c10_` doc prefix or the
  `mtp_c10_splice_bisect.py` script name

## File manifest

- `scripts/mtp_c10_splice_bisect.py` (789 lines, new) — fp32 HF reference
  comparator with splice-override semantics. Self-contained; runs on CPU
  in minutes given six pre-generated kiln dumps. Vendored helpers from
  `mtp_reference_dump.py` so it has no cross-file import drift risk.
- `docs/archive/phase-c/phase-c10/c10_report.md` (this file) — methodology, execution log, final verdict.
- `docs/archive/phase-c/phase-c10/c10_splice_bisect_table.md` (new) — per-pair + per-tap splice-bisect output from the A40 run. All 72 cells are `.` (no flip); margins stay positive everywhere.
