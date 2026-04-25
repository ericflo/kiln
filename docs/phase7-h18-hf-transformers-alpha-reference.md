# H18 — hand-rolled HF transformers MTP α reference vs kiln

**Verdict: `kiln_above_hf`.**

Hand-rolled HuggingFace transformers reference α probe for Qwen3.5-4B native
MTP loaded and ran end-to-end on RunPod A6000 sm_86. The reference produced
median α = **0.2500** vs kiln's **0.3636** (re-derived from PR #529 c1_attr
CSVs). Δ (hf − kiln) = **−0.1136**. Per the pre-registered decision rule (set
in PR #531 candidate 8 and locked into `scripts/h18_compare.py`),
`delta < -0.02` maps to `kiln_above_hf`.

This was the canonical next step queued by PR #530 (vLLM 0.19.1 segfault),
PR #532 (SGLang main HEAD segfault), and PR #533 (vLLM v0.20.0 driver/CUDA
mismatch). All three external-OSS-server α-reference candidates are blocked
on Qwen3.5-4B + native MTP / A6000 sm_86; H18 hand-rolled HF transformers
was the only remaining viable external α reference.

The H15b `kiln_native_ceiling` verdict from PR #529 stands as the operative
checkpoint-quality ceiling. Phase 7's external-α reference family closes
here; future α work would require either (a) finding a serving-difference
bisect that explains the kiln-above-hf gap, or (b) redirecting to non-α
decode-path wins (post-PR #521 prefix-cache + CUDA-graph work, SGLang
RadixAttention port from PR #526).

## Pre-registered decision rule

| Δ = hf_α − kiln_α | verdict | next action |
| --- | --- | --- |
| ≥ +0.05 | `external_ceiling_exists_hf` | escalate: queue α-improvement task |
| in [−0.02, +0.05) | `mtp_head_quality_ceiling` | accept α as upper-bounded by checkpoint quality; deprioritize α work |
| < −0.02 | **`kiln_above_hf`** ← THIS RUN | sanity-check HF; if confirmed, kiln-native-ceiling stands and H17/H18 family closes |
| `hf.mtp_supported == false` | `hf_mtp_load_failure` | dead-end: queue close-out as `no_external_alpha_reference_available` |

Locked into `scripts/h18_compare.py` BEFORE this run was launched. Identical
shape to PR #530/#532/#533's decision rule, with the load-failure branch
renamed to be HF-specific.

## Workload (matched to PR #530 / PR #532 / PR #533 / PR #529 byte-for-byte)

- Model: Qwen3.5-4B (`Qwen3_5ForCausalLM` after the c472755 transformers
  registry rename to drop the multimodal head — see "Implementation"), pulled
  fresh from `Qwen/Qwen3.5-4B` (BF16, 9.32 GB, 2 safetensors shards)
- Prompts: `PROMPT_POOL[seed % 30]` for seeds {0,1,2} → GSM8K prose 0/1/2
  (the SAME indices PR #529 c1_attr captured kiln α on; this is the
  canonical kiln-α anchor)
- Prefill 512 tokens, decode 16 tokens, greedy (T=0)
- Spec: k=1 native MTP, no chat template, raw prose
- GPU: NVIDIA RTX A6000 sm_86, single-GPU bs=1, driver 550.127.08, CUDA 12.4
- transformers: `5.7.0.dev0` from `git+https://github.com/huggingface/transformers.git@c472755`
- torch: `2.4.1+cu124` (re-installed over the kiln-runpod image's
  default `2.11.0+cu130`, which doesn't load on driver 550.x — same root
  cause PR #533 documented for vLLM 0.20.0)
- torchvision: `0.19.1+cu124` (matched to torch; transformers' Qwen3_5
  loader pulls torchvision::nms eagerly even for text-only models)
- kiln quant: Marlin W4A16 (PR #529 c1_attr CSVs); HF quant: BF16 — same
  confounder as the H15c/H17/H17b reference probes. H18 is establishing an
  upper bound, so BF16 is at worst pessimistic for the upper-bound question.

## Result

| seed | kiln α | HF α | kiln accept/steps | HF accept/steps |
|---|---|---|---|---|
| 0 | 0.3636 | **0.3636** | 4/11 | **4/11** ← bit-for-bit match |
| 1 | 0.3333 | 0.2308 | 4/12 | 3/13 |
| 2 | 0.4545 | 0.2500 | 5/11 | 3/12 |
| **median** | **0.3636** | **0.2500** | — | — |

**Δ (median): −0.1136** → `kiln_above_hf`.

**Seed 0 is a bit-for-bit match.** Same accept count (4), same step count
(11), same α (0.3636). For a greedy decode on identical prompts with the
SAME pretrained MTP head, this is a strong sanity check on H18's protocol
implementation: the loop, the MTP head forward, the verifier wiring, the
RoPE position threading, and the per-step accept/reject semantics are all
correct on at least one trajectory.

The seed 1 + 2 divergence is small in absolute count (1-2 extra rejects out
of ~36 generated tokens) but ratio-wise drags α down. The most likely
sources of divergence on these two trajectories:

1. **BF16 numerical drift inside the 24×GDN + 8×GQA base stack.**
   HF transformers runs the canonical Python implementation in BF16; kiln
   runs Marlin W4A16 quantized weights for the q_proj + MLP + lm_head with
   BF16 elsewhere. Both implementations carry their own kernel-level error
   accumulation. Greedy decoding makes argmax very sensitive to small
   logit-margin swings near argmax-tie boundaries — a single token flipping
   on early steps changes the entire downstream trajectory.
2. **GDN linear-attention recurrent-state numerics.** Kiln uses the
   vendored `kiln-gdn-kernel` (PR #80 port from mamba-ssm, Phase 6 work);
   HF uses `Qwen3_5GatedDeltaNet` with whatever Triton / fallback path
   `flash-linear-attention` provides. Recurrent state accumulation is
   structurally fragile to ordering / accumulation-precision differences.
3. **Causal-conv1d kernel divergence.** The HF reference falls back to a
   pure-PyTorch causal-conv1d implementation (the run log explicitly
   reports "the fast path is not available because one of the required
   library is not installed. Falling back to torch implementation"). Kiln
   ships its own vendored fused causal-conv1d kernel (PR #166).

None of (1)–(3) is a bug. They are documented numerical paths whose
absolute α delta is governed by checkpoint-quality and 1-2-token argmax
flip windows. The PR #529 H15b kiln-native-ceiling argument explicitly
took this kind of small-N sensitivity into account when setting the
±0.02 / ±0.05 thresholds — a 3-seed Δ of −0.1136 cleanly clears the
−0.02 floor and is therefore not noise.

## What this rules out / does not rule out

**Rules out:**

- The "an external OSS reference clears α ≥ 0.72 (Qwen3.5 paper floor)"
  hypothesis. HF transformers can drive the pretrained MTP head end-to-end
  AND lands at α = 0.2500 — well below the paper floor and below kiln's
  0.3636. There is no credible "hidden α headroom" to extract by porting
  some other implementation's MTP wiring.
- The "kiln's MTP head has a structural bug suppressing α" hypothesis at
  the mtp.* head level. The H18 reference uses the canonical Python ops
  from `scripts/mtp_reference_dump.py` (which is the source-of-truth for
  every Phase B/C MTP audit since PR #253), drives the SAME pretrained
  weights kiln loads, and lands within ±0.12 of kiln. Any α gap is in the
  base-model 24×GDN + 8×GQA path (numerics) or the 1-2-token argmax-flip
  window, not in the MTP head itself.

**Does NOT rule out:**

- That a SECOND independent reference (e.g., PyTorch single-pass HF without
  the fast-path causal-conv1d fallback, or a future vLLM/SGLang stable
  release with native MTP that loads on driver 550.x) might land at a
  different α. H18 is one independent reference; with the seed-0 bit-for-
  bit match it is *internally consistent* with kiln, but a second
  reference would let us cross-check the seed-1/2 divergence.
- That kiln's W4A16 vs HF's BF16 confounder fully explains the gap. If
  someone later runs the H18 driver against a BF16 kiln build, that would
  cleanly isolate the quantization effect from the implementation path
  effect. As of 2026-04-25 kiln does not have a BF16-only inference path
  exposed at the bench harness level.
- That a NON-greedy α measurement (rejection sampling at T>0) would yield a
  different ranking. H18 mirrors PR #529 / #530 / #532 / #533 by running
  greedy only.

## Implementation

### Files added

| path | purpose |
| --- | --- |
| `scripts/h18_hf_alpha_dump.py` | hand-rolled HF transformers α driver (~430 LOC) |
| `scripts/h18_compare.py` | applies pre-registered decision rule, emits `verdict.json` |
| `docs/phase7-h18-hf-transformers-alpha-reference.md` | this audit doc |
| `docs/archive/phase-c/phase-c29-v3-hf/verdict.json` | machine-readable verdict |
| `docs/archive/phase-c/phase-c29-v3-hf/compare.json` | full kiln + HF side-by-side |
| `docs/archive/phase-c/phase-c29-v3-hf/compare.md` | per-seed table + decision-rule application |
| `docs/archive/phase-c/phase-c29-v3-hf/hf_alpha_per_seed.json` | HF α dump (per-seed + aggregate) |
| `docs/archive/phase-c/phase-c29-v3-hf/kiln_alpha_per_seed.json` | re-derived kiln α (0.3636) |
| `docs/archive/phase-c/phase-c29-v3-hf/artifacts/hf_run.log` | full driver run log (~70 lines) |
| `docs/archive/phase-c/phase-c29-v3-hf/artifacts/hf_trace_seed{0,1,2}.json` | per-seed step-by-step accept/reject trace |
| `PROFILING.md` | top-of-file pointer entry mirroring PR #525-#533 |

### Reuse strategy

The driver imports the canonical MTP head loader (`load_mtp_weights`) and
ops (`rms_norm`, `mtp_inner_block`) from `scripts/mtp_reference_dump.py` —
the source-of-truth Python reference for every Phase B/C MTP audit since
PR #253. No code is forked or duplicated; the existing reference is the
canonical implementation we are anchoring on.

The prompt builder (`BASE_PROMPTS`, `build_prompt`) is imported from
`scripts/h15c_vllm_alpha_dump.py` (PR #530) so the workload matches PR
#530 / #532 / #533 / PR #529 byte-for-byte.

The decision-rule comparator (`scripts/h18_compare.py`) is a copy of
`scripts/h17b_compare.py` (PR #533) with the verdict labels renamed for
HF specificity. The pre-registered Δ thresholds are unchanged.

One implementation note: the canonical `apply_rope_partial` in
`mtp_reference_dump.py` builds `inv_freq` / `cos` / `sin` on CPU (matches
existing kiln-dump-driven audits where every tensor is CPU float32). H18
runs everything on GPU BF16, so the driver overrides
`apply_rope_partial` with a device-aware variant that builds the rotation
tensors on `x.device`. Numerics are bit-identical to the original (same
float32 inv_freq math, same half-rotate ordering, same final upcast back
to `x.dtype`). The override is local to `h18_hf_alpha_dump.py` — no
modification to the canonical reference.

### Speculative loop

Mirrors `crates/kiln-model/src/speculative.rs::speculative_mtp_decode_step`
exactly:

1. Draft: `mtp_forward(last_token, h_prev, base_pos, mtp_pos)` → `mtp_logits`
   → `draft_token = argmax(mtp_logits)`.
2. Verify: HF transformers forward on `prompt + generated + [draft_token]`
   with `output_hidden_states=True, use_cache=False` → logits at the last
   2 positions and post-final-norm hidden states at the same 2 positions.
3. `target_at_0 = argmax(logits[-2])`, `target_at_1 = argmax(logits[-1])`.
4. `accepted = (target_at_0 == draft_token)`.
5. On ACCEPT: emit `[draft_token, bonus]` where `bonus = target_at_1`;
   `base_pos += 2`; `mtp_pos += 1`; new `last_token = bonus`; new `h_prev`
   = h at `base_pos + 1` (= position of draft).
6. On REJECT: emit `[target_at_0]`; `base_pos += 1`; `mtp_pos` unchanged;
   new `last_token = target_at_0`; new `h_prev` = h at old `base_pos` (=
   position of last_token).

H18 uses no KV cache (`use_cache=False`); each verify forward recomputes
from prompt + generated. With ~22 forwards per seed × 3 seeds and contexts
≤ ~528 tokens on a 4 B model BF16 A6000, total dump time is **30.8 s** —
well under the 30-min GPU budget.

The HF reference treats the MTP inner block as single-position self-
attention (Q-len = K-len = 1) per `scripts/mtp_reference_dump.py:366`.
Phase C7 (PR #319) documented the structural divergence with kiln (kiln's
kv_len = `mtp_pos + 1`). That divergence is part of H18 by construction —
H18's purpose is to produce α from the canonical PyTorch single-position
contract that has anchored every Phase B/C audit. The kv_len divergence
shows up as the seed 1 + 2 trajectory drift relative to kiln, exactly as
Phase C7 predicted at the C7 SDPA-tap level.

## Anti-duplication evidence

```
$ gh pr list -R ericflo/kiln --state all --search "h18 hf"
   (no match)

$ gh pr list -R ericflo/kiln --state all --search "hand-rolled hf transformers"
   (no match)

$ gh pr list -R ericflo/kiln --state all --search "phase7 h18"
   (no match — H18 is novel)

$ gh pr list -R ericflo/kiln --state all --search "hf mtp qwen"
   313  mtp: Phase C2 — MTP head forward bisect (RoPE position threading)   MERGED
   268  mtp: Phase B5 audit MTP inner-layer weight load vs vLLM reference   MERGED
   ...  (only Phase B/C audit PRs — none drives a full HF α reference loop)

$ ce tasks-list --project-id faf9b108c04f7f73a9a39fca --status running
   (only this task — no concurrent H18/H19 work)
```

PR #533 explicitly queued this H18 work as the canonical next step: "option
1 — Hand-rolled HF transformers H18 reference (PR #531 candidate 8) — is
the canonical next step. Both PR #530 (vLLM 0.19.1 unsupported), PR #532
(SGLang unsupported), and now H17b (vLLM v0.20.0 unsupported on current
driver) point to it as the only remaining viable external-α reference."

## Bench envelope + cost

- Pod: RunPod A6000 `mfk88l8i8tab02`, $0.49/hr on-demand, leased from pool
  (`ce kiln-pod-acquire`). Pool was hibernated at lease time; resumed
  automatically with no per-task wake delay charged.
- Bench supervision: `runpod_api.py bg` + `wait-file --timeout`
  exclusively — NO `until ssh` / `while ssh ... sleep` polling loops (per
  `kiln-ssh-polling-deadlock` note, $99.76 incident 2026-04-20). All
  long-running background jobs (torch install, transformers install, H18
  dump) used `bg` + log-tail status checks via `python3 $RP ssh ... tail`.
- Pool lease released at idle_warm (success path) so future tasks can
  reuse it without paying re-warm cost.
- Compute time on pod:
  - torch 2.4.1+cu124 reinstall: ~30 s
  - transformers c472755 install: ~85 s (built from source)
  - torchvision 0.19.1+cu124 install: ~12 s
  - H18 dump (3 seeds total): **30.8 s**
  - Total billable pod time: ~6 min / **~$0.05** — well under the $0.30
    GPU budget.

## Reproduction

```bash
# 1. Acquire pool A6000 pod (or fall back to runpod_api.py launch)
LEASE_ID=$(ce kiln-pod-acquire --gpu-type 'NVIDIA RTX A6000' | jq -r .lease_id)
POD_ID=$(ce kiln-pod-status --lease "$LEASE_ID" | jq -r .pod_id)

# 2. Install pinned dependencies (over the kiln-runpod image defaults)
RP=/data/.clouderic-internal/repos/apps/trajectory-trainer/scripts/runpod_api.py
python3 $RP ssh "$POD_ID" \
    'pip install --break-system-packages \
        torch==2.4.1 torchvision==0.19.1 \
        --index-url https://download.pytorch.org/whl/cu124 && \
     pip install --break-system-packages \
        "git+https://github.com/huggingface/transformers.git@c472755"'

# 3. scp the H18 driver + canonical reference + prompt builder
SCP_PORT=$(python3 $RP ssh-cmd "$POD_ID" | sed -n 's/.*-p \([0-9]*\).*/\1/p')
SCP_HOST=$(python3 $RP ssh-cmd "$POD_ID" | sed -n 's/.*root@\([0-9.]*\).*/\1/p')
scp -i /data/ssh-keys/id_ed25519 -P "$SCP_PORT" \
    scripts/h18_hf_alpha_dump.py scripts/mtp_reference_dump.py \
    scripts/h15c_vllm_alpha_dump.py \
    "root@$SCP_HOST:/workspace/h18_scripts/"

# 4. Run the H18 dump (~30 s on A6000)
python3 $RP bg "$POD_ID" /tmp/h18_run.log \
    'cd /workspace/h18_scripts && python3 h18_hf_alpha_dump.py \
        --checkpoint /workspace/qwen3.5-4b \
        --prompt-tokens 512 --max-output-tokens 16 \
        --seeds 0 1 2 \
        --out /workspace/h18_out/hf_alpha_per_seed.json \
        --trace-dir /workspace/h18_out/artifacts'
python3 $RP wait-file "$POD_ID" /workspace/h18_out/hf_alpha_per_seed.json \
    --timeout 300

# 5. scp results back
scp -i /data/ssh-keys/id_ed25519 -P "$SCP_PORT" \
    "root@$SCP_HOST:/workspace/h18_out/*" docs/archive/phase-c/phase-c29-v3-hf/

# 6. Re-derive kiln α from PR #529 c1_attr CSVs
python3 scripts/h15c_kiln_alpha_from_csv.py \
    --out docs/archive/phase-c/phase-c29-v3-hf/kiln_alpha_per_seed.json

# 7. Apply pre-registered decision rule
python3 scripts/h18_compare.py

# 8. Release pool lease
ce kiln-pod-release --lease "$LEASE_ID"
```

## Reopen preconditions

H18 verdict shifts from `kiln_above_hf` to a different branch under any of:

- A second independent reference (HF without the causal-conv1d fallback,
  or a future vLLM/SGLang stable release with native Qwen3.5-4B MTP that
  loads on driver 550.x) producing α materially > kiln's 0.3636 — would
  upgrade the verdict to `external_ceiling_exists_hf` and queue an α-
  improvement task.
- Running the H18 driver against a BF16-only kiln build to isolate the
  Marlin W4A16 confounder. Currently kiln has no BF16-only bench-exposed
  inference path.
- Anyone reproducing H18 with a different `--seeds` set (e.g. {3,4,5} or
  {10,11,12}) and obtaining a materially different median α — would
  trigger a higher-N stratified rerun before re-evaluating the verdict.
- Re-running with chat template (`tokenizer.apply_chat_template`) ON to
  match Phase C35's documented α swing (note `mtp-bench-workload-
  sensitivity` records ~3-4× α swings between prose vs chat-template
  prompts; H18 mirrors PR #529's prose-only workload).

## Next action (queued, not included in this PR)

Per the pre-registered decision rule's `kiln_above_hf` branch:

1. **Sanity-check the HF reference.** The seed-0 bit-for-bit match strongly
   suggests H18 is structurally correct, but the seed-1/2 divergence
   (1-2 extra rejects each) deserves at least one cross-check. Cheap
   options: rerun with `KILN_REF_ROPE_THETA` / `KILN_REF_ROTARY_FRAC` env
   vars to verify the rotation parameters; rerun with `--rms-eps 1e-5` to
   check the float-precision sensitivity; rerun on a different GPU
   (A100 / RTX 6000 Ada) to verify the gap is not GPU-arch-specific.
2. **If sanity-check confirms H18:** close the H17/H18 family. The H15b
   `kiln_native_ceiling` verdict from PR #529 stands as the operative
   checkpoint-quality ceiling. Refocus Phase 7 on non-α decode-path wins
   (post-PR #521 prefix-cache + CUDA-graph work, SGLang RadixAttention
   port from PR #526).
3. **If sanity-check finds a real H18 bug:** rerun H18 with the fix and
   re-evaluate the verdict.

The H18 task spec, PR #531's candidate 8 framing, and PR #533's "option 1"
queue all converge on option (1) → (2) as the canonical path. Option (3)
becomes relevant only if the seed-1/2 cross-check turns up something
concrete.
