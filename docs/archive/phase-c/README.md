# Phase C Archive

Frozen artifacts and verdicts from the Phase 6/7 MTP (Multi-Token Prediction) acceptance investigation. 71 phase directories archived here; live work and ongoing references continue from `PROFILING.md`, `BENCHMARKS.md`, and the active scripts under `scripts/phase-c*/`.

These dirs were moved out of the top-level `docs/` to keep the repo navigable. All inline citations across `PROFILING.md`, `BENCHMARKS.md`, the helper scripts in `scripts/`, and sibling phase docs now point at `docs/archive/phase-c/phase-cXX/` paths.

## Phase groupings

### C1, C5–C28 — early MTP attribution and splice bisects
Initial MTP α-collapse investigation (α dropped from ~0.72 baseline to ~0.12). Established that the issue was in the kiln implementation, not the model checkpoint, by splicing kiln intermediates into a known-good HF reference and bisecting which sub-op introduced the divergence. Includes the first Marlin W4A16 audit (C11), FP32 head verification (C12), and the C14 splice-cosine refutation of the lm_head / fc_norm / final_layernorm head trio.

- `phase-c1` — initial seed scan
- `phase-c5`–`phase-c10` — splice bisect across full attention block
- `phase-c11` — Marlin W4A16 audit
- `phase-c12`–`phase-c16` — FP32 head verification, h_main drift audit, plumbing analysis
- `phase-c17`–`phase-c28` — phase-by-phase static audits and intermediate-tensor splices

### C29 family + external comparators (C29 v2, v3-hf, v3-sglang, v3-vllm, v3-vllm-020)
Multi-implementation cross-reference: dumped MTP intermediate tensors from kiln, vLLM (two versions), SGLang, and HF reference, then compared cosine similarity / top-K Jaccard / α reconstruction across all four. Established that kiln's MTP intermediates are bit-exact vs HF fp32 reference (top-K Jaccard 100.00% across 49 paired dumps). Refutes any "kiln math diverges" hypothesis class.

- `phase-c29` — initial cross-impl comparison
- `phase-c29-v2` — re-run with tightened tolerances
- `phase-c29-v3-hf` — HuggingFace transformers reference dumps
- `phase-c29-v3-sglang` — SGLang reference dumps
- `phase-c29-v3-vllm` — vLLM 0.6.x reference dumps
- `phase-c29-v3-vllm-020` — vLLM 0.7.x reference dumps

### C30–C45 — h-main / row-norm bisect
Layer-1 sub-op bisect localizing the seed-1 upstream `h_main` divergence to a layer-1 input_norm broadcast tolerance boundary. Includes the C36/C37 bench reproductions (their helper scripts now live at `scripts/phase-c36/` and `scripts/phase-c37/`) and the C40 family which split into C40a–C40f sub-investigations of row-norm fusion alternatives (their helper scripts now live at `scripts/phase-c40a/`, `scripts/phase-c40b/`, `scripts/phase-c40f/`).

- `phase-c30`–`phase-c35` — h_main upstream divergence bisect
- `phase-c36`, `phase-c37` — repro benches
- `phase-c38`–`phase-c40` — row-norm initial cuts
- `phase-c40a`–`phase-c40f` — row-norm fusion alternatives, including h15a determinism analysis
- `phase-c41`–`phase-c45` — input_norm broadcast tolerance localization

### C48–C66 — post-#468 MTP profiler attribution and beyond
After the C45 verdict landed and the next α-acceptance baselines were re-measured, the work shifted to NVTX profiler attribution (which kernel is actually consuming the wall-clock that needs to come down). Includes conv-child NVTX profiles (C54), post-#476 MTP decode profiles (C56), and the trailing series of post-merge re-measurements through C66.

- `phase-c48`–`phase-c54` — attribution + conv-child NVTX
- `phase-c56`–`phase-c66` — post-#476 decode profiles and successor re-measurements

(Phases C46, C47, C55 were skipped or rolled into adjacent phases during the live investigation.)

## How to read these

Each `phase-cXX/` dir typically contains:
- A `verdict.md` or `*-verdict.md` / `*-report.md` with the conclusion of that phase
- One or more `.json` / `.csv` / `.log` artifacts captured during the investigation
- Occasionally an `artifacts/` subdir for larger captures

For the active live story, start from `PROFILING.md` at the repo root — it cites individual phase verdicts inline and is the authoritative current state of the optimization frontier.
