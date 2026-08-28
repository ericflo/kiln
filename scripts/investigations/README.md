# investigations/ — frozen one-off investigation scripts

**FROZEN EVIDENCE — DO NOT EDIT.** Every file in this directory is retained
evidence for a completed, frozen investigation (phase-c MTP acceptance
probes, #1082 candle-removal audits, multi-engine α alignment probes, and
one-off pod/hardware validation drivers). They document past runs whose
results live in `bench-results/`, `docs/archive/`, and `docs/audits/`.
Do not refactor, "fix", modernize, or extend them; their exact form is part
of the evidence record. If an investigation must be re-run, write a new
script — do not touch these.

These scripts were moved out of the flat `scripts/` root (2026-09 round 153)
to separate frozen evidence from live tooling. Sibling scripts in this
directory import each other by co-location (`Path(__file__).parent`,
same-directory imports); that relationship is preserved here.

## MTP acceptance & splice probes (phase B/C)

- `mtp_reference_dump.py` — Phase B6: pure PyTorch reference implementation of the Qwen3.5-4B MTP forward pass (canonical reference for the whole family).
- `mtp_h_main_reference_dump.py` — Phase B10/B11/B12/C41–C43: pure-Python reference `h_main` dump with optional layer/GDN sub-op taps.
- `mtp_compare.py` — Phase B6/B7: per-tap numerical comparison of kiln vs reference MTP intermediates.
- `mtp_c10_splice_bisect.py` — Phase C10: fp32 HF reference comparator + top1-flip per-site bisect (Class B rejection audit).
- `mtp_c1_summarize.py` — summarize Phase C1 MTP acceptance-rate attribution CSVs (α, top-k match, Class A/B rows).
- `c11_marlin_audit.py` — Phase C11: Marlin W4A16 per-channel scale drift audit against the fp32-equivalence band.
- `c12_activation_weighted_probe.py` — Phase C12: activation-weighted Marlin drift probe (C11 weight drift × activation energy).
- `c13_hf_reference_dump.py` — Phase C13: HF reference sidecar for pre-projection MTP splice dumps (per-tap cos_sim / max|Δ|); drives `mtp_reference_dump.py` as a co-located subprocess.
- `c14_hf_reference_dump.py` — Phase C14: HF reference sidecar for post-MTP-transformer-block splice dumps; drives `mtp_reference_dump.py` as a co-located subprocess.
- `c15_h_main_drift_audit.py` — Phase C15: `h_main` drift audit across decode steps against a chained HF reference forward.
- `c16_plumbing_analyze.py` — Phase C16: audit the MTP accept/reject plumbing hypotheses (H1–H4) from C1 attribution CSVs.
- `c29_hf_reference_dump.py` — Phase C29: loop `c14_hf_reference_dump.py` over a multi-prompt kiln dump tree (thin scheduler; both co-located).
- `c29_logits_compare.py` — Phase C29: empirical MTP-logits comparator (top-1, top-K Jaccard, KL, top-1 mass).
- `c29_logits_compare_v2.py` — Phase C29 v2: the same comparator stratified by accepted vs rejected draft rows (H15b probe).

## Multi-engine MTP α alignment probes (H15c / H17 / H17b / H18)

- `h15c_compare.py` — H15c: apply the pre-registered decision rule to vLLM-vs-kiln MTP α medians.
- `h15c_vllm_alpha_dump.py` — H15c: vLLM MTP α microbench (same workload as the kiln capture, A6000 bs=1); imported by `h17b_vllm_020_alpha_dump.py` and `h18_hf_alpha_dump.py` (co-located).
- `h17_compare.py` — H17: pre-registered decision rule for SGLang-vs-kiln MTP α.
- `h17_sglang_alpha_dump.py` — H17: SGLang MTP α microbench (Qwen3.5-4B, k=1 speculative decode, seeds 0–2).
- `h17b_compare.py` — H17b: decision rule for the vLLM v0.20.0 retest against kiln α.
- `h17b_vllm_020_alpha_dump.py` — H17b: vLLM v0.20.0 MTP α microbench (thin re-export of `h15c_vllm_alpha_dump.py`, co-located).
- `h18_compare.py` — H18: decision rule for the hand-rolled HF-transformers α against kiln α.
- `h18_hf_alpha_dump.py` — H18: hand-rolled HF transformers reference α probe (base/verifier + raw `mtp.*` head); imports `mtp_reference_dump.py` + `h15c_vllm_alpha_dump.py` (co-located).

> `h15c_kiln_alpha_from_csv.py` intentionally stays at `scripts/` root:
> `check_config_schema.py` pins its exact path in
> `RETIRED_ENV_REFERENCE_ALLOWLIST`, so it is live-gated, not frozen.

## One-off validation drivers

- `capture-pre-migration-baseline.sh` — Phase 0.10: one-off capture of pre-migration candle-path numbers (evidence frozen in `bench-results/pre-migration-baseline/`; path tracked by `audit-substrate-status.sh` line 0.10).
- `cuda_qwen_sft_smoke.sh` — one-off RunPod A6000 driver: fetch Qwen3.5-4B, build kiln-bench with CUDA, run one real SFT training step.
- `phase7_cuda_graph_prefix_cache_verify.sh` — one-off verification of CUDA-graphs + prefix-cache behavior against a running kiln server.

## Phase-c family directories (run drivers + analysis)

- `phase-c36/` — C36 H14a decode-length sweep driver (`run_c36_bench.sh`, Cell D × {128, 256, 512, 1024} × seeds 0–2).
- `phase-c37/` — C37 variance re-anchor single-cell sweep driver (`run_c37_bench.sh`).
- `phase-c40a/` — C40a analysis (`analyze_c40a.py`: HumanEval, chat-template OFF, bootstrap CI on α).
- `phase-c40b/` — C40b analysis (`analyze_c40b.py`: N=20 seeds, bootstrap CI on the α median).
- `phase-c40f/` — C40f run set: `run-all-seeds.sh`, `analyze.py`, and `h15a_correlation.py` (Marlin pack determinism × MTP acceptance correlation).

## Live-vs-frozen decision rule (do not re-litigate)

A `scripts/` root script is **LIVE — locked at its exact path** when it is
referenced by any of: `.github/workflows/**`, any `scripts/check_*.py` or
gate script, `contracts/**`, the `bench-results/README.md` regenerate table,
`kiln.example.toml`, or `docs/` (live docs) — or when it generates
checked-in artifacts that the site/CI consume (the two screenshot capture
scripts). Everything else — one-off investigation/audit scripts with no such
reference — is **FROZEN evidence** and belongs in this directory.

Classification precedent (round 153):

- LIVE and kept at `scripts/` root: all `check_*` CI gates,
  `check_unification_gates.sh`, the five schema generators +
  `json_schema_subset.py` + `generate_backend_capability_report.py`, the six
  backend-latency-fixture scripts, the eight bench-results regenerate
  scripts, `bench-concurrent-batch.py` / `bench-trajectory-turns.py` /
  `run-serving-benchmark-campaign.py`, `cargo-bounded.sh`,
  `setup-build-cache.sh`, `vllm_teacher.py`,
  `h15c_kiln_alpha_from_csv.py` (gate allowlist),
  `phase2_validation_steps_1_2_3.sh` (content-asserted by
  `check_runtime_defaults.mjs`), `opd_phase0_pod_validation.sh`
  (docs/guides/VIGNETTES.md), `capture-screenshots.mjs` /
  `capture-desktop-screenshots.mjs` (generators of the tracked
  `docs/site/assets/server-ui-*` and `docs/desktop/*.png` artifacts), and
  `c2_artifacts/` (raw dumps explicitly excluded by the live
  `scripts/qualification/source_tree_hash.py` harness).
- Ambiguous and deliberately kept at root: `push-build-cache.sh` (zero
  references; build-infrastructure utility, not an investigation) and
  `issue40_actual_model_regressions.sh` (re-runnable regression driver
  cited by the live owner-managed `capabilities/KILN_IMPROVEMENT_ISSUES.md`).
