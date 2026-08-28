# Scripts

Index of the 75 top-level scripts in `scripts/` (51 `.py`, 17 `.sh`, 7 `.mjs`)
plus its 9 subdirectories. Every top-level script appears exactly once;
one-liners are read from each script's own header. `(CI)` marks
load-bearing scripts referenced by `.github/workflows/`.

## Campaign gates

- `check_production_file_budget.py` (CI) — enforce the reviewed physical-line budget for production source files against `contracts/production-file-budget-v1.json`.
- `check_repository_artifacts.py` (CI) — enforce the checked-in artifact retention and file-size policy across the repository.
- `check_unification_gates.sh` — local chain runner: backend-capability contract, kiln-model/tensor/optim/graph test lanes, and the latency-fixture pipeline self-tests.

## CI contract gates

- `check_config_schema.py` (CI) — validate the canonical Kiln configuration schema and its published inputs.
- `check_desktop_ui_smoke.mjs` (CI) — smoke-validate the desktop UI HTML/JS against the runtime-defaults and thinking-budget contracts.
- `check_docs_site_smoke.mjs` (CI) — smoke-validate the docs site: heading slugs, sitemap, `llms.txt`, and generated-doc invariants.
- `check_http_api_contract.py` (CI) — validate the canonical OpenAPI operation and transport contract.
- `check_miniopenenv_interop.sh` (CI) — interop-check the Kiln OpenEnv arcade/game inventory against the adjacent miniopenenv oracle Makefile.
- `check_openenv_contract.py` (CI) — validate the checked-in OpenEnv artifact contract without third-party packages.
- `check_release_versions.py` (CI) — guard user-facing examples against stale release-version and CLI drift.
- `check_runtime_defaults.mjs` (CI) — cross-check `contracts/runtime-defaults-v1.json` against server, eval-CLI, and desktop sources.
- `check_runtime_env_contract.py` (CI) — ratchet direct environment access in crate-owned source; also writes `docs/RUNTIME_ENVIRONMENT_INVENTORY.md`.
- `check_server_ui_smoke.mjs` (CI) — headless smoke of the server dashboard UI (markup, styles, JS demo fixtures).
- `check_source_parsing_tests.py` (CI) — inventory and ratchet tests that inspect implementation source as text; writes `docs/VERIFICATION_TEST_INVENTORY.md`.
- `check_thinking_budget_contract.mjs` (CI) — validate the thinking-budget schema/conformance vectors and the generated doc block.

## Contract & schema generators

- `generate_artifact_schema.py` (CI) — generate `contracts/kiln-artifacts-v1.schema.json` (adapter, HF/TRL, receipt, and teacher APIs).
- `generate_backend_capability_report.py` — generate `docs/backend-capability-report.md` (+ `.json`) from the live source tree.
- `generate_control_plane_schema.py` (CI) — generate `contracts/kiln-control-plane-v1.schema.json` (training, agent, and product control-plane APIs).
- `generate_eval_schema.py` (CI) — generate `contracts/kiln-evals-v1.schema.json` (eval, dataset-synthesis, and judgment APIs).
- `generate_observability_schema.py` (CI) — generate the closed read-only serving and observability JSON Schema.
- `json_schema_subset.py` (CI) — dependency-free Draft 2020-12 subset validator imported by the contract checkers and generators; path-scoped in `pages.yml`.

## Benchmark harness & latency fixtures

- `bench-concurrent-batch.py` — fail-closed OpenAI-compatible serving concurrency benchmark (Kiln and vLLM on identical request bodies).
- `bench-trajectory-turns.py` — run Kiln throughput sweeps over trajectory-trainer materialized turns.
- `build-parity-tolerance.py` — Phase 0.4: build `bench-results/parity-tolerance.csv`, the per-op parity tolerance matrix.
- `run-serving-benchmark-campaign.py` — run the complete Kiln/vLLM serving benchmark profile matrix.
- `check_backend_latency_fixtures.py` (CI) — validate the backend hardware-latency fixture manifest.
- `import_backend_latency_artifact.py` (CI) — import backend latency fixture artifacts downloaded from GitHub Actions.
- `lock_backend_latency_thresholds.py` (CI) — lock backend latency fixture thresholds from checked result artifacts.
- `plan_backend_latency_fixture_dispatch.py` (CI) — plan `gh workflow run` dispatches and runner checks for the latency fixtures.
- `run_backend_latency_fixture.py` (CI) — run one backend latency fixture and materialize its result artifact.
- `write_backend_latency_result_artifact.py` (CI) — write a backend latency result artifact from a fixture benchmark log.

## One-off investigations

Candle-removal (#1082) audits and the phase-c MTP acceptance probes; each is
retained evidence for a frozen `docs/archive/` investigation.

- `audit-candle-usage.sh` — Phase 0.1: audit the candle API surface (call sites by module + symbol) into `bench-results/candle-api-surface.csv`.
- `audit-customop.py` — Phase 0.2: audit `CustomOp1/2/3` impl blocks to design the kiln-tensor forward/backward seam.
- `audit-dtype-usage.py` — Phase 0.5: audit dtype usage (incl. crate-local FP8/Marlin/FP4 types) into `bench-results/dtype-usage.{csv,md}`.
- `audit-multi-gpu-seam.sh` — Phase 0.6: record hardcoded device-0 literals for the centralized device-accessor migration.
- `audit-preserve-list.sh` — Phase 0.7: record the NVTX range names, `KILN_*` env gates, and Tensor seams the migration must preserve.
- `audit-substrate-status.sh` — Phase 1.31: dashboard of #1082 substrate deliverable status from file existence + workspace membership.
- `c11_marlin_audit.py` — Phase C11: Marlin W4A16 per-channel scale drift audit against the fp32-equivalence band.
- `c12_activation_weighted_probe.py` — Phase C12: activation-weighted Marlin drift probe (C11 weight drift × activation energy).
- `c13_hf_reference_dump.py` — Phase C13: HF reference sidecar for pre-projection MTP splice dumps (per-tap cos_sim / max|Δ|).
- `c14_hf_reference_dump.py` — Phase C14: HF reference sidecar for post-MTP-transformer-block splice dumps.
- `c15_h_main_drift_audit.py` — Phase C15: `h_main` drift audit across decode steps against a chained HF reference forward.
- `c16_plumbing_analyze.py` — Phase C16: audit the MTP accept/reject plumbing hypotheses (H1–H4) from C1 attribution CSVs.
- `c29_hf_reference_dump.py` — Phase C29: loop `c14_hf_reference_dump.py` over a multi-prompt kiln dump tree.
- `c29_logits_compare.py` — Phase C29: empirical MTP-logits comparator (top-1, top-K Jaccard, KL, top-1 mass).
- `c29_logits_compare_v2.py` — Phase C29 v2: the same comparator stratified by accepted vs rejected draft rows (H15b probe).
- `h15c_compare.py` — H15c: apply the pre-registered decision rule to vLLM-vs-kiln MTP α medians.
- `h15c_kiln_alpha_from_csv.py` — H15c: derive kiln per-seed MTP α from PR #529's c1_attr CSVs.
- `h15c_vllm_alpha_dump.py` — H15c: vLLM MTP α microbench (same workload as the kiln capture, A6000 bs=1).
- `h17_compare.py` — H17: pre-registered decision rule for SGLang-vs-kiln MTP α.
- `h17_sglang_alpha_dump.py` — H17: SGLang MTP α microbench (Qwen3.5-4B, k=1 speculative decode, seeds 0–2).
- `h17b_compare.py` — H17b: decision rule for the vLLM v0.20.0 retest against kiln α.
- `h17b_vllm_020_alpha_dump.py` — H17b: vLLM v0.20.0 MTP α microbench (thin re-export of the H15c driver).
- `h18_compare.py` — H18: decision rule for the hand-rolled HF-transformers α against kiln α.
- `h18_hf_alpha_dump.py` — H18: hand-rolled HF transformers reference α probe (base/verifier + raw `mtp.*` head).
- `mtp_c10_splice_bisect.py` — Phase C10: fp32 HF reference comparator + top1-flip per-site bisect (Class B rejection audit).
- `mtp_c1_summarize.py` — summarize Phase C1 MTP acceptance-rate attribution CSVs (α, top-k match, Class A/B rows).
- `mtp_compare.py` — Phase B6/B7: per-tap numerical comparison of kiln vs reference MTP intermediates.
- `mtp_h_main_reference_dump.py` — Phase B10/B11/B12/C41–C43: pure-Python reference `h_main` dump with optional layer/GDN sub-op taps.
- `mtp_reference_dump.py` — Phase B6: pure PyTorch reference implementation of the Qwen3.5-4B MTP forward pass.

## Capture & UI smoke

- `capture-desktop-screenshots.mjs` — headless-Chromium capture of the Tauri shell windows with canned `window.__TAURI__` demo data.
- `capture-pre-migration-baseline.sh` — Phase 0.10: freeze pre-migration candle-path numbers so the Phase 9 "≥ baseline" gates are enforceable.
- `capture-screenshots.mjs` — dashboard demo screenshots (PNG + 720/1440/2880 WebP delivery variants), no GPU pod required.

## Training & serving drivers, pod validation

- `cuda_qwen_sft_smoke.sh` — RunPod A6000: fetch Qwen3.5-4B, build kiln-bench with CUDA, run one real SFT training step.
- `issue40_actual_model_regressions.sh` — issue #40 real-model regression driver: train an adapter, then run serving latency cycles.
- `opd_phase0_pod_validation.sh` — §13 Phase 0 real-hardware validation: OPD loss-kernel bench vs baseline + kiln-train/server suites on a CUDA pod.
- `phase2_validation_steps_1_2_3.sh` — bounded Vulkan training smoke on the single qualified route with an 8 GiB memory guard.
- `phase7_cuda_graph_prefix_cache_verify.sh` — verify CUDA-graphs + prefix-cache behavior against a running kiln server.
- `vllm_teacher.py` — fingerprint and launch an immutable vLLM prompt-logprob teacher for OPD scoring.

## Utility & build infrastructure

- `cargo-bounded.sh` — run one Cargo job under an aggregate memory ceiling (systemd scope, WSL2 cgroup, or macOS sandbox boundary).
- `ci_rust_scope.py` (CI) — select the host-testable workspace packages affected by a git diff for CI scoping.
- `push-build-cache.sh` — push compiled artifacts to B2 after a successful build, for future RunPod pods.
- `setup-build-cache.sh` — RunPod pre-build: install sccache with a B2 S3-compat backend and pull cached artifacts.
- `runpod-substrate-validate.sh` — Phase 2 substrate validation on a RunPod A6000: workspace check, substrate tests, optional GPU smoke.
- `runpod-validate-substrate-orchestrator.sh` — outside-the-pod orchestrator: acquire a pod, run the in-pod substrate validation, release the lease.

## Subdirectories

- `c2_artifacts/` — Phase C2 MTP bisect evidence: kiln/ref safetensors dumps (`kiln_pos0..2.st`, `ref_pos0..2.st`) plus comparator stdout (`c2_compare.txt`).
- `docs-site/` — docs-site build (`build.mjs`, `lib.mjs`), social-preview renderer, `tailwind.css`, and unit tests (`test/`); driven by `pages.yml`.
- `hf_trl/` — HF/TRL reference trainer `train_sft.py` (Kiln SFT/recorded-GRPO handoff via PEFT) plus the locked `requirements-sft.lock`.
- `phase-c36/` — C36 H14a decode-length sweep driver (`run_c36_bench.sh`, Cell D × {128, 256, 512, 1024} × seeds 0–2).
- `phase-c37/` — C37 variance re-anchor single-cell sweep driver (`run_c37_bench.sh`).
- `phase-c40a/` — C40a analysis (`analyze_c40a.py`: HumanEval, chat-template OFF, bootstrap CI on α).
- `phase-c40b/` — C40b analysis (`analyze_c40b.py`: N=20 seeds, bootstrap CI on the α median).
- `phase-c40f/` — C40f run set: `run-all-seeds.sh`, `analyze.py`, and `h15a_correlation.py` (Marlin pack determinism × MTP acceptance correlation).
- `qualification/` — qualification harness: `run.py` workload runner, oracle/receipt modules, `serve_*` workloads, `tests/`, and `validate_retained_evidence.sh`.

## Reference census & orphan queue (wave 3b, 2026-08-28)

Census method: for each top-level script, count tracked files outside
`scripts/<self>` (and outside `.git/`) that reference its basename in
`.github/`, `crates/`, `docs/`, `scripts/`, and the root docs.

**Result: zero orphans.** All 75 top-level scripts have ≥1 external
reference, so there is no orphan-candidate queue to adjudicate:

- **Heavily cited (evidence provenance):** `mtp_reference_dump.py` (34),
  `mtp_compare.py` (33), `cargo-bounded.sh` (24),
  `mtp_h_main_reference_dump.py` (22), `vllm_teacher.py` (19) — cited by
  `docs/archive/` investigation reports and audit receipts.
- **Campaign gates (ledger-cited):** `check_production_file_budget.py`
  (6 external + 13 ledger), `check_repository_artifacts.py` (7 external
  + 11 ledger).
- **Lightest citation:** `issue40_actual_model_regressions.sh` (1),
  `runpod-validate-substrate-orchestrator.sh` (1) — each referenced by
  exactly one script or doc; retained (pod-workflow entry points).

Family sizes: campaign gates 3, CI contract gates 12,
contract/schema generators 6, benchmark harness 10, one-off
investigations 29, capture/UI smoke 3, training/serving drivers 6,
utility/build 6.
