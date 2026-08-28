# Scripts

Index of the 50 top-level scripts in `scripts/` (29 `.py`, 14 `.sh`, 7 `.mjs`)
plus its 6 subdirectories. Every top-level script appears exactly once;
one-liners are read from each script's own header. `(CI)` marks
load-bearing scripts referenced by `.github/workflows/`. The 25 frozen
one-off investigation scripts + 5 phase-c family directories that used to sit
here now live in `investigations/` (see its README for the full index and
the live-vs-frozen decision rule).

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
- `check_runtime_env_contract.py` (CI) — ratchet direct environment access in crate-owned source; also writes `docs/contracts/RUNTIME_ENVIRONMENT_INVENTORY.md`.
- `check_server_ui_smoke.mjs` (CI) — headless smoke of the server dashboard UI (markup, styles, JS demo fixtures).
- `check_source_parsing_tests.py` (CI) — inventory and ratchet tests that inspect implementation source as text; writes `docs/policies/VERIFICATION_TEST_INVENTORY.md`.
- `check_thinking_budget_contract.mjs` (CI) — validate the thinking-budget schema/conformance vectors and the generated doc block.

## Contract & schema generators

- `generate_artifact_schema.py` (CI) — generate `contracts/kiln-artifacts-v1.schema.json` (adapter, HF/TRL, receipt, and teacher APIs).
- `generate_backend_capability_report.py` — generate `docs/contracts/backend-capability-report.md` (+ `.json`) from the live source tree.
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

## One-off investigations (kept at root: live-locked)

Candle-removal (#1082) regenerate tooling (LOCKED — `bench-results/README.md`
regenerate table) plus the one gate-pinned H15c driver. The other 22 one-off
investigation scripts (c11–c16, c29, h15c/h17/h17b/h18, mtp_*) are frozen
evidence in `investigations/`.

- `audit-candle-usage.sh` (LOCKED) — Phase 0.1: audit the candle API surface (call sites by module + symbol) into `bench-results/candle-api-surface.csv`.
- `audit-customop.py` (LOCKED) — Phase 0.2: audit `CustomOp1/2/3` impl blocks to design the kiln-tensor forward/backward seam.
- `audit-dtype-usage.py` (LOCKED) — Phase 0.5: audit dtype usage (incl. crate-local FP8/Marlin/FP4 types) into `bench-results/dtype-usage.{csv,md}`.
- `audit-multi-gpu-seam.sh` (LOCKED) — Phase 0.6: record hardcoded device-0 literals for the centralized device-accessor migration.
- `audit-preserve-list.sh` (LOCKED) — Phase 0.7: record the NVTX range names, `KILN_*` env gates, and Tensor seams the migration must preserve.
- `audit-substrate-status.sh` (LOCKED) — Phase 1.31: dashboard of #1082 substrate deliverable status from file existence + workspace membership.
- `h15c_kiln_alpha_from_csv.py` — H15c: derive kiln per-seed MTP α from PR #529's c1_attr CSVs. Stays at root because `check_config_schema.py` pins its exact path in `RETIRED_ENV_REFERENCE_ALLOWLIST` (gate-locked).

## Capture & UI smoke

- `capture-desktop-screenshots.mjs` — headless-Chromium capture of the Tauri shell windows with canned `window.__TAURI__` demo data; generates the tracked `docs/desktop/*.png` (keep: artifact generator).
- `capture-screenshots.mjs` — dashboard demo screenshots (PNG + 720/1440/2880 WebP delivery variants); generates the tracked `docs/site/assets/server-ui-*` (keep: artifact generator).

## Training & serving drivers, pod validation

- `issue40_actual_model_regressions.sh` — issue #40 real-model regression driver: train an adapter, then run serving latency cycles.
- `opd_phase0_pod_validation.sh` — §13 Phase 0 real-hardware validation: OPD loss-kernel bench vs baseline + kiln-train/server suites on a CUDA pod.
- `phase2_validation_steps_1_2_3.sh` — bounded Vulkan training smoke on the single qualified route with an 8 GiB memory guard (keep: content-asserted by `check_runtime_defaults.mjs`).
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
- `investigations/` — FROZEN one-off investigation scripts + 5 phase-c family directories (`phase-c36/` … `phase-c40f/`); do-not-edit evidence home, see its README for the per-script index and the live-vs-frozen decision rule.
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

**Round 153 (2026-09) reorg:** the 25 non-locked one-off investigation
scripts and the 5 phase-c family directories moved to `investigations/`;
the eight bench-results regenerate scripts, both screenshot generators, and
every gate/contract/tooling script stayed at root. Current family sizes:
campaign gates 3, CI contract gates 12, contract/schema generators 6,
benchmark harness 10, one-off investigations (root, locked) 7, capture/UI
smoke 2, training/serving drivers 4, utility/build 6, frozen
investigations (`investigations/`) 25 + 5 family dirs.
