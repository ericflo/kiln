# Qualification

Retained hardware-qualification evidence: the workload definitions,
server launch specs and configs that define a run, the oracle requests
and comparison results, and the compact per-run receipts that record
it. Receipts carry hashes (commit, `source_tree_sha256`, server log
sha256, model identity) so the raw logs and profiler output they
reference stay untracked (see the raw-output policy in the root
`.gitignore`) while remaining reproducible.

Enforced in the "Portable qualification evidence" step of
`.github/workflows/repository-hygiene.yml` (scoped to relevant changes):
`python3 -m unittest discover -s scripts/qualification/tests`, then
every `qualification/receipts/*.json` validated by
`scripts/qualification/receipt.py`, then
`scripts/qualification/validate_retained_evidence.sh`.
`.github/workflows/qualification-contract.yml` is a manual entry point
for the same checks. The supporting harness lives in
`scripts/qualification/` (see `scripts/README.md`).

## Subdirectories (255 tracked files)

| path | files | contents |
|---|---|---|
| receipts/ | 163 | Per-run compact receipts, grouped by backend then machine: `receipts/cuda/rtx4090-laptop-wsl2/` (4), `receipts/metal/macbook-air-m1/` (5), `receipts/rocm/strix-halo/` (116), `receipts/vulkan/strix-halo/` (38) |
| workloads/ | 29 | Deterministic workload/case definitions (`workload_id`, `kind`, `variables`, `variants`, determinism policy) |
| server-launch/ | 20 | Owned benchmark-server launch specs (schema `kiln.serving-benchmark-server-launch.v1`): command, readiness polling, startup/shutdown timeouts, acceptable exit codes |
| oracle-results/ | 17 | Oracle comparison/divergence results under `oracle-results/rocm/strix-halo/` (HF next-token first-divergence, path- and layer-attribution) |
| server-config/ | 15 | TOML server configs used by the qualification runs (mirror the `server-launch/` IDs) |
| schema/ | 8 | JSON Schemas for the tree (`receipt-v1`, `workload-v1`, `case-result-v1`, `serving-benchmark-server-launch-v1`, the HF-oracle request/oracle/attribution schemas); several are also served on the documentation site via `docs/site/docs-manifest.json` |
| runtime/ | 2 | vLLM teacher identity manifests (schema `kiln.teacher-identity.v1`) under `runtime/vllm/rocm/strix-halo/` |
| oracles/ | 1 | HF next-token request fixture (schema `kiln.hf-next-token-request.v1`) bound to the first-divergence kiln receipt |

`schema/receipt-v1.schema.json` is the JSON Schema companion of the
`scripts/qualification/receipt.py` validator (the harness tests exercise
both); the remaining schemas describe the workloads, launch specs, and
oracle artifacts.

Receipt filenames follow the same convention as `benchmarks/receipts/`
(UTC timestamp, backend, machine slug, workload ID, intent tag or
content hash, `v1` version suffix).
