# Kiln Server Changelog

## Unreleased

### Phase 10 — training-time streaming GDN prefill (impl)
- Implemented the audit-recommended remediation §1 from PR #634: added a streaming branch inside `model_forward_segment` that tiles GDN layers along T while threading `LinearAttentionState` per tile (full-attention layers always run monolithic — training has no KV cache to thread). New helper `gated_deltanet_forward_streaming` slices `[B,T,hidden]` into tiles of `KILN_STREAMING_TILE_TOKENS` (default 8192, must be a multiple of `GDN_CHUNK_SIZE=64`), calls `gated_deltanet_forward` per tile, and `Tensor::cat`s the results back into `[B,T,hidden]`. Bit-exact with the monolithic call by construction (the state hand-off matches `model_forward_paged_streaming_with`). Two new CPU parity tests (`test_gated_deltanet_forward_streaming_matches_monolithic_cpu`, `test_model_forward_segment_streaming_matches_monolithic_cpu`) pass. **A6000 SFT bench result: AMBER — dispatch is reachable now (CPU parity passes; T=8192 STREAMING ON tile=4096 takes 1.3 s of GPU work before OOM vs 0.7 s for monolithic; T=2048 STREAMING-ON loss matches STREAMING-unset bit-for-bit) but T=8192 SFT still OOMs at the 48 GB A6000 ceiling because the autograd recompute saves all per-tile activations simultaneously.** Net: necessary infrastructure for any future training-time GDN streaming work, but on its own this PR doesn't unblock T=8192 SFT — the next slice is the audit's remediation §2 (per-tile forward+backward inside `checkpointed_forward_backward`). See `docs/audits/PHASE10_GDN_TRAINING_STREAMING_IMPL.md` for the full validation, dispatch-reachability evidence, and root-cause analysis. Raw log: `docs/flce_phase_a_streaming_impl_raw_2026-04-29.log`.

### Audits
- Audit-recommended FleCE T=16384 OOM probe on A40 (sm_86, 46 GiB ceiling — A6000 capacity-blocked at task start, A40 is the audit's named arch-equivalent fallback). Two arms: (1) default A40 path (`fused_path=OFF`, A40 below the PR #644 ≥47 GiB gate) → status=OOM, peak=45,447 MiB / 46,068 MiB, delta=29,104 MiB above post-load baseline, step=9.44 s; (2) force-fused (`KILN_FORCE_RMSNORM_KERNEL=1`, simulates A6000's gate-ON dispatch on A40) → status=OOM, peak≥45,095 MiB / 46,068 MiB, delta=28,752 MiB, step=0.89 s (poller likely missed the actual peak in the early-failure case). **§3 closes for FleCE — zero ROI even though T=16384 OOMs.** The ~29 GiB delta is GDN/MLP saved activations across the 8 grad-checkpoint segments, *not* lm_head logits (Phase B's chunked-vocab CustomOp1 already avoids `[T_active, V]` materialization). FleCE Phase C — Liger-style vocab-axis chunking on top of Phase B's existing vocab-axis chunking — addresses the head, which is already <1 GiB at T=16384 with Phase B; it does not move the GDN/MLP-dominated bottleneck. Path forward for long-context SFT is GDN training-time streaming (PR #635/#636/#637 follow-ups), not a head-side fusion. Both audit branches resolve to "no kernel" under the actual mechanism. See the `Addendum 2026-04-29 — FleCE T=16384 OOM probe (closure)` section in `docs/audits/PHASE10_S3_CANDIDATE_PREFLIGHT.md` for full results, mechanism, A6000 implications, and reproduction. Raw logs: `docs/flce_phase_b_t16384_oom_probe_a40_raw_2026-04-29.log` (arm 1) and `docs/flce_phase_b_t16384_oom_probe_a40_fused_raw_2026-04-29.log` (arm 2). Also lands the `kiln-server` cuda → `kiln-train/cuda` Cargo.toml propagation fix that PR #647 and PR #649 each had to apply locally on the pod (see agent note `kiln-server-cuda-doesnt-propagate-to-flce`).

- Audited whether the existing Phase 7 streaming GDN prefill (`KILN_STREAMING_PREFILL`) is reachable from the SFT training path as a remediation for Finding 2 of the Phase A validation (T=8192/T=16384 SFT OOM on A6000). **Result: RED — streaming dispatch lives only in `model_forward_paged_streaming*`, none of which the trainer calls; the env flag is parsed but has no effect on training.** Empirical confirmation on A6000: T=2048 STREAMING ON peak VRAM matches STREAMING unset to 0 MiB and loss is bit-identical; T=8192 STREAMING ON at three tile sizes (default/4096/2048) all OOM identically at the 48 GB ceiling in 0.7 s. Closing Finding 2 requires net-new implementation work (e.g., a streaming branch inside `model_forward_segment`), not configuration. See `docs/audits/PHASE10_GDN_TRAINING_STREAMING.md` for source/empirical evidence and the remediation menu. Raw log: `docs/flce_phase_a_streaming_raw_2026-04-29.log`.

### Validation
- Validated Phase 10 Phase A FLCE on A6000 (48 GB) at T ∈ {2048, 8192, 16384} with rank-8 LoRA and gradient checkpointing ON. **Result: RED — Phase A is insufficient on A6000.** Two findings: (1) `KILN_USE_FLCE=1` fails before completing a step at T=2048 with `matmul is only supported for contiguous tensors` — the V-axis chunk of `embed_tokens_t` is non-contiguous and the chunked-vocab matmul rejects it; (2) even with that bug fixed, T=8192 and T=16384 still pin peak VRAM at the 48 GB A6000 ceiling, indicating GDN-side activations (not the head) dominate at long T. See `docs/audits/PHASE10_FLCE_PREFLIGHT.md` (Phase A validation section, 2026-04-29) for the table, raw log, and required follow-ups. Bench: `crates/kiln-server/examples/flce_phase_a_validation_bench.rs`.

## kiln-v0.2.8 — 2026-04-29

Patch release: post-v0.2.7 dep bumps and CI hygiene. First release that ships
[Sigstore-signed build provenance attestations](SECURITY.md#supply-chain-provenance)
for `kiln-v*` artifacts and the GHCR image.

### Reproducibility / release
- ci: attach build provenance attestations to release artifacts via `actions/attest-build-provenance`. `server-release.yml` and `docker-server-release.yml` now mint Sigstore-signed attestations bound to each artifact's sha256 and the workflow run; verify with `gh attestation verify <artifact> -R ericflo/kiln` (#627).

### CI / release
- ci: gate Apple codesign secrets (`APPLE_*`) to tagged releases so non-tag CI runs (PRs, branches) skip macOS code signing instead of failing on missing secrets (#625).
- ci: also gate Tauri signing secrets (`TAURI_SIGNING_*`) to tagged releases for the same reason as #625 (#626).
- chore(actions): bump `actions/upload-artifact` from 4 to 7 (#614).
- chore(actions): bump `actions/checkout` from 4 to 6 (#615).
- chore(actions): bump `docker/metadata-action` from 5 to 6 (#613).

### Dependencies
- chore(deps): bump `rand` from 0.9.4 to 0.10.1 (#616).
- chore(deps): bump `safetensors` from 0.5.3 to 0.7.0 (#617).
- chore(deps): bump `toml` from 0.8.23 to 1.1.2+spec-1.1.0 (#618).
- chore(deps): bump `tokenizers` from 0.22.2 to 0.23.1 (#619).

### Documentation
- docs(site): scaffold landing page + GitHub Pages workflow under `docs/site/` so https://ericflo.github.io/kiln/ has a published presence (Phase 9) (#624).

### Tests
- fix(test): serialize env-mutating config tests to eliminate flakiness when run in parallel (#623).

## kiln-v0.2.7 — 2026-04-28

Security release: closes all 12 findings in `docs/audits/security-audit-v0.1.md`.
Adds path-traversal hardening on adapter routes, per-route body-size limits,
webhook redirect-disable, training queue + tracked-jobs caps, adapter-dir
disk caps, LRU eviction for the composed-adapter cache, loopback default bind,
and reproducible Docker builds. No new features and no GPU/runtime breaking
changes.

### Security
- Fix path traversal in `DELETE /v1/adapters/:name` and `POST /v1/adapters/load` (Phase 9 audit §2b/§2c, HIGH).
- Validate source adapter names in `POST /v1/adapters/merge` (Phase 9 audit §2d, LOW).
- Per-route body-size limits on training + completions endpoints: 64 MiB for `/v1/train/sft` and `/v1/train/grpo`, 8 MiB for `/v1/chat/completions` and `/v1/completions/batch` (Phase 9 audit §1, LOW).
- Disable HTTP redirects on the training-completion webhook client to prevent server-side redirect chasing into internal infra (Phase 9 audit §7, item 10, LOW).
- Cap training queue depth (audit MEDIUM §4 part 1) — bounded queue with explicit reject when full (#607).
- Cap tracked-jobs map size and TTL Completed/Failed entries (audit MEDIUM §4 part 2) — prevents unbounded memory growth (#608).
- Cap total adapter_dir disk usage on upload (audit LOW §8 / item 9) — `KILN_ADAPTERS_DIR_MAX_BYTES` rejects uploads that would exceed the cap (#620).
- LRU eviction for `.composed/` adapter cache (audit LOW §8 / item 8) — bounded byte-size + entry-count caps with oldest-mtime-first eviction (#621). Closes the last open finding in security-audit-v0.1; all 12 findings now resolved.

### Changed
- Default server listen host changed from `0.0.0.0` to `127.0.0.1` (loopback). Set `server.host = "0.0.0.0"` or `KILN_HOST=0.0.0.0` to accept remote connections; pair with a trusted reverse proxy. Closes security-audit-v0.1 MEDIUM §9.

### Reproducibility / release
- docker: use `--locked` in `deploy/Dockerfile` cargo builds so the published `ghcr.io/ericflo/kiln-server` image matches the exact Cargo.lock dependency set (#601)
- Pin cargo-deny-action to a specific commit SHA and document deny.toml schema (audit LOW §11) (#612).

### Documentation
- docs(security): document training-data trust invariants (security audit §3 / item 12) in README + QUICKSTART. Calls out that `/v1/train/sft` and `/v1/train/grpo` apply a faithful gradient update to anything POSTed — kiln validates structure, not semantics — so the operator's training corpus must be treated as security-sensitive.

## kiln-v0.2.6 — 2026-04-26

Patch release: 3 server bug fixes + first release with bundled
THIRD_PARTY_LICENSES.md asset (cargo-about, MIT/Apache/BSD-only).

### Bug fixes
- metal: disable candle SDPA full path entirely to eliminate intermittent NaN on Apple Silicon (ff84800)
- server: re-emit prefilled `<think>\n` opener in chat-completion responses so streaming clients see it (7548e5a)
- server: split `<think>...</think>` content into llama.cpp-shaped `reasoning_content` field on chat completions (b1ae711)

### CI / release
- First release to ship THIRD_PARTY_LICENSES.md alongside binaries (#598, #599)

## kiln-v0.2.5 — 2026-04-26

Patch release: 4 server bug fixes since v0.2.4.

### Bug fixes
- server: close use-after-free race in prefix-cache streaming path (fad7c6b)
- server: make `stream: true` actually stream tokens in real time (11062f8)
- server: load model chat template so Qwen3.5 gets the `<think>\n` prefix in the rendered chat (e1fcc16)
- metal: bypass candle SDPA full kernel for `8 < q_seq < bq` to avoid a kernel crash (c7cf1ab)

## kiln-v0.2.4 — 2026-04-26

CI / release-prep release. No user-facing API or behavior changes since
v0.2.3; this cut exists to publish the kiln-server Docker image to GHCR
and to validate the auto-publish-on-platforms-green workflow end-to-end.

### CI / release
- Auto-publish the GitHub Release once all 3 platform jobs succeed, instead of leaving each tag in Draft (#592)
- Publish prebuilt server Docker image to `ghcr.io/ericflo/kiln-server` on every `kiln-v*` tag (#593)

## kiln-v0.2.3 — 2026-04-26

Phase 8 advanced features release: batch generation, adapter upload/download,
TIES + concatenation merge modes, per-request adapter composition, and webhook
notifications on training completion. Also lands the Phase 7 `/ui` adapter
controls, refreshed Phase 8 documentation (QUICKSTART, README, ARCHITECTURE,
plus a new docs/GRPO_GUIDE.md), and governance hygiene marking all workspace
crates `publish=false` and tightening cargo-deny wildcards to `deny`.

### Phase 8 advanced features
- POST /v1/completions/batch — efficient multi-prompt batch generation API for GRPO (#583)
- POST /v1/adapters/upload — multipart tar.gz import (#577)
- GET /v1/adapters/{name}/download — streaming tar.gz export (#575)
- TIES merge mode for /v1/adapters/merge (#578)
- Concatenation merge mode for /v1/adapters/merge (#579)
- Per-request adapter composition: stack multiple LoRAs with scaling on /v1/chat/completions (#581)
- Webhook notifications on training completion (#582)

### Phase 7 UI
- Add adapter download / upload / merge controls to `/ui` dashboard (#586)

### Docs
- Document Phase 8 API surface in QUICKSTART.md (#584)
- Refresh README + CHANGELOG for Phase 8 (upload/download/merge modes/composition/batch/webhooks) (#585)
- Refresh ARCHITECTURE.md for Phase 8 (upload/download, TIES/concat merge, composition, webhooks, batch generation) (#587)
- Add docs/GRPO_GUIDE.md with worked verifiable-rewards examples (math, JSON, code) (#588)

### Cleanup
- Move audit/preflight docs into docs/audits/, drop runtime log (#574)

### Governance / hygiene
- Mark workspace crates `publish=false`; tighten cargo-deny wildcards from warn to deny (#589)

### Test fixes
- Rewrite test_upload_rejects_path_escape_in_archive to actually emit a traversal tarball (#580)

## kiln-v0.2.2 — 2026-04-25

Coordinated release aligned with desktop-v0.2.2. Supersedes the unpublished
kiln-v0.2.1 draft; all v0.2.1 changes are included here. Highlights since
v0.2.0 are the radix prefix cache reuse path, more Metal/CUDA decode
fusions, governance docs, and dependency hygiene.

### Phase 7 prefix cache + decode reuse
- Implement radix prefix cache core (#512)
- Wire real append prefix cache (#515) and streaming real prefix cache reuse (#520)
- Use prefix cache with CUDA graphs and warn when bypassed (#518, #521)
- Expose prefix cache metrics (#513)
- Speed up greedy paged prefill defaults (#519)
- Default CUDA streaming prefill for long prompts and lower Metal threshold (#511)
- Refresh post-#521 profiling artifacts (#522)

### Phase 6 / Phase 7 Metal + CUDA fusions
- Fuse Metal attention output gate (#514)
- Fuse Metal GDN gates with recurrent decode
- Fuse Metal contiguous paged decode attention (#501)
- Fuse Metal GDN prefill conv split (#499) and Metal paged KV slot writes (#497)
- Fuse GDN recurrent RMSNorm decode (#496)
- Add CUDA GDN qk norm GQA fast path (#500)
- Add opt-in CUDA GDN decode fuse hook (#498)
- Route shared Metal greedy decode through argmax (#510)
- Defer transposed cache writer (#508); make transposed cache writes reliable (#506)
- Precompile Metal kernels before prewarm lock (#505)
- Batch MTP verifier argmax (#493)

### MTP audits and α-stability work
- H15c stratified C29 v2 reject-row probe (#529)
- H17 SGLang and H15c/H17b/H15a vLLM α microbenches (#530, #532, #533)
- H18 hand-rolled HF transformers MTP α reference (#534)
- H16 external-α reference options audit (#531)
- MTP acceptance-rate state-of-play audit (#527)
- End-to-end native-MTP self-spec decode bench post-#535 (#536)

### Phase 7 CLI / UX
- Recent requests panel on `/ui` dashboard (last 100) (#551)
- Live decode tok/s + p50/p99 ITL on `/ui` dashboard (#550)
- `kiln health` pretty-printed tree output + `--json` escape hatch (#549)
- `kiln train status` CLI subcommand + fix post-submit hint (#548)
- Surface structured server error hints in CLI (#545)
- `KILN_LOG_FORMAT=auto` — TTY-detect pretty vs JSON default (#544)
- GPU name + VRAM in startup banner (#543)
- ProgressBars for model load, SFT, GRPO (#540, #541, #542)

### Server runtime
- Move health adapter scan off runtime
- Document phase 7 prefix cache reuse benchmark
- Audit kiln radix prefix cache vs SGLang RadixAttention (#526)
- Audit vLLM fused_recurrent_gated_delta_rule against kiln-gdn-kernel (#525)
- Kill-switch bisection ruled out a single fused-kernel owner of the post-#166 decode gap (#524)
- Prefix-cache A/B ruled out cache hooks as bench regression source (#523)
- Fast-guard disabled MTP debug taps; reduce safetensors loader map allocations
- RunPod task tasks no longer pin `KILN_CUDA_ARCHS` (#494)

### Governance + CI
- Add Dependabot config for cargo + GitHub Actions (#558)
- Add `cargo-deny` license/source/bans policy and CI check job (#555, #556)
- Add CONTRIBUTING.md, SECURITY.md, CODE_OF_CONDUCT.md (#552, #553, #554)
- Add GitHub issue + PR templates (#557)

### Dependencies
- Bump tokenizers 0.21.4 → 0.22.2 (#565)
- Bump indicatif 0.17.11 → 0.18.4 (#564)
- Bump console 0.15.11 → 0.16.3 (#563)
- Migrate to rand 0.9 (#567)
- Bump cc in cargo-minor-and-patch group (#562)
- Bump docker/login-action 3 → 4 (#561) and docker/build-push-action 5 → 7 (#560)
- Bump Jimver/cuda-toolkit (#559)

### Docs / repo cleanup
- Refresh ARCHITECTURE.md for post-Phase-6 outcomes (#539)
- Refresh BENCHMARKS.md with post-#536 numbers + add vLLM/SGLang comparison (#537)
- A6000 llama.cpp re-bench at 512 → 128 (#538)
- README + QUICKSTART refresh (`/ui`, banner GPU/VRAM, logging defaults) (#546, #547)
- Archive 71 phase-cXX docs subdirs into `docs/archive/phase-c/` (#570)
- Archive frozen profiling/bench MD reports into `docs/archive/` (#568)
- Purge profiling artifact dirs from working tree (#569)

### CI fixes carried over from the unpublished v0.2.1
- Bump Jimver/cuda-toolkit from v0.2.19 to v0.2.35 to handle NVIDIA's renamed installer URLs (#469)
- Install MSVC dev env on Windows before CUDA build; fixes `M_LOG2E` undefined in `flash_api_c.cu` under MSVC (#472)
- Force static MSVC CRT on Windows CUDA build; fixes CRT mismatch between `esaxx-rs` and `kiln-marlin-gemm` (#477)

## kiln-v0.2.1 — 2026-04-24

(unpublished — superseded by kiln-v0.2.2)


Server re-cut to include CI fixes that were missing from kiln-v0.2.0. No
user-facing behavior changes from v0.2.0 in the core server; this cut also
picks up phase-6 Metal and CUDA kernel work landed on main between v0.2.0 and
v0.2.1.

### CI fixes shipped for the full platform matrix
- Bump Jimver/cuda-toolkit from v0.2.19 to v0.2.35 to handle NVIDIA's renamed
  installer URLs (#469)
- Install MSVC dev env on Windows before CUDA build; fixes M_LOG2E undefined
  in flash_api_c.cu under MSVC (#472)
- Force static MSVC CRT on Windows CUDA build; fixes CRT mismatch between
  esaxx-rs and kiln-marlin-gemm (#477)

### Phase 6 CUDA decode work
- Fuse CUDA GDN gated RMSNorm (#466)
- Add CUDA conv1d prefill fast path and fix conv1d prefill launch bounds
  (#481)
- Document post-466 and post-468 MTP decode profiles and post-476 MTP profile
  failure (#468, #480)
- Audit GDN conv decode hotspot and refresh post-#481 current-main profile
  (#473, #483)

### Phase 6 Metal decode work
- Fuse Metal LM-head argmax for greedy decode and reduce Metal LM-head argmax
  on GPU (#471)
- Speed up Metal decode GEMV and route GDN out-proj through Metal decode GEMV
- Fuse Metal full-attention QKV projections and fuse Metal GDN decode QKV
  conv norm
- Persist transposed weight cache asynchronously

### Infrastructure
- Cap cargo and nvcc parallelism and add OOM postmortem helper for RunPod
  builds (#474)

## kiln-v0.2.0 — 2026-04-24

Coordinated release aligned with desktop-v0.2.0. Headline work is the Metal
decode path for Apple Silicon: a new fused GDN kernel family, MTP speculative
decoding improvements, and a batch of macOS startup and prefill reductions.

### Metal GDN kernel fusion
- Fuse Metal GDN decode input projections
- Fuse Metal GDN chunk prep
- Fuse Metal RoPE for prefill (#418)
- Fuse Metal GQA qk norm expansion (#393)
- Default Metal MLP gate-up fusion (#447)
- Add Metal full-chunk GDN prefill (#394), head-last layout (#395)
- Speed up Metal GDN recurrent prefill (#398) and avoid zeroing recurrent outputs (#419)
- Use direct GDN chunk slices on Metal (#391); read full chunks from strided views (#455)
- Use unexpanded GDN QK for Metal decode (#456)
- Use head-major KV read for Metal decode (#452) and head-major SDPA for paged decode (#342)
- Use uninitialized Metal outputs for full-write kernels (#449)
- Parallelize Metal conv1d prefill

### MTP (multi-token prediction) decode
- Route MTP prefill through Metal streaming (#400)
- Speed up macOS default MTP decode
- Mirror desktop speculative routing in bench (#404)
- Align skip-layer bench draft state (#410)
- Defer MTP upload and trim draft state (#408)
- Avoid native MTP during Metal prewarm (#402)
- Guard non-streaming MTP final window (#454)
- Raise Metal skip-layer crossover to 4096 (#442)
- Route long macOS decode through paged skip-layer

### macOS startup and prefill
- Reduce macOS startup and KV prefill overhead
- Speed up macOS startup and skip-layer prefill
- Improve macOS startup and short-prompt routing (#440) and speculative routing (#437)
- Defer Metal precompile until background prewarm (#453); precompile Metal kernels during startup
- Move tokenizer warmup after listen (#460)
- Prewarm macOS speculative path (#386) and make prewarm opportunistic
- Tune macOS Metal hot paths (#385) and streaming prefill defaults (#377)
- Enable tiled prefill by default on Metal (#367)
- Route Metal prefix attention through head-major SDPA (#366)
- Optimize Metal paged KV prefill reads (#416)
- Speed up Metal LM head decode
- Gate Metal readiness prewarm (#332)
- Harden Metal prewarm and KV auto-sizing
- Drop redundant Metal embedding upload (#335)
- Batch Metal auxiliary weight uploads (#443); stream transposed weight cache reads (#445); mmap transposed weight cache hits (#461)

### Server runtime
- Keep default GPU sampling on device (#336); speed up default sampling and fix speculative KV advance (#328)
- Avoid zero-filling server KV pools (#337)
- Hoist paged decode debug gates (#333)

### Profiling and phase 6 kernel work
- Extensive phase 6 decode profiling work (C35–C50) and MTP α-stability re-benches documented in PROFILING.md and PROFILING-MTP-*.md

## kiln-v0.1.2 — 2026-04-20
- See the GitHub release for details.

## kiln-v0.1.1 — 2026-04-20
- See the GitHub release for details.

## kiln-v0.1.0 — 2026-04-18
- Initial public release.
