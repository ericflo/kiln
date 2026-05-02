# Phase 11 — issue #686 long-prefill 408 bisect

Date: 2026-05-01
Status: **Bisect complete — regression predates the 4-commit window. All four commits reproduce the 305 s timeout.**
Hardware: NVIDIA RTX A6000 (49 140 MiB VRAM), CUDA 12.4
Pod: RunPod pool lease `pod-7f437c35edd0bcb5fc28e2d0` (`jzueh9sibinx6f`)
Bisect SHAs (in order):

| # | SHA | Description | PRs included |
|---|-----|-------------|--------------|
| 1 | `e6f417f` | First commit on `main` after #672 (GpuCoordinationLock workers=2 serialize shim) lands | #666, #672 |
| 2 | `d06163a` | After #674 (prefix-cache evicted_blocks refcount fix) lands on top of #672 | #666, #672, #674 |
| 3 | `2318343` | After #675 (revert of #672) lands; serialize shim removed but #674 still in | #666, #674 |
| 4 | `76281c6` | Current `main` / `v0.2.9` (HEAD at audit time) | #666, #674 |

Companion to the Phase 11 operational-readiness checklist, which verified release artifacts and SLOs at the unit and CI level. This audit closes the open production-load question raised in issue #686.

## TL;DR

**The 408 timeout under workers = 2 against the canonical 43 814-input-token correction request reproduces on all four bisect commits.** The regression — if it is a regression at all — predates `e6f417f` (the oldest commit in the chosen window). Within this window:

| SHA | Status | Wall (s) | Peak `kiln_blocks_used` | /metrics snapshots over the run |
|-----|--------|---------:|------------------------:|---------------------------------:|
| `e6f417f` | both clients `timeout/urlerror` at 305.11 s | 305.11 | **2 153** | 62 |
| `d06163a` | both clients `timeout/urlerror` at 305.10 s | 305.10 | **2 153** | 62 |
| `2318343` | both clients `timeout/urlerror` at 305.10 s | 305.10 | **4 306** | 62 |
| `76281c6` | both clients `timeout/urlerror` at 305.11 s | 305.11 | **4 306** | 62 |

(`/v1/chat/completions` returns zero `status=200` log entries on every commit. Server `/metrics` GETs continue to respond in sub-millisecond throughout.)

The block-allocation pattern is exactly what the serialize shim was designed to enforce. With shim ON (`e6f417f`, `d06163a`) only one of the two concurrent requests is admitted to the engine at a time, so `kiln_blocks_used` plateaus at one request's KV-cache footprint (≈ 2 153 blocks). With shim OFF (`2318343`, `76281c6`) both requests admit and `kiln_blocks_used` plateaus at exactly 2 × the single-request footprint (4 306 = 2 × 2 153). **In neither mode does the prefill complete inside the 305 s client timeout.**

This means the v0.2.9 launch-blocker frame in the issue title — "we shipped a regression that broke long-prefill" — is **not supported by this bisect**. The serialize shim removal in #675 doubles peak block usage but does not change the timeout outcome. Both modes time out at the same wall-clock. The actual problem is upstream of the four-commit window: prefill of a single 43 814-token request on A6000 is too slow (or stuck) to land first-token within 305 s under this hot-path config.

## Procedure

1. Acquired RunPod A6000 via `ce kiln-pod-acquire --gpu-type 'NVIDIA RTX A6000'`. Lease `pod-7f437c35edd0bcb5fc28e2d0` → pod `jzueh9sibinx6f`.
2. `kiln-setup --clone` (sccache + B2 backend; Qwen3.5-4B weights at `/workspace/qwen3.5-4b`).
3. Pulled the canonical rejected payload from B2 (`b2://clouderic/issue-686/turn-3ad7e3582f31c244c8b6aa98.json`, 185 KB, 43 814 input tokens, `tools` pruned to oracle-set, `messages_json` intact). Stored at `/tmp/turn.json`.
4. For each `(SHA)` in `e6f417f, d06163a, 2318343, 76281c6`:
   - `git fetch origin && git checkout <sha>` (detached HEAD).
   - `KILN_CUDA_ARCHS=86 cargo build --release --features cuda --bin kiln` (sccache cache hit ≥ 95 % on warm pod).
   - `bash /tmp/run_one.sh <sha>` (per-commit harness; see "Bench harness" below).
   - Server brought up with `KILN_NUM_BLOCKS=16384`, `KILN_KV_CACHE_FP8=1`, `KILN_REQUEST_TIMEOUT_SECS=305`, `KILN_W4A16=0`, model `/workspace/qwen3.5-4b` served as `Qwen/Qwen3.5-4B`. Waited up to 240 s for `/v1/models` ready.
   - Repro driver `python3 /tmp/repro_686.py --workers 2 --timeout 305 --metrics-interval 5` fires two concurrent threaded `urllib.request.urlopen` POSTs to `/v1/chat/completions` with the canonical body (`temperature=0`, `top_p=1`, `max_tokens=2048`, `seed=42`).
   - Per-commit artifacts written to `/tmp/bisect-<sha>/{server.log, server.pid, repro.json, repro.log, done}`.
5. After all four runs, summarized statuses + wall-clock + `kiln_blocks_used` peak across the snapshot log.

Each run had a clean teardown (`SIGTERM` → 5 s grace → `SIGKILL`; `pkill -9 -f target/release/kiln` between commits). All four `done` markers report `REPRO_EXIT=0` (the driver exits cleanly even when both requests time out — non-200 statuses are recorded, not raised).

## Bench harness

`/tmp/run_one.sh` and `/tmp/repro_686.py` are the per-commit harness used at all four SHAs:

* `run_one.sh` is identical across commits. It only sets env vars and starts/stops the freshly-built `./target/release/kiln serve`. The kiln server CLI does not expose `--workers` or `--server-timeout` (verified against `crates/kiln-server/src/cli.rs` on each SHA — only `--served-model-id` and `--config` are present). Concurrency is therefore client-side, not server-side, and the "workers = 2" framing in the original issue refers to two simultaneous client POSTs.
* `repro_686.py` imports `build_chat_messages` and `extract_openai_tools` from `corrections-experiment/experiment/preflight/collect_elicited.py`, builds the same chat body that `kiln_chat()` would have built in production, and fires `args.workers` concurrent POSTs against `/v1/chat/completions`. It captures HTTP status, wall-clock, and `/metrics` snapshots every 5 s. The `temperature`, `top_p`, `seed`, and `max_tokens` settings match the production call site verbatim.
* `KILN_REQUEST_TIMEOUT_SECS=305` is the **server-side** request timeout. The driver's HTTP client also uses 305 s as its timeout, so both ends agree on when to give up.

The `kiln_*` metric filter inside `repro_686.py` was scoped to the prefix list (`kiln_blocks_used`, `kiln_blocks_total`, `kiln_running_requests`, `kiln_waiting_requests`, `kiln_prefix_cache`, `kiln_prefill_chunks`, `kiln_request_latency`, `kiln_request_total`). Cross-checked against `crates/kiln-server/src/metrics.rs` on `76281c6`: actual exposed names are `kiln_active_requests`, `kiln_scheduler_running`, `kiln_scheduler_waiting`, `kiln_blocks_used`, `kiln_blocks_total`, `kiln_vram_*`, `kiln_prefix_cache_*`, `kiln_requests_total`. The filter caught `kiln_blocks_used`, `kiln_blocks_total`, and `kiln_prefix_cache_*` (the load-bearing block-allocation signals) but missed `kiln_active_requests` / `kiln_scheduler_{running,waiting}` because the prefix list said `kiln_running_requests` / `kiln_waiting_requests`. The block plateau differential between shim-on and shim-off is unambiguous on its own — see "Mechanism" below — so the missing gauges did not change the bisect verdict.

## Mechanism — what the four runs reveal

The serialize shim (PR #672, in for `e6f417f` and `d06163a`) holds a single `GpuCoordinationLock` so only one request reaches the model engine at a time. With the shim engaged, only request 0 (whichever wins the lock) gets to allocate KV-cache blocks. Request 1 stays waiting on the coordination gate. The block plateau is therefore **one** request's worth: 2 153 blocks ≈ 43 814 tokens / 16 tokens-per-block × small overhead = roughly the expected paged-decode footprint for a single 43 814-token prefill in FP8 KV.

After PR #675 reverts #672 (commits `2318343` and `76281c6`), both requests admit concurrently. The block plateau exactly doubles to 4 306. This is consistent with PR #675's stated mechanism — there is no surprise here.

**The point of confusion is what the shim was supposed to fix and what it actually fixed.** The 408 timeout is *not* caused by two requests fighting for KV-cache blocks (16 384 blocks total, 4 306 used at the worst point — 26 % occupancy, not block-pool exhaustion). It's caused by prefill itself not finishing. The shim made block usage cleaner without making prefill faster. Removing the shim made block usage messier without making prefill slower. **The 305 s wall-clock outcome is identical in both directions.**

Server log analysis confirms this. On every commit:

```
$ grep -c '/v1/chat/completions' /tmp/bisect-<sha>/server.log
1                                              # the route entry, no response
$ grep -E 'response.*status=200.*chat/completions' /tmp/bisect-<sha>/server.log
(no matches)
```

The chat-completion span is opened, the request is admitted to the engine, and it never closes within the run. `/metrics` GETs continue to return in 60-250 µs throughout the entire 305 s, so the server's HTTP layer and the metrics-gathering path are not blocked. The block of work is inside the prefill engine.

## Hypotheses (in order of plausibility)

These are now the open lines of investigation for the v0.2.10 work — the four-commit bisect rules them in or out as needed:

1. **The 43 814-token prefill simply takes longer than 305 s on A6000 under the hot-path config.** This is the simplest explanation and is consistent with every datum we have: identical wall-clock outcome across all four commits; healthy metrics path; healthy server health-check; no coordination-lock pile-up; identical block-plateau math in both shim-on and shim-off variants. The implication is that the 305 s server timeout is the wrong knob for long-prefill — we need either a bigger budget (with chunked-prefill that yields control between chunks) or a different scheduler decision.
2. **A scheduler regression from a commit older than `e6f417f`.** The bisect window was chosen on the assumption that one of #666 / #672 / #674 / #675 introduced the timeout. All four commits reproduce, so if the timeout is in fact a regression, the offending commit is on `main` *before* `e6f417f`. The next slice should re-bisect a wider window starting from a known-good commit (e.g., the v0.2.7 or v0.2.6 tag), keeping the same harness.
3. **A prefill-throughput problem in the GDN/GQA decode kernels at very high `T`.** This is the same end-state as (1) but with a more specific cause: a kernel hot-spot that grows super-linearly in T or that hits a slow path at T ≥ 32 K. Worth profiling with NVTX-instrumented kiln-bench against the same 43 814-token corpus.
4. **A cancel-drain bug in #666 that completes the request internally but never sends the HTTP body.** Plausible but contradicted by the chat-completion span never closing — if the engine had completed and the cancel-drain swallowed the response, we'd expect to see span-close-with-error in the log. We see the span open and never close, which is consistent with the engine still working on prefill at SIGTERM.

What this audit explicitly rules out:

* **Block-pool exhaustion as the proximate cause of the 408.** Peak `kiln_blocks_used` of 4 306 against `KILN_NUM_BLOCKS=16384` is 26 %.
* **The `GpuCoordinationLock` shim being responsible for the timeout.** Removing it (#675) does not change the outcome.
* **The prefix-cache eviction refcount fix in #674 being responsible.** The shim-on commit *with* #674 (`d06163a`) and the shim-on commit *without* #674 (`e6f417f`) are bit-equal in outcome and within tens-of-milliseconds of each other in wall-clock.

## Recommended next slice (v0.2.10)

1. **Re-bisect against an older window.** Start from the v0.2.7 release tag (or the most recent commit known to have served this canonical request inside 305 s in production), and bisect forward against the same `/tmp/run_one.sh` + `/tmp/repro_686.py` harness. The timeout is older than the four-commit window we chose. The harness is portable to any commit that builds the `kiln` binary against the canonical Qwen3.5-4B; only `KILN_REQUEST_TIMEOUT_SECS=305` and the env-var config matter at the server side.
2. **Profile the prefill path against the canonical 43 814-token request.** Use `kiln-bench --paged --prompt-tokens 43814 --max-output-tokens 0 --skip-training` with NVTX-instrumented release build (`--features cuda,nvtx`). Capture an `nsys` trace of the first 30 s of prefill and confirm whether the engine is making forward progress through chunked prefill or stuck in a single chunk. The Phase 6 history doc (`docs/audits/PHASE6_*.md`-equivalents — see PROFILING.md) has the recipe.
3. **Add request-level cancellation telemetry.** Today `/metrics` exposes `kiln_active_requests` and `kiln_scheduler_running` but not "how many tokens of this request have been prefilled so far." A `kiln_request_prefill_tokens_completed` gauge keyed by request id would have made this audit much faster — we could see at a glance whether prefill was making progress at 1 token / s or 1 000 tokens / s, which decides between hypothesis (1) and (3).
4. **Decide on the right server-side timeout.** 305 s is currently both the production timeout and the timeout this bisect ran against. If long-prefill on A6000 needs more, bump `KILN_REQUEST_TIMEOUT_SECS` and re-run. If the production SLA is 305 s, the load-bearing fix is to either chunk-yield more aggressively, accept long-prefill rejections at the API layer, or split the canonical request into two calls.

## Evidence

All four per-commit log bundles (server log, repro JSON, repro log, `done` marker) are bundled and uploaded:

* `b2://clouderic/issue-686-bisect/<timestamp>.tar.gz`

Recovery: `b2 file download b2://clouderic/issue-686-bisect/<timestamp>.tar.gz /tmp/issue-686-bisect.tar.gz && tar -xzf /tmp/issue-686-bisect.tar.gz -C /tmp/`.

Each `/tmp/bisect-<sha>/server.log` is the JSONL `RUST_LOG=kiln=info,kiln_server=info,warn` stream from the brought-up kiln server. Each `/tmp/bisect-<sha>/repro.json` is the structured driver output (statuses, per-worker elapsed, full `/metrics` snapshot history).
