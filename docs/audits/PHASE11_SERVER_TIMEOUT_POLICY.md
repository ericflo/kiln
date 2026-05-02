# Phase 11 — server-side timeout policy for v0.2.10

Date: 2026-05-01
Auditor: Cloud Eric
Status: **decision-pending** — recommendation below; final pick lands when the v0.2.10 implementation PR opens.
Trigger: [`PHASE11_ISSUE_686_BISECT.md`](PHASE11_ISSUE_686_BISECT.md) §`Recommended next slice (v0.2.10)` item 4: "Decide on the right server-side timeout."
Companion to: [`PHASE11_ISSUE_686_BISECT.md`](PHASE11_ISSUE_686_BISECT.md), [`PHASE11_PRELAUNCH_OPS_CHECKLIST.md`](PHASE11_PRELAUNCH_OPS_CHECKLIST.md).

This doc is the gating design decision for v0.2.10. PR #689's items 1–3 (wider re-bisect, NVTX prefill profile, `kiln_request_prefill_tokens_completed` gauge) all need GPU pod time and a committed direction; without a decision here those items don't yet know what they're trying to prove. Items 1–3 are explicitly out of scope for this audit — see §6.

## 1. Problem statement

Production line at `kiln-v0.2.9` (HEAD `76281c6`) ships with `server.request_timeout_secs = 300` as the default ([`crates/kiln-server/src/config.rs:330`](../../crates/kiln-server/src/config.rs#L330)), declared at [`crates/kiln-server/src/config.rs:40`](../../crates/kiln-server/src/config.rs#L40), env-overridable as `KILN_REQUEST_TIMEOUT_SECS` at [`crates/kiln-server/src/config.rs:526-530`](../../crates/kiln-server/src/config.rs#L526), validated `must be > 0` at [`crates/kiln-server/src/config.rs:707-709`](../../crates/kiln-server/src/config.rs#L707), wired into the engine as `request_timeout: Duration` at [`crates/kiln-server/src/state.rs:440-441`](../../crates/kiln-server/src/state.rs#L440) and applied via the wrapper paths at [`crates/kiln-server/src/state.rs:525-546`](../../crates/kiln-server/src/state.rs#L525) and [`crates/kiln-server/src/state.rs:582-841`](../../crates/kiln-server/src/state.rs#L582). The default is mirrored in [`kiln.example.toml:13`](../../kiln.example.toml#L13).

When the timeout fires, kiln returns an HTTP 408 with `code: "request_timeout"` and the hint string at [`crates/kiln-server/src/error.rs:88`](../../crates/kiln-server/src/error.rs#L88): *"Try reducing max_tokens, or increase the server's request_timeout_secs in the config file."* PR #666 made the timeout safe by cancel-draining the underlying `spawn_blocking` so the cache/scheduler state isn't poisoned by the timed-out request.

The bisect harness in PR #689 used `KILN_REQUEST_TIMEOUT_SECS=305` — the same number plus a 5-second client-side cushion — so when the doc says "all four commits time out at 305 s", the server-side timeout firing at 300 s is the proximate cause; the harness then waits five more seconds for the HTTP body that never arrives.

The concrete failure mode: `workers=2` against a single 43 814-input-token correction request on `Qwen3.5-4B`, A6000, `KILN_NUM_BLOCKS=16384`, `KILN_KV_CACHE_FP8=1`, all four bisect commits (`e6f417f`, `d06163a`, `2318343`, `76281c6`) reproduce identically — the chat-completion span opens, prefill begins, and 305 s later the client gives up. Peak `kiln_blocks_used` is 4 306 / 16 384 (26 % occupancy) so block-pool exhaustion is ruled out. The metrics path stays sub-millisecond throughout. The block of work is inside the prefill engine itself.

The v0.2.9 release artifact ships a documented workaround at [`QUICKSTART.md:296-326`](../../QUICKSTART.md#L296) (§4.3): either run with `workers=1` (client-side serialize the prompt) **or** raise `server.request_timeout_secs` to ≥600 s. The CHANGELOG.md `kiln-v0.2.9 — 2026-05-01` entry references the same workaround. **There is no committed default-fix direction for v0.2.10.** This audit picks one.

## 2. Constraints and goals

- **Doc-only PR.** No code changes in this audit. All design choices need to be implementable from the recommendation alone.
- **No regressions for in-the-wild configs.** Anything that explicitly sets `server.request_timeout_secs` in TOML (or `KILN_REQUEST_TIMEOUT_SECS` in env) must keep its current behavior bit-for-bit.
- **The fix must be bounded.** A request that runs forever ties up an engine slot forever; PR #666's cancel-drain is the floor — anything we change must compose with that floor, not erode it.
- **Phase 11 is the onboarding-prep phase.** Real prefill-scheduler work (Option E below) is v0.3.x territory; v0.2.10 wants the smallest correct change that closes the release-readiness reliability story without committing the project to a multi-week kernel rewrite.
- **The 4-commit bisect rules out a regression in {#666, #672, #674, #675}.** Whatever the right server-side default is, it can be picked without first identifying an offending commit.

## 3. Options

Each option lists **What**, **Pros**, **Cons**, **Code shape**, **Migration**, **Risk**.

### Option A — status quo (300 s default + workaround docs)

- **What.** Keep `request_timeout_secs = 300`. Rely on QUICKSTART §4.3 + the existing `prompt_too_long`-style hint string at `error.rs:88` to teach users.
- **Pros.** Zero code change. Zero migration. The cancel-drain in PR #666 means the 408 doesn't leak engine state.
- **Cons.** Every cold-reader who first hits a long-prefill prompt sees a 408 and has to find QUICKSTART §4.3 (or read the error hint, which currently only points at `request_timeout_secs`, not at the workers=1 alternative). The onboarding story is "by the way, there's a known regression — read this section first." That's a worse first impression than any of B/C/D.
- **Code shape.** Zero.
- **Migration.** Zero.
- **Risk.** Reputational, not technical. The bisect already rules out a fix in the {#666, #672, #674, #675} commit window, so there is no known scoped code change that closes the issue without going to Option E.

### Option B — raise default to 600 s

- **What.** Change `request_timeout_secs` default from 300 → 600 in [`config.rs:330`](../../crates/kiln-server/src/config.rs#L330) and the matching value in [`kiln.example.toml:13`](../../kiln.example.toml#L13). Add a CHANGELOG entry under v0.2.10. Refresh QUICKSTART §4.3 to drop the now-stale "raise to ≥600" workaround sentence (the workers=1 workaround stays — it's still useful for very long prefills).
- **Pros.** Single-line config change. Matches the existing QUICKSTART §4.3 workaround so the docs converge instead of diverging. The 305 s timeouts in the bisect are roughly half of 600 s, so this gives the long-prefill case enough headroom to land first-token under most reasonable interpretations of hypothesis (1) in PR #689.
- **Cons.** A stuck request now ties up a slot for 10 min instead of 5 min — but PR #666's cancel-drain bounds the worst-case engine-state exposure and the metrics path stays healthy throughout, so the worst-case is "one engine slot is blocked for an extra 5 min before it fails clean." Users who explicitly set a 300 s timeout in TOML keep their current behavior.
- **Code shape.** Trivial — three lines (config default, example TOML, CHANGELOG bullet). One config-default test in [`crates/kiln-server/src/config.rs:768`](../../crates/kiln-server/src/config.rs#L768) needs to be updated from `300` to `600`.
- **Migration.** Zero for explicit configs (TOML override wins). Implicit configs (default-only) get the new value silently. CHANGELOG should call this out.
- **Risk.** Low. The worst-case slot-occupancy doubles, but the cancel-drain floor (#666) is unchanged. The hypothesis-1 "prefill needs more than 305 s on A6000 at this T" case is exactly what this fix addresses.

### Option C — per-request override via HTTP header

- **What.** Add `X-Kiln-Request-Timeout-Secs: <int>` HTTP header parsed in the chat-completions handler at [`crates/kiln-server/src/api/completions.rs`](../../crates/kiln-server/src/api/completions.rs) (3 189 lines — re-verify exact handler symbol during edit; the file holds both `/v1/chat/completions` and `/v1/completions`). Bound to `[1, server.max_request_timeout_secs]` with a new optional `ServerConfig` field `max_request_timeout_secs: Option<u64>` defaulting to e.g. 1800 s. Add validator following the pattern at [`config.rs:707-709`](../../crates/kiln-server/src/config.rs#L707) (cap must be `>= request_timeout_secs` if set). New `ApiError::request_timeout_header_invalid` variant.
- **Pros.** Most flexible — the slow-prompt user gets a knob without forcing a global default change. Makes the contract per-request explicit, which composes well with future per-request `max_prompt_tokens` checks (Option D).
- **Cons.** API surface widens. Every future client library needs to know about the header. Tests need to cover header parsing edge cases (negative, zero, non-integer, larger-than-cap). The default header-absent path still runs into the 300 s default unless we *also* raise the default (i.e., this is additive to Option B, not a substitute).
- **Code shape.** ~80–120 lines: header parse in completions handler + bounds check + new config field + validator + integration tests + CHANGELOG.
- **Migration.** Zero for clients that don't send the header — current behavior is preserved.
- **Risk.** Medium. New API surface is a release-eve commitment; once shipped, removing or renaming the header is a breaking change. Better timing is later, after users have actually asked for per-request control.

### Option D — API-layer rejection of long-prefill at submission

- **What.** Add an optional `max_prompt_tokens: Option<usize>` field to `ServerConfig`, default `None` (no cap, current behavior). When set and the inbound chat-completions request's `prompt_tokens` exceeds the cap, reject with a structured `prompt_too_long` 4xx error before the engine touches GPU work. The token count is already available to the chunked-prefill scheduler so the plumbing is short. New `ApiError::prompt_too_long { prompt_tokens, cap }` variant; mirrors the `request_timeout` hint pattern at [`error.rs:83-91`](../../crates/kiln-server/src/error.rs#L83).
- **Pros.** Zero-default = zero-migration. Operators who care can pin a per-deployment cap without forcing a behavior change on operators who don't. Rejected requests use no GPU time. The cap is per-deployment, not per-request, so it composes cleanly with Option B's bigger default.
- **Cons.** Doesn't help any user who *needs* long prefill — it just shortens the failure path for users who don't want long prefill on their deployment. Not a substitute for B; complementary.
- **Code shape.** ~60–100 lines: new config field with default `None`, validator (cap must be `>= 0` if set; `0` means reject everything which is silly so `> 0` if set), token-count check in the chat-completions handler before submitting to the engine, new error variant + hint, integration test (one rejection case + one passes-through case), CHANGELOG.
- **Migration.** Zero by default. Explicit cap is opt-in.
- **Risk.** Low. The check happens before `chunked-prefill` begins so no GPU work is wasted on rejections. The default-`None` shape mirrors how `kv_cache_fp8` and `num_blocks` already handle "opt-in feature" defaults in [`config.rs:75-86`](../../crates/kiln-server/src/config.rs#L75).

### Option E — real fix: chunk-yield more aggressively in prefill scheduler

- **What.** Hypothesis (1) in PR #689's `Recommended next slice` calls out chunked-prefill as the actual root-cause fix. Today, chunked-prefill admits a request and runs it through to first-token before yielding to other admitted requests. Under `workers=2` with two long prompts, each prompt monopolizes the GPU until *its* first token, so neither completes inside 305 s. Yielding more aggressively between chunks would let two concurrent long prompts each get half the GPU instead of starving each other.
- **Pros.** This is the actual fix. Closes #686 properly, not via a bigger budget.
- **Cons.** Live scheduler work in [`crates/kiln-scheduler/`](../../crates/kiln-scheduler/) — re-verify the exact module layout and the existing chunk-yield policy before designing the change; do **not** guess line numbers in this audit. Behavior change may regress single-stream throughput if yield granularity is wrong. Needs a GPU bench cycle (NVTX trace + per-yield-policy A/B) plus the `kiln_request_prefill_tokens_completed` gauge from PR #689 item 3 to even know whether the fix is doing what it should. This is the v0.2.x → v0.3.x scheduler line of work.
- **Code shape.** Requires its own audit doc with GPU bench numbers. This audit explicitly defers detailed design.
- **Migration.** Behavior change — chunk-yield policy is observable in throughput shape. Needs a migration paragraph in CHANGELOG plus an opt-out env var for the first release that ships it.
- **Risk.** High. Real scheduler work during release-readiness cleanup is exactly the kind of v0.3.x commitment Phase 11 is supposed to defer until after the onboarding baseline is stable.

## 4. Recommendation

**Pick Option B + D combined for v0.2.10.** Explicitly defer Option C until a real customer asks for it (YAGNI). Explicitly defer Option E to its own audit doc + GPU bench cycle scheduled after the release-readiness milestone.

**Why B + D, specifically.**

- **B alone** would fix the documented failure mode (the QUICKSTART §4.3 workaround already prescribes raising to 600 s) but leave operators who want a hard ceiling on prompt size with no knob — they would have to wrap kiln in a reverse proxy that rejects oversized prompts at the edge.
- **D alone** would give operators a hard ceiling but leave the default 300 s timeout in place, which means deployments that *want* to accept long prompts still hit the documented regression on the first request.
- **B + D together** make the default behavior good (long prefills get enough budget) **and** give operators who want to clamp their deployment a single TOML knob to do so.
- The combined surface is small: one default change + one new optional config field + one new error variant. Total under ~150 lines of code, all in `kiln-server`, no engine or scheduler changes.

**Why defer C.** Per-request override is real flexibility but it widens the API contract during release-readiness cleanup. Once shipped it's a breaking change to remove. Better to wait for users to tell us they want it.

**Why defer E.** The real scheduler work is the v0.3.x prefill story. Trying to land it during release-readiness cleanup risks dragging v0.2.10 into a multi-week debugging arc and delays onboarding polish. Ship B + D for v0.2.10, then take Option E up properly with PR #689 items 1–3 as the supporting evidence.

**Alternative primary recommendations and what would justify picking them.**

- *Pick D alone* if you believe the production SLA must stay at exactly 5 min and you'd rather reject than wait. B + D collapses to D alone in that case (no default change, just the cap). Trade-off: every long-prefill user has to know about the cap and configure a workaround, reverting to status quo for the on-by-default user.
- *Pick A* if the release-readiness baseline should ship with no v0.2.10 reliability changes at all and the workaround docs are deemed sufficient. Cheapest option; worst first impression for cold readers; no engineering investment.
- *Pick E now* if Eric believes the release-readiness story can absorb the schedule risk. This audit doesn't recommend it but the argument is open: the bisect rules out a fix in the four-commit window, so the only "real" fix path is Option E. If schedule slip is acceptable in exchange for shipping v0.2.10 with the actual prefill regression closed, Option E is the technically-correct choice.

## 5. Concrete v0.2.10 PR shopping list (post-decision)

If Eric (or a future planning-loop cycle) accepts the B + D recommendation, the v0.2.10 work decomposes into a single PR plus a small docs sweep. The numbered items below are what the implementation task should cover; this audit doc does not block on them.

1. **Bump default `request_timeout_secs` from 300 → 600** in [`crates/kiln-server/src/config.rs:330`](../../crates/kiln-server/src/config.rs#L330) + [`kiln.example.toml:13`](../../kiln.example.toml#L13) + the assertion in the config-default test at [`crates/kiln-server/src/config.rs:768`](../../crates/kiln-server/src/config.rs#L768). (The other two `300` test references at [`config.rs:914`](../../crates/kiln-server/src/config.rs#L914) and [`config.rs:973`](../../crates/kiln-server/src/config.rs#L973) — re-read each one in context to decide whether they assert the new default or are pinned for unrelated reasons.) Add CHANGELOG entry under `kiln-v0.2.10` calling out the default change explicitly so any operator reading release notes sees the slot-occupancy implication.
2. **Add `max_prompt_tokens: Option<usize>` to `ServerConfig`** following the `Option<usize>` shape already used by `memory.num_blocks` at [`config.rs:76`](../../crates/kiln-server/src/config.rs#L76). Default `None`. Validator (in the same `validate()` function at [`config.rs:702`](../../crates/kiln-server/src/config.rs#L702)) bails if set to `0`. Env override `KILN_MAX_PROMPT_TOKENS` following the `KILN_REQUEST_TIMEOUT_SECS` pattern at [`config.rs:526-530`](../../crates/kiln-server/src/config.rs#L526). Mirror in [`kiln.example.toml`](../../kiln.example.toml) as a commented-out line under `[server]` so it's discoverable.
3. **Plumb the cap into the chat-completions handler** in [`crates/kiln-server/src/api/completions.rs`](../../crates/kiln-server/src/api/completions.rs) (3 189 lines — find the request-validation site near where `max_tokens` is currently checked; the precise insertion point should be re-verified during edit, not guessed here). Reject with a new `ApiError::prompt_too_long { prompt_tokens, cap }` variant whose hint reads "Reduce input length, or increase server.max_prompt_tokens (currently <cap>) to accept this prompt." Status code `413 Payload Too Large` is the natural fit; alternative is `400 Bad Request` for symmetry with the existing 4xx errors — this audit recommends 413 because the error is specifically about size, but the implementation PR can pick either with a one-line CHANGELOG note.
4. **Refresh QUICKSTART §4.3** at [`QUICKSTART.md:296-326`](../../QUICKSTART.md#L296) to: (a) drop the now-stale "raise `server.request_timeout_secs` to at least 600" workaround sentence (the new default IS 600 — workers=1 stays as the recommended workaround for very long prefills); (b) add a one-paragraph reference to the new `max_prompt_tokens` cap with a TOML example.
5. **Optional, low-priority**: update the `request_timeout` error hint at [`crates/kiln-server/src/error.rs:88`](../../crates/kiln-server/src/error.rs#L88) from "*Try reducing max_tokens, or increase the server's request_timeout_secs in the config file.*" to also mention the new `max_prompt_tokens` cap as an upstream remediation. One-line text edit. Low value but improves the cold-reader experience.

Integration test coverage for the implementation PR:

- New config-default test asserts `request_timeout_secs == 600`.
- New roundtrip test sets `[server] max_prompt_tokens = 100` in TOML, fires a chat-completions request with 200-token prompt, asserts 413 + `code: "prompt_too_long"`.
- New roundtrip test with `max_prompt_tokens` unset (default `None`) asserts no rejection regardless of prompt size (current behavior preserved).
- New env-override test sets `KILN_MAX_PROMPT_TOKENS=50` and asserts the cap takes effect.

## 6. Out of scope

- **Wider re-bisect against the v0.2.7 tag** (PR #689 item 1). Separate GPU task once Option E gets prioritized.
- **NVTX prefill profile against the canonical 43 814-token request** (PR #689 item 2). Separate GPU task; the trace decides between hypothesis (1) "prefill is just slow" and hypothesis (3) "kernel hot-spot at high T" in the bisect doc.
- **`kiln_request_prefill_tokens_completed` gauge** (PR #689 item 3). Separate code+GPU task. Note: this gauge is a prerequisite for any Option E follow-up — without it, the chunked-prefill yield-policy A/B can't tell whether yielding is helping. The implementation PR should sequence it before any Option E work.
- **Any change to Qwen3.5-4B chunked-prefill chunk size or scheduler behavior** — that's Option E territory, deferred.
- **The desktop / `/ui` surface area.** This audit is scoped to the HTTP API; UI behavior on long prefill is a separate Phase 11 polish task.
- **`KILN_REQUEST_TIMEOUT_SECS` semantics.** The env var stays exactly as it is — `Option<u64>` override of the config default. Recommendation B + D does not change the env-var contract.

## 7. Decision log

```yaml
decision: pending
picked_option: pending
picked_at: pending
rationale: pending
linked_pr: pending
```

A future cycle (or Eric in PR review) fills this block in when the v0.2.10 implementation PR opens. The decision-log block lives at the bottom of this doc so it's easy to find with `tail`.
