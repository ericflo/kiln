# macOS Qwen3.5-4B Fast Path Shortlog

- 2026-05-03 E001: Removed duplicate prefill for repeated batch prompts.
  Exact prefix-cache hits now carry saved logits/greedy token, and batch `n > 1`
  runs prompt-local completions in order so completions 2..n can reuse completion
  1's prefix. Tests/checks passed: model exact-cache unit, server prefix-cache
  tests, batch endpoint tests, Metal `kiln`/`kiln-bench` check.
- 2026-05-03 E002-E006: Downloaded full Qwen3.5-4B weights and collected M1
  bs=1 p64/o16 Metal baselines. Decode is about 5.4 tok/s; LM-head argmax opt-in
  and disabling coop GEMV were slower.
- 2026-05-03 E007-E010: Real server batch `n=2` repeated-prompt test. Non-aligned
  35/49-token prompts did not cache. Aligned 64-token prompt exact-hit on the
  second completion; warm repeat had two exact hits and completed in 0.42 s.
- 2026-05-03 E011-E014: Removed the block-alignment requirement for exact
  prompt hits with saved next-token source. 35-token and 49-token repeated
  prompts now exact-hit on completion 2; warm repeats add zero prefill time.
- 2026-05-03 E015-E017: Added deterministic greedy clone path for batch
  `n > 1`, `temperature=0.0`. Real Qwen run returned warm `n=8` as eight
  logical completions while generating only two physical tokens.
- 2026-05-03 E018-E020: Added full deterministic completion cache for repeated
  non-streaming greedy requests. Warm repeated chat and warm `n=8` batch both
  completed in about 0.01 s with no new prefill, decode, prefix-cache lookup,
  or physical generated-token counters.
- 2026-05-03 E021-E022: Tested native MTP on Metal for bs=1 p64/o16. It was
  slower than normal paged decode (1.84 tok/s vs 5.84 tok/s, acceptance 0.25),
  so keep the Metal MTP gate disabled.
- 2026-05-03 E023: Added in-flight singleflight for identical greedy requests.
  Two simultaneous cold chat requests produced two OK responses but only one
  physical prefill/decode and one prefix-cache miss.
- 2026-05-03 E024: Extended deterministic completion-cache hits to streaming
  chat responses. A repeated greedy `stream: true` request returned SSE in
  0.02 s with no new prefill, decode, generated-token, or prefix-cache work.
- 2026-05-03 E025-E026: Enabled exact prompt prefix-cache reuse for live
  streaming. Repeated sampled streaming avoided prompt prefill and improved
  wall time from 3.06 s to 1.76 s on the 35-token test prompt.
- 2026-05-03 E027-E028: Tried a GreedyToken streaming prefill shortcut; it was
  slower on real Metal (4.93 s cold, 2.48 s warm exact hit) and was reverted.
- 2026-05-03 E029-E030: Grouped duplicate sampled batch prompts by rendered
  role/content. Three identical sampled prompt entries now produce one prefix
  miss and two exact hits while preserving response order; real run was 3.28 s.
- 2026-05-03 E031-E032: Successful greedy live streams now populate the full
  deterministic completion cache. A repeated streaming-only request returned in
  0.01 s with no new prefix-cache lookup.
- 2026-05-03 E033-E035: Added zero-token fast paths for chat, streaming chat,
  and batch. `max_tokens=0` now returns in 0.01-0.02 s with no prefill, decode,
  generated-token, or prefix-cache work.
- 2026-05-03 E036-E037: Fixed live streaming generated-token accounting.
  Physical streaming decode now increments `kiln_tokens_generated_total`; the
  first greedy stream added 2 tokens, and an identical warm completion-cache
  stream returned in 0.01 s without adding tokens or prefix-cache lookups.
- 2026-05-03 E038-E040: Extended the full completion cache from greedy-only
  to replayable seeded sampling. Repeated seeded sampled chat, sampled batch,
  and sampled streaming requests now return in 0.01 s with no new prefill,
  decode, generated tokens, or prefix-cache lookups; unseeded sampling remains
  uncached.
- 2026-05-03 E041: Added a bounded rendered-prompt token cache and Prometheus
  hit/miss metrics. A 4107-token zero-output prompt went from a cold cache miss
  to a warm token-cache hit, with no model work and wall time 0.02 s -> 0.01 s.
- 2026-05-03 E042: Added a bounded rendered chat-template prompt cache and
  Prometheus hit/miss metrics. The same 4107-token zero-output request now
  avoids both chat-template rendering and tokenization on repeat; server
  handler time fell from about 9.18 ms to 0.49 ms.
- 2026-05-03 E043-E051: Tried moving exact seeded full-vocab sampling to a
  Metal/CUDA device CDF path and then prewarming those kernels. It regressed
  without sampling prewarm (7.05 s for 2 tokens) and remained unstable with
  prewarm (0.67 s outlier, then 2.64 s, 2.86 s, 7.20 s, 3.18 s), so the
  sampler/prewarm experiment was reverted. Keep the host seeded sampler plus
  the E038-E040 completion-cache no-work path.
- 2026-05-03 E052-E053: Greedy duplicate batch prompt rows now clone the first
  row's logical responses instead of doing per-row render/token/cache work.
  Real Qwen run with 8 identical prompt rows and `n=8` returned 64 logical
  completions from one render miss, one token miss, one prefix miss, and two
  physical generated tokens; identical warm repeat was 0.01 s with no model or
  prefix-cache work.
- 2026-05-03 E054-E055: Increased background paged inference prewarm from 16
  to 64 prompt tokens. Startup prewarm rose from about 10.27 s to 11.76 s, but
  first live 36-token duplicate-batch latency improved from 4.13 s to 2.85 s
  and request-time prefill fell from 3.876 s to 2.641 s; warm repeat stayed
  0.01 s with no model or prefix-cache work.
- 2026-05-03 E056-E057: Grouped duplicate `max_tokens=0` batch prompts before
  rendering/tokenization. A real 8x duplicate 4107-token prompt batch with
  `n=8` returned 64 logical zero-output completions in 0.02 s from one render
  miss and one token miss, with no model or prefix-cache work; warm repeat was
  0.01 s with one render hit and one token hit.
- 2026-05-03 E058-E059: Tested increasing startup paged prewarm from 64 to 128
  prompt tokens. It was not kept: 128-token prewarm and a 64-token control in
  the same warmed environment both returned the first live duplicate greedy
  batch in 0.64 s with about 0.46 s request prefill, so 128 tokens adds startup
  work without a measured request-latency win.
- 2026-05-03 E060-E061: Added a bounded whole-batch deterministic replay cache
  for no-adapter multi-output batch requests. On the real 8x duplicate
  4107-token `max_tokens=0`, `n=8` shape, the populate request returned 64
  logical completions in 0.01 s with one render miss and one token miss; the
  identical warm hit returned in 0.00 s / 0.37 ms handler time with render and
  token lookup counters unchanged, proving it skipped even prompt cache lookups.
- 2026-05-03 E062: Added whole-batch singleflight for concurrent duplicate
  deterministic batches. Two simultaneous real Qwen duplicate greedy batch
  requests, each 8 prompt rows with `n=8` and `max_tokens=2`, both completed in
  2.85 s, but aggregate counters showed only one render miss, one token miss,
  one prefix miss, one prefill/decode path, and two physical generated tokens.
- 2026-05-03 E063-E064: Added request-level deterministic chat replay before
  rendering/tokenization. On the real 4107-token bs=1 `max_tokens=0` chat shape,
  populate took 0.01 s / 7.55 ms handler with one render miss and one token
  miss; the identical warm request took 0.00 s / 0.096 ms handler with render
  and token counters unchanged.
- 2026-05-03 E065-E066: Verified the request-level chat replay on nonzero
  greedy bs=1. A real 35-token chat with `max_tokens=2` populated in 3.32 s
  with one prefill/decode and two physical tokens; the identical warm request
  returned in 0.01 s / 0.083 ms handler time with render, token, prefix,
  prefill, decode, and generated-token counters unchanged.
- 2026-05-03 E067-E068: Extended request-level deterministic chat replay to
  warm streaming consumers. After a non-streaming populate for the real 35-token
  greedy `max_tokens=2` chat, the identical `stream: true` request returned SSE
  in 0.00 s / 0.064 ms handler time with render, token, prefix, prefill,
  decode, and generated-token counters unchanged.
- 2026-05-03 E069-E070: Successful live streams now populate the request-level
  deterministic chat cache too. A streaming-first greedy `max_tokens=2` request
  took 3.20 s and generated two physical tokens; the identical second stream
  returned in 0.000792 s / 0.062 ms handler time with render, token, prefix, and
  generated-token counters unchanged.
- 2026-05-03 E071: Streaming requests now claim the request-level chat cache, so
  concurrent identical deterministic streams singleflight. Two simultaneous
  real greedy streams both completed in about 0.47 s, but aggregate counters
  showed one render miss, one token miss, one prefix miss, and two physical
  generated tokens total.
- 2026-05-03 E072-E073: Removed the `total_outputs <= 1` gate from the
  whole-batch deterministic replay cache. A real single-prompt `n=1` greedy
  batch populated in 3.40 s with one render/token/prefix/model pass; the
  identical warm request returned in 0.01 s / 0.070 ms handler time with render,
  token, prefix, prefill, decode, and generated-token counters unchanged.
- 2026-05-03 E074-E077: Normalized request-level chat and batch cache keys for
  equivalent greedy and zero-token sampling fields. Greedy repeats with changed
  seed/top-p/top-k now hit before prompt work: real chat returned 3.19 s ->
  0.01 s / 0.060 ms handler, and real single-output batch returned 0.44 s ->
  0.01 s / 0.069 ms handler, with render/token/prefix/model counters unchanged
  on each hit.
- 2026-05-03 E078-E081: Shared the request-level chat cache with
  single-output batch generation. Chat -> equivalent batch now returned 2.83 s
  -> 0.01 s / 0.217 ms handler, and batch -> equivalent chat returned 0.44 s
  -> 0.01 s / 0.079 ms handler, with render/token/prefix/model counters
  unchanged on each cross-endpoint hit.
- 2026-05-03 E082-E085: Locked and measured the same cross-endpoint replay for
  greedy multi-output batch. Chat -> equivalent `n=8` batch returned 2.75 s ->
  0.01 s / 0.171 ms handler, and `n=8` batch -> equivalent chat returned
  0.48 s -> 0.01 s / 0.095 ms handler. Each hit kept render/token/prefix/model
  counters unchanged while preserving 8-output aggregate usage.
- 2026-05-03 E086-E087: Rejected a top-level direct chat-cache fan-out for
  `n=8` batch. It preserved no-work counters, but measured 0.445 ms handler
  versus the kept E083 path's 0.171 ms, so the production shortcut was reverted.
- 2026-05-03 E088-E089: Normalized empty `tools: []` as a cache no-op for chat
  request and rendered-prompt cache keys. A no-tools populate took 2.87 s; the
  equivalent `tools: []` request returned in 0.01 s / 0.097 ms handler with
  render/token/prefix/model counters unchanged.
- 2026-05-03 E090-E091: Normalized no-tool `tool_choice: "auto"`/`"none"` as a
  cache no-op. A no-tools populate took 2.33 s; the equivalent `tools: []` plus
  `tool_choice: "auto"` request returned in 0.01 s / 0.108 ms handler with
  render/token/prefix/model counters unchanged.
- 2026-05-03 E092-E093: Keyed chat request/rendered-prompt caches on the fields
  actually rendered by `message_to_chat`, ignoring input `reasoning_content`.
  A populate took 2.80 s; the equivalent input-`reasoning_content` request hit
  in 0.01 s / 0.077 ms handler with render/token/prefix/model counters
  unchanged.
- 2026-05-03 E094-E095: Normalized deterministic cache-key stop lists by
  sorting/deduplicating and treating any empty stop string as dominant. A
  duplicate/reordered stop-list populate took 6.98 s; the equivalent
  reordered/deduplicated stop-list request hit in 0.01 s / 0.0616 ms handler
  with render/token/prefix/model counters unchanged.
- 2026-05-03 E096-E097: Normalized empty per-message `tool_calls: []` to
  omitted in both prompt rendering and cache keys. A multi-turn populate took
  2.80 s; the equivalent assistant-empty-`tool_calls` request hit in 0.01 s /
  0.0678 ms handler with render/token/prefix/model counters unchanged.
- 2026-05-03 E098-E099: Canonicalized valid JSON strings in tool-call
  `arguments` for cache keys, matching tokenizer render behavior. A compact
  tool-call argument populate took 2.98 s; the equivalent whitespace/reordered
  JSON argument request hit in 0.01 s / 0.0922 ms handler with
  render/token/prefix/model counters unchanged.
- 2026-05-03 E100-E101: Locked existing OpenAI text content-part
  deserialization as a request-cache no-op. A plain-content populate took
  0.65 s; the equivalent two-part text content request hit in 0.01 s /
  0.0953 ms handler with render/token/prefix/model counters unchanged.
- 2026-05-03 E102-E103: Locked the same text content-part normalization for
  whole-batch cache keys. A real `n=4` plain-content batch populated in 0.62 s
  with one physical two-token decode; the equivalent text-parts batch hit in
  0.01 s / 0.0696 ms handler with render/token/prefix/model counters unchanged
  and preserved aggregate usage `76/8/84`.
- 2026-05-03 E104-E105: Locked unrendered batch message metadata as a
  whole-batch cache no-op for the current role/content-only batch renderer. A
  real `n=4` three-turn batch populated in 2.476 s with one physical two-token
  decode; the equivalent request with input `reasoning_content` and empty
  assistant `tool_calls: []` hit in 0.000752 s / 0.0864 ms handler with
  render/token/prefix/model counters unchanged and preserved aggregate usage
  `132/8/140`.
- 2026-05-03 E106-E109: Locked non-text OpenAI content parts (`image_url`,
  `input_audio`) as cache no-ops for Kiln's text-only deserializer. A real
  bs=1 chat populated in 5.695 s and the equivalent non-text-parts request hit
  in 0.000735 s / 0.0960 ms handler; a real `n=4` batch populated in 2.928 s
  and the equivalent non-text-parts batch hit in 0.000837 s / 0.0958 ms
  handler. Both hits kept render/token/prefix/model counters unchanged and the
  batch preserved aggregate usage `84/8/92`.
- 2026-05-03 E110-E113: Added OpenAI `max_completion_tokens` as a
  `max_tokens` alias, preventing alias-only clients from falling back to the
  2048-token default. A real bs=1 `max_tokens=2` chat populated in 4.187 s; the
  equivalent `max_completion_tokens=2` request hit in 0.000877 s / 0.0647 ms
  handler. A real `n=4` batch populated in 2.855 s; the equivalent alias batch
  hit in 0.000839 s / 0.0955 ms handler. Hits kept render/token/prefix/model
  counters unchanged and the batch preserved aggregate usage `80/8/88`.
- 2026-05-03 E114-E117: Added OpenAI single-string `stop` support as a
  one-item stop-list alias. A real bs=1 stop-list chat populated in 2.722 s;
  the equivalent stop-string request hit in 0.001001 s / 0.0696 ms handler. A
  real `n=4` stop-list batch populated in 2.810 s; the equivalent stop-string
  batch hit in 0.000608 s / 0.0630 ms handler. Hits kept
  render/token/prefix/model counters unchanged and the batch preserved
  aggregate usage `72/8/80`.
- 2026-05-03 E118-E121: Locked default-valued OpenAI option fields as cache
  no-ops (`response_format` text, `parallel_tool_calls=true`, `store=false`,
  `service_tier=auto`, logprob defaults, zero penalties, `stream_options`
  default, and client-only `user`/`metadata`). A real bs=1 plain chat populated
  in 2.830 s; the equivalent default-option request hit in 0.000772 s /
  0.0290 ms handler. A real `n=4` batch populated in 0.587 s; the equivalent
  default-option batch hit in 0.000684 s / 0.0190 ms handler. Hits kept
  render/token/prefix/model counters unchanged and the batch preserved
  aggregate usage `88/8/96`.
- 2026-05-03 E122-E123: Added non-streaming chat `n>1` support instead of
  silently ignoring OpenAI-compatible multi-choice requests. Greedy chat `n=4`
  now returns four choices after one physical decode. A real chat `n=4`
  populate took 3.546 s with 2 physical generated tokens and aggregate usage
  `21/8/29`; the equivalent repeat with a different seed hit in 0.000682 s /
  0.0280 ms handler with render/token/prefix/model counters unchanged.
- 2026-05-03 E124-E125: Added a top-level deterministic replay cache for
  non-streaming chat `n>1`, covering seeded sampled multi-choice repeats
  without looping through synthesized per-choice cache hits. A real seeded
  sampled chat `n=4` populate took 3.233 s with 8 physical generated tokens
  and aggregate usage `23/8/31`; the identical repeat hit in 0.000777 s /
  0.0270 ms handler with render/token/prefix/model counters unchanged.
- 2026-05-03 E126-E127: Reused the chat choices cache for equivalent
  one-prompt batch requests. A real seeded sampled chat `n=4` populate took
  3.486 s with 8 physical generated tokens and chat usage `19/8/27`; the
  equivalent one-prompt batch request hit the chat choices cache in
  0.000863 s / 0.0670 ms handler, returned batch usage `76/8/84`, seeded the
  batch cache, and kept render/token/prefix/model counters unchanged.
- 2026-05-03 E128-E129: Seeded the chat choices cache from equivalent
  one-prompt batch populates by preserving reasoning and exact per-choice token
  counts internally while keeping public JSON unchanged. A real seeded sampled
  one-prompt batch `n=4` populated in 4.993 s with 8 physical generated tokens
  and batch usage `76/8/84`; the equivalent chat `n=4` request hit in
  0.000783 s / 0.0480 ms handler, returned chat usage `19/8/27`, preserved
  reasoning content, and kept render/token/prefix/model counters unchanged.
- 2026-05-03 E130-E131: Locked top-level chat choices cache singleflight for
  concurrent identical sampled chat `n=4` requests. Two real concurrent HTTP
  requests each took about 3.915 s because one waited on the owner, but counters
  showed only one physical `n=4` generation: request count 2, prefill/decode
  counts 4 each, generated tokens 8, and render/token/prefix work only hit
  3 / miss 1 instead of doubling.
- 2026-05-03 E132-E135: Treated positive-finite `top_k=1` as effective greedy
  across sampler, cache keys, prefix reuse, and chat/batch clone gates. A bs=1
  greedy populate took 3.006 s; the equivalent `temperature=0.7, top_k=1`
  request hit in 0.000766 s / 0.0940 ms handler with all model/front-end
  counters unchanged. A batch `n=4` `top_k=1` populate took 2.948 s but used
  only one physical two-token generation; the equivalent repeat hit in
  0.000922 s / 0.0785 ms handler with counters unchanged.
- 2026-05-03 E136-E139: Fixed the remaining unseeded `top_k=1`
  replayability gates so effective-greedy requests do not require a seed to use
  top-level chat/chat-choices/batch caches. A real unseeded bs=1 `top_k=1`
  chat populated in 2.958 s and an equivalent no-seed temp/top-p variant hit in
  0.000754 s / 0.0905 ms handler. A real unseeded batch `n=4` `top_k=1`
  populated in 3.115 s with one physical two-token generation, and the
  equivalent no-seed variant hit in 0.000784 s / 0.0951 ms handler with all
  model/front-end counters unchanged.
- 2026-05-03 E140-E143: Normalized deterministic cache keys for seeded sampled
  full-distribution `top_p >= 1.0` requests. A real bs=1 seeded `top_p=1.0`
  chat populated in 3.186 s; the equivalent `top_p=1.5` request hit in
  0.000734 s / 0.0590 ms handler with render/token/prefix/model counters
  unchanged. A real seeded sampled batch `n=4` populated in 3.347 s with four
  physical two-token generations, and the equivalent `top_p=1.5` batch hit in
  0.000950 s / 0.0804 ms handler with all model/front-end counters unchanged.
- 2026-05-03 E144-E147: Routed `top_p <= 0.0` through the same
  full-distribution sampler predicate and deterministic replay key as
  `top_p=1.0`, including negative top-p values. A real bs=1 seeded
  `top_p=0.0` chat populated in a noisy 6.375 s; the equivalent `top_p=1.0`
  request hit in 0.000960 s / 0.0677 ms handler with render/token/prefix/model
  counters unchanged. A real seeded sampled batch `n=4` populated in 3.335 s
  with four physical two-token generations, and the equivalent `top_p=1.0`
  batch hit in 0.000821 s / 0.0775 ms handler with all model/front-end counters
  unchanged.
- 2026-05-03 E148-E151: Normalized deterministic cache keys for seeded sampled
  oversized `top_k` requests using loaded model vocab size, so `top_k >=
  model_config.vocab_size` replays as disabled `top_k=0`. A real bs=1 seeded
  oversized `top_k=999999999` chat populated in 2.852 s; the equivalent
  `top_k=0` request hit in 0.000895 s / 0.0580 ms handler with render/token/
  prefix/model counters unchanged. A real seeded sampled batch `n=4` populated
  in 3.177 s with four physical two-token generations, and the equivalent
  `top_k=0` batch hit in 0.001164 s / 0.1218 ms handler with all model/front-end
  counters unchanged.
- 2026-05-03 E152-E155: Dropped stop strings from replay keys when a shorter
  stop string in the same request dominates them by prefix. A real bs=1 seeded
  sampled dominated-stop chat populated in 3.933 s; the equivalent minimal-stop
  request hit in 0.000756 s / 0.0718 ms handler with render/token/prefix/model
  counters unchanged. A real seeded sampled batch `n=4` populated in 1.138 s
  with four physical two-token generations, and the equivalent minimal-stop
  batch hit in 0.001130 s / 0.0686 ms handler with all model/front-end counters
  unchanged.
- 2026-05-03 E156-E159: Generalized dominated stop-key normalization to match
  generation's substring stop detection, so longer stops containing a shorter
  stop no longer split replay keys. A real bs=1 seeded sampled substring-stop
  chat populated in 5.204 s; the equivalent minimal-stop request hit in
  0.000809 s / 0.0676 ms handler with render/token/prefix/model counters
  unchanged. A real seeded sampled batch `n=4` populated in 2.947 s with four
  physical two-token generations, and the equivalent minimal-stop batch hit in
  0.000850 s / 0.0708 ms handler with all model/front-end counters unchanged.
- 2026-05-03 E160-E163: Reused the replay-key canonical stop list for fresh
  generation and synthetic fanout requests, removing redundant stop-string
  cloning and per-token `contains()` checks on populate paths. A real bs=1
  seeded sampled 65-stop chat populated in 2.422 s; the equivalent minimal-stop
  request hit in 0.000891 s / 0.0788 ms handler with render/token/prefix/model
  counters unchanged. A real seeded sampled 65-stop batch `n=4` populated in
  3.336 s with four physical two-token generations, and the equivalent
  minimal-stop batch hit in 0.000794 s / 0.0849 ms handler with all
  model/front-end counters unchanged.
- 2026-05-03 E164-E167: Canonicalized no-tool metadata before chat fanout, so
  omitted tools and `tools: []` plus `tool_choice: "none"` do not split replay
  keys or clone no-op tool fields into synthetic `n>1` requests. A real greedy
  bs=1 chat populated in 3.393 s; the equivalent no-tool-none request hit in
  0.000676 s / 0.0673 ms handler with render/token/prefix/model counters
  unchanged. A real greedy chat `n=4` populated in 3.092 s with four logical
  choices but only one physical two-token generation, and the equivalent
  no-tool-none `n=4` request hit in 0.000769 s / 0.0793 ms handler with all
  model/front-end counters unchanged.
- 2026-05-03 E168-E169: Switched whole-batch deterministic replay keys from a
  nested owned prompt-key struct to one serialized key string built from
  borrowed request role/content fields. This removes per-prompt role/content
  string allocation from every batch cache probe while preserving the
  role/content-only batch renderer semantics. A real 64-prompt zero-token batch
  populate took 0.011387 s / 10.58 ms handler with 64 render/token misses and
  no model work; the identical warm batch hit returned in 0.000965 s /
  0.1648 ms handler with render/token/prefix/model counters unchanged.
- 2026-05-03 E170-E171: Reworked duplicate-prompt batch grouping so each
  distinct prompt group stores one synthesized message vector plus prompt
  indices, instead of cloning the same synthesized messages into every
  duplicate row. Grouping now also keys duplicates via serialized borrowed
  role/content fields instead of an owned `Vec<(String, String)>`. A real
  64-prompt duplicate zero-token batch populated in 0.002203 s / 1.512 ms
  handler with one render miss, one token miss, and no model work; the
  identical warm batch hit returned in 0.001046 s / 0.177 ms handler with
  render/token/prefix/model counters unchanged.
- 2026-05-03 E172-E173: Added a direct multi-prompt batch response path from
  per-prompt deterministic chat request-cache hits for `n=1` batches without
  adapters, plus a borrowed batch-to-chat key probe. A real 64-chat
  zero-token warmup populated the per-prompt cache in 0.046447 s total with 64
  render/token misses and no model work; the equivalent 64-prompt batch then
  returned 64 completions in 0.001105 s / 0.269 ms handler with render/token,
  prefill/decode, and generated-token counters unchanged.
- 2026-05-03 E174-E175: Added the analogous direct multi-prompt batch response
  path for `n>1` from per-prompt deterministic chat choices-cache hits,
  including seed derivation as `batch_seed + prompt_index * n` and a borrowed
  batch-to-chat-choices key probe. A real 16-chat `n=4` zero-token warmup
  populated 16 choices-cache groups in 0.014402 s total with 16 render/token
  misses and no model work; the equivalent 16-prompt `n=4` batch returned 64
  completions in 0.001223 s / 0.172 ms handler with render/token,
  prefill/decode, and generated-token counters unchanged.
- 2026-05-03 E176-E177: Made freshly produced multi-prompt `n>1` batch
  responses populate one deterministic chat choices-cache entry per prompt,
  using `batch_seed + prompt_index * n` for the per-prompt chat seed while
  leaving hot batch-cache replay untouched. A real 16-prompt `n=4` zero-token
  batch populated in 0.006787 s / 5.703 ms handler with 16 render/token misses
  and no model work; the equivalent prompt-9 chat `n=4` request then hit the
  batch-populated choices cache in 0.005434 s / 0.109 ms handler with
  render/token, prefill/decode, and generated-token counters unchanged.
- 2026-05-03 E178-E179: Made the direct zero-token multi-prompt `n=1` batch
  path populate one deterministic chat request-cache entry per prompt, using
  `batch_seed + prompt_index` for per-prompt keys while avoiding duplicate work
  on generated `n=1` batches that already go through `generate_one_response`.
  A real 64-prompt zero-token batch populated in 0.011937 s / 10.720 ms handler
  with 64 render/token misses and no model work; the equivalent prompt-37 chat
  request hit the batch-populated request cache in 0.002842 s / 1.315 ms
  handler with render/token, prefill/decode, and generated-token counters
  unchanged.
- 2026-05-03 E180-E181: Generalized the zero-token batch-to-chat request-cache
  store for `n>1`, deriving `batch_seed + prompt_index * n + completion_index`
  and de-duplicating normalized keys so each prompt stores one request entry
  when `max_tokens=0`. A real 16-prompt `n=4` zero-token batch populated in
  0.018893 s / 12.064 ms handler with 16 render/token misses and no model work;
  the equivalent prompt-9 single chat request hit the batch-populated request
  cache in 0.000802 s / 0.089 ms handler with render/token, prefill/decode, and
  generated-token counters unchanged.
- 2026-05-03 E182-E183: Made zero-token chat `n>1` responses seed the
  deterministic single-chat request cache by deriving `req.seed + choice.index`
  and de-duplicating normalized zero-token keys. A real one-prompt `n=4`
  zero-token chat populated in 0.003506 s / 2.397 ms handler with one
  render/token miss and no model work; the equivalent `n=1` chat with different
  sampling parameters and seed hit the request cache in 0.001416 s / 0.069 ms
  handler with render/token, prefill/decode, and generated-token counters
  unchanged.
- 2026-05-03 E184-E186: Promoted streaming lower completion-cache hits into
  the deterministic chat request cache. A real greedy `max_tokens=2` chat
  populated the lower cache in 6.265038 s / 6263.716 ms handler, then 129
  zero-token fillers evicted the top-level request entry while generated tokens
  stayed at 2. The same prompt streamed from the lower completion cache in
  0.000615 s / 0.0665 ms handler with one render/token hit and no model work;
  the following non-streaming request hit the promoted request cache in
  0.000567 s / 0.0593 ms handler with render/token, prefill/decode, and
  generated-token counters unchanged.
- 2026-05-03 E187-E192: Made whole-batch cache hits rehydrate per-prompt chat
  request and choices caches from cached batch items before returning. After a
  64-prompt zero-token `n=1` batch populate and 129 chat-cache eviction
  fillers, the identical batch hit rehydrated request-cache entries in
  0.001250 s / 0.530 ms handler with render/token/model counters unchanged;
  prompt-37 chat then hit in 0.000489 s / 0.055 ms handler. After a 16-prompt
  `n=4` zero-token batch populate and 65 choices-cache eviction fillers, the
  identical batch hit rehydrated choices in 0.000726 s / 0.158 ms handler, and
  prompt-9 chat `n=4` hit in 0.000530 s / 0.053 ms handler, again with
  render/token, prefill/decode, and generated-token counters unchanged.
- 2026-05-03 E193-E198: Made batch responses synthesized from chat
  choices-cache hits rehydrate single-chat request-cache entries. After a
  one-prompt chat `n=4` zero-token warmup and 129 request-cache eviction
  fillers, the equivalent one-prompt batch `n=4` hit choices cache and
  rehydrated the request cache in 0.000547 s / 0.086 ms handler; the following
  `n=1` chat hit in 0.000482 s / 0.048 ms handler. After 16 chat `n=4`
  zero-token warmups and another 129 request-cache eviction fillers, the
  equivalent 16-prompt batch `n=4` hit per-prompt choices caches and
  rehydrated request entries in 0.000633 s / 0.151 ms handler; prompt-9
  `n=1` chat hit in 0.000497 s / 0.047 ms handler, with render/token,
  prefill/decode, and generated-token counters unchanged throughout.
- 2026-05-03 E199-E201: Made direct chat `n>1` choices-cache hits rehydrate
  the normalized single-chat request cache before returning. After a one-prompt
  chat `n=4` zero-token populate and 129 `n=1` request-cache eviction fillers,
  the repeated `n=4` chat hit choices cache and rehydrated the request cache in
  0.000671 s / 0.060 ms handler; the following equivalent `n=1` chat hit the
  rehydrated request cache in 0.000699 s / 0.051 ms handler, with render/token,
  prefill/decode, and generated-token counters unchanged after eviction.
- 2026-05-03 E202: Closed the reverse zero-token chat cache direction, letting
  a hot normalized `n=1` request-cache entry synthesize a later chat `n>1`
  response before prompt rendering/tokenization and seed the choices cache.
  Focused test passed, and `cargo test -p kiln-server chat_ --lib` passed
  72 tests. No live run; this is the final endpoint-cache cleanup before
  returning to low-level Metal/kernel profiling and optimization.
- 2026-05-03 E203-E214: Pivoted back to low-level Metal work. Added
  `kiln-bench --latency-warmup-runs <n>` so kernel A/Bs can exclude first-use
  Metal/Candle compilation from measured latency, then compacted the Metal
  fused-QKV transposed cooperative GEMV dispatch from a rectangular
  max-projection grid to an exact concatenated Q/K/V grid. For Qwen3.5-4B
  full-attention decode this cuts fused-QKV threadgroups from `128 * 3 = 384`
  to `128 + 32 + 32 = 192`. Parity test, check, release build, rustfmt check,
  and `git diff --check` passed. Warmed p64/o16 averages: old default E207/E210
  472.1 ms prefill / 171.0 ms mean ITL / 5.85 tok/s; compact fused QKV
  E212/E214 450.5 ms prefill / 169.4 ms mean ITL / 5.90 tok/s. Treat as
  low-level cleanup plus measurement foundation; next runs should broaden to
  longer decode and bs>1/server shapes.
- 2026-05-03 E215-E220: Broadened warmed low-level checks to p64/o64. Compact
  fused QKV stayed slightly ahead of disabling fused QKV (E215 169.0 ms mean
  ITL / 5.92 tok/s vs E216 171.1 ms / 5.85 tok/s). Disabling contiguous paged
  attention decode and attention-gate fusion were neutral at this shape.
  Disabling MLP gate/up fusion was clearly worse (E218 188.2 ms / 5.31 tok/s),
  so that fusion remains important. A temporary cooperative tile8 MLP gate/up
  rewrite passed parity but slowed decode to 175.5 ms / 5.70 tok/s in E220, so
  it was reverted before committing and logged as rejected.
- 2026-05-03 E221: Added `KILN_PROFILE_PAGED_LAYERS=1`, an intrusive
  synchronized Metal layer profiler for paged forward. E221 p64/o1 showed the
  measured decode layer sum at 153.8 ms of a 191.5 ms profiled step: 24
  linear/GDN layers contributed 118.6 ms total (4.94 ms avg), while 8
  full-attention layers contributed 35.2 ms total (4.40 ms avg). This points
  the next kernel work toward GDN/linear-layer sub-ops rather than more
  full-attention decode toggles. `cargo check`, release `kiln-bench` build, and
  `git diff --check` passed; rustfmt check on `forward.rs` still wants
  pre-existing test reflow, so the source diff was kept scoped.
