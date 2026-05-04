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
- 2026-05-03 E222: Added `KILN_PROFILE_GDN_STAGES=1`, an intrusive
  synchronized GDN sub-stage profiler for paged linear-attention decode.
  E222 p64/o1, with both layer and GDN-stage profiling enabled, reported
  262.8 ms mean ITL due to instrumentation overhead. The measured decode layer
  sum was 225.5 ms: 24 linear/GDN layers contributed 188.2 ms total (7.84 ms
  avg), while 8 full-attention layers contributed 37.3 ms total (4.66 ms avg).
  Across the measured decode linear layers, GDN stages summed to 91.9 ms:
  `in_proj` 36.1 ms, `out_proj` 17.0 ms, `gates` 12.4 ms, `gated_norm` 9.8 ms,
  `recurrent` 9.5 ms, `qkv_conv_norm` 7.1 ms, and `post_transpose` ~0.0 ms.
  Treat this as target selection, not a latency baseline: the next low-level
  target is Metal GDN input projection.
- 2026-05-03 E223-E224: Ablated Metal GDN input projection. Disabling
  `KILN_DISABLE_METAL_GDN_IN_PROJ_FUSION=1` made warmed p64/o64 decode much
  worse than the E215 baseline: 191.8 ms mean ITL / 5.21 tok/s vs 169.0 ms /
  5.92 tok/s, so the fused kernel is necessary. A temporary tile8 cooperative
  GDN in-proj rewrite passed
  `cargo test -p kiln-model --features metal test_gdn_in_proj_decode_matches_broadcast_matmul --lib`
  but still slowed warmed p64/o64 decode to 177.7 ms / 5.63 tok/s, so it was
  reverted. Release `kiln-bench` was rebuilt after the revert. Next
  input-projection work needs a different kernel design, not the existing
  cooperative transposed-GEMV template.
- 2026-05-03 E225-E233: Matched Metal GDN gates/rmsnorm kernels to the real
  checkpoint dtype envelope: `A_log` and `linear_attn.norm.weight` are F32,
  while `dt_bias` is BF16. Updating the gates, gated RMSNorm, and fused
  gates+recurrent+rmsnorm kernels to read F32 aux weights enabled the fused
  decode path. Focused Metal parity tests passed, release `kiln-bench` build
  passed, `cargo check` passed, rustfmt for `metal.rs` passed, and
  `git diff --check` passed. Same-session warmed p64/o64 improved from E229
  174.4 ms mean ITL / 5.73 tok/s to E231 168.4 ms / 5.94 tok/s and E232
  172.2 ms / 5.81 tok/s. Synchronized p64/o1 profiling dropped measured
  linear/GDN layer sum from E228 187.3 ms to E233 163.0 ms; GDN stage sum
  dropped from 91.2 ms to 70.0 ms as split `gates` + `recurrent` +
  `gated_norm` became fused `gates_recur_gated_norm`.
- 2026-05-03 E234: Rebuilt release `kiln` and ran a real-server bs=4
  distinct-prompt `/v1/completions/batch` smoke after the F32-aux kernel
  change. Four cold prompts with `max_tokens=2` completed in 4.44 s wall /
  4,420.024 ms handler, generated 8 physical tokens, returned 68 prompt tokens
  + 8 completion tokens, and showed 0 prefix-cache hits with 4 render/token
  misses. This validates the current branch on a non-cache bs>1 server shape;
  future bs>1 speed work still needs true batched model-forward or scheduler
  continuous batching.
- 2026-05-03 E235-E237: Reduced decode hot-path Metal output allocation count
  for fused full-attention QKV projection and GDN input projection by backing
  each logical output tuple with one sliced output tensor. Focused Metal parity
  tests passed, release `kiln-bench` build passed, and rustfmt for `metal.rs`
  passed. Warmed p64/o64 measured E235 169.0 ms mean ITL / 5.92 tok/s and E236
  168.1 ms / 5.95 tok/s. Synchronized E237 p64/o1 showed projection kernel
  time basically unchanged (`in_proj` 36.754 ms, `out_proj` 17.302 ms across
  24 GDN layers), so treat this as a small allocation-overhead cleanup, not a
  new GEMV-throughput breakthrough.
- 2026-05-03 E238: Rejected a specialized full-attention Q/Gate/K/V projection
  candidate. The temporary Metal kernel split gated q_proj columns directly
  into contiguous Q and gate outputs, passed a focused split-reference parity
  test, and release `kiln-bench` built, but warmed p64/o64 regressed to
  188.2 ms mean ITL / 5.31 tok/s. The candidate was reverted before commit;
  the extra in-kernel routing cost outweighed removing the post-projection
  narrow/contiguous copies.
- 2026-05-03 E239: Ran the same real-server bs=4 distinct cold batch as E234
  with `KILN_PREFIX_CACHE_ENABLED=false`. Timing was effectively identical:
  4.43 s wall / 4,416.317 ms handler versus E234 4.44 s / 4,420.024 ms. Metrics
  showed prefix cache disabled with 0 lookups and 0 retained entries. This
  rules out prefix-cache registration as the bs=4 bottleneck for that shape;
  future bs>1 work should target actual batched/continuous model execution.
- 2026-05-03 E240: Compared the E234/E239 four distinct prompts as sequential
  no-prefix `/v1/chat/completions` calls. Sequential wall time was 6.01 s with
  5,921.841 ms handler sum and 8 generated tokens; per-request curl timings
  were 4.30 s, 0.57 s, 0.55 s, and 0.54 s. This was slower than the 4.43 s
  no-prefix batch run, so naive serial batch handling is not the bs>1 fix.
- 2026-05-03 E241-E243: Rejected a temporary tile16 Metal cooperative GEMV
  candidate. Synthetic Qwen-shaped projection timings improved by 1.009-1.074x
  versus tile8, but full warmed p64/o64 decode did not: tile8 control was
  174.0 ms mean ITL, tile16 was 173.5 ms then 177.3 ms on repeat. The candidate
  was reverted before commit; future GEMV work needs more than simple
  column-width tuning.
- 2026-05-03 E244: Accepted shared server-state startup prewarm. The previous
  prewarm used a temporary `BlockManager`/`PagedKvCache`; it now runs the same
  64-token/2-token synthetic generation through the live shared block manager
  and paged KV cache via `generate_paged_shared_tokens`, which frees its
  reservation afterward. Fresh no-prefix bs=4 distinct batch improved from E239
  4.43 s wall / 4,416.317 ms handler to 2.48 s / 2,461.483 ms with the same 8
  generated tokens and 0 prefix-cache lookups.
- 2026-05-03 E245-E249: Fixed the default prefix-enabled short-miss path after
  shared prewarm. E245/E246 showed the default prefix-enabled bs=4 short-prompt
  shape at 5.22 s / 5,210.563 ms and 5.06 s / 5,052.511 ms while retaining
  27.5 MB of GDN state per 17-token prompt. E247 skipped retention below
  64 tokens but still took 5.99 s because the prefix generation path built a
  registration snapshot before rejecting it. The accepted E248 change routes a
  short prefix-cache miss directly to `generate_paged_shared_tokens()` and keeps
  `RealPrefixCache` from registering prompts below 64 tokens; E248 measured
  4.48 s / 4,474.376 ms with 4 prefix misses, 0 cached entries, and 0 retained
  state bytes. E249 no-prefix control was contaminated by severe memory
  pressure and is logged only as a caveat, not a regression verdict.
- 2026-05-03 E250-E252: Rejected a GDN input-projection unroll4 Metal kernel.
  The opt-in candidate passed focused Metal parity and the synthetic
  Qwen-shaped GDN in-proj microbench improved from 1198.901 us to 994.100 us
  (1.206x), but full warmed p64/o64 decode regressed badly: scalar control
  E251 was 414.9 ms prefill / 166.9 ms mean ITL / 5.99 tok/s, while unroll4
  E252 was 683.1 ms / 305.1 ms / 3.28 tok/s. The candidate was reverted before
  commit; future GDN in-proj work should reduce material work or improve weight
  layout rather than tweaking the scalar loop shape.
- 2026-05-03 E253: Rejected sharing one backing allocation across the GDN
  qkv-conv-norm Q/K/V outputs. The wrapper-only candidate passed focused Metal
  parity and release `kiln-bench` built, but warmed p64/o64 regressed to
  442.0 ms prefill / 285.3 ms mean ITL / 3.51 tok/s versus E251's
  414.9 ms / 166.9 ms / 5.99 tok/s control. The candidate was reverted before
  commit; more view-sharing around existing launches is not the right low-level
  direction for this path.
- 2026-05-03 E254: Rejected routing LM-head logits through the generic
  cooperative transposed GEMV kernel. The candidate passed focused Metal
  LM-head parity and release `kiln-bench` built, but warmed p64/o64 regressed
  to 617.7 ms prefill / 295.0 ms mean ITL / 3.39 tok/s versus E251's
  414.9 ms / 166.9 ms / 5.99 tok/s control. `memory_pressure` showed 74% free
  memory after the run, so this is treated as a real rejection; the scalar
  LM-head materialization remains the faster default.
- 2026-05-03 E255: Refreshed the synchronized target-selection profile on the
  clean branch head after rebuilding release `kiln-bench`. The profiled run is
  not a latency baseline, but the measured second section still ranks GDN
  `in_proj` first: decode linear/GDN layers summed to 196.322 ms, full-attn
  layers to 47.005 ms, and decode GDN stages were `in_proj` 56.742 ms,
  `out_proj` 21.512 ms, `gates_recur_gated_norm` 8.498 ms, and
  `qkv_conv_norm` 5.663 ms. Next low-level work should target real GDN
  input-projection work reduction, not more wrapper allocation swaps.
- 2026-05-03 E256-E257: Rejected an opt-in fused QKV projection +
  qkv-conv/norm decode path intended to remove materialized `mixed_qkv`. The
  candidate passed focused Metal parity and release `kiln-bench` built, but
  p64/o64 warmed timing lost to same-session default: E256 opt-in was
  477.2 ms prefill / 178.0 ms mean ITL / 5.62 tok/s, while E257 default was
  424.6 ms / 172.7 ms / 5.79 tok/s. The candidate was reverted before commit;
  future GDN `in_proj` work needs a projection algorithm or weight-layout
  change, not just fusing around the existing column-strided projection.
- 2026-05-03 E258-E261: Rejected an opt-in Metal residual-add + RMSNorm decode
  fusion. The candidate passed focused Metal parity under a loose BF16
  tolerance and release `kiln-bench` built, but two p64/o64 warmed A/B pairs
  were not robust: E258 opt-in was 167.2 ms mean ITL versus E259 default
  168.8 ms, while repeat E260 opt-in lost at 171.0 ms versus E261 default
  169.5 ms. The candidate was reverted before commit; future residual/norm
  work needs a larger fused boundary or tighter arithmetic, not this small
  post-attention fusion.
- 2026-05-03 E262: Rejected a combined GDN input-projection weight layout.
  The opt-in candidate packed `[hidden, qkv+z+a+b]` at load time and passed the
  focused Metal GDN in-proj parity test, but full warmed p64/o64 collapsed to
  9108.9 ms prefill / 203.1 ms mean ITL / 4.92 tok/s. `memory_pressure`
  reported ~58% free pages after the run, so this is a real rejection; future
  GDN projection work must avoid extra resident copies and improve the
  reduction/packing algorithm itself.
- 2026-05-03 E263-E264: Rejected an MLP down-projection + residual Metal
  fusion. The candidate wrote `residual + down_proj(hidden)` directly from a
  cooperative GEMV kernel and passed focused split-reference parity, but the
  full warmed p64/o64 same-binary A/B showed no decode improvement: fused was
  507.9 ms prefill / 197.3 ms mean ITL / 5.07 tok/s, while the disabled control
  was 470.7 ms / 197.5 ms / 5.06 tok/s. The candidate was reverted and the
  clean-source release `kiln-bench` rebuild passed; future low-level work
  should keep targeting GDN projection math/packing instead of small residual
  boundaries.
- 2026-05-04 E265-E266: Rejected a row-major GDN input-projection Metal
  kernel. It used the already-loaded `[out, hidden]` weights to avoid
  transposed column-strided reads and passed focused parity, but same-binary
  warmed p64/o64 lost to the current transposed fused kernel: row-major was
  463.7 ms prefill / 202.6 ms mean ITL / 4.94 tok/s, while control was
  463.8 ms / 188.5 ms / 5.30 tok/s. The candidate was reverted and the
  clean-source release `kiln-bench` rebuild passed.
- 2026-05-04 E267: Accepted a bounded short-prompt prefix lookup bypass. Since
  the production prefix cache registers only prompts with at least 64 tokens,
  shorter prompts cannot hit, so non-streaming and streaming real generation
  now skip `lookup()` and route directly to the shared paged path. Focused
  server tests/checks passed. A fresh default prefix-enabled release-server
  bs=4 short batch measured 5.60 s wall / 5,595.400 ms handler with
  0 prefix hits, 0 prefix misses, 0 cached entries, 0 retained state bytes,
  4 render misses, and 4 token misses.
  Because `memory_pressure` reported only 34% free memory, E267 is logged as a
  semantics-preserving no-work cleanup, not as a latency win. Next work returns
  to low-level/kernel and batching changes.
- 2026-05-04 E268-E271: Rejected a cooperative tile8 Metal GDN input-projection
  kernel. The candidate split the hidden reduction across SIMDGROUP lanes for
  the four GDN projections and passed focused parity, but same-binary warmed
  p64/o64 A/Bs lost twice: E268 was 455.3 ms prefill / 167.4 ms mean ITL /
  5.97 tok/s versus E269 scalar control 419.3 ms / 163.9 ms / 6.10 tok/s;
  repeat E270 was 449.0 ms / 166.8 ms / 6.00 tok/s versus E271 control
  425.5 ms / 164.5 ms / 6.08 tok/s. Memory pressure was 80-81% free around
  the repeat. The candidate source was reverted and clean-source release
  `kiln-bench` rebuild passed.
- 2026-05-04 E272-E273: Rejected a GDN `out_proj` + residual fusion. The
  candidate added a tile8 transposed cooperative GEMV epilogue that writes
  `residual + out_proj(gated_norm)` directly, passed focused split-reference
  parity, and release `kiln-bench` built. Same-binary warmed p64/o64 lost
  clearly: fused was 418.6 ms prefill / 167.3 ms mean ITL / 5.98 tok/s with
  239.2 ms P99, while disabled control was 417.4 ms / 159.9 ms / 6.25 tok/s
  with 178.9 ms P99. Memory pressure was 78% free after E273. The candidate
  source was reverted and clean-source release `kiln-bench` rebuild passed.
- 2026-05-04 E274-E277: Rejected a weighted Metal LM-head greedy decode path.
  The candidate used the fact that final RMSNorm's inverse-RMS scalar cannot
  change greedy argmax, then projected `(hidden * final_norm_weight)` directly
  in a new Metal BF16 LM-head kernel. Focused Metal argmax parity passed and
  release `kiln-bench` built, but two same-binary p64/o64 A/B pairs lost:
  E274 weighted was 452.7 ms prefill / 167.6 ms mean ITL / 5.97 tok/s versus
  E275 disabled control 416.7 ms / 165.2 ms / 6.05 tok/s; repeat E276 was
  416.0 ms / 167.8 ms / 5.96 tok/s versus E277 control 422.3 ms / 162.4 ms /
  6.16 tok/s. Memory pressure was 81% free after E277. The candidate source
  was reverted and clean-source release `kiln-bench` rebuild passed.
- 2026-05-04 E278-E281: Accepted a narrower weighted-hidden prep for Metal
  BF16 greedy LM-head decode. Instead of pushing the final norm weight multiply
  into every vocab-column dot product, the accepted path materializes
  `(hidden * final_norm_weight)` once and then calls the existing LM-head
  argmax path, skipping the final RMSNorm reduction only for Metal BF16
  `[1, 1, H]`. Focused CPU/Metal tests passed and same-binary p64/o64 won both
  A/B pairs: E278 was 410.5 ms prefill / 163.3 ms mean ITL / 6.13 tok/s versus
  E279 control 424.4 ms / 165.7 ms / 6.04 tok/s; repeat E280 was 422.5 ms /
  162.8 ms / 6.14 tok/s versus E281 control 419.7 ms / 163.2 ms / 6.13 tok/s.
  Memory pressure was 81% free after E281.
- 2026-05-04 E282-E283: Accepted prepared prompt reuse inside batch prompt
  groups. The real `/v1/completions/batch` path still fans out synthetic
  single-chat requests, but now each distinct prompt group renders/tokenizes
  once and shares the prepared prompt text/tokens across physical completions.
  Focused batch tests and `cargo check --locked -p kiln-server --features metal
  --bin kiln --bin kiln-bench` passed. A live same-prompt `n=4`,
  `max_tokens=2`, sampled release-server A/B measured E282 prepared at 7.07 s
  wall / 7,063.568 ms handler versus E283 same-binary disabled control at
  25.31 s / 25,303.646 ms. This is a server no-work cleanup for the current
  fan-out architecture, not true model-forward batching.
- 2026-05-04 E284-E285: Rejected a threadgroup-scalar variant of the fused
  Metal GDN decode `gates_recur_gated_norm` kernel. The candidate computed
  per-head `beta`/`decay` once in `tid == 0` and shared them through
  threadgroup memory instead of recomputing sigmoid/softplus/exp in every
  value lane. Focused Metal parity and release `kiln-bench` build passed, but
  the same-binary p64/o64 A/B lost clearly: E284 measured 444.7 ms prefill /
  168.3 ms mean ITL / 5.94 tok/s / 225.6 ms P99, while E285 scalar-gate
  control measured 420.5 ms / 162.9 ms / 6.14 tok/s / 183.5 ms P99. Memory
  pressure was 81% free after E285. The candidate source was reverted.
- 2026-05-04 E286: Refreshed current synchronized target selection from a
  clean-source release `kiln-bench` rebuild. With profiling sync enabled, the
  measured p64/o1 section reported 479.6 ms prefill and 209.6 ms mean ITL.
  Decode layer sums were 145.346 ms across 24 GDN layers and 35.369 ms across
  8 full-attention layers. Decode GDN stage ranking remains projection-led:
  `in_proj` 34.354 ms, `out_proj` 17.142 ms, `gates_recur_gated_norm` 8.100
  ms, `qkv_conv_norm` 6.253 ms. Prefill GDN also still ranks `in_proj` first
  at 83.101 ms. Memory pressure was 81% free. Treat this as target selection,
  not a latency baseline.
- 2026-05-04 E287-E288: Rejected a GDN decode project-z-in-recurrent boundary
  move. The temporary opt-in path skipped materializing `z` in decode
  `in_proj` and instead projected `z` inside the fused
  gates+recurrent+RMSNorm Metal kernel. Focused Metal parity tests and release
  `kiln-bench` build passed while applied, but same-binary warmed p64/o64 lost
  clearly: E287 measured 448.7 ms prefill / 171.3 ms mean ITL / 5.84 tok/s /
  189.6 ms P99, while E288 materialized-`z` control measured 418.1 ms /
  159.9 ms / 6.25 tok/s / 173.2 ms P99. Memory pressure was 81% free after
  E288. The candidate source was reverted.
- 2026-05-04 E289: Added an env-gated synchronized full-attention stage
  profiler (`KILN_PROFILE_FULL_ATTN_STAGES=1`) and used it for target
  selection. The intrusive p64/o1 profile measured 450.2 ms prefill and
  193.6 ms mean ITL, with 76% memory free after the run. Decode full-attention
  stage sums across 8 layers rank `qkv_proj` first at 10.641 ms, `o_proj`
  second at 5.360 ms, then `qkv_split` 2.454 ms and
  `decode_attn_contiguous` 2.257 ms. Full-attention decode is projection-led,
  not paged-attention-kernel-led; treat this as target selection, not a
  latency baseline.
- 2026-05-04 E290: Added an env-gated synchronized MLP stage profiler
  (`KILN_PROFILE_MLP_STAGES=1`) and used it for target selection across all
  32 layers. The intrusive p64/o1 profile measured 493.2 ms prefill and
  203.6 ms mean ITL, with 77% memory free after the run. Decode MLP stage sums
  rank `gate_up_fused` first at 63.710 ms and `down_proj` second at
  36.905 ms, for 100.615 ms total across 32 layers. Prefill MLP sums rank
  `down_proj` 71.021 ms, `gate_proj` 64.677 ms, and `up_proj` 63.921 ms.
  This redirects the next low-level pass toward MLP projection mechanics or
  true model-forward batching, not cache-only reuse and not another narrow
  down-projection residual epilogue.
- 2026-05-04 E291-E292: Rejected an opt-in MLP gate/up threadgroup-`x` cache.
  The candidate kept the current one-thread-per-output-column gate/up mapping,
  cached the `[H]` input vector in threadgroup memory per 256-column group, and
  passed focused Metal parity plus `cargo check`, but same-binary warmed
  p64/o64 lost badly: E291 measured 455.0 ms prefill / 202.8 ms mean ITL /
  4.93 tok/s / 221.6 ms P99, while E292 current gate/up control measured
  421.5 ms / 162.0 ms / 6.17 tok/s / 173.2 ms P99. Memory pressure was 80%
  free after E292. The candidate source was reverted and clean-source
  `kiln-bench` rebuild passed.
- 2026-05-04 E293-E294: Rejected an opt-in MLP gate/up `fast::exp` sigmoid.
  The candidate passed focused Metal parity plus `cargo check`, but same-binary
  warmed p64/o64 did not improve decode: E293 measured 408.9 ms prefill /
  162.8 ms mean ITL / 6.14 tok/s / 176.2 ms P99, while E294 current control
  measured 418.6 ms / 162.6 ms / 6.15 tok/s / 179.7 ms P99. Memory pressure
  was 81% free after E294. The candidate source was reverted and clean-source
  `kiln-bench` rebuild passed.
- 2026-05-04 E295: Refreshed the current release-server bs=4 distinct baseline
  after the recent accepted/rejected cache, prewarm, profiling, and low-level
  work. Four distinct prompts with `max_tokens=2` measured 7.22 s wall /
  7,202.326 ms handler for 8 generated tokens, with 4 rendered-prompt misses,
  4 token-cache misses, and no prefix-cache lookup work. Memory pressure was
  80% free after shutdown. Treat this as target selection for true
  model-forward/continuous batching and further low-level kernel work, not as a
  cache optimization.
- 2026-05-04 E296-E300: Rejected MLP gate/up threadgroup-width tuning. A
  temporary env knob tested widths 128 and 512 against the current 256 default
  in the same release binary. The first pass was slightly favorable
  (128/512 at 161.6/161.7 ms mean ITL versus 256 control at 163.7 ms), but the
  repeat tied (256 control 163.4 ms, 128 repeat 163.3 ms). Source was reverted;
  clean-source release `kiln-bench` rebuild and `git diff --check` passed.
  The pass also confirmed true batch speedups need a broader model-forward API
  change: current batch fan-out still routes physical outputs through
  single-sequence generation with one `BlockTable`/linear state.
- 2026-05-04 E301-E305: Accepted a low-level MLP gate/up decode kernel change.
  The fused Metal BF16 gate/up kernel now computes two adjacent output columns
  per thread, reducing repeated input-vector loads without the threadgroup
  barrier cost that sank E291. Same-source A/Bs held: E301/E303 two-column
  candidate measured 158.1/158.3 ms mean ITL versus E302/E304 old one-column
  controls at 161.2/162.4 ms. Final default E305 measured 418.0 ms prefill /
  157.8 ms mean ITL / 6.34 tok/s / 168.7 ms P99, with 76% memory free.
  Focused Metal parity, `cargo check`, release `kiln-bench` build, and
  `git diff --check` passed.
- 2026-05-04 E306: Refreshed synchronized MLP target selection after the
  accepted two-column gate/up kernel. The intrusive p64/o1 profile measured
  492.2 ms prefill and 202.5 ms mean ITL, with 78% memory free. Decode MLP
  sums still rank `gate_up_fused` first at 60.977 ms and `down_proj` second at
  37.154 ms across 32 layers. Prefill MLP ranks `down_proj` 71.457 ms,
  `gate_proj` 64.126 ms, and `up_proj` 64.100 ms. This supports one more
  structural gate/up variant before moving to MLP down-projection or a broader
  projection/materialization boundary.
- 2026-05-04 E307-E308: Rejected an opt-in MLP gate/up four-column decode
  kernel. The candidate computed four adjacent output columns per thread and
  used `bfloat4` adjacent weight loads, but same-binary warmed p64/o64 lost to
  the current two-column default: E307 measured 439.1 ms prefill / 161.5 ms
  mean ITL / 6.19 tok/s / 214.6 ms P99, while E308 control measured 419.1 ms /
  157.8 ms / 6.34 tok/s / 167.7 ms P99. Memory pressure was 80% free. The
  candidate source was reverted and clean-source release `kiln-bench` rebuild
  plus `git diff --check` passed.
- 2026-05-04 E309-E310: Rejected a Qwen3.5-shape-specialized Metal MLP
  `down_proj` decode kernel before full-model testing. The current synthetic
  transposed-GEMV refresh measured the down-projection shape at
  `broadcast_matmul=1186.880 us`, tile4 `966.188 us`, and tile8 `939.508 us`.
  The temporary exact-shape kernel removed generic dimension/tail branches for
  `[1,1,9216] x [9216,2560]`, matched tile8 bit-for-bit, but lost in the direct
  same-binary comparison: tile8 `974.933 us` versus specialized `1068.878 us`.
  Source was reverted; `rustfmt --check`, `git diff --check`, and release
  `kiln-bench` rebuild passed. This rules out a narrow down-projection
  specialization that only erases generic branch overhead.
- 2026-05-04 E311: Measured current warmed release-server batch fan-out against
  four sequential single chat requests. A 4-prompt distinct batch (`n=1`,
  `max_tokens=2`, greedy) took 5,330.886 ms handler / 5.353 s curl wall for
  8 physical generated tokens, with 4 render misses, 4 token misses, and no
  prefix-cache lookups. Four similar uncached single requests generated the
  same 8 physical tokens with handler timings 3,257.725 + 553.875 + 556.278 +
  608.478 ms = 4,976.357 ms. This confirms the current batch endpoint is not
  true model-forward batching; distinct prompt groups still serialize on the
  shared paged-cache/model-forward path and can be slightly slower than
  sequential singles.
- 2026-05-04 E312-E315: Rejected a sequential distinct-prompt batch scheduler.
  The temporary path serialized `/v1/completions/batch` prompt groups on the
  real backend instead of spawning one task per distinct prompt group. E313
  looked fast once at 2,275.571 ms handler / 2.292 s wall versus E312 fan-out
  control at 4,479.301 ms / 4.497 s, but two promoted-default repeats
  regressed badly: E314 10,215.447 ms / 10.231 s and E315 10,793.169 ms /
  10.80 s. All runs generated 8 physical tokens with 4 render misses,
  4 token misses, and no prefix-cache lookups. Source was reverted. This is
  not a viable substitute for true model-forward batching; next bs>1 work needs
  per-sequence cache tables and batched linear/GDN state, alongside continued
  low-level Metal kernel work.
- 2026-05-04 E316: Rejected an opt-in exact-shape Qwen3.5 MLP gate/up
  `bfloat2` load/store variant. The candidate kept the accepted two-column
  gate/up work shape but removed the second-column branch for
  `[1,1,2560] x [2560,9216]` and loaded adjacent weights with `bfloat2`.
  It matched current output exactly, but the same-binary synthetic bench lost:
  current two-column `1636.741 us` versus `bfloat2` `1683.015 us`
  (`0.973x`). Source was reverted after `rustfmt --check`, focused gate/up
  parity, and the ignored synthetic bench.
- 2026-05-04 E317: Rejected an exact-shape Qwen3.5 MLP down-proj x-cache
  kernel. The temporary kernel cached the 9216-element BF16 input vector in
  threadgroup memory once per 32-output-column group, then reused it across the
  four tile8 SIMD groups. It matched current tile8 exactly, but the same-binary
  synthetic bench lost: current tile8 `907.185 us` versus x-cache `964.268 us`
  (`0.941x`). Source was reverted after `rustfmt --check`, focused transposed
  GEMV parity, and the ignored synthetic bench.
- 2026-05-04 E318: Accepted decode-batch MLP gate/up fusion support. The
  existing two-column Metal kernel already handled multiple flattened rows, so
  the support gate now allows nonzero BF16 `[B,1,H] x [H,I]` while still
  rejecting prefill `[1,T,H]`. Added batch parity coverage and an ignored
  Qwen3.5 synthetic bench. Same-binary synthetic results were strong versus the
  broadcast fallback for future batched decode rows: batch1 `2568.350 us` ->
  `1800.531 us` (1.426x), batch2 `120632.192 us` -> `1884.446 us` (64.015x),
  batch4 `268723.294 us` -> `2146.144 us` (125.212x), batch8
  `590190.196 us` -> `7215.167 us` (81.799x), all with max abs diff
  <= `5.960464e-8`. `cargo check` for `kiln`/`kiln-bench` passed. This is a
  bs>1 building block, not full endpoint batching.
- 2026-05-04 E319: Accepted decode-batch transposed GEMV support. Added a
  separate tile8 Metal kernel for BF16 `[B,1,K] x [K,N]`, `B > 1`, and routed
  decode linear projections through it while leaving the existing bs=1 kernels
  untouched. Focused batch GEMV parity passed. Qwen3.5 down-proj-shaped
  synthetic results versus broadcast fallback were batch2 `66577.996 us` ->
  `1644.683 us` (40.481x), batch4 `146361.521 us` -> `3203.308 us`
  (45.691x), and batch8 `320277.131 us` -> `7039.210 us` (45.499x), with
  exact output match. `cargo check` for `kiln`/`kiln-bench` and release
  `kiln-bench` build passed. This pairs with E318 as a true-batching building
  block; endpoint/model-forward batching still remains.
- 2026-05-04 E320: Rejected a decode-batch fused QKV projection. The temporary
  batch variant fused Q/K/V into one Metal dispatch for BF16 `[B,1,2560]` using
  Qwen3.5 gated full-attention dims (`q_t=[2560,8192]`,
  `k_t/v_t=[2560,1024]`) and matched the current separate E319 batch GEMVs
  exactly, but lost in same-binary synthetic timing: batch2 `2194.850 us` vs
  `2448.417 us` (0.896x), batch4 `4309.402 us` vs `4703.504 us` (0.916x),
  and batch8 `8273.379 us` vs `9448.035 us` (0.876x). Source was reverted.
  This keeps the focus on true model-forward batching and lower-level kernels
  with demonstrated wins, not simple QKV launch fusion.
- 2026-05-04 E321: Accepted decode-batch GDN qkv-conv/norm support. The
  existing Metal kernel already had batch-indexed output and conv-state
  addressing, so the support gate now allows nonzero BF16 `[B,1,H]` decode
  batches and the launcher dispatches row-within-batch by batch index. Focused
  parity against per-row fused execution passed. Qwen3.5 GDN synthetic results
  versus split conv+QK-norm work were batch1 `658.700 us` -> `73.708 us`
  (8.937x), batch2 `798.927 us` -> `84.123 us` (9.497x), batch4 `829.923 us`
  -> `88.669 us` (9.360x), and batch8 `838.171 us` -> `96.650 us` (8.672x),
  with exact row q/k/v/state matches. This is a true-batching building block;
  remaining work is GDN gate/recurrent/state batching plus per-sequence cache
  tables, attention state, scheduler, and model-forward integration.
- 2026-05-04 E322: Accepted decode-batch GDN gates+recurrent+RMSNorm support.
  The existing Metal kernels already indexed q/k/v, gates, recurrent state,
  and output by `batch_idx`, so the support gate now allows nonzero
  `[B,1,H]` decode batches with checked launch bounds. Batch-4 focused parity
  passed. Qwen3.5 synthetic results versus split Metal gates + recurrent +
  gated RMSNorm were batch1 `504.527 us` -> `252.727 us` (1.996x), batch2
  `778.104 us` -> `172.983 us` (4.498x), batch4 `983.356 us` -> `249.662 us`
  (3.939x), and batch8 `1237.192 us` -> `504.165 us` (2.454x), with output
  max diff <= `9.765625e-4` and state max diff `2.441406e-4`. E321+E322 keep
  most of the GDN decode body fused for future true `[B,1,H]` rows; remaining
  blockers include batched GDN input projection and model-forward/scheduler
  state plumbing.
- 2026-05-04 E323: Accepted decode-batch GDN input-projection support. The
  scalar fused Metal in-proj kernel now indexes by batch, while `B > 1`
  returns separate contiguous qkv/z/a/b outputs so the following fused GDN
  kernels do not pay hidden copies. Batch-4 parity and contiguous-output
  checks passed. Qwen3.5 synthetic results versus four broadcast projection
  matmuls were batch1 `2517.021 us` -> `1251.346 us` (2.011x), batch2
  `84133.521 us` -> `2103.188 us` (40.003x), batch4 `185698.783 us` ->
  `3394.900 us` (54.699x), and batch8 `409072.888 us` -> `5578.967 us`
  (73.324x), with exact output match. The GDN decode-batch kernel chain now
  has accepted support from in-proj through recurrent RMSNorm; the remaining
  bs>1 work is model-forward state/cache/scheduler plumbing.
- 2026-05-04 E324: Rejected batched paged-decode SDPA gather. The temporary
  Metal backend path accepted `q=[B,1,16,256]` and per-row
  `block_table=[B,32]` by flattening all block rows, gathering K/V to
  `[B,511,4,256]`, and calling one batched Candle Metal SDPA. It was
  numerically correct, but slower than rowwise gather+SDPA at every tested
  batch: batch1 `2566.121 us` vs `2610.800 us` (0.983x), batch2
  `4611.233 us` vs `5409.046 us` (0.853x), batch4 `8686.367 us` vs
  `11646.946 us` (0.746x), and batch8 `17172.817 us` vs `25354.850 us`
  (0.677x). Source was reverted; keep attention batching focused on a
  purpose-built decode kernel or real model-forward/scheduler plumbing.
- 2026-05-04 E325: Accepted attention output-gate decode-batch support. The
  Metal sigmoid/mul kernel already linearized all elements by `gid`; its
  support gate now accepts nonzero BF16 `[B,1,H]` rows with matching gate shape
  and checked dispatch bounds. Batch-4 parity passed. Qwen3.5 full-attention
  gate synthetic wins versus unfused sigmoid+mul were batch1 `222.165 us` ->
  `48.819 us` (4.551x), batch2 `218.188 us` -> `147.700 us` (1.477x), batch4
  `244.819 us` -> `52.275 us` (4.683x), and batch8 `240.867 us` ->
  `52.350 us` (4.601x), with max abs diff `1.953125e-3`. This removes another
  full-attention `[B,1,H]` kernel blocker; endpoint batching still requires
  model-forward state/cache/scheduler plumbing.
- 2026-05-04 E326: Accepted batched LM-head full-logits route. E254 already
  rejected generic cooperative GEMV for the one-row LM-head, so bs=1 still uses
  the scalar Metal LM-head. For `B > 1`, `lm_head_forward` now routes BF16
  `[B,1,H] x [H,V]` through the E319 batch GEMV kernel instead of
  `broadcast_matmul`. Batch-4 forward parity passed. Exact Qwen3.5 shape
  synthetic results, `x=[B,1,2560]`, `weight_t=[2560,248320]`, were batch2
  `1793751.167 us` -> `258241.125 us` (6.946x), batch4 `3823733.791 us` ->
  `568938.375 us` (6.721x), and batch8 fallback failed to allocate while the
  fused path completed in `922537.958 us`. This removes a large full-logits
  broadcast cliff for future true batching.
- 2026-05-04 E327: Accepted a batched paged-KV token-major write primitive.
  Added a Metal BF16 writer for `k/v=[B,1,heads,head_dim]` plus `u32` slot ids,
  with shape checks, checked dispatch bounds, a kernel-side slot guard, parity
  coverage, and a Qwen3.5-style synthetic bench. Current production bs=1 cache
  writes stay on the existing one-row writer because batch1 lost
  `49.633 us` vs `53.369 us` (0.930x). For future true decode batches the raw
  write primitive won: batch2 `94.185 us` -> `70.394 us` (1.338x), batch4
  `173.939 us` -> `48.290 us` (3.602x), and batch8 `325.680 us` -> `47.168 us`
  (6.905x), with exact pool matches. This is low-level launch-count work, not
  cache-reuse tuning; endpoint speed still requires model-forward
  scheduler/cache plumbing to provide B per-sequence slots.
- 2026-05-04 E328: Accepted a batched contiguous paged-decode attention
  primitive. This follows E324's rejection with a purpose-built custom kernel
  instead of one batched Candle SDPA over gathered rows. The new Metal kernel
  handles BF16 `q=[B,16,1,256]`, token-major K/V pools `[total_slots,4,256]`,
  and `u32 start_slots=[B]`. Batch-4 parity against rowwise custom-kernel
  launches passed. Same-binary synthetic results with `seq_len=512` were
  batch1 `217.892 us` vs `222.863 us` (0.978x, so bs=1 stays on the current
  path), batch2 `565.992 us` -> `555.942 us` (1.018x), batch4 `806.212 us` ->
  `514.329 us` (1.568x), and batch8 `1259.971 us` -> `777.429 us` (1.621x),
  all with exact output matches. Endpoint use still needs model-forward
  scheduler/cache plumbing to provide B contiguous cache runs.
- 2026-05-04 E329: Rejected full-attention QK-norm+RoPE decode fusion for the
  production path. The temporary Metal BF16 kernel fused per-head Q/K RMSNorm
  with RoPE for `seq_len == 1`, matched the split path exactly, and won the
  synthetic stage benchmark: batch1 `170.188 us` -> `105.040 us` (1.620x),
  batch2 `259.421 us` -> `109.523 us` (2.369x), batch4 `179.427 us` ->
  `119.438 us` (1.502x), and batch8 `216.608 us` -> `143.946 us` (1.505x).
  Real same-binary endpoint A/Bs did not improve: p64/o16 fused measured
  `164.9 ms` mean ITL versus disabled control `163.8 ms`, and the more stable
  p64/o64 fused measured `163.9 ms` versus disabled control `163.5 ms`.
  Source was reverted; this PR should keep prioritizing low-level changes that
  move endpoint latency/throughput, not standalone microbench wins.
- 2026-05-04 E330: Rejected row-aware Metal RMSNorm threadgroup tuning. The
  temporary candidate kept Qwen3.5 single-token decode RMSNorm on the current
  1024-thread reduction, but used 256 threads for `hidden=2560` multi-row
  RMSNorm. Synthetic results were exact and showed `[1,64,2560]` improving
  from 1024-thread `72.323 us` to 256-thread `60.490 us` (1.196x), while
  `[1,1,2560]` stayed effectively on the current path. Clean p512/o1 endpoint
  A/B after discarding machine-busy runs was only `3060.8 ms` TTFT control
  versus `3050.0 ms` tuned, with no decode improvement (`219.1 ms` vs
  `220.7 ms` mean ITL over two tokens). Source was reverted because the
  endpoint signal is too small for another production branch.
- 2026-05-04 E331: Refreshed current synchronized target profile with paged
  layers, full-attention stages, and MLP stages enabled. This is an intrusive
  profile, not a latency baseline: p64/o1 measured `514.1 ms` prefill and
  `224.6 ms` mean ITL. Decode layer sums were linear/GDN `133.121 ms` across
  24 layers and full attention `62.251 ms` across 8 layers. Decode MLP still
  dominates: `gate_up_fused` `64.276 ms` plus `down_proj` `37.381 ms` across
  32 layers. Full-attention decode remains projection-led but smaller:
  `qkv_proj` `10.788 ms`, `o_proj` `5.515 ms`, then `qkv_split` `2.468 ms`.
  Takeaway: next production work should change the MLP projection/materialized
  hidden boundary or implement true model-forward/scheduler batching, not more
  standalone QK-norm/RoPE/RMSNorm-style microbench wins.
- 2026-05-04 E332: Rejected prefill MLP gate/up fusion through the decode
  kernel. The temporary opt-in allowed `[1,T,2560]` prefill rows through the
  existing serial fused gate/up kernel to test whether removing separate
  `gate_proj`, `up_proj`, SiLU, and multiply materialization would improve
  TTFT. It was exact but much slower than the current prefill matmul path:
  seq16 `4071.883 us` vs `22487.812 us` (0.181x), seq64 `4283.308 us` vs
  `127059.996 us` (0.034x), and seq128 `7977.654 us` vs `262751.358 us`
  (0.030x). Source was reverted; useful prefill MLP fusion would need a real
  matrix-kernel shape, not the decode GEMV-style kernel.
- 2026-05-04 E333: Accepted batched RoPE position tables as a true bs>1
  batching primitive. The Metal rotary Q/K kernel still supports the existing
  shared 2D `[seq_len, half_rotary]` tables, and now also supports per-row 3D
  `[batch, seq_len, half_rotary]` tables through one `table_batch_stride`
  argument. This removes a model-forward blocker for continuous decode batches
  whose rows have different absolute positions. The focused Metal test matched
  rowwise launches exactly. Qwen3.5 decode-shape synthetic results
  `q=[B,1,16,256]`, `k=[B,1,4,256]`, `cos/sin=[B,1,32]` were batch1
  `73.211 us` -> `72.430 us` (1.011x), batch2 `326.505 us` -> `71.146 us`
  (4.589x), batch4 `646.147 us` -> `72.520 us` (8.910x), and batch8
  `1284.864 us` -> `75.740 us` (16.964x), all exact. This is not yet an
  endpoint win; scheduler/model-forward batching still needs to feed B rows.
- 2026-05-04 E334: Rejected MLP down-projection residual epilogue fusion. The
  temporary Metal kernel fused the down-proj GEMV epilogue with the following
  BF16 residual add, matching current `GEMV + add` semantics exactly by
  rounding the GEMV accumulator to BF16 before adding the residual. The
  Qwen3.5 decode-shape synthetic bench lost: current `917.338 us` versus fused
  `925.611 us` (0.991x), with exact output. Source was reverted before
  endpoint testing.
- 2026-05-04 E335: Accepted `PagedKvCache::write_token_major_native_batch` as
  model-forward batching plumbing. E327 had the fast Metal batch KV-write
  kernel, but the cache owner still only exposed one-sequence writes. The new
  API accepts one `BlockTable` and absolute write position per row plus
  token-major K/V tensors `[batch,1,num_kv_heads,head_dim]`, routes Metal BF16
  shapes through the E327 batch primitive, falls back rowwise elsewhere, and
  preserves FP8's `false` fallback contract. CPU and Metal three-row roundtrip
  tests passed. This is not an endpoint win yet; it removes one model-forward
  plumbing gap for continuous batching.
- 2026-05-04 E336: Accepted backend exposure for batched contiguous
  paged-attention. Added
  `BackendRuntime::flash_attn_paged_decode_contiguous_batch`, implemented it
  for Metal through the E328 custom kernel, extended parity so the trait path
  matches rowwise direct-kernel output, and added a cache helper for computing
  one contiguous start slot per batch row. Fresh Qwen3.5 synthetic results for
  `q=[B,16,1,256]`, `pools=[4096,4,256]`, `seq_len=512` were batch1
  `118.494 us` -> `86.335 us` (1.372x), batch2 `214.759 us` -> `116.427 us`
  (1.845x), batch4 `387.776 us` -> `211.215 us` (1.836x), and batch8
  `752.128 us` -> `618.280 us` (1.216x), all exact. This remains batching
  plumbing rather than a current endpoint win because production model-forward
  is still single-row and the kernel currently requires common sequence length
  plus one contiguous KV run per row.
- 2026-05-04 E337: Rejected MLP gate/up shared-X threadgroup caching. The
  temporary Qwen-shape Metal kernel loaded the `x=[B,1,2560]` row once per
  threadgroup into threadgroup memory to avoid global reloads in every output
  thread. It was exact, but lost the primary bs=1 target (`1598.787 us`
  current vs `1648.015 us` shared-X, 0.970x), barely lost batch2
  (`0.989x`), cratered batch4 (`1901.101 us` vs `3217.815 us`, 0.591x), and
  only won batch8 (`1.242x`). Source was reverted; do not carry this
  Qwen-specific shared-X layout.
- 2026-05-04 E338: Accepted batch-aware `LinearAttentionState` construction.
  Added `LinearAttentionState::new_with_batch(config, batch, device)` while
  preserving `new` as the batch-1 wrapper. CPU and Metal tests verify batch-3
  recurrent state `[3,nv,dk,dv]`, conv state `[3,conv_dim,k-1]`, and the
  existing Metal BF16 recurrent/F32 conv dtype policy. This is true-batching
  plumbing, not a current endpoint win; it removes the hard-coded batch-1 GDN
  state allocation blocker.
- 2026-05-04 E339: Accepted `LinearAttentionState` row assembly/scatter APIs.
  Added `batch_size`, `from_batch_rows`, `split_batch_rows`, and
  `scatter_batch_rows` so one-row per-request GDN states can be concatenated
  into `[B,...]` state for batched decode and split back afterward. CPU tests
  verify exact nonzero recurrent/conv state roundtrips and error handling;
  Metal BF16 tests verify assemble/split dimensions and dtype policy. This
  removes the explicit per-sequence GDN state row assembly/scatter blocker,
  but still needs scheduler/model-forward integration for endpoint wins.
- 2026-05-04 E340: Accepted
  `gqa_attention_paged_decode_contiguous_batch` as low-level full-attention
  batching plumbing. The helper projects Q/K/V for `[B,1,H]`, applies Q/K
  norm + RoPE, writes one token-major K/V row per request, dispatches the
  batched contiguous paged-attention backend kernel, then applies the attention
  gate and `o_proj`. It is intentionally strict: common `start_pos`,
  non-FP8 cache, contiguous live KV windows, and backend acceptance are
  required. A Metal two-row parity test with prepopulated prefix K/V matched
  rowwise `gqa_attention_paged` exactly (`max_abs_diff=0`, `mean_abs_diff=0`
  for both rows). This is not an endpoint win yet; it gives the scheduler/model
  forward path a real full-attention batch decode primitive to call.
- 2026-05-04 E341: Accepted
  `transformer_block_paged_decode_contiguous_batch` as block-level
  model-forward batching plumbing. The helper wraps the E340 full-attention
  batch decode primitive with pre/post RMSNorm, residuals, and MLP for
  full-attention layers only, preserving the same strict common-position and
  contiguous-KV constraints. A Metal two-row parity test with prefix K/V
  matched rowwise `transformer_block_paged` exactly (`max_abs_diff=0`,
  `mean_abs_diff=0` for both rows). This moves the scheduler integration seam
  from attention-only to a full block operation.
- 2026-05-04 E342: Accepted
  `model_forward_paged_decode_contiguous_batch` as strict model-forward
  batching plumbing. The helper embeds one token per row, loops all layers,
  routes full-attention layers through the E341 block helper, routes GDN
  layers through batch-shaped `LinearAttentionState`, then runs final norm and
  LM head over `[B,1,H]`. A Metal two-row parity test with prefix K/V matched
  two rowwise `model_forward_paged` calls exactly (`max_abs_diff=0`,
  `mean_abs_diff=0` for both rows). This is not a live endpoint win yet; the
  scheduler still needs to admit compatible decode rows into the helper.
- 2026-05-04 E343: Accepted
  hybrid correctness coverage for `model_forward_paged_decode_contiguous_batch`.
  The Metal test uses three production-shaped GDN layers plus one
  production-shaped full-attention layer with `attn_output_gate=true`,
  nonzero assembled `[B,...]` GDN state, and prepopulated paged K/V prefix
  data. Two-row batched logits matched rowwise `model_forward_paged` exactly
  (`max_abs_diff=0`, `mean_abs_diff=0` for both rows), and every post-decode
  GDN recurrent/conv state row matched exactly for all three linear layers.
  This removes the main correctness blocker before scheduler admission.
- 2026-05-04 E344: Accepted
  `ModelRunner::decode_next_tokens_paged_contiguous_batch_greedy` as the first
  scheduler-facing decode batch admission primitive. It accepts one ready
  decode token/block table/position per row, assembles per-request GDN states
  into batch state, calls `model_forward_paged_decode_contiguous_batch` under
  the shared paged-cache lock, scatters state back, and samples all rows with
  `greedy_sample_rows`. A Metal `ModelRunner` test using Qwen full-attention
  decode geometry matched two-row batched greedy tokens against rowwise
  `model_forward_paged` logits plus `greedy_sample`. This is not live request
  batching yet; it gives the eventual scheduler one call to execute.
- 2026-05-04 E345: Accepted the first live greedy streaming decode batcher.
  `kiln-server` can create a shared Metal-only `DecodeBatcher` when
  `KILN_DECODE_BATCHER=1` is set and pass it into non-speculative streaming
  decode, including prefix-cache paths. Greedy Metal decode steps submit
  one-token jobs carrying their `BlockTable`, position, and one-row GDN state;
  the worker groups same-position jobs up to `KILN_DECODE_BATCH_MAX` and
  optionally waits `KILN_DECODE_BATCH_WAIT_US` for peers. A Metal test drove
  two same-position jobs through the live worker, observed
  `max_observed_batch == 2`, and matched rowwise greedy tokens. This moves true
  batching into the serving path as an opt-in path for measurement.
- 2026-05-04 E346: Added live decode-batcher Prometheus metrics and rejected
  default enablement. Four concurrent streaming chat requests showed the
  batcher really coalesces rows (`KILN_DECODE_BATCHER=1`, zero wait: `28`
  submitted jobs, `16` worker batches, `28` rows, max batch `2`), but endpoint
  time regressed from disabled `6.245348s` for 32 generated tokens to
  zero-wait enabled `7.735135s` for the same 32 generated tokens. A `500us`
  wait observed max batch `3` but slowed to `10.076978s` and one request ended
  early. The batcher is now opt-in (`KILN_DECODE_BATCHER=1`) until a batched
  greedy-tail/argmax path or better admission policy can beat rowwise
  `model_forward_paged_next_token_greedy`.
- 2026-05-04 E347: Accepted batched Metal LM-head argmax and enabled zero-wait
  live greedy batching by default on Metal. The new BF16 batch-row argmax avoids
  materialized `[B,1,V]` logits in batched greedy decode and won the Qwen-shaped
  synthetic LM-head bench by `4.382x` at batch 2, `4.305x` at batch 4, and
  `2.378x` at batch 8. Live four-request streaming improved from disabled
  `7.127789s` for 32 generated tokens to zero-wait enabled `5.230091s`
  (`28` jobs, `14` batches, `28` rows, max batch `3`); `500us` wait got max
  batch `4` but slowed to `5.565906s`. Single streaming also favored enabled
  zero-wait (`2.269265s` -> `1.619216s`, max batch `1`). `KILN_DECODE_BATCHER=0`
  and `KILN_DISABLE_METAL_LM_HEAD_ARGMAX_ROWS=1` remain kill switches.
- 2026-05-04 E348: Added synchronized live batch decode stage profiling and
  used it to choose the next low-level target. The batched contiguous decode
  helper now honors `KILN_PROFILE_FULL_ATTN_STAGES`,
  `KILN_PROFILE_GDN_STAGES`, and `KILN_PROFILE_MLP_STAGES`; full-attention
  batch work gets `_batch` stage names. A profiled four-request streaming
  probe (`max_tokens=3`) returned all `200`s, generated `12` tokens, and sent
  `8` jobs through `4` worker batches with max batch `3`. Excluding the
  background prewarm, live decode stage totals were MLP `657.623 ms`, GDN
  `461.869 ms`, and full attention `224.225 ms`; top buckets were
  `mlp:gate_up_fused` `375.207 ms`, `mlp:down_proj` `282.416 ms`,
  `gdn:in_proj` `251.382 ms`, and `gdn:out_proj` `111.212 ms`. Next work
  should target projection-heavy MLP/GDN decode kernels, not cache plumbing.
- 2026-05-04 E349: Rejected and reverted a simdgroup-cooperative rewrite of
  the Metal MLP fused gate/up kernel. It passed single-row and decode-batch
  parity, but the Qwen-shaped synthetic bench regressed from baseline
  `1968.889/2265.722/2336.778/7300.972 us` at batch `1/2/4/8` to
  `2455.736/5197.055/10557.667/22828.167 us`. Do not return to this
  eight-column cooperative gate/up shape.
- 2026-05-04 E350: Measured the existing MLP down-projection batch GEMV before
  changing it. The Qwen-shaped `[B,1,9216] x [9216,2560]` synthetic bench is
  already strong: fused `1742.167 us` at batch 2, `3337.611 us` at batch 4,
  and `7118.903 us` at batch 8, about `40-47x` faster than broadcast matmul.
  No source change; avoid speculative tile-size churn here until a better
  target emerges.
- 2026-05-04 E351: Rejected and reverted a simdgroup-cooperative rewrite of
  the Metal GDN in-projection kernel. Odd-dimension parity passed, but the
  Qwen-shaped synthetic bench regressed from baseline
  `1250.139/2411.611/3767.250/5612.819 us` at batch `1/2/4/8` to
  `1358.014/3554.653/5294.875/10503.195 us`. The simple per-column fused
  kernel remains faster; avoid this cooperative in-proj shape.
- 2026-05-04 E352: Accepted `200us` as the default Metal live decode-batcher
  admission wait. E348 showed zero-wait can let one request run ahead after
  prefill; this same-binary sweep filled the gap between zero and the earlier
  losing `500us` wait. Four concurrent streaming requests with `max_tokens=8`
  generated `32` tokens each run: wait `0us` took `5.283006s` with `14`
  worker batches and max batch `3`; `50us` took `5.144448s`; `100us` took
  `5.059628s`; `200us` won at `4.951647s` with `8` batches and max batch `4`;
  `300us` rose to `5.060804s`. `KILN_DECODE_BATCH_WAIT_US=0` remains the
  zero-wait override and `KILN_DECODE_BATCHER=0` remains the kill switch.
- 2026-05-04 E353: Checked bs=1 streaming latency after E352. Same prompt,
  prewarmed release server, greedy `max_tokens=32`, and `32` generated tokens:
  default `200us` wait took `5.504201s`; `KILN_DECODE_BATCH_WAIT_US=0` took
  `5.529497s`. Both paths submitted `31` one-row batcher jobs, executed `31`
  batches, and observed max batch `1`. No measurable single-user latency
  regression from the `200us` default in this probe.
