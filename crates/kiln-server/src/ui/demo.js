/* =====================================================================
   Demo fixture mode (?demo=1)
   ---------------------------------------------------------------------
   Used to capture realistic README screenshots without a live GPU pod.
   Returns canned Qwen3.5-4B data shaped like the production endpoints.
   No effect when the query string is absent.
   ===================================================================== */
(function () {
  const params = new URLSearchParams(window.location.search);
  if (params.get('demo') !== '1') return;

  const start = Date.now() - 4527 * 1000;
  // The demo paints Kiln as what it actually is: a local inference server that
  // coding agents (pi, opencode) point at, with a flywheel of trained adapters.
  const demoState = {
    active: 'pi-coder-v4',
  };

  function uptimeSeconds() { return Math.floor((Date.now() - start) / 1000); }

  // Shared fixture helpers so every page (not just Overview) has agent-centric
  // data in the demo. ISO timestamps are anchored to load time so "x ago" reads
  // naturally; MODEL/ACTIVE mirror the live server's identifiers.
  const NOW = Date.now();
  const MB = 1024 * 1024;
  const GIB = 1024 ** 3;
  const MODEL = 'Qwen3.5-4B';
  const ACTIVE_ADAPTER = demoState.active;
  const ISO = (ago) => new Date(NOW - ago).toISOString();

  const evalDatasets = () => ({
    datasets: [
      { name: 'pi-traces', format: 'sft_chat', description: 'Curated pi agent trajectories with tool calls', num_rows: 12_480, size_bytes: 184 * MB, created_at: ISO(20 * 86400_000), updated_at: ISO(2 * 86400_000), stats: { num_assistant_turns: 38_210, num_tool_messages: 21_004, num_with_tool_calls: 18_902, max_messages_per_conv: 42, max_content_chars: 18_400, avg_messages_per_conv: 11.4, sample_role_patterns: ['system→user→assistant→tool→assistant'] } },
      { name: 'rust-review-pairs', format: 'sft_chat', description: 'PR review comments paired with diffs', num_rows: 5_120, size_bytes: 72 * MB, created_at: ISO(14 * 86400_000), updated_at: ISO(5 * 86400_000), stats: { num_assistant_turns: 5_120, num_tool_messages: 0, num_with_tool_calls: 88, max_messages_per_conv: 4, max_content_chars: 9_200, avg_messages_per_conv: 3.0, sample_role_patterns: ['system→user→assistant'] } },
      { name: 'sql-explain-gold', format: 'sft_chat', description: 'SQL queries with natural-language explanations', num_rows: 3_004, size_bytes: 28 * MB, created_at: ISO(9 * 86400_000), updated_at: ISO(9 * 86400_000), stats: { num_assistant_turns: 3_004, num_tool_messages: 0, num_with_tool_calls: 0, max_messages_per_conv: 2, max_content_chars: 4_100, avg_messages_per_conv: 2.0, sample_role_patterns: ['user→assistant'] } },
    ],
  });
  const evalSuites = () => ({
    suites: [
      { name: 'pi-toolcall-eval', description: 'Does the agent emit the correct tool call for the task?', num_examples: 240, default_scorer_kind: 'tool_call' },
      { name: 'rust-review-quality', description: 'LLM-judge scored review usefulness vs gold comments', num_examples: 128, default_scorer_kind: 'judge' },
      { name: 'sql-exact-match', description: 'Normalized SQL equivalence against the gold query', num_examples: 300, default_scorer_kind: 'exact_match' },
      { name: 'commit-msg-contains', description: 'Conventional-commit prefix + scope presence', num_examples: 96, default_scorer_kind: 'contains' },
    ],
  });
  const evalSuiteDetail = (name) => ({
    name, description: 'Eval suite ' + name,
    default_scorer: { kind: 'exact_match', case_sensitive: false, strip_whitespace: true },
    generation: { temperature: 0.0, top_p: 1.0, top_k: 0, max_tokens: 256, n: 1, stop: [], seed: null },
    examples: [
      { id: 'ex1', messages: [{ role: 'user', content: 'List files importing legacy_client' }], target: 'grep_repo legacy_client', tags: ['tool_call'] },
      { id: 'ex2', messages: [{ role: 'user', content: 'Write the commit message' }], target: 'fix(cache): ...', tags: ['commit'] },
    ],
    schema_version: 1,
  });
  // NOTE: the real server returns jobs sorted NEWEST-first (descending
  // submitted_at_iso) — keep this array in that order so demo renders match.
  const evalJobs = () => ({
    jobs: [
      { job_id: 'eval_44e5f607-0000-1111-2222-333344445555', suite_name: 'commit-msg-contains', adapters: ['commit-msg-writer'], submission_kind: 'on_demand', state: 'queued', progress: { examples_completed: 0, examples_total: 96, running_accuracy: 0, running_mean_score: 0 }, finished_runs: [], headline_accuracy: null, error: null, submitted_at_iso: ISO(30_000) },
      { job_id: 'eval_77a1c0de-aa11-bb22-cc33-dd44ee55ff66', suite_name: 'pi-toolcall-eval', adapters: [ACTIVE_ADAPTER], submission_kind: 'on_demand', state: 'running', progress: { examples_completed: 162, examples_total: 240, running_accuracy: 0.913, running_mean_score: 0.91 }, finished_runs: [], headline_accuracy: null, error: null, submitted_at_iso: ISO(180_000) },
      { job_id: 'eval_31b4f0a9-1234-5678-9abc-def012345678', suite_name: 'rust-review-quality', adapters: [null, ACTIVE_ADAPTER], submission_kind: 'compare', state: 'completed', progress: { examples_completed: 128, examples_total: 128, running_accuracy: 0, running_mean_score: 0 }, headline_accuracy: 0.84, error: null, submitted_at_iso: ISO(3 * 3600_000), finished_runs: [
        { suite_name: 'rust-review-quality', adapter: null, metrics: { num_examples: 128, num_pass: 79, accuracy: 0.617, mean_score: 0.62, pass_rate_by_tag: { correctness: 0.71, style: 0.55, security: 0.58 } }, outcomes: [] },
        { suite_name: 'rust-review-quality', adapter: ACTIVE_ADAPTER, metrics: { num_examples: 128, num_pass: 108, accuracy: 0.844, mean_score: 0.85, pass_rate_by_tag: { correctness: 0.92, style: 0.80, security: 0.81 } }, outcomes: [] },
      ] },
      { job_id: 'eval_9c2d1e8f-aaaa-bbbb-cccc-dddddddddddd', suite_name: 'sql-exact-match', adapters: [ACTIVE_ADAPTER], submission_kind: 'on_demand', state: 'completed', progress: { examples_completed: 300, examples_total: 300, running_accuracy: 0, running_mean_score: 0 }, headline_accuracy: 0.91, error: null, submitted_at_iso: ISO(8 * 3600_000), finished_runs: [
        { suite_name: 'sql-exact-match', adapter: ACTIVE_ADAPTER, metrics: { num_examples: 300, num_pass: 273, accuracy: 0.91, mean_score: 0.91, pass_rate_by_tag: { select: 0.96, join: 0.88, window: 0.79 } }, outcomes: [] },
      ] },
      { job_id: 'eval_55f6a708-9999-8888-7777-666655554444', suite_name: 'pi-toolcall-eval', adapters: ['tool-call-tuned-v1'], submission_kind: 'on_demand', state: 'failed', progress: { examples_completed: 41, examples_total: 240, running_accuracy: 0, running_mean_score: 0 }, finished_runs: [], headline_accuracy: null, error: 'adapter tool-call-tuned-v1 failed to load: rank mismatch (expected 32, got 16)', submitted_at_iso: ISO(2 * 86400_000) },
    ],
  });
  const evalJobDetail = (id) => ({
    job_id: id, state: 'completed',
    runs: [ { suite_name: 'sql-exact-match', adapter: ACTIVE_ADAPTER, metrics: { num_examples: 300, num_pass: 273, accuracy: 0.91, mean_score: 0.91, pass_rate_by_tag: { select: 0.96, join: 0.88, window: 0.79 } }, outcomes: [
      { example_id: 'ex1', kind: 'pass', score: 1.0, output: 'SELECT * FROM users WHERE active = true', expected: 'SELECT * FROM users WHERE active = true', tags: ['select'] },
      { example_id: 'ex2', kind: 'fail', score: 0.0, output: 'SELECT id FROM t', expected: 'SELECT id, name FROM t', tags: ['select'] },
    ] } ],
    progress: null,
  });
  const judgments = () => ({
    judgments: [
      { name: 'review-tone-ab', description: 'Which review comment is more actionable?', num_rows: 142, winner_histogram: { a: 64, b: 58, tie: 20 } },
      { name: 'sql-explain-clarity', description: 'Clearer SQL explanation A vs B', num_rows: 88, winner_histogram: { a: 41, b: 39, tie: 8 } },
      { name: 'commit-msg-quality', description: 'Better conventional-commit message', num_rows: 36, winner_histogram: { a: 19, b: 14, tie: 3 } },
    ],
  });
  const teachers = () => ({
    teachers: [
      { spec: { alias: 'served-base', kind: 'local', model_id: 'Qwen/Qwen3.5-4B' }, capabilities: { teacher_id: 'served-base', vocab_size: 248_320, max_top_k: 32 }, status: 'configured', usable: true },
      { spec: { alias: 'qwen-vllm', kind: 'remote', provider: 'vllm', model_id: 'qwen35-27b-teacher', url: 'http://127.0.0.1:8000', identity: { max_model_len: 32_768, max_prompt_logprob_candidates: 1_000_000 } }, capabilities: { teacher_id: 'qwen-vllm', vocab_size: 248_320, max_top_k: 20 }, status: 'verified', usable: true, identity_revision: 'sha256:8ef148923cb8b57955555ff4b57f875af6462f41818584bc573c869f40f1a0a7' },
      { spec: { alias: 'deterministic-fixture', kind: 'fixture', model_id: 'fixture-v1' }, capabilities: { teacher_id: 'deterministic-fixture', vocab_size: 248_320, max_top_k: 32 }, status: 'configured', usable: true },
    ],
  });
  const recipes = () => ({
    recipes: [
      { name: 'cold-start-distill', description: 'OPD warm-up then GRPO refinement for a new domain', num_steps: 3 },
      { name: 'review-quality-loop', description: 'SFT on review pairs, judge LoRA, then GRPO against the judge', num_steps: 4 },
      { name: 'tool-call-bootstrap', description: 'Synthesize tool-call eval, OPD distill, gate on eval recovery', num_steps: 3 },
    ],
  });
  const library = () => ({
    adapters: [
      { id: 'kiln/pi-coder-v4', name: 'pi-coder-v4', source_kind: 'local', description: 'Tool-calling coding agent, GRPO-tuned', uploader: 'floguy', size_bytes: 86 * MB, post_eval: { 'pi-toolcall-eval': 0.93 } },
      { id: 'kiln/rust-reviewer-v2', name: 'rust-reviewer-v2', source_kind: 'local', description: 'Rust PR review assistant', uploader: 'floguy', size_bytes: 64 * MB, post_eval: { 'rust-review-quality': 0.84 } },
      { id: 'hub/math-tutor-distilled', name: 'math-tutor-distilled', source_kind: 'huggingface', description: 'Distilled from qwen3.6-27b for grade-school math', uploader: 'community', size_bytes: 41 * MB, post_eval: {} },
    ],
    note: 'Showing 3 local + 1 remote entry. Configure a registry URL to see more.',
  });
  const cacheStats = () => ({
    root: '/home/floguy/.kiln/logit-cache',
    stats: { total_entries: 482_104, total_bytes: 14 * 1024 * MB, per_teacher: { 'qwen3.6-27b@local': 391_002, 'gpt-4.1@openai': 84_220, 'claude-opus-4@anthropic': 6_882 } },
  });
  const preflightCompat = () => ({
    matches: [
      { teacher: 'qwen3.6-27b-local', student: MODEL, domain: 'coding-agent', predicted_initial_overlap: 0.72, recommended_rank: 32, expected_gpu_hours: 4.5, expected_cost_usd: 6.30, validation_eval: 'pi-toolcall-eval', expected_eval_delta_points: 11.2, cold_start_epochs: 1 },
      { teacher: 'qwen3.6-27b-local', student: MODEL, domain: 'sql', predicted_initial_overlap: 0.81, recommended_rank: 16, expected_gpu_hours: 2.1, expected_cost_usd: 2.94, validation_eval: 'sql-exact-match', expected_eval_delta_points: 6.8, cold_start_epochs: 1 },
      { teacher: 'gpt-frontier', student: MODEL, domain: 'review', predicted_initial_overlap: 0.41, recommended_rank: 64, expected_gpu_hours: 9.0, expected_cost_usd: 38.0, validation_eval: 'rust-review-quality', expected_eval_delta_points: 14.5, cold_start_epochs: 2 },
    ],
    note: '3 compatibility rows from the teacher registry.',
  });
  const preflightTiers = () => ({
    tiers: [
      { tier: 'fast', default_logit_source: 'remote', default_loss: 'teacher_top_k', default_top_k: 16, lora_rank: 16, batch_size: 8, samples_per_prompt_default: 2, samples_per_prompt_data_multiplier: 1, max_rollout_tokens: 512, auto_checkpoint_cadence_steps: 100, cost_cap_default_usd: 10, cold_start_overlap_threshold: 0.5, mixture_distillation_golden_fraction: 0.1, eval_gate_required: false, notifications_channels: [] },
      { tier: 'balanced', default_logit_source: 'local', default_loss: 'teacher_top_k', default_top_k: 32, lora_rank: 32, batch_size: 8, samples_per_prompt_default: 4, samples_per_prompt_data_multiplier: 2, max_rollout_tokens: 1024, auto_checkpoint_cadence_steps: 250, cost_cap_default_usd: 50, cold_start_overlap_threshold: 0.6, mixture_distillation_golden_fraction: 0.15, eval_gate_required: true, notifications_channels: ['slack'] },
      { tier: 'quality', default_logit_source: 'local', default_loss: 'teacher_top_k', default_top_k: 32, lora_rank: 64, batch_size: 4, samples_per_prompt_default: 8, samples_per_prompt_data_multiplier: 3, max_rollout_tokens: 2048, auto_checkpoint_cadence_steps: 500, cost_cap_default_usd: null, cold_start_overlap_threshold: 0.7, mixture_distillation_golden_fraction: 0.2, eval_gate_required: true, notifications_channels: ['slack', 'email'] },
    ],
  });
  const agentTraces = () => ({
    traces: [
      { id: 'pi-2026-06-08-a1b2c3', working_dir: '/home/floguy/dev/kiln', num_turns: 24, num_tool_calls: 41, forked: false, parent_id: null, first_event_at: ISO(3 * 3600_000), last_event_at: ISO(2 * 3600_000), outcome: { ended_with_exit_0: true, user_edited_agent_files: ['src/cache.rs'], has_followup_attempt: false } },
      { id: 'pi-2026-06-07-d4e5f6', working_dir: '/home/floguy/dev/webapp', num_turns: 12, num_tool_calls: 18, forked: true, parent_id: 'pi-2026-06-07-aaaa', first_event_at: ISO(28 * 3600_000), last_event_at: ISO(27 * 3600_000), outcome: { ended_with_exit_0: false, user_edited_agent_files: [], has_followup_attempt: true } },
      { id: 'pi-2026-06-06-998877', working_dir: '/home/floguy/dev/kiln', num_turns: 31, num_tool_calls: 52, forked: false, parent_id: null, first_event_at: ISO(52 * 3600_000), last_event_at: ISO(51 * 3600_000), outcome: { ended_with_exit_0: true, user_edited_agent_files: ['src/api/eval.rs', 'src/ui.html'], has_followup_attempt: false } },
    ],
  });
  const agentDiscover = () => ({ indexed: 3, path: '/home/floguy/.pi/agent/sessions/' });
  const trainJobDetail = (id) => ({
    job_id: id, state: 'Running', job_type: 'GRPO', adapter_name: 'rust-reviewer-v3', progress: 0.58, current_loss: 0.286, elapsed_secs: 1_842, submitted_unix_ms: NOW - 1_842_000,
    loss_history: Array.from({ length: 40 }, (_, i) => ({ step: i + 1, loss: 0.9 * Math.exp(-i / 14) + 0.22 + (Math.sin(i) * 0.01) })),
    config: { lora_rank: 32, batch_size: 8, samples_per_prompt: 4 },
  });
  const adapterDetail = (name) => ({
    name, active: name === ACTIVE_ADAPTER, size_bytes: 86 * MB,
    base_model: 'Qwen/Qwen3.5-4B', lora_rank: 32, lora_alpha: 64, target_modules: ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    created_at: ISO(6 * 3600_000),
    files: [ { name: 'adapter_config.json', size_bytes: 612 }, { name: 'adapter_model.safetensors', size_bytes: 85 * MB } ],
    post_eval: { 'pi-toolcall-eval': 0.93 },
  });

  const responses = {
    '/v1/config': () => ({
      serving_profile: { profile: 'stable', source: 'default' },
      vram: {
        probe_selector: 'nvidia:0',
        unified: false,
        physical_capacity_bytes: 51 * GIB,
        physical_capacity_gib: 51.0,
        physical_capacity_source: 'nvidia-smi',
        configured_capacity_bytes: null,
        configured_capacity_gib: null,
        effective_capacity_bytes: 51 * GIB,
        effective_capacity_gib: 51.0,
        effective_capacity_source: 'nvidia-smi',
        configured_capacity_clamped: false,
        live: {
          total_bytes: 51 * GIB,
          total_gib: 51.0,
          used_bytes: 20.25 * GIB,
          used_gib: 20.25,
          available_bytes: 30.75 * GIB,
          available_gib: 30.75,
          effective_capacity_available_bytes: 30.75 * GIB,
          effective_capacity_available_gib: 30.75,
          usable_after_governor_floor_bytes: 29.75 * GIB,
          usable_after_governor_floor_gib: 29.75,
          soft_reserved_bytes: 0,
          soft_reserved_gib: 0,
          pressure: 'Comfortable',
          source: 'nvidia-smi',
          raw_observations: {
            probe_failed: false,
            driver_total_bytes: 51 * GIB,
            driver_used_bytes: 20.25 * GIB,
            driver_free_bytes: 30.75 * GIB,
            driver_vram_total_bytes: null,
            driver_vram_used_bytes: null,
            driver_gtt_total_bytes: null,
            driver_gtt_used_bytes: null,
            host_total_bytes: null,
            host_available_bytes: null,
            cgroup_limit_bytes: null,
            cgroup_current_bytes: null,
            cgroup_remaining_bytes: null,
            unified_reserve_bytes: null,
          },
        },
        governor: {
          floor_bytes: 1 * GIB,
          floor_gib: 1.0,
          capacity_limit_bytes: 51 * GIB,
          capacity_limit_gib: 51.0,
          probe_ms: 500,
          reclaim_mode_requested: 'off',
          reclaim_mode_effective: 'off',
          reclaim_mode_source: 'default',
          reclaim_disabled_by_serving_profile: true,
        },
      },
      kv_cache: { num_blocks: 528, num_blocks_source: 'auto', fp8_enabled: true },
      training: {
        runtime_device: 'cuda:0',
        model_weight_device: 'cuda:0',
        native_training_supported: true,
        checkpoint_policy: { mode: 'auto' },
        checkpoint_segments: 4,
        checkpoint_segments_source: 'auto',
        checkpointing_enabled: true,
      },
      memory_budget: {
        total_vram_bytes: 51 * GIB,
        total_vram_gib: 51.0,
        model_bytes: 8.25 * GIB,
        model_gib: 8.25,
        kv_cache_bytes: 12 * GIB,
        kv_cache_gib: 12.0,
        training_budget_bytes: 4.125 * GIB,
        training_budget_gib: 4.125,
        inference_memory_fraction: 0.55,
      },
      generation: {
        default_thinking_enabled: true,
        default_thinking_budget_tokens: 64,
        default_thinking_budget_ms: 1500,
        fold_reasoning_into_content: false,
      },
    }),
    '/health': () => ({
      status: 'ok',
      model: 'Qwen/Qwen3.5-4B',
      backend: 'cuda',
      uptime_seconds: uptimeSeconds(),
      active_adapter: demoState.active,
      gpu_memory: {
        total_vram_gb: 47.5,
        model_gb: 8.2,
        kv_cache_gb: 12.0,
        training_budget_gb: 4.1,
      },
      scheduler: { waiting: 0, running: 1, blocks_used: 142, blocks_free: 386 },
      checks: [
        { pass: true, name: 'cuda_runtime' },
        { pass: true, name: 'model_loaded' },
        { pass: true, name: 'kv_cache' },
        { pass: true, name: 'lora_engine' },
      ],
    }),
    // Jittered so the live tok/s sparkline reads as a trend, not a flat line.
    '/v1/stats/decode': () => {
      const t = Date.now() / 1000;
      const tps = 138 + Math.sin(t / 7) * 11 + Math.sin(t / 2.3) * 4;
      const p50 = 6.6 + Math.sin(t / 5) * 0.7;
      return {
        window_secs: 60,
        sample_count: 9,
        tok_per_sec: Math.round(tps * 10) / 10,
        p50_itl_ms: Math.round(p50 * 10) / 10,
        p99_itl_ms: Math.round((p50 + 4.2) * 10) / 10,
        mean_itl_ms: Math.round((p50 + 0.6) * 10) / 10,
      };
    },
    '/v1/stats/recent-requests': () => {
      const now = Date.now();
      return [
        {
          id: 'chatcmpl-9f2a7c41-0e3b-4a18-b9d2-1c8e7a4f02d1',
          model: 'Qwen3.5-4B', adapter: 'pi-coder-v4', streamed: true,
          timestamp_unix_ms: now - 2_400, duration_ms: 1_812, ttft_ms: 38,
          prompt_tokens: 4_182, completion_tokens: 268, finish_reason: 'stop',
          temperature: 0.2, top_p: 0.95, max_tokens: 2048,
          user_agent: 'pi/1.2.0',
          prompt_preview: 'Refactor src/auth/session.rs to use the new TokenStore trait and update call sites',
          completion_preview: "I'll update `Session::new` to take a `TokenStore` and migrate the three call sites in `handlers/`. Running cargo check…",
          prompt_full: 'You are pi, a terminal coding agent.\n\nRefactor src/auth/session.rs to use the new TokenStore trait and update call sites.',
          completion_full: "I'll update `Session::new` to take a `TokenStore` and migrate the three call sites in `handlers/`. Running cargo check now to confirm the trait bounds line up.",
        },
        {
          id: 'chatcmpl-7b1d5e92-aa14-4c0f-9e71-2f6b3d9c8a55',
          model: 'Qwen3.5-4B', adapter: 'pi-coder-v4', streamed: true,
          timestamp_unix_ms: now - 9_100, duration_ms: 642, ttft_ms: 29,
          prompt_tokens: 2_011, completion_tokens: 96, finish_reason: 'tool_calls',
          temperature: 0.0, top_p: 1.0, max_tokens: 1024,
          user_agent: 'pi/1.2.0',
          prompt_preview: 'What files import the deprecated `legacy_client` module?',
          completion_preview: 'grep_repo({"pattern":"legacy_client","glob":"**/*.rs"})',
          prompt_full: 'What files import the deprecated `legacy_client` module?',
          completion_full: 'grep_repo({"pattern":"legacy_client","glob":"**/*.rs"})',
        },
        // Dashboard-originated rows (`client: 'dashboard'` from the
        // X-Kiln-Client header): labeled honestly in the list, but they never
        // count toward "Agent connected", the client tally, or the Connect
        // panel auto-collapse — only external traffic does.
        {
          id: 'chatcmpl-d05b3a92-4c17-4e8b-a6f0-8b2c9d1e7f43',
          model: 'Qwen3.5-4B', adapter: 'pi-coder-v4', streamed: true,
          timestamp_unix_ms: now - 14_700, duration_ms: 1_104, ttft_ms: 35,
          prompt_tokens: 64, completion_tokens: 118, finish_reason: 'stop',
          temperature: 0.7, top_p: 0.95, max_tokens: 1024,
          user_agent: 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/126.0 Safari/537.36',
          client: 'dashboard',
          prompt_preview: 'Compare base vs adapter: explain this regex',
          completion_preview: 'This regex matches semantic version strings: an optional `v` prefix, then MAJOR.MINOR.PATCH digits…',
          prompt_full: 'Compare base vs adapter: explain this regex: ^v?(\\d+)\\.(\\d+)\\.(\\d+)$',
          completion_full: 'This regex matches semantic version strings: an optional `v` prefix, then MAJOR.MINOR.PATCH digit groups anchored to the whole string.',
        },
        {
          id: 'chatcmpl-e1c47b08-9d35-4a62-b7e9-5f0a8c3d2b17',
          model: 'Qwen3.5-4B', adapter: 'pi-coder-v4', streamed: false,
          timestamp_unix_ms: now - 16_200, duration_ms: 287, ttft_ms: null,
          prompt_tokens: 18, completion_tokens: 2, finish_reason: 'stop',
          temperature: 0, top_p: 1.0, max_tokens: 8,
          user_agent: 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/126.0 Safari/537.36',
          client: 'dashboard',
          prompt_preview: 'Reply with the single word: connected',
          completion_preview: 'connected',
          prompt_full: 'Reply with the single word: connected',
          completion_full: 'connected',
        },
        {
          id: 'chatcmpl-3c8f0a17-7d22-49b6-8c41-0a5e6f2b1d34',
          model: 'Qwen3.5-4B', adapter: 'pi-coder-v4', streamed: true,
          timestamp_unix_ms: now - 21_500, duration_ms: 3_204, ttft_ms: 41,
          prompt_tokens: 6_540, completion_tokens: 512, finish_reason: 'stop',
          temperature: 0.2, top_p: 0.95, max_tokens: 4096,
          user_agent: 'opencode/0.4.2 (@ai-sdk/openai-compatible)',
          prompt_preview: 'Write a unit test for the new rate limiter covering burst + steady-state',
          completion_preview: '```rust\n#[tokio::test]\nasync fn limiter_allows_burst_then_throttles() {\n    let lim = RateLimiter::new(5, Duration::from_secs(1));',
          prompt_full: 'Write a unit test for the new rate limiter covering burst + steady-state behaviour.',
          completion_full: '```rust\n#[tokio::test]\nasync fn limiter_allows_burst_then_throttles() {\n    let lim = RateLimiter::new(5, Duration::from_secs(1));\n    for _ in 0..5 { assert!(lim.try_acquire().await); }\n    assert!(!lim.try_acquire().await);\n}\n```',
        },
        {
          id: 'chatcmpl-5e9a2b48-3f61-4d7a-a2b8-9c0d1e4f7a26',
          model: 'Qwen3.5-4B', adapter: null, streamed: true,
          timestamp_unix_ms: now - 47_800, duration_ms: 988, ttft_ms: 33,
          prompt_tokens: 1_204, completion_tokens: 142, finish_reason: 'stop',
          temperature: 0.3, top_p: 0.95, max_tokens: 1024,
          user_agent: 'pi/1.2.0',
          prompt_preview: 'Summarize the diff in this PR and flag anything risky',
          completion_preview: 'The PR swaps the blocking Mutex for an async RwLock in `cache.rs`. Risk: the `.read()` guard is now held across an await on line 88…',
          prompt_full: 'Summarize the diff in this PR and flag anything risky.',
          completion_full: 'The PR swaps the blocking Mutex for an async RwLock in `cache.rs`. Risk: the `.read()` guard is now held across an await on line 88, which can deadlock under contention.',
        },
        {
          id: 'chatcmpl-8a3c1f76-6b94-4e2c-bf03-7d5a9e8c2b41',
          model: 'Qwen3.5-4B', adapter: 'pi-coder-v4', streamed: false,
          timestamp_unix_ms: now - 92_300, duration_ms: 421, ttft_ms: null,
          prompt_tokens: 812, completion_tokens: 54, finish_reason: 'stop',
          temperature: 0.0, top_p: 1.0, max_tokens: 256,
          user_agent: 'curl/8.7.1',
          prompt_preview: 'Generate a conventional-commit message for the staged changes',
          completion_preview: 'fix(cache): hold read guard only for the lookup, not across the network fetch',
          prompt_full: 'Generate a conventional-commit message for the staged changes.',
          completion_full: 'fix(cache): hold read guard only for the lookup, not across the network fetch',
        },
        {
          id: 'chatcmpl-2d6b9e03-1a48-4f57-9b21-3e8c0a7d5f62',
          model: 'Qwen3.5-4B', adapter: 'pi-coder-v4', streamed: true,
          timestamp_unix_ms: now - 158_400, duration_ms: 2_140, ttft_ms: 36,
          prompt_tokens: 3_902, completion_tokens: 318, finish_reason: 'length',
          temperature: 0.2, top_p: 0.95, max_tokens: 320,
          user_agent: 'OpenAI/Python 1.40.0',
          prompt_preview: 'Explain why the integration test `flaky_reconnect` is failing intermittently',
          completion_preview: 'The test asserts reconnection within 100ms but the backoff jitter can push the first retry to 150ms under load. Either widen…',
          prompt_full: 'Explain why the integration test `flaky_reconnect` is failing intermittently.',
          completion_full: 'The test asserts reconnection within 100ms but the backoff jitter can push the first retry to 150ms under load. Either widen the assertion window or seed the jitter RNG in the test harness.',
        },
      ];
    },
    '/v1/adapters': () => ({
      active: demoState.active,
      available: [
        { name: 'pi-coder-v4',  size_bytes: 86 * 1024 * 1024 },
        { name: 'rust-reviewer-v3',   size_bytes: 64 * 1024 * 1024 },
        { name: 'sql-explainer-v2',   size_bytes: 28 * 1024 * 1024 },
        { name: 'commit-msg-writer',  size_bytes: 12 * 1024 * 1024 + 600 * 1024 },
        { name: 'tool-call-tuned-v1', size_bytes: 41 * 1024 * 1024 },
      ],
    }),
    '/v1/train/queue': () => ({
      running: { job_id: 'job_43_rust_reviewer_grpo', state: 'Running', progress: 0.58, current_loss: 0.286, adapter_name: 'rust-reviewer-v3', elapsed_secs: 1_842, job_type: 'GRPO' },
      queued:  [
        { job_id: 'job_44_sql_explainer_sft', state: 'Queued', adapter_name: 'sql-explainer-v2', position: 1, job_type: 'SFT' },
      ],
      completed: [
        // §8.7 promotion-gate verdict: machine-readable gate_outcome next
        // to the prose verdict, exactly as the live server stamps them —
        // the demo's queue panel shows the green "promoted" pill.
        { job_id: 'job_42_pi_grpo',  state: 'Completed', progress: 1.0, current_loss: 0.214, adapter_name: 'pi-coder-v4', elapsed_secs: 5_280, job_type: 'GRPO', gate_outcome: 'promoted', post_eval_verdict: 'PASSED: accuracy 0.913 >= 0.850; adapter `pi-coder-v4` promoted to active' },
        { job_id: 'job_41_commit_sft',     state: 'Completed', progress: 1.0, current_loss: 0.331, adapter_name: 'commit-msg-writer', elapsed_secs: 1_212, job_type: 'SFT' },
        { job_id: 'job_40_corrections_sft', state: 'Completed', progress: 1.0, current_loss: 0.402, adapter_name: 'codebase-corrections', elapsed_secs: 318, job_type: 'SFT' },
      ],
    }),
    '/v1/models': () => ({ data: [{ id: 'Qwen3.5-4B', object: 'model' }] }),
    // The embedded pi terminal needs a live server (PTY + WebSocket) — be
    // honest about that in the static demo instead of a broken Launch.
    '/v1/terminal/status': () => ({
      enabled: false,
      disabled_reason: 'The embedded pi terminal needs a live Kiln server — this is the static demo. Run `kiln serve` and open /ui there.',
      pi_available: false, pi_path: null, cwd: '—', session_active: false,
    }),
    // Train / Eval / Distill pages so the whole demo is a complete agent-backend
    // showcase, not just the Overview.
    '/v1/eval/datasets': evalDatasets,
    '/v1/eval/suites': evalSuites,
    '/v1/eval/jobs': evalJobs,
    '/v1/judgments': judgments,
    '/v1/teachers': teachers,
    '/v1/recipes': recipes,
    '/v1/library': library,
    '/v1/cache/stats': cacheStats,
    '/v1/preflight/compatibility': preflightCompat,
    '/v1/preflight/tiers': preflightTiers,
    '/v1/agent/traces': agentTraces,
  };

  // Parameterized GETs (detail drills) — matched by pattern, not exact path.
  function paramResponse(method, path) {
    let m;
    if (method === 'GET' && (m = path.match(/^\/v1\/adapters\/(.+)\/detail$/))) return adapterDetail(decodeURIComponent(m[1]));
    if (method === 'GET' && (m = path.match(/^\/v1\/eval\/suites\/(.+)$/))) return evalSuiteDetail(decodeURIComponent(m[1]));
    if (method === 'GET' && (m = path.match(/^\/v1\/eval\/jobs\/(.+)$/))) return evalJobDetail(decodeURIComponent(m[1]));
    if (method === 'GET' && (m = path.match(/^\/v1\/train\/jobs\/(.+)$/))) return trainJobDetail(decodeURIComponent(m[1]));
    if (method === 'GET' && (m = path.match(/^\/v1\/eval\/datasets\/(.+)\/rows$/))) {
      // Small representative SFT-chat rows so the "use an uploaded dataset"
      // training path works in the demo.
      return Array.from({ length: 24 }, (_, i) => ({
        messages: [
          { role: 'user', content: `Demo dataset row ${i + 1}: explain what this function does.` },
          { role: 'assistant', content: `Row ${i + 1}: it validates the input, then maps each entry through the parser and returns the collected results.` },
        ],
      }));
    }
    if (method === 'POST' && path === '/v1/agent/traces/discover') return agentDiscover();
    return undefined;
  }

  const realFetch = window.fetch.bind(window);
  window.fetch = function (input, init) {
    const url = typeof input === 'string' ? input : (input && input.url) || '';
    const path = url.startsWith('http') ? new URL(url).pathname : url.split('?')[0];
    const method = ((init && init.method) || (typeof input !== 'string' && input && input.method) || 'GET').toUpperCase();
    const ok = (body) => Promise.resolve(new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } }));
    if (responses[path]) return ok(responses[path]());
    // Demo honesty: load/unload actually move the active adapter, so the
    // ACTIVE pill and flywheel respond to a swap like the real server.
    if (method === 'POST' && path === '/v1/adapters/load') {
      try { demoState.active = JSON.parse(init && init.body || '{}').name || demoState.active; } catch {}
      return ok({ ok: true, status: 'ok', message: 'Demo: loaded ' + demoState.active });
    }
    if (method === 'POST' && path === '/v1/adapters/unload') {
      demoState.active = null;
      return ok({ ok: true, status: 'ok', message: 'Demo: unloaded' });
    }
    const param = paramResponse(method, path);
    if (param !== undefined) return ok(param);
    // Any other mutation under /v1 → generic success ack so buttons don't error.
    if (path.startsWith('/v1/') && (method === 'POST' || method === 'DELETE' || method === 'PUT')) {
      return ok({ ok: true, status: 'ok', message: 'Demo mode: ' + method + ' ' + path + ' acknowledged', job_id: 'job_demo_' + path.replace(/\W+/g, '_') });
    }
    // Any other /v1 GET → empty-but-valid envelope (keeps panels from erroring).
    if (path.startsWith('/v1/') && method === 'GET') return ok({});
    return realFetch(input, init);
  };
})();
