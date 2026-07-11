use std::path::PathBuf;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn read(path: &str) -> String {
    std::fs::read_to_string(workspace_root().join(path)).unwrap()
}

fn source_between<'a>(source: &'a str, start_marker: &str, end_marker: &str) -> &'a str {
    let start = source
        .find(start_marker)
        .unwrap_or_else(|| panic!("missing start marker: {start_marker}"));
    let rest = &source[start..];
    let end = rest
        .find(end_marker)
        .unwrap_or_else(|| panic!("missing end marker: {end_marker}"));
    &rest[..end]
}

#[test]
fn grpo_uses_settled_group_boundaries_instead_of_a_job_long_gpu_writer() {
    let queue = read("crates/kiln-server/src/training_queue.rs");
    let branch = source_between(
        &queue,
        "QueuedJob::Grpo(mut req) => {",
        "QueuedJob::Opd(mut req) => {",
    );
    assert!(
        !branch.contains("gpu_coordination_write_guard_while_healthy"),
        "GRPO must not retain one GPU writer across dataset I/O and the full job"
    );
    assert!(branch.contains("Some(trainer::GpuStepCoordination::new("));

    let trainer = read("crates/kiln-train/src/trainer.rs");
    let helper = source_between(
        &trainer,
        "fn run_coordinated_grpo_gpu_phase<T>(",
        "pub fn sft_train(",
    );
    for required in [
        "catch_unwind",
        "runtime_synchronize_external_yield",
        "backend_health.quarantine",
        "drop(guard)",
    ] {
        assert!(
            helper.contains(required),
            "coordinated GRPO phases must settle and fail closed before yielding: {required}"
        );
    }

    let inline = source_between(
        &trainer,
        "pub fn grpo_train_to_with_coordination(",
        "pub fn grpo_dry_run_jsonl(",
    );
    let streamed = source_between(
        &trainer,
        "pub fn grpo_train_jsonl_to_with_coordination(",
        "/// Tokenized data for a single completion",
    );
    for (route, source) in [("inline", inline), ("streamed", streamed)] {
        assert!(
            source.contains("optimizer group"),
            "{route} GRPO step is not coordinated"
        );
        assert!(source.contains("checkpoint device snapshot"));
        assert!(source.contains("final adapter snapshot"));
        assert!(source.contains("adapter smoke test and cleanup"));
        assert!(
            source.contains("save_peft(&ckpt_dir") && source.contains("save_peft(&output_dir"),
            "{route} GRPO must publish captured CPU state outside its GPU phases"
        );
    }
}

#[test]
fn rocm_disk_cache_persistence_has_no_server_call_sites() {
    let main = read("crates/kiln-server/src/main.rs");
    for forbidden in [
        "rocm_load_algo_cache_from_disk",
        "rocm_flush_algo_cache_to_disk",
        "hipblaslt autotune cache restored from disk",
        "hipblaslt autotune cache flushed to disk",
        "hipblaslt autotune cache flushed on shutdown",
    ] {
        assert!(
            !main.contains(forbidden),
            "ROCm hipBLASLt disk persistence must stay absent from server startup/shutdown: {forbidden}"
        );
    }
}

#[test]
fn rocm_tensor_has_no_public_disk_cache_api() {
    let lib = read("crates/kiln-tensor/src/lib.rs");
    let rocm_matmul = read("crates/kiln-tensor/src/rocm_matmul.rs");
    for forbidden in [
        "hipblaslt_cache_path",
        "rocm_load_algo_cache_from_disk",
        "rocm_flush_algo_cache_to_disk",
        "rocm_restore_into_shared_cache",
        "rocm_snapshot_algo_cache",
    ] {
        assert!(
            !lib.contains(forbidden),
            "ROCm disk/cache internals must not be public tensor API: {forbidden}"
        );
        assert!(
            !rocm_matmul.contains(&format!("pub fn {forbidden}")),
            "ROCm disk/cache helper must not be reintroduced as a public function: {forbidden}"
        );
    }
}

#[test]
fn blaslt_workspace_is_stream_owned_not_handle_global() {
    for path in [
        "crates/kiln-blas/src/cublaslt_handle.rs",
        "crates/kiln-rocblas/src/hipblaslt_handle.rs",
    ] {
        let src = read(path);
        assert!(
            src.contains("workspace_by_stream"),
            "{path}: BLASLt workspace must be keyed by the typed stream owner"
        );
        assert!(
            !src.contains("workspace_buf: Mutex<Option"),
            "{path}: a single handle-global workspace buffer is not concurrency-safe"
        );
        assert!(
            !src.contains(".default_stream()\n                .alloc"),
            "{path}: workspace allocation must use the caller's active stream"
        );
        assert!(
            src.contains("stream: &Arc<"),
            "{path}: matmul must accept a typed stream owner, not only a raw stream pointer"
        );
    }
}

#[test]
fn paged_kv_row_scatter_materializes_rows_before_slice_set() {
    let src = read("crates/kiln-model/src/paged_kv_cache_kt.rs");
    assert!(
        src.contains("fn row_for_slice_set("),
        "paged-KV row-scatter must keep a single helper for slice_set-ready row materialization"
    );
    assert!(
        src.contains("contiguous row {row_idx}"),
        "paged-KV row-scatter helper must materialize zero-offset contiguous rows before slice_set"
    );

    for required in [
        "Self::row_for_slice_set(&k_flat, i, \"token_major k\")",
        "Self::row_for_slice_set(&v_flat, i, \"token_major v\")",
        "Self::row_for_slice_set(&k_flat, i, \"write_native k\")",
        "Self::row_for_slice_set(&v_flat, i, \"write_native v\")",
        "Self::row_for_slice_set(&k_q, i, \"write_fp8 k\")",
        "Self::row_for_slice_set(&v_q, i, \"write_fp8 v\")",
    ] {
        assert!(
            src.contains(required),
            "paged-KV row-scatter fallback must route through contiguous row materialization: {required}"
        );
    }
}

#[test]
fn process_shared_persistence_uses_resource_layer() {
    let resource = read("crates/kiln-resource/src/lib.rs");
    for required in [
        "pub fn locked_update",
        "create_new(true)",
        "file.sync_all()",
        "std::fs::rename",
        "pub fn lock_path_for",
        "lock_owner_is_dead",
    ] {
        assert!(
            resource.contains(required),
            "kiln-resource must own the cross-process locked atomic write contract: {required}"
        );
    }

    for path in [
        "crates/kiln-blas/src/algo_cache.rs",
        "crates/kiln-rocblas/src/algo_cache.rs",
        "crates/kiln-server/src/api/teachers.rs",
        "crates/kiln-server/src/api/agent_traces.rs",
    ] {
        let src = read(path);
        assert!(
            src.contains("kiln_resource::"),
            "{path}: process-shared persistence must go through kiln-resource"
        );
        assert!(
            !src.contains("std::fs::write(path, bytes"),
            "{path}: direct overwrite writes are not concurrency-safe"
        );
    }
}

#[test]
fn prefix_cache_ownership_is_request_scoped_through_stream_cleanup() {
    let state = read("crates/kiln-server/src/state.rs");
    let batching = read("crates/kiln-server/src/batching_engine.rs");
    let completions = read("crates/kiln-server/src/api/completions.rs");
    let generate = read("crates/kiln-model/src/generate.rs");

    for required in [
        "pub struct RealPrefixCacheRequest",
        "impl Drop for RealPrefixCacheRequest",
        "global_generation",
        "adapter_generations",
        "pending_release_entries",
    ] {
        assert!(
            state.contains(required),
            "prefix cache must retain its request/generation ownership contract: {required}"
        );
    }
    for (path, source) in [
        ("batching_engine.rs", batching.as_str()),
        ("api/completions.rs", completions.as_str()),
    ] {
        assert!(
            !source.contains("hit_entry_id"),
            "{path}: raw cache entry IDs must not escape the move-only request owner"
        );
        assert!(
            !source.contains("release_hit("),
            "{path}: production callers must not manually release prefix hits"
        );
    }
    for required in [
        "run_prefix_cached_stream_worker",
        "post_decode: F",
        "PrefixCachedStreamingCleanup",
        "prefix_stream_decode_panicked",
    ] {
        assert!(
            generate.contains(required),
            "threaded prefix streaming must keep worker-owned cleanup: {required}"
        );
    }
    for forbidden in ["block_free_signal", "free_rx.recv()"] {
        assert!(
            !generate.contains(forbidden),
            "split prefix-stream cleanup ownership must not return: {forbidden}"
        );
    }
}

#[test]
fn rocm_graph_state_is_decode_row_owned() {
    let rocm_graph = read("crates/kiln-model/src/rocm_graph.rs");
    let generate = read("crates/kiln-model/src/generate.rs");

    for required in [
        "enum RocmGraphOwner",
        "DecodeRow(u64)",
        "struct RocmGraphCacheKey",
        "owner: RocmGraphOwner",
        "captured: HashMap<RocmGraphCacheKey, CapturedDecodeGraphRocm>",
        "decode_timelines: HashMap<RocmGraphOwner, RocmGraphOwnerTimeline>",
        "fn prepare_owner_decode",
        "remove_graphs_owned_by(&mut self.captured, owner)",
        "RocmGraphCacheKey::new(owner, requested_key.clone())",
        "RocmGraphCacheKey::new(owner, key)",
    ] {
        assert!(
            rocm_graph.contains(required),
            "ROCm HIP graph decode state must be keyed by decode-row owner: {required}"
        );
    }

    for forbidden in [
        "captured: HashMap<RocmGraphKey, CapturedDecodeGraphRocm>",
        "last_decode_seq_len: None",
        "last_decode_block0: None",
        "Anonymous",
        "graph_row_id: Option<u64>",
        "RocmGraphOwner::from_row_id",
    ] {
        assert!(
            !rocm_graph.contains(forbidden),
            "ROCm HIP graph runner must not keep a runner-wide decode timeline: {forbidden}"
        );
    }

    let batched_rocm = source_between(
        &generate,
        "// R.9: ROCm HIP-graph single-row decode",
        "let sampled = if let Some(tokens)",
    );
    assert_eq!(
        batched_rocm.matches("row_ids[0],").count(),
        2,
        "greedy and sampled ROCm serving decode must pass a concrete stable row id"
    );
    assert!(
        !batched_rocm.contains("Some(row_ids[0])"),
        "ROCm graph APIs must require a concrete row id instead of an optional owner"
    );
}

#[test]
fn direct_rocm_graph_decode_uses_scoped_process_unique_owners() {
    let generate = read("crates/kiln-model/src/generate.rs");
    let rocm_graph = read("crates/kiln-model/src/rocm_graph.rs");
    let bench = read("crates/kiln-server/src/bench.rs");

    for required in [
        "static DECODE_ROW_NEXT_ID",
        "fn allocate_decode_row_id(",
        "fn next_decode_row_id() -> u64",
        "struct RocmDecodeOwnerLease",
        "impl Drop for RocmDecodeOwnerLease",
    ] {
        assert!(
            generate.contains(required),
            "direct and batched ROCm decode must share one owner namespace: {required}"
        );
    }
    assert_eq!(
        generate.matches("id: next_decode_row_id(),").count(),
        3,
        "the direct lease and both batching-state paths must use the shared owner allocator"
    );
    assert!(generate.contains("0 => None"));
    assert!(generate.contains("u64::MAX => Some(0)"));

    let lease_drop = source_between(
        &generate,
        "impl Drop for RocmDecodeOwnerLease",
        "/// Build a strict-prefix prefix-cache registration",
    );
    assert!(lease_drop.contains("graph.release_decode_row(self.row_id)"));
    assert!(lease_drop.contains("poisoned.into_inner().release_decode_row(self.row_id)"));
    for forbidden in [".unwrap()", ".expect(", "panic!("] {
        assert!(
            !lease_drop.contains(forbidden),
            "decode-owner cleanup must not panic while unwinding: {forbidden}"
        );
    }

    for (start, end) in [
        (
            "fn decode_from_prefill_token(",
            "/// Decode one greedy token for multiple compatible paged requests",
        ),
        (
            "fn generate_from_tokens_paged_interleaved(",
            "/// CUDA-graph variant of the interleaved decode path",
        ),
        (
            "pub(crate) fn run_stream_decode_loop_with_first(",
            "fn generate_from_tokens_streaming_paged_speculative_interleaved(",
        ),
    ] {
        let decode_loop = source_between(&generate, start, end);
        let lease = decode_loop
            .find("RocmDecodeOwnerLease::new(&self.rocm_graph")
            .unwrap_or_else(|| panic!("direct decode loop lacks scoped owner: {start}"));
        let loop_start = decode_loop
            .find("for _step in 0..params.max_tokens")
            .unwrap_or_else(|| panic!("direct decode loop lacks token loop: {start}"));
        assert!(
            lease < loop_start,
            "owner lease must cover every early success/error/cancel/drop exit in {start}"
        );
        assert!(decode_loop.contains("rocm_owner.row_id()"));
    }

    let direct_dispatch = source_between(
        &generate,
        "fn decode_next_token_paged_interleaved(",
        "fn decode_next_token_paged_interleaved_or_batched(",
    );
    assert!(direct_dispatch.contains("graph_row_id: u64"));
    assert!(direct_dispatch.contains("graph_row_id,"));

    assert_eq!(
        rocm_graph.matches("graph_row_id: u64").count(),
        3,
        "all three ROCm bs=1 graph APIs must require a concrete owner"
    );
    assert!(!rocm_graph.contains("graph_row_id: Option<u64>"));
    assert!(!rocm_graph.contains("Anonymous"));
    assert!(bench.contains("let rocm_graph_row_id = 1_u64;"));
    assert_eq!(
        bench.matches("rocm_graph_row_id,").count(),
        2,
        "the single-generation latency runner must pass its explicit fixed owner"
    );
}

#[test]
fn rocm_graph_decode_row_state_is_released_before_finish_work() {
    let rocm_graph = read("crates/kiln-model/src/rocm_graph.rs");
    let generate = read("crates/kiln-model/src/generate.rs");
    let batching_engine = read("crates/kiln-server/src/batching_engine.rs");

    for required in [
        "pub fn release_decode_row(&mut self, row_id: u64)",
        "remove_graphs_owned_by(&mut self.captured, owner)",
        "self.decode_timelines.remove(&owner)",
    ] {
        assert!(
            rocm_graph.contains(required),
            "ROCm graph request cleanup must release captured graphs and timelines: {required}"
        );
    }

    let finish = source_between(
        &generate,
        "pub fn finish_paged_batched_decode(",
        "fn completed_prompt_registration(",
    );
    let release = finish
        .find("graph.release_decode_row(state.id)")
        .expect("batched decode finish must release its ROCm graph owner");
    let destructure = finish
        .find("let PagedBatchedDecodeState {")
        .expect("batched decode finish must destructure its state");
    let tokenize = finish
        .find(".tokenizer")
        .expect("batched decode finish must decode output tokens");
    assert!(
        release < destructure && release < tokenize,
        "decode-row graph cleanup must precede all fallible finish work"
    );

    assert!(
        batching_engine
            .matches("finish_paged_batched_decode(state,")
            .count()
            >= 2,
        "normal and discarded batching requests must share the owner-releasing finish path"
    );
}

#[test]
fn control_plane_uses_published_batching_snapshot_without_actor_await() {
    for path in [
        "crates/kiln-server/src/api/health.rs",
        "crates/kiln-server/src/api/metrics.rs",
        "crates/kiln-server/src/api/debug_model_state.rs",
    ] {
        let source = read(path);
        assert!(
            source.contains("cached_snapshot()"),
            "{path}: control-plane reads must use the published batching snapshot"
        );
        assert!(
            !source.contains("engine.snapshot().await"),
            "{path}: control-plane reads must not wait on the batching actor"
        );
    }

    let batching_engine = read("crates/kiln-server/src/batching_engine.rs");
    for required in [
        "pub async fn snapshot(&self) -> Result<BatchingEngineSnapshot>",
        "pub fn cached_snapshot(&self) -> BatchingEngineSnapshot",
        "published_at: Instant",
        "snapshot_age_ms",
        "self.refresh_snapshot();\n        let mut slots:",
    ] {
        assert!(
            batching_engine.contains(required),
            "batching snapshot cache must preserve the control/barrier contract: {required}"
        );
    }
}

#[test]
fn stream_stall_grace_is_strict_startup_config_not_actor_environment() {
    let config = read("crates/kiln-server/src/config.rs");
    for required in [
        "pub stream_stall_grace_ms: StreamStallGrace",
        "config.apply_stream_stall_grace_env_override()?",
        "STREAM_STALL_GRACE_MIN_MS",
        "STREAM_STALL_GRACE_MAX_MS",
        "ConfigValueSource::Environment",
    ] {
        assert!(
            config.contains(required),
            "stream-stall config must be strict, bounded, and source tracked: {required}"
        );
    }

    let batching = read("crates/kiln-server/src/batching_engine.rs");
    assert!(batching.contains("response_delivery_policy: ResponseDeliveryPolicy"));
    assert!(batching.contains("self.response_delivery_policy.stream_stall_grace"));
    for forbidden in [
        "KILN_STREAM_STALL_GRACE_MS",
        "stalled_client_send_grace",
        "OnceLock<Duration>",
        "set_var(\"KILN_STREAM_STALL_GRACE_MS\"",
    ] {
        assert!(
            !batching.contains(forbidden),
            "batching actor/test must not use process-global stall config: {forbidden}"
        );
    }

    let state = read("crates/kiln-server/src/state.rs");
    assert!(
        state.contains("response_delivery_policy: crate::batching_engine::ResponseDeliveryPolicy")
    );
    assert!(state.contains("stream_stall_grace_source = %response_delivery_policy"));

    for path in [
        "crates/kiln-server/src/api/health.rs",
        "crates/kiln-server/src/api/debug_model_state.rs",
    ] {
        let source = read(path);
        assert!(source.contains("stream_stall_grace_ms"), "{path}");
        assert!(source.contains("stream_stall_grace_source"), "{path}");
    }
}

#[test]
fn response_delivery_is_off_actor_ordered_and_non_blocking() {
    let batching = read("crates/kiln-server/src/batching_engine.rs");
    let delivery = read("crates/kiln-server/src/response_delivery.rs");

    let queued = source_between(
        &batching,
        "struct QueuedRequest {",
        "enum ActiveDeliveryState",
    );
    let active = source_between(
        &batching,
        "struct ActiveRequest {",
        "struct EngineDeliveryResultSink",
    );
    assert!(
        !queued.contains("response_tx") && !active.contains("response_tx"),
        "compute-owned request rows must hold delivery keys, not public response senders"
    );
    assert!(!batching.contains("blocking_send(EngineEvent"));
    assert!(!batching.contains("try_send(EngineEvent"));
    assert!(batching.contains("DeliveryKey"));
    assert!(batching.contains("ActiveDeliveryState::InFlight"));
    assert!(batching.contains("DeliveryCommand::DeliverMany"));
    assert!(batching.contains("delivery_outbox"));
    assert!(batching.contains("DeliveryWorker::barrier"));
    assert!(batching.contains("Sender<Vec<DeliveryResult>>"));
    assert!(batching.contains("for result in results"));
    assert!(batching.contains("mpsc::WeakSender<EngineCommand>"));

    for required in [
        "struct DeliveryWorker",
        "response_tx.try_send(event)",
        "A terminal batch may include the final token",
        "generation: u64",
        "newly_ready_lanes",
        "cadence_blocked_lanes",
        "fn notify(&mut self)",
        "PendingDeliveryBarrier",
        "results_published",
        "DeliveryResult::ProtocolError",
    ] {
        assert!(
            delivery.contains(required),
            "response delivery worker contract is missing: {required}"
        );
    }
    assert!(
        !delivery.contains("blocking_send"),
        "the delivery worker must never wait on a public response channel"
    );
}

#[test]
fn rocm_graph_owner_lifecycle_is_bounded_and_observable() {
    let rocm_graph = read("crates/kiln-model/src/rocm_graph.rs");
    let generate = read("crates/kiln-model/src/generate.rs");
    let health = read("crates/kiln-server/src/api/health.rs");

    for field in [
        "decode_owner_release_count",
        "decode_owner_graph_release_count",
        "tracked_decode_owner_count",
    ] {
        assert!(
            rocm_graph.contains(field),
            "ROCm graph lifecycle stats must expose {field}"
        );
        assert!(
            health.contains(field),
            "health must preserve ROCm graph lifecycle field {field}"
        );
    }
    let release = source_between(
        &rocm_graph,
        "pub fn release_decode_row(&mut self, row_id: u64)",
        "fn prepare_owner_decode",
    );
    for field in [
        "record_decode_owner_release(evicted_graphs)",
        "event = \"rocm_graph_decode_owner_released\"",
        "row_id",
        "evicted_graphs",
        "removed_timeline",
        "tracked_decode_owner_count = self.decode_timelines.len()",
    ] {
        assert!(
            release.contains(field),
            "ROCm graph release event must preserve cleanup field {field}"
        );
    }

    let owner_start = source_between(
        &rocm_graph,
        "fn prepare_owner_decode",
        "fn max_cached_graphs",
    );
    for field in [
        "event = \"rocm_graph_decode_owner_started\"",
        "row_id = owner.row_id()",
        "seq_len",
        "block0 = block0.unwrap_or_default()",
        "block0_present = block0.is_some()",
    ] {
        assert!(
            owner_start.contains(field),
            "ROCm graph start event must preserve request-bound field {field}"
        );
    }
    assert!(generate.contains("event = \"direct_decode_receiver_dropped\""));
    assert!(generate.contains("row_id = rocm_owner.row_id()"));
}

#[test]
fn cuda_graph_state_is_decode_row_owned() {
    let cuda_graph = read("crates/kiln-model/src/cuda_graph.rs");
    let generate = read("crates/kiln-model/src/generate.rs");

    for required in [
        "enum CudaGraphOwner",
        "DecodeRow(u64)",
        "struct CudaGraphCacheKey",
        "owner: CudaGraphOwner",
        "captured: HashMap<CudaGraphCacheKey, CapturedDecodeGraph>",
        "decode_timelines: HashMap<CudaGraphOwner, CudaGraphOwnerTimeline>",
        "fn prepare_owner_decode",
        "self.captured.retain(|key, _| key.owner != owner)",
        "CudaGraphCacheKey::new(owner, requested_key.clone())",
        "CudaGraphCacheKey::new(owner, key)",
    ] {
        assert!(
            cuda_graph.contains(required),
            "CUDA graph decode state must be keyed by decode-row owner: {required}"
        );
    }

    for forbidden in [
        "captured: HashMap<CudaGraphKey, CapturedDecodeGraph>",
        "last_decode_seq_len: None",
        "last_decode_block0: None",
    ] {
        assert!(
            !cuda_graph.contains(forbidden),
            "CUDA graph runner must not keep a runner-wide bs=1 decode timeline: {forbidden}"
        );
    }

    assert!(
        generate.contains("Some(row_ids[0])"),
        "batched serving decode must pass the stable row id into graph owner keys"
    );
}

#[test]
fn rocm_sampled_batches_have_native_hidden_decode_path() {
    let generate = read("crates/kiln-model/src/generate.rs");
    for required in [
        "ROCm sampled serving batches need a native decode path",
        "kiln_tensor::Device::Rocm(_)",
        "decode_hidden_paged_contiguous_batch_with_ids",
        "sample ROCm hidden batch",
        "row_count > 1",
    ] {
        assert!(
            generate.contains(required),
            "ROCm sampled continuous batches must not fall through to generic fallback: {required}"
        );
    }
}

#[test]
fn rocm_graph_replay_failure_is_a_circuit_breaker() {
    let rocm_graph = read("crates/kiln-model/src/rocm_graph.rs");
    assert!(
        rocm_graph.contains("disabling ROCm HIP graphs for this runner"),
        "ROCm graph replay failures must disable the runner instead of recapturing forever"
    );
    assert!(
        rocm_graph.matches("self.enabled = false;").count() >= 3,
        "ROCm graph replay/capture failure paths must set a runner-local circuit breaker"
    );
    assert!(
        rocm_graph.matches("self.captured.clear();").count() >= 4,
        "ROCm graph replay failures must clear captured graph state before eager fallback"
    );
}

#[test]
fn rocm_graph_warm_pass_prewarms_capture_stream() {
    let rocm_graph = read("crates/kiln-model/src/rocm_graph.rs");
    let capture = source_between(
        &rocm_graph,
        "fn try_capture_hidden(",
        "fn new_token_buffer(",
    );
    let warm_pass = source_between(
        capture,
        "let htod_before = kiln_tensor::rocm_htod_count();",
        "if let Err(err) = warm_result",
    );

    assert!(
        capture.contains("ROCm graph capture: sync kt default stream before warm pass"),
        "ROCm graph capture must make default-stream input fills visible before warming the capture stream"
    );
    assert!(
        warm_pass.contains("kiln_tensor::with_active_rocm_stream(stream.clone(), ||"),
        "ROCm graph warm pass must run on the same stream that will be captured, so hipBLASLt per-stream workspace is preallocated before begin_capture"
    );
    assert!(
        capture.contains("sync capture stream after ROCm graph warm pass"),
        "ROCm graph capture must wait for the warm pass before restoring state or falling back"
    );

    let warm_stream = capture
        .find("kiln_tensor::with_active_rocm_stream(stream.clone(), ||")
        .expect("warm pass should install active ROCm stream");
    let begin_capture = capture
        .find(".begin_capture()")
        .expect("capture path should call begin_capture");
    assert!(
        warm_stream < begin_capture,
        "ROCm capture stream must be warmed before hipStreamBeginCapture"
    );
}

#[test]
fn rocm_capture_arena_suppresses_active_stream_synchronizes() {
    let rocm_storage = read("crates/kiln-tensor/src/rocm_storage.rs");
    let compute_sync = source_between(
        &rocm_storage,
        "pub fn rocm_synchronize_compute_stream(",
        "/// Block until the active stream for a ROCm tensor",
    );
    let tensor_sync = source_between(
        &rocm_storage,
        "pub fn rocm_synchronize_tensor_stream(",
        "/// Refresh `dst`'s contents in place",
    );
    let contiguous = source_between(
        &rocm_storage,
        "pub fn rocm_contiguous(",
        "fn rocm_view_is_physically_compact(",
    );
    let concat = read("crates/kiln-tensor/src/rocm_ops/concat.rs");
    let bf16_matmul = read("crates/kiln-tensor/src/rocm_ops/bf16_matmul.rs");

    assert!(
        compute_sync.contains("if crate::rocm_capture_arena_active()"),
        "ROCm active compute-stream sync must no-op under HIP graph capture"
    );
    assert!(
        tensor_sync.contains("if crate::rocm_capture_arena_active()"),
        "ROCm tensor-stream sync must no-op under HIP graph capture"
    );
    assert!(
        contiguous
            .matches("if !crate::rocm_capture_arena_active()")
            .count()
            >= 3,
        "ROCm contiguous must not call hipStreamSynchronize inside capture"
    );
    assert!(
        concat
            .matches("if !crate::rocm_capture_arena_active()")
            .count()
            >= 3,
        "ROCm concat must not call hipStreamSynchronize inside capture"
    );
    assert!(
        bf16_matmul.contains("if !crate::rocm_capture_arena_active()"),
        "ROCm BF16 matmul fallback must not call hipStreamSynchronize inside capture"
    );
}

#[test]
fn hip_runtime_errors_are_cleared_at_wrapper_boundary() {
    let hip = read("crates/kiln-hip/src/lib.rs");
    let sys = read("crates/kiln-hip/src/sys.rs");
    let rocm_graph = read("crates/kiln-model/src/rocm_graph.rs");

    assert!(
        sys.contains("pub fn hipGetLastError()"),
        "kiln-hip must bind hipGetLastError so failed runtime calls can clear sticky error state"
    );
    assert!(
        hip.contains("let _ = unsafe { sys::hipGetLastError() };"),
        "kiln-hip::check must clear sticky HIP runtime errors after surfacing direct API failures"
    );
    assert!(
        hip.contains("sticky per host thread"),
        "the sticky-error boundary is a concurrency/resource invariant and should stay documented"
    );

    for required in [
        "*linear_state = gdn_snapshot;",
        "*linear_state = capture_snapshot;",
        "freeze-pointers warm (Record) pass failed",
        "forward pass failed during graph capture",
        "end_capture failed",
        "execute captured decode graph (first run)",
    ] {
        assert!(
            rocm_graph.contains(required),
            "ROCm graph capture failures must restore decode state before eager fallback: {required}"
        );
    }
}
