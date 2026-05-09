# WSL CUDA Qwen3.5-4B Shortlog

Date: 2026-05-09

- Rejected CUDA graph stable paged metadata after a `libcuda.so.1.1` SIGSEGV
  and WSL `dxgkio_make_resident: -12` memory pressure.
- Rejected `KILN_CUDA_GRAPH_CACHE_MAX=128` for 16 GiB GPUs: warmed latency can
  improve, but cold-fill and memory risk are not shippable.
- Accepted inference-only fused RMSNorm dispatch:
  - untracked inference tensors use `kiln_rmsnorm_kernel::fused_rmsnorm`;
  - tracked/autograd tensors keep the existing 47 GiB training gate;
  - RMSNorm kill switches still force fallback.
- Patched default server, no force override, warmed 64-token latency:
  `2.7625, 2.7347, 2.7066, 2.7377, 2.7352` seconds.
- Recovered fallback baseline warmed latency:
  `3.296, 3.296, 3.284, 3.273, 3.371` seconds.
- Net warmed gain: 17.2% lower request latency, 20.8% higher completion
  throughput.
- Training smoke completed on the same 4090 Laptop GPU:
  `state=completed`, `final_loss=4.8016462326049805`.
- Training log showed `fused_path="OFF"` for autograd on this 16 GiB GPU, so
  the small-GPU training OOM guard remains active.
- Validation:
  `cargo fmt --all --check`;
  `cargo build --release --features cuda --bin kiln`;
  `cargo test --release -p kiln-model --features cuda rms_norm --lib`.
- Follow-up accepted: rate-limit CUDA graph cache-full warnings to one WARN per
  graph runner, DEBUG afterward, with adapter invalidation resetting the flag.
- Six-request default-log probe warning count dropped from 369 to 1.
- Warmed default-log latency moved from 2.7687s to 2.7435s for 64 tokens
  (+0.9% completion throughput); quiet-log control was 2.7251s.
- Follow-up validation:
  `cargo fmt --all --check`;
  `cargo build --release --features cuda --bin kiln`;
  `cargo test --release -p kiln-model --features cuda cuda_graph --lib`.
