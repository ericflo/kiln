//! Phase 0.8 — cublasLt vs cublas-default MLP gate||up probe.
//!
//! Runs the MLP gate||up matmul `[B*T, 2560] @ [2560, 18432]` via:
//!   (1) `cublasGemmEx` with `CUBLAS_GEMM_DEFAULT_TENSOR_OP` (the locked-in
//!       candle path)
//!   (2) `cublasLtMatmul` with explicit `cublasLtMatmulAlgoGetHeuristic`
//!       and a workspace
//!
//! at `B*T ∈ {1024, 2048, 4096, 8192}` and writes a JSON report to stdout
//! (and optionally to a file if `--out <path>` is passed).
//!
//! See `crates/kiln-blas/csrc/cublaslt_probe.cu` for the C++ implementation.
//!
//! Run on RunPod (cublasLt is only available on a CUDA host):
//!
//! ```text
//! cargo run --release \
//!   -p kiln-blas --features probe \
//!   --example cublaslt_mlp_probe -- \
//!   --out bench-results/cublaslt_mlp_probe-<gpu-sku>.json
//! ```

use kiln_blas::probe_ffi::probe;
use serde::Serialize;
use std::path::PathBuf;

#[derive(Serialize)]
struct ShapeReport {
    bt: i32,
    k: i32,
    n: i32,
    iters: i32,
    ms_cublas_default: f32,
    ms_cublaslt_heuristic: f32,
    speedup_x: f32,
    chosen_algo_id: i32,
    chosen_workspace_bytes: u64,
}

#[derive(Serialize)]
struct Report {
    gpu_query: String,
    qwen3p5_4b_mlp_gate_up_shape: &'static str,
    note: &'static str,
    per_shape: Vec<ShapeReport>,
}

fn main() {
    let mut out_path: Option<PathBuf> = None;
    let mut iters: i32 = 32;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--out" => {
                out_path = args.next().map(PathBuf::from);
            }
            "--iters" => {
                if let Some(s) = args.next() {
                    if let Ok(v) = s.parse::<i32>() {
                        iters = v;
                    }
                }
            }
            "-h" | "--help" => {
                eprintln!(
                    "usage: cublaslt_mlp_probe [--out PATH] [--iters N]\n\
                     \n\
                     Benchmarks the Qwen3.5-4B MLP gate||up matmul shape\n\
                     [B*T, 2560] @ [2560, 18432] at B*T in {{1024, 2048, 4096, 8192}}\n\
                     via cublasGemmEx (DEFAULT_TENSOR_OP) vs cublasLtMatmul (heuristic).\n"
                );
                return;
            }
            other => eprintln!("ignoring unknown arg: {other}"),
        }
    }

    let bts = [1024, 2048, 4096, 8192];
    let k = 2560;
    let n = 18432;

    // Best-effort GPU name via `nvidia-smi --query-gpu=name --format=csv,noheader`.
    let gpu_query = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader"])
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "unknown".to_string());

    println!("# Phase 0.8 cublasLt vs candle gemm probe");
    println!("# GPU: {}", gpu_query);
    println!(
        "# shape: [B*T, k={}] @ [k, n={}]  →  [B*T, n]  (BF16 inputs, BF16 output, FP32 compute)",
        k, n
    );
    println!("# iters per shape (median of): {}", iters);
    println!(
        "{:>5} | {:>16} | {:>20} | {:>9} | {:>10} | {:>15}",
        "B*T", "cublas-default ms", "cublasLt-heuristic ms", "speedup", "algo_id", "workspace KB"
    );

    let mut per_shape = Vec::new();
    for &bt in &bts {
        match probe(bt, k, n, iters) {
            Ok(r) => {
                let speedup = if r.ms_cublaslt_heuristic > 0.0 {
                    r.ms_cublas_default / r.ms_cublaslt_heuristic
                } else {
                    f32::NAN
                };
                println!(
                    "{:>5} | {:>16.3} | {:>20.3} | {:>8.3}x | {:>10} | {:>15.1}",
                    r.bt,
                    r.ms_cublas_default,
                    r.ms_cublaslt_heuristic,
                    speedup,
                    r.chosen_algo_id,
                    (r.chosen_workspace_bytes as f64) / 1024.0,
                );
                per_shape.push(ShapeReport {
                    bt: r.bt,
                    k: r.k,
                    n: r.n,
                    iters: r.iters,
                    ms_cublas_default: r.ms_cublas_default,
                    ms_cublaslt_heuristic: r.ms_cublaslt_heuristic,
                    speedup_x: speedup,
                    chosen_algo_id: r.chosen_algo_id,
                    chosen_workspace_bytes: r.chosen_workspace_bytes,
                });
            }
            Err(rc) => {
                eprintln!("probe failed at B*T={}: err_code={}", bt, rc);
                std::process::exit(1);
            }
        }
    }

    let report = Report {
        gpu_query,
        qwen3p5_4b_mlp_gate_up_shape: "[B*T, 2560] @ [2560, 18432]",
        note: "BF16 inputs, BF16 output, FP32 compute. (1) cublasGemmEx with \
               CUBLAS_GEMM_DEFAULT_TENSOR_OP. (2) cublasLtMatmul with heuristic.",
        per_shape,
    };
    let s = serde_json::to_string_pretty(&report).expect("serialize");
    if let Some(p) = out_path {
        std::fs::write(&p, &s).expect("write report");
        eprintln!("wrote {}", p.display());
    } else {
        println!();
        println!("# JSON report");
        println!("{}", s);
    }
}
