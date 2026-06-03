//! Phase R.6 — hipBLASLt vs hipBLASLt-default MLP gate||up probe.
//!
//! ROCm analog of `crates/kiln-blas/examples/cublaslt_mlp_probe.rs`.
//! Runs the MLP gate||up matmul `[B*T, 2560] @ [2560, 18432]` via:
//!   (1) `hipblasLtMatmul` with the implicit heuristic (default algo
//!       selection) — the "default" baseline.
//!   (2) `hipblasLtMatmul` with explicit `hipblasLtMatmulAlgoGetHeuristic`
//!       and a workspace.
//!
//! at `B*T ∈ {1024, 2048, 4096, 8192}` and writes a JSON report to stdout
//! (and optionally to a file if `--out <path>` is passed).
//!
//! See `crates/kiln-rocblas/csrc/hipblaslt_probe.cu` for the C++
//! implementation.
//!
//! Run on a ROCm host (hipBLASLt requires a HIP device):
//!
//! ```text
//! cargo run --release \
//!   -p kiln-rocblas --features probe \
//!   --example hipblaslt_mlp_probe -- \
//!   --out bench-results/hipblaslt_mlp_probe-<gpu-sku>.json
//! ```

use kiln_rocblas::probe_ffi::probe;
use serde::Serialize;
use std::path::PathBuf;

#[derive(Serialize)]
struct ShapeReport {
    bt: i32,
    k: i32,
    n: i32,
    iters: i32,
    ms_rocblas_default: f32,
    ms_hipblaslt_heuristic: f32,
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
                    "usage: hipblaslt_mlp_probe [--out PATH] [--iters N]\n\
                     \n\
                     Benchmarks the Qwen3.5-4B MLP gate||up matmul shape\n\
                     [B*T, 2560] @ [2560, 18432] at B*T in {{1024, 2048, 4096, 8192}}\n\
                     via hipblasLtMatmul (default heuristic) vs hipblasLtMatmul (explicit heuristic).\n"
                );
                return;
            }
            other => eprintln!("ignoring unknown arg: {other}"),
        }
    }

    let bts = [1024, 2048, 4096, 8192];
    let k = 2560;
    let n = 18432;

    // Best-effort GPU name via `rocm-smi --showproductname`.
    let gpu_query = std::process::Command::new("rocm-smi")
        .args(["--showproductname", "--csv"])
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

    println!("# Phase R.6 hipBLASLt heuristic vs default probe");
    println!("# GPU: {}", gpu_query);
    println!(
        "# shape: [B*T, k={}] @ [k, n={}]  →  [B*T, n]  (BF16 inputs, BF16 output, FP32 compute)",
        k, n
    );
    println!("# iters per shape (median of): {}", iters);
    println!(
        "{:>5} | {:>16} | {:>20} | {:>9} | {:>10} | {:>15}",
        "B*T", "hipLt-default ms", "hipLt-heuristic ms", "speedup", "algo_id", "workspace KB"
    );

    let mut per_shape = Vec::new();
    for &bt in &bts {
        match probe(bt, k, n, iters) {
            Ok(r) => {
                let speedup = if r.ms_hipblaslt_heuristic > 0.0 {
                    r.ms_rocblas_default / r.ms_hipblaslt_heuristic
                } else {
                    f32::NAN
                };
                println!(
                    "{:>5} | {:>16.3} | {:>20.3} | {:>8.3}x | {:>10} | {:>15.1}",
                    r.bt,
                    r.ms_rocblas_default,
                    r.ms_hipblaslt_heuristic,
                    speedup,
                    r.chosen_algo_id,
                    (r.chosen_workspace_bytes as f64) / 1024.0,
                );
                per_shape.push(ShapeReport {
                    bt: r.bt,
                    k: r.k,
                    n: r.n,
                    iters: r.iters,
                    ms_rocblas_default: r.ms_rocblas_default,
                    ms_hipblaslt_heuristic: r.ms_hipblaslt_heuristic,
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
        note: "BF16 inputs, BF16 output, FP32 compute. (1) hipblasLtMatmul with the \
               implicit heuristic (default algo). (2) hipblasLtMatmul with explicit heuristic.",
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
