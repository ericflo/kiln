#![cfg(feature = "metal")]
//! Metal GEMM config sweep (#1082) — compiles many matrix-core GEMM
//! variants in-process and benchmarks each at a full-tile Qwen3.5-4B
//! prefill shape to find the hardware-maxing tile / K-tile / staging
//! config on Apple Silicon. `#[ignore]` — run explicitly:
//!   cargo test -p kiln-tensor --features metal --test metal_gemm_sweep -- --ignored --nocapture

use kiln_tensor::{GemmCfg, bench_gemm_cfg, bench_steel_cfg};

#[test]
#[ignore]
fn sweep_gemm_configs() {
    if kiln_tensor::primary_metal_companion(0).is_err() {
        eprintln!("no Metal device; skipping");
        return;
    }
    // Full-tile shape (M,N multiples of every tile under test, K multiple
    // of 8/16/32) so boundary handling never skews the comparison.
    let (m, k, n, iters) = (512usize, 2560usize, 4096usize, 30usize);
    let gflop = 2.0 * m as f64 * k as f64 * n as f64 / 1e9;
    println!(
        "\n=== Metal GEMM sweep  [{m}x{k}]@[{k}x{n}]  ({gflop:.2} GFLOP/iter, {iters} iters) ==="
    );

    // The live candle `call_mlx_gemm` reference comparison was removed
    // (#1082 final step) when the candle_metal_kernels dependency was
    // dropped. The kiln steel-GEMM perf was already validated against it.

    let cfgs = [
        // single naive baseline (established ~140 GFLOP/s, F32-staging plateau)
        GemmCfg {
            bm: 32,
            bn: 64,
            bk: 16,
            wm: 1,
            wn: 2,
            stg: "float",
        },
    ];

    let mut results: Vec<(String, f64)> = Vec::new();
    for c in &cfgs {
        match bench_gemm_cfg(c, m, k, n, iters) {
            Ok(gflops) => {
                println!("  naive {:28}  {:8.1} GFLOP/s", label(c), gflops);
                results.push((format!("naive {}", label(c)), gflops));
            }
            Err(e) => println!("  naive {:28}  FAILED: {e}", label(c)),
        }
    }

    println!("\n--- kiln STEEL kernel (MLX-technique port) ---");
    // All keep TM*TN<=8 (the M1 register-spill cliff). Explore BK (reuse),
    // thread count (128/256/512), and tile aspect ratio.
    let steel_cfgs = [
        // *** the exact config candle's MLX selects for BF16/nn on M1 ***
        GemmCfg {
            bm: 64,
            bn: 64,
            bk: 16,
            wm: 1,
            wn: 2,
            stg: "bfloat",
        }, // TM*TN=32
        GemmCfg {
            bm: 64,
            bn: 64,
            bk: 32,
            wm: 1,
            wn: 2,
            stg: "bfloat",
        },
        GemmCfg {
            bm: 64,
            bn: 64,
            bk: 16,
            wm: 2,
            wn: 2,
            stg: "bfloat",
        }, // was 41 w/o fast-math
        // prior best (re-measure with fast-math)
        GemmCfg {
            bm: 64,
            bn: 32,
            bk: 32,
            wm: 4,
            wn: 2,
            stg: "bfloat",
        },
        GemmCfg {
            bm: 32,
            bn: 32,
            bk: 64,
            wm: 2,
            wn: 2,
            stg: "bfloat",
        },
        GemmCfg {
            bm: 32,
            bn: 32,
            bk: 16,
            wm: 2,
            wn: 2,
            stg: "bfloat",
        },
        // other MLX tiles
        GemmCfg {
            bm: 32,
            bn: 64,
            bk: 16,
            wm: 1,
            wn: 2,
            stg: "bfloat",
        }, // TILE_32_64_16_1_2
        GemmCfg {
            bm: 64,
            bn: 32,
            bk: 32,
            wm: 2,
            wn: 2,
            stg: "bfloat",
        }, // TILE_64_32_32_2_2
    ];
    for c in &steel_cfgs {
        match bench_steel_cfg(c, m, k, n, iters) {
            Ok(gflops) => {
                println!("  steel {:28}  {:8.1} GFLOP/s", label(c), gflops);
                results.push((format!("steel {}", label(c)), gflops));
            }
            Err(e) => println!("  steel {:28}  FAILED: {e}", label(c)),
        }
    }
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("\n--- ranked ---");
    for (lbl, g) in &results {
        println!("  {lbl:28}  {g:8.1} GFLOP/s");
    }
    if let Some((best, g)) = results.first() {
        println!("\nBEST: {best}  @ {g:.1} GFLOP/s");
    }
}

fn label(c: &GemmCfg) -> String {
    format!(
        "BM{} BN{} BK{} W{}x{} {}",
        c.bm, c.bn, c.bk, c.wm, c.wn, c.stg
    )
}
