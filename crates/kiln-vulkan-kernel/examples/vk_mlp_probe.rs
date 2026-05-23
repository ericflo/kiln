//! Phase 0.9 — Vulkan MLP gate||up probe.
//!
//! Runs the Qwen3.5-4B MLP gate||up fused path
//! `[B*T, 2560] @ [2560, 9216]` (gate) and `[B*T, 2560] @ [2560, 9216]` (up)
//! followed by `silu(gate) * up`, via the existing
//! `dispatch_mlp_gate_up_decode_cached` entry point.
//!
//! Reports per-`B*T` median ms over `--iters` iterations (default 32) plus
//! a JSON report keyed by GPU name (from `vulkaninfo`).
//!
//! Run on RunPod (vulkan-enabled image) per the kiln-vulkan-kernel README:
//!
//! ```text
//! cargo run --release \
//!     -p kiln-vulkan-kernel \
//!     --example vk_mlp_probe -- \
//!     --out bench-results/vk_mlp_probe-a6000.json \
//!     --iters 32
//! ```
//!
//! ## Why this is asymmetric to the cublasLt probe
//!
//! The cublasLt probe at `cublaslt_mlp_probe` times the **matmul alone**
//! at prefill-range `B*T ∈ {1024, 2048, 4096, 8192}`. The existing
//! `dispatch_mlp_gate_up_decode_cached` is a **fused** kernel
//! (gate matmul + up matmul + silu*mul → output), and the decode-path
//! shape is `[B*T, 1, 2560]` rather than 2-D `[B*T, 2560]`. The probe
//! sweeps `B*T ∈ {1, 4, 16, 64, 256, 1024}` to cover the realistic
//! decode-batch range.
//!
//! Phase 2's Vulkan track will resolve the asymmetry by either:
//!   (a) adding a matmul-only probe entry point that strips the fused
//!       activation off the end (so the Vulkan vs cublasLt comparison is
//!       apples-to-apples), or
//!   (b) writing a CUDA fused-kernel equivalent of
//!       `dispatch_mlp_gate_up_decode_cached` and benching against that
//!       instead.
//!
//! Either way, the choice is informed by which has the better speedup
//! ceiling at the most-used B*T.

use std::time::Instant;

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use half::bf16;
use kiln_vulkan_kernel::buffer::VulkanBuffer;
use kiln_vulkan_kernel::device::VulkanDevice;
use kiln_vulkan_kernel::kernels::{dispatch_mlp_gate_up_decode_cached, upload_tensor_bf16_packed_buffer};

const HIDDEN: usize = 2560;
const INTERMEDIATE: usize = 9216;

const DEFAULT_ITERS: usize = 32;
const WARMUP_ITERS: usize = 8;
const BATCH_SWEEP: &[usize] = &[1, 4, 16, 64, 256, 1024];

fn make_bf16_weight(rows: usize, cols: usize, seed: u64) -> Result<Tensor> {
    // Same deterministic-pattern fill as `decode_microbench.rs` so the two
    // are comparable.
    let n = rows * cols;
    let data: Vec<bf16> = (0..n)
        .map(|i| {
            let v = (seed.wrapping_mul(i as u64 + 1) ^ 0x9E3779B97F4A7C15) & 0xFFFF;
            let f = ((v as f32) - 32768.0) / 327680.0;
            bf16::from_f32(f)
        })
        .collect();
    Tensor::from_vec(data, (rows, cols), &Device::Cpu).map_err(Into::into)
}

fn make_x(batch: usize) -> Result<Tensor> {
    // Decode dispatch shape: [batch, 1, hidden], FP32.
    let n = batch * HIDDEN;
    let data: Vec<f32> = (0..n)
        .map(|i| ((i % 31) as f32 - 15.0) * 0.01)
        .collect();
    Tensor::from_vec(data, (batch, 1, HIDDEN), &Device::Cpu).map_err(Into::into)
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn time_one_shape(
    vk_device: &VulkanDevice,
    gate_buf: &VulkanBuffer,
    up_buf: &VulkanBuffer,
    batch: usize,
    iters: usize,
) -> Result<f64> {
    let x = make_x(batch)?;
    for _ in 0..WARMUP_ITERS {
        let _ = dispatch_mlp_gate_up_decode_cached(
            vk_device, &x, gate_buf, up_buf, HIDDEN, INTERMEDIATE,
        )?;
    }
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        let _ = dispatch_mlp_gate_up_decode_cached(
            vk_device, &x, gate_buf, up_buf, HIDDEN, INTERMEDIATE,
        )?;
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    Ok(median(times))
}

fn gpu_name_via_vulkaninfo() -> String {
    std::process::Command::new("vulkaninfo")
        .args(["--summary"])
        .output()
        .ok()
        .and_then(|o| {
            if !o.status.success() {
                return None;
            }
            let s = String::from_utf8_lossy(&o.stdout).to_string();
            // Heuristic: first `deviceName = ...` line.
            for line in s.lines() {
                let line = line.trim();
                if let Some(rest) = line.strip_prefix("deviceName") {
                    return Some(
                        rest.trim_start_matches([' ', '=', '\t'])
                            .trim()
                            .to_string(),
                    );
                }
            }
            None
        })
        .unwrap_or_else(|| "unknown".to_string())
}

fn main() -> Result<()> {
    let mut iters = DEFAULT_ITERS;
    let mut out_path: Option<std::path::PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--iters" => {
                if let Some(s) = args.next() {
                    iters = s.parse().context("--iters not a number")?;
                }
            }
            "--out" => {
                out_path = args.next().map(std::path::PathBuf::from);
            }
            "-h" | "--help" => {
                eprintln!(
                    "usage: vk_mlp_probe [--iters N] [--out PATH]\n\
                     \n\
                     Benchmarks dispatch_mlp_gate_up_decode_cached at the\n\
                     Qwen3.5-4B MLP gate||up shape across batch sizes.\n"
                );
                return Ok(());
            }
            other => eprintln!("ignoring unknown arg: {other}"),
        }
    }

    let vk_device = VulkanDevice::new().context("init Vulkan device")?;
    let gpu_name = gpu_name_via_vulkaninfo();

    // Build BF16 weights once, upload to device-local buffers (warm).
    let w_gate = make_bf16_weight(HIDDEN, INTERMEDIATE, 0xDEADBEEF)?;
    let w_up = make_bf16_weight(HIDDEN, INTERMEDIATE, 0xC0FFEE)?;
    let gate_buf = upload_tensor_bf16_packed_buffer(&vk_device, &w_gate)?;
    let up_buf = upload_tensor_bf16_packed_buffer(&vk_device, &w_up)?;

    println!("# Phase 0.9 Vulkan MLP gate||up probe");
    println!("# GPU: {}", gpu_name);
    println!(
        "# fused shape: gate=[B,1,{}] @ [{}, {}], up=[B,1,{}] @ [{}, {}], silu*mul",
        HIDDEN, HIDDEN, INTERMEDIATE, HIDDEN, HIDDEN, INTERMEDIATE,
    );
    println!("# iters per shape (median of): {}", iters);
    println!("{:>5} | {:>14}", "B*T", "vk-fused ms");

    let mut json_rows: Vec<serde_json::Value> = Vec::new();
    for &b in BATCH_SWEEP {
        let ms = time_one_shape(&vk_device, &gate_buf, &up_buf, b, iters)?;
        println!("{:>5} | {:>14.3}", b, ms);
        json_rows.push(serde_json::json!({
            "bt": b,
            "hidden": HIDDEN,
            "intermediate": INTERMEDIATE,
            "iters": iters,
            "ms_vk_fused": ms,
        }));
    }

    let report = serde_json::json!({
        "gpu": gpu_name,
        "qwen3p5_4b_mlp_gate_up": {
            "hidden": HIDDEN,
            "intermediate_each_half": INTERMEDIATE,
            "fused": true,
            "kernel": "dispatch_mlp_gate_up_decode_cached",
        },
        "note": "Fused gate+up+silu*mul through the existing Vulkan decode-path \
                 entry point. Asymmetric to the cublasLt probe (which is \
                 matmul-only) — see vk_mlp_probe.rs lib doc.",
        "per_shape": json_rows,
    });
    let s = serde_json::to_string_pretty(&report)?;
    if let Some(p) = out_path {
        std::fs::write(&p, &s)?;
        eprintln!("wrote {}", p.display());
    } else {
        println!();
        println!("# JSON report");
        println!("{}", s);
    }
    Ok(())
}
