//! Candle-free microbench for Vulkan GDN chunkwise prefill.
//!
//! The timed region reuses already-uploaded q/k/v/beta/g/state tensors, then
//! runs either the existing per-dispatch path or the fused single-submit path.
//! It includes VkTensor output/intermediate allocation, command recording,
//! submits, and GPU waits. It excludes host-to-device input upload.
//!
//! Example:
//! ```bash
//! cargo run --release -p kiln-vulkan-kernel \
//!     --example gdn_chunkwise_prefill_microbench -- \
//!     --seq-len 128 --iters 10 --warmup 3
//! ```

use anyhow::{Context, Result, bail, ensure};
use kiln_vulkan_kernel::vk_ops::gdn_chunkwise::{
    vk_gdn_chunkwise_forward_no_grad, vk_gdn_chunkwise_forward_no_grad_single_submit,
};
use kiln_vulkan_kernel::{VkTensor, VulkanDevice};
use std::env;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug)]
struct Args {
    batch: usize,
    heads: usize,
    seq_len: usize,
    dk: usize,
    dv: usize,
    chunk: usize,
    warmup: usize,
    iters: usize,
    repeats: usize,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            batch: 1,
            heads: 32,
            seq_len: 128,
            dk: 128,
            dv: 128,
            chunk: 64,
            warmup: 3,
            iters: 10,
            repeats: 3,
        }
    }
}

fn parse_usize_flag(args: &[String], idx: &mut usize, name: &str) -> Result<usize> {
    *idx += 1;
    let raw = args
        .get(*idx)
        .with_context(|| format!("missing value for {name}"))?;
    raw.parse::<usize>()
        .with_context(|| format!("invalid usize value for {name}: {raw}"))
}

fn parse_args() -> Result<Args> {
    let argv: Vec<String> = env::args().skip(1).collect();
    let mut cfg = Args::default();
    let mut idx = 0usize;
    while idx < argv.len() {
        match argv[idx].as_str() {
            "--batch" => cfg.batch = parse_usize_flag(&argv, &mut idx, "--batch")?,
            "--heads" => cfg.heads = parse_usize_flag(&argv, &mut idx, "--heads")?,
            "--seq-len" => cfg.seq_len = parse_usize_flag(&argv, &mut idx, "--seq-len")?,
            "--dk" => cfg.dk = parse_usize_flag(&argv, &mut idx, "--dk")?,
            "--dv" => cfg.dv = parse_usize_flag(&argv, &mut idx, "--dv")?,
            "--chunk" => cfg.chunk = parse_usize_flag(&argv, &mut idx, "--chunk")?,
            "--warmup" => cfg.warmup = parse_usize_flag(&argv, &mut idx, "--warmup")?,
            "--iters" => cfg.iters = parse_usize_flag(&argv, &mut idx, "--iters")?,
            "--repeats" => cfg.repeats = parse_usize_flag(&argv, &mut idx, "--repeats")?,
            "--help" | "-h" => {
                println!(
                    "Usage: gdn_chunkwise_prefill_microbench [--batch N] [--heads N] \
                     [--seq-len N] [--dk N] [--dv N] [--chunk N] [--warmup N] \
                     [--iters N] [--repeats N]"
                );
                std::process::exit(0);
            }
            other => bail!("unknown argument: {other}"),
        }
        idx += 1;
    }

    ensure!(cfg.batch > 0, "--batch must be > 0");
    ensure!(cfg.heads > 0, "--heads must be > 0");
    ensure!(cfg.seq_len > 0, "--seq-len must be > 0");
    ensure!(cfg.dk > 0, "--dk must be > 0");
    ensure!(cfg.dv > 0, "--dv must be > 0");
    ensure!(
        cfg.chunk > 0 && cfg.chunk <= 64,
        "--chunk must be in 1..=64 for the current shaders"
    );
    ensure!(cfg.dv <= 256, "--dv must be <= 256 for the current shaders");
    ensure!(cfg.warmup > 0, "--warmup must be > 0");
    ensure!(cfg.iters > 0, "--iters must be > 0");
    ensure!(cfg.repeats > 0, "--repeats must be > 0");
    Ok(cfg)
}

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let bits = ((*seed >> 40) as u32) & 0x00ff_ffff;
    (bits as f32) / 16_777_216.0
}

fn deterministic_data(len: usize, seed: u64, scale: f32, bias: f32) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            let x = lcg(&mut state) * 2.0 - 1.0;
            bias + scale * x
        })
        .collect()
}

fn upload_inputs(
    dev: &Arc<VulkanDevice>,
    cfg: Args,
) -> Result<(VkTensor, VkTensor, VkTensor, VkTensor, VkTensor, VkTensor)> {
    let qkv_n = cfg.batch * cfg.heads * cfg.seq_len * cfg.dk;
    let v_n = cfg.batch * cfg.heads * cfg.seq_len * cfg.dv;
    let gate_n = cfg.batch * cfg.heads * cfg.seq_len;
    let state_n = cfg.batch * cfg.heads * cfg.dk * cfg.dv;

    let q = VkTensor::from_f32_slice(
        &deterministic_data(qkv_n, 0x5157, 0.025, 0.0),
        vec![cfg.batch, cfg.heads, cfg.seq_len, cfg.dk],
        Arc::clone(dev),
    )
    .context("upload q")?;
    let k = VkTensor::from_f32_slice(
        &deterministic_data(qkv_n, 0x4b17, 0.025, 0.0),
        vec![cfg.batch, cfg.heads, cfg.seq_len, cfg.dk],
        Arc::clone(dev),
    )
    .context("upload k")?;
    let v = VkTensor::from_f32_slice(
        &deterministic_data(v_n, 0x7171, 0.05, 0.0),
        vec![cfg.batch, cfg.heads, cfg.seq_len, cfg.dv],
        Arc::clone(dev),
    )
    .context("upload v")?;
    let beta = VkTensor::from_f32_slice(
        &deterministic_data(gate_n, 0x3e7a, 0.15, 0.5),
        vec![cfg.batch, cfg.heads, cfg.seq_len],
        Arc::clone(dev),
    )
    .context("upload beta")?;
    let g = VkTensor::from_f32_slice(
        &deterministic_data(gate_n, 0x9d31, 0.04, -0.08),
        vec![cfg.batch, cfg.heads, cfg.seq_len],
        Arc::clone(dev),
    )
    .context("upload g")?;
    let state = VkTensor::from_f32_slice(
        &deterministic_data(state_n, 0x57a7e, 0.01, 0.0),
        vec![cfg.batch, cfg.heads, cfg.dk, cfg.dv],
        Arc::clone(dev),
    )
    .context("upload state")?;

    Ok((q, k, v, beta, g, state))
}

fn run_once(
    cfg: Args,
    q: &VkTensor,
    k: &VkTensor,
    v: &VkTensor,
    beta: &VkTensor,
    g: &VkTensor,
    initial_state: &VkTensor,
    single_submit: bool,
) -> Result<(VkTensor, VkTensor)> {
    let mut state = initial_state.clone();
    let out = if single_submit {
        vk_gdn_chunkwise_forward_no_grad_single_submit(q, k, v, beta, g, &mut state, cfg.chunk)?
    } else {
        vk_gdn_chunkwise_forward_no_grad(q, k, v, beta, g, &mut state, cfg.chunk)?
    };
    Ok((out, state))
}

fn time_path(
    cfg: Args,
    q: &VkTensor,
    k: &VkTensor,
    v: &VkTensor,
    beta: &VkTensor,
    g: &VkTensor,
    initial_state: &VkTensor,
    single_submit: bool,
) -> Result<Duration> {
    for _ in 0..cfg.warmup {
        let _ = run_once(cfg, q, k, v, beta, g, initial_state, single_submit)?;
    }

    let mut samples = Vec::with_capacity(cfg.repeats);
    for _ in 0..cfg.repeats {
        let start = Instant::now();
        for _ in 0..cfg.iters {
            let _ = run_once(cfg, q, k, v, beta, g, initial_state, single_submit)?;
        }
        samples.push(start.elapsed() / cfg.iters as u32);
    }
    samples.sort_unstable();
    Ok(samples[samples.len() / 2])
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn main() -> Result<()> {
    let cfg = parse_args()?;
    let dev = Arc::new(VulkanDevice::new().context("create Vulkan device")?);
    let (q, k, v, beta, g, initial_state) = upload_inputs(&dev, cfg)?;

    let (legacy_out, legacy_state) = run_once(cfg, &q, &k, &v, &beta, &g, &initial_state, false)
        .context("legacy correctness run")?;
    let (single_out, single_state) = run_once(cfg, &q, &k, &v, &beta, &g, &initial_state, true)
        .context("single-submit correctness run")?;
    let out_err = max_abs_diff(&legacy_out.to_vec_f32()?, &single_out.to_vec_f32()?);
    let state_err = max_abs_diff(&legacy_state.to_vec_f32()?, &single_state.to_vec_f32()?);

    let legacy =
        time_path(cfg, &q, &k, &v, &beta, &g, &initial_state, false).context("time legacy path")?;
    let single = time_path(cfg, &q, &k, &v, &beta, &g, &initial_state, true)
        .context("time single-submit path")?;

    let legacy_ms = legacy.as_secs_f64() * 1.0e3;
    let single_ms = single.as_secs_f64() * 1.0e3;
    let tokens = (cfg.batch * cfg.seq_len) as f64;

    println!("device: {}", dev.device_name());
    println!(
        "shape: batch={} heads={} seq_len={} dk={} dv={} chunk={}",
        cfg.batch, cfg.heads, cfg.seq_len, cfg.dk, cfg.dv, cfg.chunk
    );
    println!(
        "timing: warmup={} iters={} repeats={} median_per_iter",
        cfg.warmup, cfg.iters, cfg.repeats
    );
    println!("correctness: out_max_abs_err={out_err:.6e} state_max_abs_err={state_err:.6e}");
    println!(
        "legacy_per_dispatch: {legacy_ms:.3} ms ({:.1} tok/s)",
        tokens / legacy.as_secs_f64()
    );
    println!(
        "single_submit:       {single_ms:.3} ms ({:.1} tok/s)",
        tokens / single.as_secs_f64()
    );
    println!("speedup:             {:.2}x", legacy_ms / single_ms);

    Ok(())
}
