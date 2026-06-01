//! Verification and timing probe for split-K paged decode attention.
//!
//! The timed region includes transient buffer allocation, host uploads,
//! command recording/submission, and output readback for both wrappers.

use anyhow::{bail, ensure, Context, Result};
use kiln_vulkan_kernel::kernels::{
    dispatch_paged_attn_decode_batch_paged_f32_bytes,
    dispatch_paged_attn_decode_batch_paged_splitk_f32_bytes, paged_attn_decode_splitk_chunks,
};
use kiln_vulkan_kernel::VulkanDevice;
use std::env;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug)]
struct Args {
    batch: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    blocks_per_seq: usize,
    block_size: usize,
    seq_len: usize,
    chunks: Option<usize>,
    warmup: usize,
    iters: usize,
    repeats: usize,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            batch: 3,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 16,
            blocks_per_seq: 3,
            block_size: 4,
            seq_len: 9,
            chunks: None,
            warmup: 1,
            iters: 3,
            repeats: 2,
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
            "--heads" => cfg.num_heads = parse_usize_flag(&argv, &mut idx, "--heads")?,
            "--kv-heads" => {
                cfg.num_kv_heads = parse_usize_flag(&argv, &mut idx, "--kv-heads")?
            }
            "--head-dim" => cfg.head_dim = parse_usize_flag(&argv, &mut idx, "--head-dim")?,
            "--blocks" => cfg.blocks_per_seq = parse_usize_flag(&argv, &mut idx, "--blocks")?,
            "--block-size" => cfg.block_size = parse_usize_flag(&argv, &mut idx, "--block-size")?,
            "--seq-len" => cfg.seq_len = parse_usize_flag(&argv, &mut idx, "--seq-len")?,
            "--chunks" => {
                let chunks = parse_usize_flag(&argv, &mut idx, "--chunks")?;
                cfg.chunks = Some(chunks);
            }
            "--warmup" => cfg.warmup = parse_usize_flag(&argv, &mut idx, "--warmup")?,
            "--iters" => cfg.iters = parse_usize_flag(&argv, &mut idx, "--iters")?,
            "--repeats" => cfg.repeats = parse_usize_flag(&argv, &mut idx, "--repeats")?,
            "--help" | "-h" => {
                println!(
                    "Usage: paged_attn_splitk_check [--batch N] [--heads N] [--kv-heads N] \
                     [--head-dim N] [--blocks N] [--block-size N] [--seq-len N] \
                     [--chunks N] [--warmup N] [--iters N] [--repeats N]"
                );
                std::process::exit(0);
            }
            other => bail!("unknown argument: {other}"),
        }
        idx += 1;
    }

    ensure!(cfg.batch > 0, "--batch must be > 0");
    ensure!(cfg.num_heads > 0, "--heads must be > 0");
    ensure!(cfg.num_kv_heads > 0, "--kv-heads must be > 0");
    ensure!(
        cfg.num_heads % cfg.num_kv_heads == 0,
        "--heads must be divisible by --kv-heads"
    );
    ensure!(
        cfg.head_dim > 0 && cfg.head_dim <= 256,
        "--head-dim must be in 1..=256"
    );
    ensure!(cfg.blocks_per_seq > 0, "--blocks must be > 0");
    ensure!(cfg.block_size > 0, "--block-size must be > 0");
    ensure!(
        cfg.seq_len > 0 && cfg.seq_len <= cfg.blocks_per_seq * cfg.block_size,
        "--seq-len must be in 1..=blocks*block-size"
    );
    if let Some(chunks) = cfg.chunks {
        ensure!(chunks > 0, "--chunks must be > 0");
    }
    ensure!(cfg.iters > 0, "--iters must be > 0");
    ensure!(cfg.repeats > 0, "--repeats must be > 0");
    Ok(cfg)
}

fn patterned_f32(len: usize, period: usize, scale: f32, bias: f32) -> Vec<f32> {
    (0..len)
        .map(|i| bias + ((i % period) as f32 - (period / 2) as f32) * scale)
        .collect()
}

fn build_seq_lens(cfg: Args) -> Vec<u32> {
    (0..cfg.batch)
        .map(|row| cfg.seq_len.saturating_sub(row % cfg.seq_len).max(1) as u32)
        .collect()
}

fn build_block_table(cfg: Args) -> Vec<u32> {
    let mut table = Vec::with_capacity(cfg.batch * cfg.blocks_per_seq);
    for row in 0..cfg.batch {
        let base = row * cfg.blocks_per_seq;
        for block in 0..cfg.blocks_per_seq {
            let permuted = (block * 2 + row + 1) % cfg.blocks_per_seq;
            table.push((base + permuted) as u32);
        }
    }
    table
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[allow(clippy::too_many_arguments)]
fn run_non_split(
    dev: &VulkanDevice,
    cfg: Args,
    q: &[f32],
    k_pool: &[f32],
    v_pool: &[f32],
    block_table: &[u32],
    seq_lens: &[u32],
    total_slots: usize,
    softmax_scale: f32,
) -> Result<Vec<u8>> {
    dispatch_paged_attn_decode_batch_paged_f32_bytes(
        dev,
        bytemuck::cast_slice(q),
        bytemuck::cast_slice(k_pool),
        bytemuck::cast_slice(v_pool),
        cfg.batch,
        cfg.num_heads,
        cfg.head_dim,
        total_slots,
        cfg.num_kv_heads,
        block_table,
        seq_lens,
        cfg.blocks_per_seq,
        cfg.block_size,
        softmax_scale,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_split(
    dev: &VulkanDevice,
    cfg: Args,
    q: &[f32],
    k_pool: &[f32],
    v_pool: &[f32],
    block_table: &[u32],
    seq_lens: &[u32],
    total_slots: usize,
    softmax_scale: f32,
    chunks: usize,
) -> Result<Vec<u8>> {
    dispatch_paged_attn_decode_batch_paged_splitk_f32_bytes(
        dev,
        bytemuck::cast_slice(q),
        bytemuck::cast_slice(k_pool),
        bytemuck::cast_slice(v_pool),
        cfg.batch,
        cfg.num_heads,
        cfg.head_dim,
        total_slots,
        cfg.num_kv_heads,
        block_table,
        seq_lens,
        cfg.blocks_per_seq,
        cfg.block_size,
        softmax_scale,
        chunks,
    )
}

#[allow(clippy::too_many_arguments)]
fn time_path<F>(
    cfg: Args,
    mut f: F,
) -> Result<Duration>
where
    F: FnMut() -> Result<Vec<u8>>,
{
    for _ in 0..cfg.warmup {
        let _ = f()?;
    }
    let mut samples = Vec::with_capacity(cfg.repeats);
    for _ in 0..cfg.repeats {
        let start = Instant::now();
        for _ in 0..cfg.iters {
            let _ = f()?;
        }
        samples.push(start.elapsed() / cfg.iters as u32);
    }
    samples.sort_unstable();
    Ok(samples[samples.len() / 2])
}

fn main() -> Result<()> {
    let cfg = parse_args()?;
    let dev = VulkanDevice::new().context("create Vulkan device")?;
    let total_slots = cfg.batch * cfg.blocks_per_seq * cfg.block_size;
    let softmax_scale = (cfg.head_dim as f32).sqrt().recip();
    let chunks = cfg
        .chunks
        .unwrap_or_else(|| paged_attn_decode_splitk_chunks(cfg.batch, cfg.blocks_per_seq));
    let seq_lens = build_seq_lens(cfg);
    let block_table = build_block_table(cfg);

    let q = patterned_f32(cfg.batch * cfg.num_heads * cfg.head_dim, 17, 0.015, 0.01);
    let k_pool = patterned_f32(
        total_slots * cfg.num_kv_heads * cfg.head_dim,
        23,
        0.012,
        -0.005,
    );
    let v_pool = patterned_f32(
        total_slots * cfg.num_kv_heads * cfg.head_dim,
        29,
        0.02,
        0.02,
    );

    let base = run_non_split(
        &dev,
        cfg,
        &q,
        &k_pool,
        &v_pool,
        &block_table,
        &seq_lens,
        total_slots,
        softmax_scale,
    )
    .context("run non-split paged attention")?;
    let split = run_split(
        &dev,
        cfg,
        &q,
        &k_pool,
        &v_pool,
        &block_table,
        &seq_lens,
        total_slots,
        softmax_scale,
        chunks,
    )
    .context("run split-K paged attention")?;

    let base_f32: &[f32] = bytemuck::cast_slice(&base);
    let split_f32: &[f32] = bytemuck::cast_slice(&split);
    let max_abs = max_abs_diff(base_f32, split_f32);
    ensure!(
        max_abs <= 1.0e-5,
        "split-K paged attention mismatch: max_abs={max_abs:.6e}"
    );

    let non_split_time = time_path(cfg, || {
        run_non_split(
            &dev,
            cfg,
            &q,
            &k_pool,
            &v_pool,
            &block_table,
            &seq_lens,
            total_slots,
            softmax_scale,
        )
    })?;
    let split_time = time_path(cfg, || {
        run_split(
            &dev,
            cfg,
            &q,
            &k_pool,
            &v_pool,
            &block_table,
            &seq_lens,
            total_slots,
            softmax_scale,
            chunks,
        )
    })?;
    let non_split_ms = non_split_time.as_secs_f64() * 1.0e3;
    let split_ms = split_time.as_secs_f64() * 1.0e3;

    println!("device: {}", dev.device_name());
    println!(
        "shape: batch={} heads={} kv_heads={} head_dim={} blocks={} block_size={} seq_len={} chunks={}",
        cfg.batch,
        cfg.num_heads,
        cfg.num_kv_heads,
        cfg.head_dim,
        cfg.blocks_per_seq,
        cfg.block_size,
        cfg.seq_len,
        chunks
    );
    println!(
        "timing: warmup={} iters={} repeats={} median_per_iter",
        cfg.warmup, cfg.iters, cfg.repeats
    );
    println!("correctness: max_abs_diff={max_abs:.6e}");
    println!("non_split: {non_split_ms:.3} ms");
    println!("split_k:   {split_ms:.3} ms");
    println!("speedup:   {:.2}x", non_split_ms / split_ms);
    Ok(())
}
