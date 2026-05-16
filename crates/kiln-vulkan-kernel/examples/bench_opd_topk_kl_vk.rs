//! Throughput microbench for the fused Vulkan OPD top-K reverse-KL kernel.
//!
//! Run on a Vulkan-capable host (RTX A6000 etc.) with:
//!
//! ```bash
//! NVIDIA_DRIVER_CAPABILITIES=all \
//!     cargo run --release --example bench_opd_topk_kl_vk -p kiln-vulkan-kernel
//! ```
//!
//! Reports per-shape:
//! - Forward kernel-path tok/s.
//! - Backward kernel-path tok/s.
//! - Median per-iteration latency.
//!
//! Shapes mirror §9.7 of the grand-plan-on-policy-distillation doc:
//! Qwen3.5-4B student (H=2560, V=32K subset), K=32, T ∈ {256, 1024, 4096}.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use kiln_vulkan_kernel::vk_ops::opd::{
    dispatch_opd_topk_kl_bwd_resident, dispatch_opd_topk_kl_fwd_resident,
};
use kiln_vulkan_kernel::vk_tensor::{VkDType, VkTensor};
use kiln_vulkan_kernel::{VulkanBuffer, VulkanDevice};
use std::sync::Arc;
use std::time::Instant;

const WARMUP_ITERS: usize = 10;
const TIMED_ITERS: usize = 30;
const REPEATS: usize = 3;

fn deterministic_hidden(t: usize, h: usize) -> Vec<f32> {
    (0..(t * h))
        .map(|i| ((i as f32) * 0.013 + 0.07).sin() * 0.5)
        .collect()
}

fn deterministic_weight(v: usize, h: usize) -> Vec<f32> {
    (0..(v * h))
        .map(|i| (((i as f32) + 7.0) * 0.0091).cos() * 0.25)
        .collect()
}

fn deterministic_topk(t: usize, v: usize, k: usize) -> (Vec<u32>, Vec<f32>) {
    let mut idx: Vec<u32> = Vec::with_capacity(t * k);
    let mut lpq: Vec<f32> = Vec::with_capacity(t * k);
    for ti in 0..t {
        let mut row: Vec<u32> = (0..k as u32)
            .map(|kk| ((ti * 17 + (kk as usize) * 31 + 5) % v) as u32)
            .collect();
        let mut seen = std::collections::HashSet::new();
        for kk in 0..k {
            while !seen.insert(row[kk]) {
                row[kk] = (row[kk] + 1) % v as u32;
            }
        }
        idx.extend_from_slice(&row);
        for kk in 0..k {
            lpq.push(-((ti as f32 + 1.0).ln() + (kk as f32) * 0.3));
        }
    }
    (idx, lpq)
}

fn upload_f32(dev: &Arc<VulkanDevice>, data: &[f32], shape: &[usize]) -> Result<VkTensor> {
    let t = Tensor::from_vec(data.to_vec(), shape.to_vec(), &Device::Cpu)?;
    VkTensor::from_candle(&t, Arc::clone(dev))
}

fn upload_bf16w(dev: &Arc<VulkanDevice>, data: &[f32], shape: &[usize]) -> Result<VkTensor> {
    let t = Tensor::from_vec(data.to_vec(), shape.to_vec(), &Device::Cpu)?
        .to_dtype(DType::BF16)?;
    VkTensor::from_candle(&t, Arc::clone(dev))
}

fn upload_u32_buf(dev: &Arc<VulkanDevice>, data: &[u32]) -> Result<Arc<VulkanBuffer>> {
    let bytes: Vec<u8> = data.iter().flat_map(|i| i.to_le_bytes()).collect();
    let buf = VulkanBuffer::create_device_local(
        dev.device(),
        dev.device_local_mem_type(),
        bytes.len().max(4) as u64,
    )?;
    VulkanBuffer::upload_data(
        dev.device(),
        dev.host_visible_mem_type(),
        dev.queue(),
        dev.queue_family_index(),
        &buf,
        &bytes,
    )?;
    Ok(Arc::new(buf))
}

fn upload_f32_buf(dev: &Arc<VulkanDevice>, data: &[f32]) -> Result<Arc<VulkanBuffer>> {
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let buf = VulkanBuffer::create_device_local(
        dev.device(),
        dev.device_local_mem_type(),
        bytes.len().max(4) as u64,
    )?;
    VulkanBuffer::upload_data(
        dev.device(),
        dev.host_visible_mem_type(),
        dev.queue(),
        dev.queue_family_index(),
        &buf,
        &bytes,
    )?;
    Ok(Arc::new(buf))
}

fn alloc_f32_buf(dev: &Arc<VulkanDevice>, n: usize) -> Result<Arc<VulkanBuffer>> {
    let buf = VulkanBuffer::create_device_local(
        dev.device(),
        dev.device_local_mem_type(),
        (n.max(1) * 4) as u64,
    )?;
    Ok(Arc::new(buf))
}

fn time_block<F: FnMut() -> Result<()>>(mut f: F, t_active: usize) -> Result<(f64, f64)> {
    for _ in 0..WARMUP_ITERS {
        f()?;
    }
    let mut best_ns = u128::MAX;
    for _ in 0..REPEATS {
        let start = Instant::now();
        for _ in 0..TIMED_ITERS {
            f()?;
        }
        let per_iter = start.elapsed().as_nanos() / (TIMED_ITERS as u128);
        if per_iter < best_ns {
            best_ns = per_iter;
        }
    }
    let ms = (best_ns as f64) / 1.0e6;
    let toks_per_sec = (t_active as f64) / (best_ns as f64 / 1.0e9);
    Ok((ms, toks_per_sec))
}

#[derive(Clone, Copy, Debug)]
struct BenchCfg {
    t: usize,
    h: usize,
    v: usize,
    k: usize,
    bf16w: bool,
}

fn bench_one(dev: &Arc<VulkanDevice>, cfg: BenchCfg) -> Result<()> {
    let hidden_data = deterministic_hidden(cfg.t, cfg.h);
    let weight_data = deterministic_weight(cfg.v, cfg.h);
    let (idx_data, lpq_data) = deterministic_topk(cfg.t, cfg.v, cfg.k);

    let hidden = upload_f32(dev, &hidden_data, &[cfg.t, cfg.h])?;
    let weight = if cfg.bf16w {
        upload_bf16w(dev, &weight_data, &[cfg.v, cfg.h])?
    } else {
        upload_f32(dev, &weight_data, &[cfg.v, cfg.h])?
    };
    let idx_buf = upload_u32_buf(dev, &idx_data)?;
    let lpq_buf = upload_f32_buf(dev, &lpq_data)?;
    let kl_buf = alloc_f32_buf(dev, cfg.t)?;
    let dh_buf = alloc_f32_buf(dev, cfg.t * cfg.h)?;
    let grad_loss_buf = upload_f32_buf(dev, &[1.0f32])?;

    let weight_is_bf16 = weight.dtype() == VkDType::Bf16;
    let h_handle = hidden.buffer().handle();
    let w_handle = weight.buffer().handle();

    // Forward
    let mut fwd = || -> Result<()> {
        dispatch_opd_topk_kl_fwd_resident(
            dev,
            h_handle,
            w_handle,
            weight_is_bf16,
            idx_buf.handle(),
            lpq_buf.handle(),
            kl_buf.handle(),
            cfg.t as u32,
            cfg.h as u32,
            cfg.v as u32,
            cfg.k as u32,
        )
    };
    let (fwd_ms, fwd_toks) = time_block(&mut fwd, cfg.t)?;

    // Backward — scalar-mean mode, scale = 1/T_active.
    let mut bwd = || -> Result<()> {
        dispatch_opd_topk_kl_bwd_resident(
            dev,
            h_handle,
            w_handle,
            weight_is_bf16,
            idx_buf.handle(),
            lpq_buf.handle(),
            grad_loss_buf.handle(),
            dh_buf.handle(),
            cfg.t as u32,
            cfg.h as u32,
            cfg.v as u32,
            cfg.k as u32,
            0, // ScalarMean
            1.0 / (cfg.t as f32),
        )
    };
    let (bwd_ms, bwd_toks) = time_block(&mut bwd, cfg.t)?;

    let dtype = if cfg.bf16w { "bf16w" } else { "f32" };
    println!(
        "t={:>5}  h={:>5}  v={:>6}  k={:>2}  {:<5}  fwd_ms={:>7.3}  fwd_tok/s={:>10.0}  bwd_ms={:>7.3}  bwd_tok/s={:>10.0}",
        cfg.t, cfg.h, cfg.v, cfg.k, dtype, fwd_ms, fwd_toks, bwd_ms, bwd_toks
    );
    Ok(())
}

fn main() -> Result<()> {
    if !VulkanDevice::probe() {
        anyhow::bail!("no Vulkan device found");
    }
    let dev = Arc::new(VulkanDevice::new().context("create Vulkan device")?);
    println!("# kiln-vulkan-kernel OPD top-K reverse-KL throughput bench");
    println!("# WARMUP={WARMUP_ITERS} TIMED_ITERS={TIMED_ITERS} REPEATS={REPEATS}");
    println!("# columns: t (active tokens), h, v, k, dtype, fwd_ms, fwd_tok/s, bwd_ms, bwd_tok/s");

    let configs = [
        BenchCfg { t: 256,  h: 2560, v: 32_000, k: 32, bf16w: false },
        BenchCfg { t: 1024, h: 2560, v: 32_000, k: 32, bf16w: false },
        BenchCfg { t: 4096, h: 2560, v: 32_000, k: 32, bf16w: false },
        BenchCfg { t: 1024, h: 2560, v: 32_000, k: 16, bf16w: false },
        BenchCfg { t: 1024, h: 2560, v: 32_000, k: 32, bf16w: true  },
        BenchCfg { t: 4096, h: 2560, v: 32_000, k: 32, bf16w: true  },
    ];

    for cfg in configs {
        if let Err(e) = bench_one(&dev, cfg) {
            eprintln!("bench failed for {:?}: {e:#}", cfg);
        }
    }
    Ok(())
}
