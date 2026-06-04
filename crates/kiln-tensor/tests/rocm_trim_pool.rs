//! On-hardware smoke for `rocm_trim_pool` — proves kiln can RETURN pooled VRAM
//! to the OS (so a coexisting process can reclaim it), which the
//! release-threshold pin otherwise prevents by design.
//!
//! Ignored by default (needs a real ROCm device + is sensitive to other memory
//! churn on the box). Run explicitly:
//!   cargo test -p kiln-tensor --features rocm --test rocm_trim_pool -- --ignored --nocapture
#![cfg(feature = "rocm")]

use kiln_tensor::{DType, Device, Tensor};

fn mem_available_bytes() -> u64 {
    let raw = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    for line in raw.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            if let Some(kib) = rest.split_whitespace().next().and_then(|s| s.parse::<u64>().ok()) {
                return kib * 1024;
            }
        }
    }
    0
}

#[test]
fn trim_pool_is_callable_and_clean() {
    if !kiln_tensor::rocm_is_available() {
        eprintln!("no ROCm device; skipping");
        return;
    }
    // A trim with nothing to release must be a clean no-op.
    kiln_tensor::rocm_trim_pool(0, 0).expect("rocm_trim_pool no-op");
}

#[test]
#[ignore]
fn trim_pool_returns_memory_to_os() {
    if !kiln_tensor::rocm_is_available() {
        eprintln!("no ROCm device; skipping");
        return;
    }
    let gb = |b: u64| b as f64 / 1e9;
    let dev_free = || kiln_tensor::rocm_mem_get_info(0).map(|(f, _)| f as u64).unwrap_or(0);
    let before = dev_free();
    let _ = mem_available_bytes; // keep the helper referenced

    // Allocate ~2 GiB on device, then drop it. With the release-threshold pin
    // the freed block stays POOLED (not auto-returned to the OS).
    //
    // Discrete GPU: `after-trim` device-free rises by ~2 GiB (the pool released
    // it). Unified APU (Strix Halo): GPU memory is GTT-paged system RAM and the
    // device counter is a large virtual budget that the trim doesn't move — there
    // the coexistence signal is system-RAM pressure (the `kiln-memory` governor),
    // not a discrete pool. So this smoke is INFORMATIONAL, not asserted.
    let n = 512usize * 1024 * 1024; // 512M f32 = 2 GiB
    {
        let _t = Tensor::zeros_on(Device::Rocm(0), vec![n], DType::F32).expect("alloc 2GiB");
        kiln_tensor::rocm_synchronize_default_stream(0).ok();
    }
    kiln_tensor::rocm_synchronize_default_stream(0).ok();
    let pooled = dev_free();

    // Trim the pool back to the OS.
    kiln_tensor::rocm_trim_pool(0, 0).expect("trim");
    let after_trim = dev_free();

    eprintln!(
        "device free (hipMemGetInfo): before={:.2}GB  after-free(pooled)={:.2}GB  after-trim={:.2}GB  \
         | reclaimed-by-trim={:.2}GB",
        gb(before),
        gb(pooled),
        gb(after_trim),
        gb(after_trim.saturating_sub(pooled)),
    );
}
