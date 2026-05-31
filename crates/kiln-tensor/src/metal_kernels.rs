//! Kiln-owned MSL op kernels (#1082) — the `candle_metal_kernels::call_*`
//! replacements.
//!
//! Each op family here owns its MSL source and dispatches it through an
//! objc2 `ComputeCommandEncoder` (the same pattern [`crate::metal_matmul`]
//! established for the GEMM) instead of calling `candle_metal_kernels::call_*`.
//! Pipelines are compiled once and cached process-wide per `(device, entry)`.
//!
//! These still reach the GPU through the `MetalCompanion`'s candle-derived
//! `Device` / command pool — that substrate is the *last* candle dependency
//! and is flipped to a pure objc2-metal substrate in a follow-up once every
//! `call_*` here is kiln-owned (so candle's `&Device`/`&Kernels` are no
//! longer required by any op).

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use crate::metal_types::{ComputePipeline, MetalCompanion};
use crate::{DType, Error, Result};

type MetalBuffer = candle_metal_kernels::metal::Buffer;

/// Compile options matching candle's GEMM path (fast math + fast FP).
pub(crate) fn compile_options() -> objc2::rc::Retained<objc2_metal::MTLCompileOptions> {
    use objc2_metal::{MTLCompileOptions, MTLMathFloatingPointFunctions, MTLMathMode};
    let opts = MTLCompileOptions::new();
    opts.setMathMode(MTLMathMode::Fast);
    opts.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Fast);
    opts
}

/// Compile + cache a compute pipeline for `(device, entry)`. `source` is the
/// MSL containing `entry`; it is compiled at most once per `(device, entry)`.
pub(crate) fn op_pipeline(
    companion: &MetalCompanion,
    source: &str,
    entry: &str,
) -> Result<ComputePipeline> {
    static CACHE: OnceLock<Mutex<HashMap<(u64, String), ComputePipeline>>> = OnceLock::new();
    let key = (companion.device_id(), entry.to_string());
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    {
        let map = cache
            .lock()
            .map_err(|e| Error::Msg(format!("metal_kernels: pipeline cache poisoned: {e}")))?;
        if let Some(p) = map.get(&key) {
            return Ok(p.clone());
        }
    }
    let opts = compile_options();
    let lib = companion
        .device()
        .new_library_with_source(source, Some(&opts))
        .map_err(|e| Error::Msg(format!("metal_kernels: compile {entry}: {e:?}")))?;
    let func = lib
        .get_function(entry, None)
        .map_err(|e| Error::Msg(format!("metal_kernels: get_function {entry}: {e:?}")))?;
    let pipeline = companion
        .device()
        .new_compute_pipeline_state_with_function(&func)
        .map_err(|e| Error::Msg(format!("metal_kernels: pipeline {entry}: {e:?}")))?;
    let mut map = cache
        .lock()
        .map_err(|e| Error::Msg(format!("metal_kernels: pipeline cache poisoned: {e}")))?;
    map.insert(key, pipeline.clone());
    Ok(pipeline)
}

/// MSL scalar type name for a kt dtype (float triple).
pub(crate) fn msl_ty(dt: DType) -> Result<&'static str> {
    match dt {
        DType::F32 => Ok("float"),
        DType::BF16 => Ok("bfloat"),
        DType::F16 => Ok("half"),
        other => Err(Error::Msg(format!("metal_kernels: unsupported dtype {other}"))),
    }
}

/// 1-D dispatch covering exactly `n` threads (non-uniform threadgroups —
/// Apple4+; the M1 is Apple7).
fn dispatch_1d(
    encoder: &candle_metal_kernels::metal::ComputeCommandEncoder,
    n: usize,
) {
    let tg = 256usize.min(n.max(1));
    let grid = objc2_metal::MTLSize { width: n, height: 1, depth: 1 };
    let tgs = objc2_metal::MTLSize { width: tg, height: 1, depth: 1 };
    encoder.dispatch_threads(grid, tgs);
}

// ----------------------------------------------------------------------
// cast — replaces candle_metal_kernels::call_cast_contiguous
// ----------------------------------------------------------------------

/// Contiguous element-wise dtype cast `out[i] = (TO)in[i]` over the float
/// triple. Kiln-owned MSL; dispatched via the companion's encoder.
pub(crate) fn cast(
    companion: &MetalCompanion,
    input: &MetalBuffer,
    output: &MetalBuffer,
    from: DType,
    to: DType,
    n: usize,
) -> Result<()> {
    let (fty, tty) = (msl_ty(from)?, msl_ty(to)?);
    let entry = format!("kt_cast_{fty}_{tty}");
    let src = format!(
        r#"
#include <metal_stdlib>
using namespace metal;
kernel void {entry}(
    device const {fty}* inp [[buffer(0)]],
    device {tty}* outp      [[buffer(1)]],
    constant uint& n        [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{{
    if (gid < n) outp[gid] = ({tty})inp[gid];
}}
"#
    );
    let pipeline = op_pipeline(companion, &src, &entry)?;
    let encoder = companion
        .command_encoder()
        .map_err(|e| Error::Msg(format!("metal_kernels::cast: encoder: {e:?}")))?;
    encoder.set_label("kt_cast");
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(input), 0);
    encoder.set_buffer(1, Some(output), 0);
    let n_u = n as u32;
    encoder.set_bytes(2, &n_u);
    dispatch_1d(&encoder, n);
    drop(encoder);
    Ok(())
}
