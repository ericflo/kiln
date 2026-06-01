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

type MetalBuffer = crate::metal_rt::Buffer;

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
    encoder: &crate::metal_rt::ComputeCommandEncoder,
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

// ----------------------------------------------------------------------
// activation_unary — replaces candle_metal_kernels::call_unary_contiguous
// ----------------------------------------------------------------------

/// MSL body computing `out = f(in)` in **float** for one activation
/// `kind_tag`, given the float input variable `x` and producing the float
/// result variable `y`. Computed entirely in `float` (load→float, compute
/// in float, store→dtype) so BF16/F16 match the kt CPU reference
/// (`ActivationOp::apply_f32`) bit-for-bit, and F32 matches candle's
/// `unary.metal` (which computes the float ops in `T == float`).
///
/// Returns `None` for an unsupported `kind_tag`.
fn unary_float_body(kind_tag: i32) -> Option<(&'static str, &'static str)> {
    // (op-name fragment for the entry symbol, MSL float expression in `x`).
    //
    // Faithful ports of candle `unary.metal`'s `define_unary_op` bodies,
    // evaluated in `float`:
    //   silu : usilu   = x / (1 + exp(-x))
    //   gelu : ugelu   = 0.5*x*(1 + tanh(M_2_SQRTPI_F*M_SQRT1_2_F*(x+0.044715*x^3)))
    //                    (candle's `x>5` early-out is OMITTED — see helper docs / risks)
    //   tanh : utanh   = precise::tanh(x)
    //   relu : urelu   = x < 0 ? 0 : x
    //   log  : ulog    = log(x)
    //   exp  : uexp    = exp(x)
    //   sin  : usin    = sin(x)
    //   cos  : ucos    = cos(x)
    //   neg  : uneg    = -x
    //   abs  : uabs    = abs(float(x))
    //   sqrt : usqrt   = sqrt(x)
    //   recip: urecip  = 1.0 / x
    //   sign : usign   = sign(x)
    //   floor: ufloor  = floor(x)
    //   ceil : uceil   = ceil(x)
    //   round: uround  = round(x)
    let r = match kind_tag {
        0 => ("silu", "x / (1.0f + exp(-x))"),
        2 => (
            "gelu",
            "0.5f * x * (1.0f + precise::tanh(\
             (M_2_SQRTPI_F * M_SQRT1_2_F) * (x + 0.044715f * x * x * x)))",
        ),
        3 => ("tanh", "precise::tanh(x)"),
        4 => ("relu", "(x < 0.0f ? 0.0f : x)"),
        5 => ("log", "log(x)"),
        6 => ("exp", "exp(x)"),
        7 => ("sin", "sin(x)"),
        8 => ("cos", "cos(x)"),
        12 => ("neg", "-x"),
        13 => ("abs", "fabs(x)"),
        14 => ("sqrt", "sqrt(x)"),
        22 => ("recip", "1.0f / x"),
        23 => ("sign", "sign(x)"),
        24 => ("floor", "floor(x)"),
        25 => ("ceil", "ceil(x)"),
        26 => ("round", "round(x)"),
        _ => return None,
    };
    Some(r)
}

/// Contiguous element-wise unary activation `out[i] = (T)f((float)in[i])`
/// over the float triple. Kiln-owned MSL replacement for
/// `candle_metal_kernels::call_unary_contiguous`; dispatched via the
/// companion's encoder. `kind_tag` follows the kt unary tag scheme
/// (0=silu, 2=gelu, 3=tanh, 4=relu, 5=log, 6=exp, 7=sin, 8=cos, 12=neg,
/// 13=abs, 14=sqrt, 22=recip, 23=sign, 24=floor, 25=ceil, 26=round).
///
/// All math is done in `float`; for BF16/F16 the value is loaded, widened
/// to float, computed, then narrowed back — matching the kt CPU reference
/// (`ActivationOp::apply_f32` and friends). For F32 this is bit-identical
/// to candle's `unary.metal` float ops.
pub(crate) fn activation_unary(
    companion: &MetalCompanion,
    input: &MetalBuffer,
    output: &MetalBuffer,
    dt: DType,
    kind_tag: i32,
    n: usize,
) -> Result<()> {
    let ty = msl_ty(dt)?;
    let (op, expr) = unary_float_body(kind_tag).ok_or_else(|| {
        Error::Msg(format!(
            "metal_kernels::activation_unary: kind_tag {kind_tag} unsupported"
        ))
    })?;
    let entry = format!("kt_unary_{op}_{ty}");
    let src = format!(
        r#"
#include <metal_stdlib>
#include <metal_math>
using namespace metal;
kernel void {entry}(
    device const {ty}* inp [[buffer(0)]],
    device {ty}* outp      [[buffer(1)]],
    constant uint& n       [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{{
    if (gid >= n) return;
    float x = (float)inp[gid];
    float y = {expr};
    outp[gid] = ({ty})y;
}}
"#
    );
    let pipeline = op_pipeline(companion, &src, &entry)?;
    let encoder = companion
        .command_encoder()
        .map_err(|e| Error::Msg(format!("metal_kernels::activation_unary: encoder: {e:?}")))?;
    encoder.set_label("kt_unary");
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(input), 0);
    encoder.set_buffer(1, Some(output), 0);
    let n_u = n as u32;
    encoder.set_bytes(2, &n_u);
    dispatch_1d(&encoder, n);
    drop(encoder);
    Ok(())
}

// ----------------------------------------------------------------------
// elementwise_binary — replaces candle_metal_kernels::call_binary_contiguous
// ----------------------------------------------------------------------

/// Contiguous element-wise binary op over the float triple, same-shape /
/// same-dtype inputs: `out[i] = a[i] <op> b[i]` for `<op>` selected by
/// `kind_tag` (0=add, 1=sub, 2=mul, 3=div). Operands loaded to `float`, the
/// op evaluated in `float`, result stored back to the element dtype —
/// matching candle's effective bf16/half arithmetic and the kt CPU reference
/// (the parity gate). Kiln-owned MSL; one pipeline per (op, dtype).
pub(crate) fn elementwise_binary(
    companion: &MetalCompanion,
    left: &MetalBuffer,
    right: &MetalBuffer,
    output: &MetalBuffer,
    dtype: DType,
    kind_tag: i32,
    n: usize,
) -> Result<()> {
    let ty = msl_ty(dtype)?;
    let (op_prefix, expr) = match kind_tag {
        0 => ("badd", "a + b"),
        1 => ("bsub", "a - b"),
        2 => ("bmul", "a * b"),
        3 => ("bdiv", "a / b"),
        other => {
            return Err(Error::Msg(format!(
                "metal_kernels::elementwise_binary: kind_tag {other} not supported \
                 (only 0=Add, 1=Sub, 2=Mul, 3=Div)"
            )));
        }
    };
    let entry = format!("kt_binary_{op_prefix}_{ty}");
    let src = format!(
        r#"
#include <metal_stdlib>
using namespace metal;
kernel void {entry}(
    device const {ty}* left  [[buffer(0)]],
    device const {ty}* right [[buffer(1)]],
    device {ty}* output      [[buffer(2)]],
    constant uint& n         [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{{
    if (gid < n) {{
        float a = (float)left[gid];
        float b = (float)right[gid];
        output[gid] = ({ty})({expr});
    }}
}}
"#
    );
    let pipeline = op_pipeline(companion, &src, &entry)?;
    let encoder = companion
        .command_encoder()
        .map_err(|e| Error::Msg(format!("metal_kernels::elementwise_binary: encoder: {e:?}")))?;
    encoder.set_label("kt_elementwise_binary");
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(left), 0);
    encoder.set_buffer(1, Some(right), 0);
    encoder.set_buffer(2, Some(output), 0);
    let n_u = n as u32;
    encoder.set_bytes(3, &n_u);
    dispatch_1d(&encoder, n);
    drop(encoder);
    Ok(())
}

// ----------------------------------------------------------------------
// softmax (last axis) — replaces candle_metal_kernels::call_last_softmax
// ----------------------------------------------------------------------

/// Build the per-row last-axis softmax MSL (online/Welford normalizer, a
/// faithful port of candle reduce.metal's `softmax_<dt>`): running max `m`
/// in dtype `T`, running sum `d` in `float`; online merge; finalize
/// `dst[i] = (T)(exp(src[i]-m) * (1/d))`. One threadgroup per row.
fn softmax_src(ty: &str, entry: &str) -> String {
    let exp_expr = match ty {
        "half" => "exp(v)".to_string(),
        "bfloat" => "static_cast<bfloat>(fast::exp(static_cast<float>(v)))".to_string(),
        _ => "fast::exp(v)".to_string(), // "float"
    };
    format!(
        r#"
#include <metal_stdlib>
using namespace metal;

struct MD_t {{
    {ty} m;
    float d;
}};

static inline {ty} kt_exp({ty} v) {{ return {exp_expr}; }}

static inline MD_t kt_md_merge(MD_t a, MD_t b) {{
    bool a_bigger = a.m > b.m;
    MD_t bigger  = a_bigger ? a : b;
    MD_t smaller = a_bigger ? b : a;
    MD_t res;
    res.d = bigger.d + smaller.d * (float)kt_exp((smaller.m - bigger.m));
    res.m = bigger.m;
    return res;
}}

kernel void {entry}(
    constant uint& src_numel    [[buffer(0)]],
    constant uint& el_per_block [[buffer(1)]],
    device const {ty}* src      [[buffer(2)]],
    device {ty}* dst            [[buffer(3)]],
    threadgroup MD_t* shared    [[threadgroup(0)]],
    uint tid       [[thread_index_in_threadgroup]],
    uint dst_id    [[threadgroup_position_in_grid]],
    uint block_dim [[threads_per_threadgroup]])
{{
    const uint offset   = dst_id * el_per_block;
    const uint stop_idx = min(el_per_block + offset, src_numel);

    MD_t md;
    md.m = -INFINITY;
    md.d = 0.0f;
    for (uint i = tid + offset; i < stop_idx; i += block_dim) {{
        MD_t e;
        e.m = src[i];
        e.d = 1.0f;
        md = kt_md_merge(md, e);
    }}

    shared[tid] = md;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = block_dim / 2; s > 0; s >>= 1) {{
        if (tid < s) {{
            shared[tid] = kt_md_merge(shared[tid], shared[tid + s]);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    const MD_t md_total = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float d_total_inverse = 1.0f / md_total.d;
    for (uint i = tid + offset; i < stop_idx; i += block_dim) {{
        dst[i] = ({ty})((float)kt_exp((src[i] - md_total.m)) * d_total_inverse);
    }}
}}
"#
    )
}

/// Contiguous last-axis softmax: view buffer as `[rows, cols]`; per row
/// `out = exp(x-max)/Σexp(x-max)`. One threadgroup per row, reduction in
/// `MD<T>` (max in `T`, sum in `float`). Faithful port of candle's
/// `softmax_<dt>`.
pub(crate) fn softmax_last_axis(
    companion: &MetalCompanion,
    input: &MetalBuffer,
    output: &MetalBuffer,
    dt: DType,
    rows: usize,
    cols: usize,
) -> Result<()> {
    if rows == 0 || cols == 0 {
        return Ok(());
    }
    let ty = msl_ty(dt)?;
    let entry = format!("kt_softmax_{ty}");
    let src = softmax_src(ty, &entry);
    let pipeline = op_pipeline(companion, &src, &entry)?;

    // candle's width: min(maxTotalThreads, (cols/2).next_pow2) — always a
    // power of two so the reduction tree halves cleanly.
    let width = pipeline
        .max_total_threads_per_threadgroup()
        .min((cols / 2).next_power_of_two().max(1));

    let encoder = companion
        .command_encoder()
        .map_err(|e| Error::Msg(format!("metal_kernels::softmax_last_axis: encoder: {e:?}")))?;
    encoder.set_label("kt_softmax_last_axis");
    encoder.set_compute_pipeline_state(&pipeline);

    let src_numel = (rows * cols) as u32;
    let el_per_block = cols as u32;
    encoder.set_bytes(0, &src_numel);
    encoder.set_bytes(1, &el_per_block);
    encoder.set_buffer(2, Some(input), 0);
    encoder.set_buffer(3, Some(output), 0);
    // shared[width] of MD_t; sizeof(MD_t) == 8 for the float triple.
    encoder.set_threadgroup_memory_length(0, width * 8);

    let groups = objc2_metal::MTLSize { width: rows, height: 1, depth: 1 };
    let tg = objc2_metal::MTLSize { width, height: 1, depth: 1 };
    encoder.dispatch_thread_groups(groups, tg);
    drop(encoder);
    Ok(())
}

// ----------------------------------------------------------------------
// rms_norm (last axis) — replaces candle_metal_kernels::call_rms_norm
// ----------------------------------------------------------------------

/// Kiln-owned RMSNorm over the trailing axis (faithful port of candle
/// reduce.metal `rms_norm`): `out[r,j] = x[r,j] * rsqrt(mean_k(x[r,k]^2)+eps)
/// * weight[j]`. Sum-of-squares + scale in `float`; the two store-side
/// multiplies in dtype `T` (matching candle). One threadgroup per row.
pub(crate) fn rms_norm(
    companion: &MetalCompanion,
    input: &MetalBuffer,
    weight: &MetalBuffer,
    output: &MetalBuffer,
    dtype: DType,
    n: usize,
    hidden: usize,
    eps: f32,
) -> Result<()> {
    if n == 0 || hidden == 0 {
        return Ok(());
    }
    let ty = msl_ty(dtype)?;
    let entry = format!("kt_rmsnorm_{ty}");
    let src = format!(
        r#"
#include <metal_stdlib>
using namespace metal;

#define KT_RMSNORM_IMPL(NAME, T)                                            \
kernel void NAME(                                                           \
    device const T* src        [[buffer(0)]],                              \
    device const T* alpha      [[buffer(1)]],                              \
    device T* dst              [[buffer(2)]],                              \
    constant uint& n           [[buffer(3)]],                              \
    constant uint& total_elems [[buffer(4)]],                              \
    constant float& eps        [[buffer(5)]],                              \
    threadgroup float* shared  [[threadgroup(0)]],                         \
    uint tid       [[thread_index_in_threadgroup]],                        \
    uint row       [[threadgroup_position_in_grid]],                       \
    uint block_dim [[threads_per_threadgroup]])                            \
{{                                                                          \
    const uint offset = row * n;                                           \
    const uint stop_idx = min(offset + n, total_elems);                    \
    float acc = 0.0f;                                                      \
    for (uint i = offset + tid; i < stop_idx; i += block_dim) {{           \
        float v = float(src[i]);                                           \
        acc += v * v;                                                      \
    }}                                                                      \
    shared[tid] = acc;                                                     \
    threadgroup_barrier(mem_flags::mem_threadgroup);                       \
    for (uint s = block_dim >> 1; s > 0; s >>= 1) {{                       \
        if (tid < s) shared[tid] += shared[tid + s];                       \
        threadgroup_barrier(mem_flags::mem_threadgroup);                   \
    }}                                                                      \
    threadgroup float total;                                              \
    if (tid == 0) {{                                                       \
        total = rsqrt(shared[0] / float(n) + eps);                         \
    }}                                                                      \
    threadgroup_barrier(mem_flags::mem_threadgroup);                       \
    const float scale = total;                                             \
    for (uint i = offset + tid; i < stop_idx; i += block_dim) {{           \
        T val = src[i] * static_cast<T>(scale);                            \
        val *= alpha[i - offset];                                          \
        dst[i] = val;                                                      \
    }}                                                                      \
}}

KT_RMSNORM_IMPL({entry}, {ty})
"#
    );

    let pipeline = op_pipeline(companion, &src, &entry)?;

    let rows = n / hidden;
    // Round the threadgroup width DOWN to a power of two so the `>>1` tree
    // reduction is exact for any `hidden` (threads beyond `hidden` simply
    // contribute 0 via the strided loop's bound).
    let mut tg = 256usize.min(hidden.max(1));
    tg = if tg.is_power_of_two() {
        tg
    } else {
        tg.next_power_of_two() >> 1
    }
    .max(1);
    let shared_bytes = tg * std::mem::size_of::<f32>();

    let encoder = companion
        .command_encoder()
        .map_err(|e| Error::Msg(format!("metal_kernels::rms_norm: encoder: {e:?}")))?;
    encoder.set_label("kt_rmsnorm");
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(input), 0);
    encoder.set_buffer(1, Some(weight), 0);
    encoder.set_buffer(2, Some(output), 0);
    let hidden_u = hidden as u32;
    let total_u = n as u32;
    encoder.set_bytes(3, &hidden_u);
    encoder.set_bytes(4, &total_u);
    encoder.set_bytes(5, &eps);
    encoder.set_threadgroup_memory_length(0, shared_bytes);

    let groups = objc2_metal::MTLSize { width: rows.max(1), height: 1, depth: 1 };
    let threads = objc2_metal::MTLSize { width: tg, height: 1, depth: 1 };
    encoder.dispatch_thread_groups(groups, threads);
    drop(encoder);
    Ok(())
}

// ----------------------------------------------------------------------
// layer_norm (last axis) — replaces candle_metal_kernels::call_layer_norm
// ----------------------------------------------------------------------

/// MSL template for [`layer_norm`]; the literal `TY` token is replaced with
/// the scalar type before compilation.
const LAYERNORM_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kt_layernorm_TY(
    device const TY* src   [[buffer(0)]],
    device TY* dst         [[buffer(1)]],
    device const TY* alpha [[buffer(2)]],
    device const TY* beta  [[buffer(3)]],
    constant uint& hidden  [[buffer(4)]],
    constant float& eps    [[buffer(5)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint row      [[threadgroup_position_in_grid]],
    uint tid      [[thread_index_in_threadgroup]],
    uint nthreads [[threads_per_threadgroup]])
{
    const uint base = row * hidden;

    float local_sum = 0.0f;
    for (uint i = tid; i < hidden; i += nthreads) {
        local_sum += float(src[base + i]);
    }
    shared[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = nthreads >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float mean = shared[0] / float(hidden);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_var = 0.0f;
    for (uint i = tid; i < hidden; i += nthreads) {
        float d = float(src[base + i]) - mean;
        local_var += d * d;
    }
    shared[tid] = local_var;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = nthreads >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float var = shared[0] / float(hidden);
    const float inv = rsqrt(var + eps);

    for (uint i = tid; i < hidden; i += nthreads) {
        float normalized = (float(src[base + i]) - mean) * inv;
        float y = normalized * float(alpha[i]) + float(beta[i]);
        dst[base + i] = TY(y);
    }
}
"#;

/// Kiln-owned LayerNorm over the trailing axis — replaces
/// `candle_metal_kernels::call_layer_norm`. Two-pass mean/variance in
/// `float`; affine in `float`, store narrows to `T`. One threadgroup/row.
#[allow(clippy::too_many_arguments)]
pub(crate) fn layer_norm(
    companion: &MetalCompanion,
    input: &MetalBuffer,
    weight: &MetalBuffer,
    bias: &MetalBuffer,
    output: &MetalBuffer,
    dt: DType,
    n_rows: usize,
    hidden: usize,
    eps: f32,
) -> Result<()> {
    if n_rows == 0 || hidden == 0 {
        return Ok(());
    }
    let ty = msl_ty(dt)?;
    let entry = format!("kt_layernorm_{ty}");
    let src = LAYERNORM_MSL.replace("TY", ty);

    let pipeline = op_pipeline(companion, &src, &entry)?;
    let encoder = companion
        .command_encoder()
        .map_err(|e| Error::Msg(format!("metal_kernels::layer_norm: encoder: {e:?}")))?;
    encoder.set_label("kt_layernorm");
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(input), 0);
    encoder.set_buffer(1, Some(output), 0);
    encoder.set_buffer(2, Some(weight), 0);
    encoder.set_buffer(3, Some(bias), 0);
    let hidden_u = hidden as u32;
    encoder.set_bytes(4, &hidden_u);
    encoder.set_bytes(5, &eps);

    // Round threadgroup width DOWN to a power of two (clean tree reduction),
    // capped at the pipeline max and never exceeding `hidden`.
    let max_tg = pipeline.max_total_threads_per_threadgroup();
    let mut tg = 256usize.min(max_tg).min(hidden.max(1));
    tg = if tg.is_power_of_two() {
        tg
    } else {
        tg.next_power_of_two() >> 1
    }
    .max(1);
    encoder.set_threadgroup_memory_length(0, tg * std::mem::size_of::<f32>());

    let groups = objc2_metal::MTLSize { width: n_rows, height: 1, depth: 1 };
    let tgs = objc2_metal::MTLSize { width: tg, height: 1, depth: 1 };
    encoder.dispatch_thread_groups(groups, tgs);
    drop(encoder);
    Ok(())
}

// ----------------------------------------------------------------------
// index_select dim0 — replaces candle_metal_kernels::call_index_select
// ----------------------------------------------------------------------

/// Kiln-owned `index_select` along dim 0, contiguous src (u32 ids) — replaces
/// candle's `call_index_select` on the `is_u32_<dt>` path. Gathers
/// `out[i*row_len + j] = input[ids[i]*row_len + j]`, with candle's sentinel
/// (`0xFFFFFFFF` → zero row) + clamp (`min(id, src_dim_size-1)`) semantics.
#[allow(clippy::too_many_arguments)]
pub(crate) fn index_select_dim0(
    companion: &MetalCompanion,
    input: &MetalBuffer,
    ids: &MetalBuffer,
    output: &MetalBuffer,
    dtype: DType,
    src_dim_size: usize,
    row_len: usize,
    ids_size: usize,
) -> Result<()> {
    let tty = msl_ty(dtype)?;
    let entry = format!("kt_index_select_dim0_u32_{tty}");
    let src = format!(
        r#"
#include <metal_stdlib>
using namespace metal;
kernel void {entry}(
    constant uint& dst_size      [[buffer(0)]],
    constant uint& src_dim_size  [[buffer(1)]],
    constant uint& right_size    [[buffer(2)]],
    constant uint& ids_size      [[buffer(3)]],
    device const {tty}* input    [[buffer(4)]],
    device const uint* input_ids [[buffer(5)]],
    device {tty}* output         [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{{
    if (tid >= dst_size) {{
        return;
    }}
    const uint id_i = (tid / right_size) % ids_size;
    const uint raw = input_ids[id_i];
    if (raw == 0xFFFFFFFFu) {{
        output[tid] = static_cast<{tty}>(0);
    }} else {{
        const uint input_i = min(raw, src_dim_size - 1u);
        const uint right_rank_i = tid % right_size;
        const uint src_i = input_i * right_size + right_rank_i;
        output[tid] = input[src_i];
    }}
}}
"#
    );
    let pipeline = op_pipeline(companion, &src, &entry)?;

    let dst_size = ids_size * row_len;
    if dst_size == 0 {
        return Ok(());
    }
    let encoder = companion
        .command_encoder()
        .map_err(|e| Error::Msg(format!("metal_kernels::index_select_dim0: encoder: {e:?}")))?;
    encoder.set_label("kt_index_select_dim0");
    encoder.set_compute_pipeline_state(&pipeline);
    let dst_size_u = dst_size as u32;
    let src_dim_size_u = src_dim_size as u32;
    let right_size_u = row_len as u32;
    let ids_size_u = ids_size as u32;
    encoder.set_bytes(0, &dst_size_u);
    encoder.set_bytes(1, &src_dim_size_u);
    encoder.set_bytes(2, &right_size_u);
    encoder.set_bytes(3, &ids_size_u);
    encoder.set_buffer(4, Some(input), 0);
    encoder.set_buffer(5, Some(ids), 0);
    encoder.set_buffer(6, Some(output), 0);
    dispatch_1d(&encoder, dst_size);
    drop(encoder);
    Ok(())
}

// ----------------------------------------------------------------------
// sdpa — replaces candle_metal_kernels::call_sdpa_{vector,vector_2pass,full}
// ----------------------------------------------------------------------

/// Packed params for the SDPA kernel (set_bytes). Field order MUST match the
/// MSL `struct SdpaParams` exactly. Strides/offsets are in ELEMENTS.
/// (`head_dim`/`NPER`/`W` are baked into the MSL as constants, not passed.)
#[repr(C)]
#[derive(Clone, Copy)]
struct SdpaParams {
    q_seq: u32,
    k_seq: u32,
    hq: u32,
    gqa: u32,
    causal: u32,
    scale: f32,
    q_s0: u32,
    q_s1: u32,
    q_s2: u32,
    q_s3: u32,
    q_off: u32,
    k_s0: u32,
    k_s1: u32,
    k_s2: u32,
    k_s3: u32,
    k_off: u32,
    v_s0: u32,
    v_s1: u32,
    v_s2: u32,
    v_s3: u32,
    v_off: u32,
}

/// Build the flash-attention SDPA MSL for a given dtype + head_dim + split `W`.
///
/// **Flash-tiled, simdgroup-cooperative, split-K.** A threadgroup of `W`
/// simdgroups (W*32 lanes) computes one (qi, head, batch) query. Within a
/// simdgroup, lane `L` owns head-dim slots `L, L+32, ...` (`NPER=ceil(D/32)`,
/// register-resident); the per-key `q·k` score is a warp reduction
/// (`simd_sum`) — **no threadgroup barrier in the hot key loop**. Online
/// softmax (m, l) + the weighted-V accumulator stay in registers.
///
/// The `W` simdgroups split the key range round-robin (`kj = sg, sg+W, ...`),
/// each producing a partial online-softmax state; a single threadgroup-memory
/// combine at the end merges the `W` partials. `W=1` (prefill, where the many
/// queries already saturate the GPU) skips the combine entirely (compile-time
/// dead-code). `W>1` (decode, q_seq small → few query-threadgroups) adds
/// key-parallelism so long-context decode isn't latency-bound on one
/// simdgroup serially walking all keys. GQA / causal / strided q/k/v as before.
fn sdpa_src(ty: &str, head_dim: usize, w: usize, entry: &str) -> String {
    let nper = head_dim.div_ceil(32);
    format!(
        r#"
#include <metal_stdlib>
using namespace metal;

constant constexpr uint D = {head_dim};
constant constexpr uint NPER = {nper};
constant constexpr uint W = {w};

struct SdpaParams {{
    uint q_seq;
    uint k_seq;
    uint hq;
    uint gqa;
    uint causal;
    float scale;
    uint q_s0; uint q_s1; uint q_s2; uint q_s3; uint q_off;
    uint k_s0; uint k_s1; uint k_s2; uint k_s3; uint k_off;
    uint v_s0; uint v_s1; uint v_s2; uint v_s3; uint v_off;
}};

kernel void {entry}(
    device const {ty}* q     [[buffer(0)]],
    device const {ty}* k     [[buffer(1)]],
    device const {ty}* v     [[buffer(2)]],
    device {ty}* out         [[buffer(3)]],
    constant SdpaParams& p   [[buffer(4)]],
    threadgroup float* shared [[threadgroup(0)]],   // W*D + 2*W floats (unused if W==1)
    uint3 tgpos [[threadgroup_position_in_grid]],    // x=qi, y=h, z=b
    uint sg     [[simdgroup_index_in_threadgroup]],  // 0..W-1
    uint lane   [[thread_index_in_simdgroup]])       // 0..31
{{
    const uint qi = tgpos.x;
    const uint h  = tgpos.y;
    const uint b  = tgpos.z;
    if (qi >= p.q_seq || h >= p.hq) return;
    const uint kv_h = h / p.gqa;

    const uint q_base = p.q_off + b * p.q_s0 + h * p.q_s1 + qi * p.q_s2;
    const uint k_base = p.k_off + b * p.k_s0 + kv_h * p.k_s1;
    const uint v_base = p.v_off + b * p.v_s0 + kv_h * p.v_s1;

    // This lane's q components: head-dim slots lane, lane+32, ... (stride 32).
    float qf[NPER];
#pragma clang loop unroll(full)
    for (uint i = 0; i < NPER; i++) {{
        uint d = lane + i * 32u;
        qf[i] = (d < D) ? (float)q[q_base + d * p.q_s3] : 0.0f;
    }}

    // Causal key limit: query qi attends keys 0..=(k_seq - q_seq + qi).
    uint key_limit = p.k_seq;
    if (p.causal != 0) {{
        key_limit = (p.k_seq + qi + 1u);
        key_limit = (key_limit >= p.q_seq) ? (key_limit - p.q_seq) : 0u;
        if (key_limit > p.k_seq) key_limit = p.k_seq;
    }}

    // Per-simdgroup partial online-softmax over its key stride (sg, sg+W, ...).
    float m = -INFINITY;
    float l = 0.0f;
    float acc[NPER];
#pragma clang loop unroll(full)
    for (uint i = 0; i < NPER; i++) acc[i] = 0.0f;

    for (uint kj = sg; kj < key_limit; kj += W) {{
        const uint k_row = k_base + kj * p.k_s2;
        float partial = 0.0f;
#pragma clang loop unroll(full)
        for (uint i = 0; i < NPER; i++) {{
            uint d = lane + i * 32u;
            if (d < D) partial += qf[i] * (float)k[k_row + d * p.k_s3];
        }}
        const float score = simd_sum(partial) * p.scale;

        const float m_new = max(m, score);
        const float corr = exp(m - m_new);
        const float pj = exp(score - m_new);
        l = l * corr + pj;
        const uint v_row = v_base + kj * p.v_s2;
#pragma clang loop unroll(full)
        for (uint i = 0; i < NPER; i++) {{
            uint d = lane + i * 32u;
            if (d < D) acc[i] = acc[i] * corr + pj * (float)v[v_row + d * p.v_s3];
        }}
        m = m_new;
    }}

    const uint out_base = ((b * p.hq + h) * p.q_seq + qi) * D;

    if (W == 1) {{
        // No split: this simdgroup owns the whole row — write directly.
#pragma clang loop unroll(full)
        for (uint i = 0; i < NPER; i++) {{
            uint d = lane + i * 32u;
            if (d < D) out[out_base + d] = ({ty})(l > 0.0f ? (acc[i] / l) : 0.0f);
        }}
        return;
    }}

    // Split-K combine: stash each simdgroup's partial, then merge in sg 0.
    threadgroup float* acc_sh = shared;          // [W][D]
    threadgroup float* m_sh   = shared + W * D;  // [W]
    threadgroup float* l_sh   = m_sh + W;        // [W]
#pragma clang loop unroll(full)
    for (uint i = 0; i < NPER; i++) {{
        uint d = lane + i * 32u;
        if (d < D) acc_sh[sg * D + d] = acc[i];
    }}
    if (lane == 0) {{ m_sh[sg] = m; l_sh[sg] = l; }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0) {{
        float gm = -INFINITY;
#pragma clang loop unroll(full)
        for (uint w = 0; w < W; w++) gm = max(gm, m_sh[w]);
        float gl = 0.0f;
#pragma clang loop unroll(full)
        for (uint w = 0; w < W; w++) gl += l_sh[w] * exp(m_sh[w] - gm);
#pragma clang loop unroll(full)
        for (uint i = 0; i < NPER; i++) {{
            uint d = lane + i * 32u;
            if (d < D) {{
                float a = 0.0f;
#pragma clang loop unroll(full)
                for (uint w = 0; w < W; w++) a += acc_sh[w * D + d] * exp(m_sh[w] - gm);
                out[out_base + d] = ({ty})(gl > 0.0f ? (a / gl) : 0.0f);
            }}
        }}
    }}
}}
"#
    )
}

/// Build the **matrix-core tiled flash-attention** MSL for prefill (large
/// `q_seq`) — a kiln-owned faithful port of candle/MLX's `steel_attention`
/// (`call_sdpa_full`). FA-2 online softmax with **both** `S = Q·Kᵀ` and
/// `O += P·V` running on Apple's simdgroup matrix units (the same idioms the
/// kiln steel GEMM uses: BF16/dtype threadgroup staging, 8×8
/// `simdgroup_matrix` fragments, `simdgroup_multiply_accumulate`, and
/// `#pragma clang loop unroll(full)` on every fragment loop so the
/// register-resident fragment arrays never spill).
///
/// Tiling (matching MLX's `bd<512` pick): `BQ=32` query rows per threadgroup,
/// `BK` key cols per K/V block, `WM=4` simdgroups down Q, `WN=1`. With
/// `TQ = BQ/(WM*8) = 1` each simdgroup owns exactly **one** 8-row Q fragment,
/// so the per-row softmax reductions are entirely intra-simdgroup (no
/// cross-simdgroup combine) — that's what makes this layout clean.
///
/// **Fragment layout** (8×8, 32 lanes, 2 elems/lane — Apple's MMA map). Lane
/// `L` owns S-tile row `sm = (qid&4)+(L/2)%4` (`qid=L/4`) and the two columns
/// `sn = (qid&2)*2 + (L%2)*2`, `sn+1`. So `S[8][BK]` lives across the 32 lanes
/// as `TK = BK/8` fragments, each lane holding 2 cols per fragment for its one
/// row. The running `m`/`l` and the `O[8][D]` accumulator (`TD = D/8`
/// fragments) are per-lane registers; the per-row online rescale by
/// `exp2(m_old-m_new)` is applied to each lane's 2 O-elements via the same row
/// index `sm` (`row_bin_op`). Row reductions (max, sum) fold a lane's 2
/// elements then `simd_shuffle_xor(.,1)` and `simd_shuffle_xor(.,8)` to sum
/// across the fragment's 8 columns — exactly MLX's `BaseMMAFrag::row_reduce`.
///
/// Softmax is done in the FA-2 `exp2` form: Q is pre-scaled by
/// `scale * log2(e)` so `exp2(s - m)` equals `exp(scale*qk - m')`. Accumulate
/// dtype is **float**. Causal masking is right-aligned (`q_off = kL - qL`):
/// key `c` is masked for query row `r` when `r + q_off < c`. K/V blocks past
/// the per-row causal limit are skipped at block granularity, the boundary
/// block is masked per element. `D` is baked per-variant (`d{D}`); a non-
/// multiple-of-8 `D` rounds the staged head-dim up to `TD*8` and zero-pads.
fn sdpa_steel_src(ty: &str, head_dim: usize, bq: usize, bk: usize, wm: usize, entry: &str) -> String {
    let td = head_dim.div_ceil(8); // head-dim fragments (zero-padded if D%8!=0)
    let dpad = td * 8; // staged/padded head dim
    let tk = bk / 8; // key fragments per block
    let tgp = wm * 32; // threads per group (WN=1)
    // Threadgroup leading dims with +pad to dodge bank conflicts (bytes→elems).
    let ld_q = dpad + 8; // Qs[BQ][dpad+8]
    let ld_k = bk + 8; // Ks[dpad][bk+8]  (K staged transposed: [d][k])
    let ld_v = dpad + 8; // Vs[bk][dpad+8]
    format!(
        r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

constant constexpr uint D    = {head_dim};
constant constexpr uint DPAD = {dpad};
constant constexpr uint BQ   = {bq};
constant constexpr uint BK   = {bk};
constant constexpr uint WM   = {wm};
constant constexpr uint TD   = {td};
constant constexpr uint TK   = {tk};
constant constexpr uint TGP  = {tgp};
constant constexpr uint LDQ  = {ld_q};
constant constexpr uint LDK  = {ld_k};
constant constexpr uint LDV  = {ld_v};
constant constexpr float LOG2E = 1.44269504088896340736f;

struct SdpaParams {{
    uint q_seq;
    uint k_seq;
    uint hq;
    uint gqa;
    uint causal;
    float scale;
    uint q_s0; uint q_s1; uint q_s2; uint q_s3; uint q_off;
    uint k_s0; uint k_s1; uint k_s2; uint k_s3; uint k_off;
    uint v_s0; uint v_s1; uint v_s2; uint v_s3; uint v_off;
}};

// Apple 8x8 MMA fragment coord: lane L owns row `fm`, cols `fn`,`fn+1`.
static inline void frag_coord(uint lane, thread uint& fm, thread uint& fn) {{
    uint qid = lane / 4u;
    fm = (qid & 4u) + (lane / 2u) % 4u;
    fn = (qid & 2u) * 2u + (lane % 2u) * 2u;
}}

kernel void {entry}(
    device const {ty}* q     [[buffer(0)]],
    device const {ty}* k     [[buffer(1)]],
    device const {ty}* v     [[buffer(2)]],
    device {ty}* out         [[buffer(3)]],
    constant SdpaParams& p   [[buffer(4)]],
    uint3 tgpos [[threadgroup_position_in_grid]],   // x=q-block, y=h, z=b
    uint sg     [[simdgroup_index_in_threadgroup]], // 0..WM-1
    uint lane   [[thread_index_in_simdgroup]])      // 0..31
{{
    const uint qb = tgpos.x;        // query block index
    const uint h  = tgpos.y;
    const uint b  = tgpos.z;
    if (h >= p.hq) return;
    const uint kv_h = h / p.gqa;
    const uint q0 = qb * BQ;        // first query row of this block

    threadgroup {ty} Qs[BQ * LDQ];        // [BQ][DPAD]   (pre-scaled)
    threadgroup {ty} Ks[DPAD * LDK];      // [DPAD][BK]   (K transposed)
    threadgroup {ty} Vs[BK * LDV];        // [BK][DPAD]

    const uint q_base = p.q_off + b * p.q_s0 + h * p.q_s1;
    const uint k_base = p.k_off + b * p.k_s0 + kv_h * p.k_s1;
    const uint v_base = p.v_off + b * p.v_s0 + kv_h * p.v_s1;

    const uint tid = sg * 32u + lane;

    // ---- Stage the Q block, pre-scaled by scale*log2(e) (FA-2 exp2). ----
    // Qs[r][d] for r in [0,BQ), d in [0,DPAD).
    const {ty} qscale = ({ty})(p.scale * LOG2E);
    for (uint idx = tid; idx < BQ * DPAD; idx += TGP) {{
        uint r = idx / DPAD;
        uint d = idx % DPAD;
        uint qr = q0 + r;
        {ty} val = (qr < p.q_seq && d < D)
            ? ({ty})(q[q_base + qr * p.q_s2 + d * p.q_s3] * qscale)
            : ({ty})0;
        Qs[r * LDQ + d] = val;
    }}

    // Each simdgroup owns one 8-row Q fragment (TQ=1). `simdgroup_load`/`mma`
    // and `thread_elements()` all use Apple's native lane->element layout
    // (== MLX get_coord), so the per-lane row index for masking/softmax is
    // `sm` and the per-lane cols are `sn`,`sn+1`.
    uint sm, sn;
    frag_coord(lane, sm, sn);
    const uint row_base = sg * 8u;      // this simd's first query row in tile

    // Online-softmax running state for this lane's single row.
    float m_run = -INFINITY;
    float l_run = 0.0f;
    // O accumulator: TD float fragments, 2 elems/lane each.
    simdgroup_matrix<float, 8, 8> Oacc[TD];
#pragma clang loop unroll(full)
    for (uint i = 0; i < TD; i++) Oacc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);

    // Causal: query row r attends keys 0..=(k_seq - q_seq + r). Right-aligned
    // offset q_off = k_seq - q_seq (may be negative if k_seq<q_seq -> clamp 0).
    const int q_off = (int)p.k_seq - (int)p.q_seq;
    uint nk = (p.k_seq + BK - 1u) / BK;     // number of key blocks
    uint kb_lim = nk;
    if (p.causal != 0) {{
        // max key index any row in this block can attend = (q0+BQ-1)+q_off.
        int qmax = (int)(q0 + BQ - 1u) + q_off;
        if (qmax < 0) qmax = 0;
        kb_lim = (uint)((qmax + (int)BK) / (int)BK);  // ceil((qmax+1)/BK)
        if (kb_lim > nk) kb_lim = nk;
    }}

    for (uint kb = 0; kb < kb_lim; kb++) {{
        const uint k0 = kb * BK;        // first key of this block

        // ---- Stage K block transposed: Ks[d][kk] = K[k0+kk][d]. ----
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint idx = tid; idx < DPAD * BK; idx += TGP) {{
            uint d  = idx / BK;
            uint kk = idx % BK;
            uint kr = k0 + kk;
            {ty} val = (kr < p.k_seq && d < D)
                ? k[k_base + kr * p.k_s2 + d * p.k_s3]
                : ({ty})0;
            Ks[d * LDK + kk] = val;
        }}
        // ---- Stage V block: Vs[kk][d] = V[k0+kk][d]. ----
        for (uint idx = tid; idx < BK * DPAD; idx += TGP) {{
            uint kk = idx / DPAD;
            uint d  = idx % DPAD;
            uint kr = k0 + kk;
            {ty} val = (kr < p.k_seq && d < D)
                ? v[v_base + kr * p.v_s2 + d * p.v_s3]
                : ({ty})0;
            Vs[kk * LDV + d] = val;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- S = Q @ K^T  (contract over D). Stile[8 q][BK k]. ----
        simdgroup_matrix<float, 8, 8> Stile[TK];
#pragma clang loop unroll(full)
        for (uint j = 0; j < TK; j++) Stile[j] = make_filled_simdgroup_matrix<float, 8>(0.0f);

#pragma clang loop unroll(full)
        for (uint dd = 0; dd < TD; dd++) {{
            simdgroup_barrier(mem_flags::mem_none);
            // Q fragment [8 q][8 d]: base row `row_base`, col `dd*8` in Qs.
            simdgroup_matrix<{ty}, 8, 8> Qf;
            simdgroup_load(Qf, Qs + row_base * LDQ + dd * 8u, LDQ);
#pragma clang loop unroll(full)
            for (uint j = 0; j < TK; j++) {{
                // K fragment [8 d][8 k] from Ks (transposed staging).
                simdgroup_matrix<{ty}, 8, 8> Kf;
                simdgroup_load(Kf, Ks + dd * 8u * LDK + j * 8u, LDK);
                simdgroup_multiply_accumulate(Stile[j], Qf, Kf, Stile[j]);
            }}
        }}

        // ---- Per-element causal / length masking on Stile. ----
        const int row_pos = (int)(q0 + row_base + sm) + q_off;  // this lane's query key-index
        if (p.causal != 0 || (k0 + BK) > p.k_seq) {{
#pragma clang loop unroll(full)
            for (uint j = 0; j < TK; j++) {{
                int col0 = (int)(k0 + j * 8u + sn);
                thread auto& el = Stile[j].thread_elements();
#pragma clang loop unroll(full)
                for (uint jj = 0; jj < 2u; jj++) {{
                    int col = col0 + (int)jj;
                    bool masked = (col >= (int)p.k_seq)
                        || (p.causal != 0 && row_pos < col);
                    if (masked) el[jj] = -INFINITY;
                }}
            }}
        }}

        // ---- Online softmax (FA-2, base-2). Each lane owns one row. ----
        // Row max over this block's BK keys (this lane's 2*TK elements +
        // cross-column shuffle).
        float s_max = -INFINITY;
#pragma clang loop unroll(full)
        for (uint j = 0; j < TK; j++) {{
            thread const auto& el = Stile[j].thread_elements();
            s_max = max(s_max, max(el[0], el[1]));
        }}
        s_max = max(s_max, simd_shuffle_xor(s_max, 1u));
        s_max = max(s_max, simd_shuffle_xor(s_max, 8u));

        const float m_new = max(m_run, s_max);
        // P = exp2(S - m_new); guard fully-masked row (m_new == -inf -> 0).
        float p_sum = 0.0f;
#pragma clang loop unroll(full)
        for (uint j = 0; j < TK; j++) {{
            thread auto& el = Stile[j].thread_elements();
#pragma clang loop unroll(full)
            for (uint jj = 0; jj < 2u; jj++) {{
                float e = (m_new == -INFINITY) ? 0.0f : fast::exp2(el[jj] - m_new);
                el[jj] = e;
                p_sum += e;
            }}
        }}
        p_sum += simd_shuffle_xor(p_sum, 1u);
        p_sum += simd_shuffle_xor(p_sum, 8u);

        // Rescale running acc + l by exp2(m_old - m_new); guard m_old==-inf.
        const float corr = (m_run == -INFINITY) ? 0.0f : fast::exp2(m_run - m_new);
        l_run = l_run * corr + p_sum;
#pragma clang loop unroll(full)
        for (uint i = 0; i < TD; i++) {{
            thread auto& o = Oacc[i].thread_elements();
            o[0] *= corr;
            o[1] *= corr;
        }}
        m_run = m_new;

        // ---- O += P @ V  (contract over BK keys). ----
        // P (=Stile) is float; cast to a {ty} fragment for the MMA so V (dtype
        // staged) matches. Stile[j] is [8 q][8 k]; Vf is [8 k][8 d].
#pragma clang loop unroll(full)
        for (uint j = 0; j < TK; j++) {{
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_matrix<{ty}, 8, 8> Pf;
            {{
                thread const auto& sf = Stile[j].thread_elements();
                thread auto& pf = Pf.thread_elements();
                pf[0] = ({ty})sf[0];
                pf[1] = ({ty})sf[1];
            }}
#pragma clang loop unroll(full)
            for (uint dd = 0; dd < TD; dd++) {{
                simdgroup_matrix<{ty}, 8, 8> Vf;
                simdgroup_load(Vf, Vs + j * 8u * LDV + dd * 8u, LDV);
                simdgroup_multiply_accumulate(Oacc[dd], Pf, Vf, Oacc[dd]);
            }}
        }}
    }}

    // ---- Normalize O by l_run and store this lane's 2 elems per d-fragment. ----
    const float inv_l = (l_run > 0.0f) ? (1.0f / l_run) : 0.0f;
    const uint qr = q0 + row_base + sm;     // this lane's global query row
    if (qr >= p.q_seq) return;
    const uint out_base = ((b * p.hq + h) * p.q_seq + qr) * D;
#pragma clang loop unroll(full)
    for (uint dd = 0; dd < TD; dd++) {{
        thread const auto& o = Oacc[dd].thread_elements();
        uint d0 = dd * 8u + sn;
        if (d0 + 0u < D) out[out_base + d0 + 0u] = ({ty})(o[0] * inv_l);
        if (d0 + 1u < D) out[out_base + d0 + 1u] = ({ty})(o[1] * inv_l);
    }}
}}
"#
    )
}

/// Pick the split-K width `W` (simdgroups per query). Prefill (q_seq>8) uses
/// W=1 — the q_seq*Hq*B queries already saturate the GPU. Decode (small q_seq,
/// long context) splits the key loop across W simdgroups for parallelism.
/// Overridable via `KILN_SDPA_SPLIT` (for tuning). Always a power of two.
fn sdpa_split_w(q_seq: usize, k_seq: usize) -> usize {
    if let Ok(s) = std::env::var("KILN_SDPA_SPLIT") {
        if let Ok(w) = s.parse::<usize>() {
            return w.clamp(1, 32).next_power_of_two();
        }
    }
    if q_seq > 8 || k_seq < 256 {
        1
    } else {
        8
    }
}

/// Q-block / K-block / simdgroups-down-Q tiling for the matrix-core prefill
/// kernel ([`sdpa_steel_src`]) — MLX's `bd<512` pick (BQ=32, WM=4) with a
/// smaller K-block at large head-dim to keep the threadgroup-memory pool under
/// the M1's 32 KiB. (WN is always 1.) `dtype_bytes` shrinks `BK` further for
/// F32 (4 B/elem) so the `Qs+Ks+Vs` BF16/F32 staging pool stays under budget
/// even at `head_dim=128` (F32 D=128 would be 38 KiB at BK=16 → use BK=8).
fn sdpa_steel_cfg(head_dim: usize, dtype_bytes: usize) -> (usize, usize, usize) {
    let bk = if head_dim < 128 {
        32
    } else if dtype_bytes >= 4 {
        8
    } else {
        16
    };
    (32, bk, 4) // (BQ, BK, WM)
}

/// Threshold (in `q_seq`) at/above which the compute-bound **matrix-core
/// tiled** prefill path is used; below it the memory-bound simd_sum/split-K
/// decode path. Overridable via `KILN_SDPA_PREFILL_MIN` (for A/B + tuning).
fn sdpa_prefill_threshold() -> usize {
    if let Ok(s) = std::env::var("KILN_SDPA_PREFILL_MIN") {
        if let Ok(t) = s.parse::<usize>() {
            return t;
        }
    }
    16
}

/// Build the packed [`SdpaParams`] for a dispatch.
#[allow(clippy::too_many_arguments)]
fn build_sdpa_params(
    sq: usize,
    sk: usize,
    hq: usize,
    gqa: usize,
    causal: bool,
    scale: f32,
    q_strides: &[usize],
    q_offset: usize,
    k_strides: &[usize],
    k_offset: usize,
    v_strides: &[usize],
    v_offset: usize,
) -> SdpaParams {
    SdpaParams {
        q_seq: sq as u32,
        k_seq: sk as u32,
        hq: hq as u32,
        gqa: gqa as u32,
        causal: u32::from(causal),
        scale,
        q_s0: q_strides[0] as u32,
        q_s1: q_strides[1] as u32,
        q_s2: q_strides[2] as u32,
        q_s3: q_strides[3] as u32,
        q_off: q_offset as u32,
        k_s0: k_strides[0] as u32,
        k_s1: k_strides[1] as u32,
        k_s2: k_strides[2] as u32,
        k_s3: k_strides[3] as u32,
        k_off: k_offset as u32,
        v_s0: v_strides[0] as u32,
        v_s1: v_strides[1] as u32,
        v_s2: v_strides[2] as u32,
        v_s3: v_strides[3] as u32,
        v_off: v_offset as u32,
    }
}

/// Kiln-owned flash-attention SDPA — replaces
/// `candle_metal_kernels::call_sdpa_vector` / `_vector_2pass` / `_full`.
/// `q [B,Hq,Sq,D] @ k [B,Hkv,Sk,D]^T`, softmax, `@ v [B,Hkv,Sk,D]` →
/// `out [B,Hq,Sq,D]` (contiguous). GQA, causal, strided q/k/v; split-K decode.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sdpa(
    companion: &MetalCompanion,
    q: &MetalBuffer,
    k: &MetalBuffer,
    v: &MetalBuffer,
    out: &MetalBuffer,
    dtype: DType,
    b: usize,
    hq: usize,
    hkv: usize,
    sq: usize,
    sk: usize,
    head_dim: usize,
    scale: f32,
    causal: bool,
    q_strides: &[usize],
    q_offset: usize,
    k_strides: &[usize],
    k_offset: usize,
    v_strides: &[usize],
    v_offset: usize,
) -> Result<()> {
    if b == 0 || hq == 0 || sq == 0 || head_dim == 0 {
        return Ok(());
    }
    if hkv == 0 || hq % hkv != 0 {
        return Err(Error::Msg(format!(
            "metal_kernels::sdpa: invalid GQA (hq={hq}, hkv={hkv})"
        )));
    }
    if q_strides.len() != 4 || k_strides.len() != 4 || v_strides.len() != 4 {
        return Err(Error::Msg(
            "metal_kernels::sdpa: q/k/v strides must be rank-4".to_string(),
        ));
    }
    let ty = msl_ty(dtype)?;
    let gqa = hq / hkv;
    let params = build_sdpa_params(
        sq, sk, hq, gqa, causal, scale, q_strides, q_offset, k_strides, k_offset, v_strides,
        v_offset,
    );

    // PREFILL (large q_seq, compute-bound): matrix-core tiled flash-attention.
    // DECODE (small q_seq, memory-bound): the simd_sum/split-K kernel.
    if sq >= sdpa_prefill_threshold() {
        let dtype_bytes = dtype.size_in_bytes();
        return sdpa_dispatch_steel(
            companion,
            q,
            k,
            v,
            out,
            ty,
            dtype_bytes,
            b,
            hq,
            sq,
            head_dim,
            &params,
        );
    }

    let split_w = sdpa_split_w(sq, sk);
    let entry = format!("kt_sdpa_{ty}_d{head_dim}_w{split_w}");
    let src = sdpa_src(ty, head_dim, split_w, &entry);
    let pipeline = op_pipeline(companion, &src, &entry)?;

    let encoder = companion
        .command_encoder()
        .map_err(|e| Error::Msg(format!("metal_kernels::sdpa: encoder: {e:?}")))?;
    encoder.set_label("kt_sdpa");
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(q), 0);
    encoder.set_buffer(1, Some(k), 0);
    encoder.set_buffer(2, Some(v), 0);
    encoder.set_buffer(3, Some(out), 0);
    encoder.set_bytes(4, &params);
    if split_w > 1 {
        // shared: W*D acc + 2*W (m,l) floats.
        let shared_floats = split_w * head_dim + 2 * split_w;
        encoder.set_threadgroup_memory_length(0, shared_floats * std::mem::size_of::<f32>());
    }

    // W simdgroups (W*32 lanes) per (qi, head, batch).
    let groups = objc2_metal::MTLSize { width: sq, height: hq, depth: b };
    let threads = objc2_metal::MTLSize { width: split_w * 32, height: 1, depth: 1 };
    encoder.dispatch_thread_groups(groups, threads);
    drop(encoder);
    Ok(())
}

/// Dispatch the matrix-core tiled flash-attention prefill kernel
/// ([`sdpa_steel_src`]). One threadgroup (`WM` simdgroups = `WM*32` lanes)
/// computes one `BQ`-row query block for one (head, batch). Grid is
/// `(ceil(q_seq/BQ), hq, b)`. Threadgroup memory is the kernel's own
/// fixed-size `Qs/Ks/Vs` arrays (declared inside the MSL), so no
/// `set_threadgroup_memory_length` is needed.
#[allow(clippy::too_many_arguments)]
fn sdpa_dispatch_steel(
    companion: &MetalCompanion,
    q: &MetalBuffer,
    k: &MetalBuffer,
    v: &MetalBuffer,
    out: &MetalBuffer,
    ty: &str,
    dtype_bytes: usize,
    b: usize,
    hq: usize,
    sq: usize,
    head_dim: usize,
    params: &SdpaParams,
) -> Result<()> {
    let (bq, bk, wm) = sdpa_steel_cfg(head_dim, dtype_bytes);
    let entry = format!("kt_sdpa_steel_{ty}_d{head_dim}_bq{bq}_bk{bk}_wm{wm}");
    let src = sdpa_steel_src(ty, head_dim, bq, bk, wm, &entry);
    let pipeline = op_pipeline(companion, &src, &entry)?;

    let encoder = companion
        .command_encoder()
        .map_err(|e| Error::Msg(format!("metal_kernels::sdpa(steel): encoder: {e:?}")))?;
    encoder.set_label("kt_sdpa_steel");
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(q), 0);
    encoder.set_buffer(1, Some(k), 0);
    encoder.set_buffer(2, Some(v), 0);
    encoder.set_buffer(3, Some(out), 0);
    encoder.set_bytes(4, params);

    let groups = objc2_metal::MTLSize {
        width: sq.div_ceil(bq),
        height: hq,
        depth: b,
    };
    let threads = objc2_metal::MTLSize { width: wm * 32, height: 1, depth: 1 };
    encoder.dispatch_thread_groups(groups, threads);
    drop(encoder);
    Ok(())
}
