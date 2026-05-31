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
    let split_w = sdpa_split_w(sq, sk);
    let entry = format!("kt_sdpa_{ty}_d{head_dim}_w{split_w}");
    let src = sdpa_src(ty, head_dim, split_w, &entry);
    let pipeline = op_pipeline(companion, &src, &entry)?;

    let gqa = hq / hkv;
    let params = SdpaParams {
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
    };

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
