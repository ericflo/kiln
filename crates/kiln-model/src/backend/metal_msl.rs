//! Metal shader source strings.
//!
//! These raw MSL snippets are compiled into the shared Metal library by
//! `backend::metal`. Keeping source text separate from dispatch code makes
//! the pipeline/source boundary explicit without changing pipeline cache
//! behavior.

pub(super) const METAL_RMSNORM_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kiln_rmsnorm_bf16(
    device const bfloat* x [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& hidden [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    constant uint& threadgroup_width [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float scratch[1024];

    const uint row = gid.y;
    if (row >= rows) {
        return;
    }

    const uint base = row * hidden;
    float sum_sq = 0.0f;
    for (uint col = tid; col < hidden; col += threadgroup_width) {
        const float xv = static_cast<float>(x[base + col]);
        sum_sq += xv * xv;
    }
    scratch[tid] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threadgroup_width / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float rms_inv = rsqrt((scratch[0] / static_cast<float>(hidden)) + eps);
    for (uint col = tid; col < hidden; col += threadgroup_width) {
        const float xv = static_cast<float>(x[base + col]);
        const float scale = 1.0f + static_cast<float>(weight[col]);
        out[base + col] = static_cast<bfloat>(xv * rms_inv * scale);
    }
}
"#;
pub(super) const METAL_ROTARY_QK_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kiln_rotary_qk_bf16(
    device const bfloat* q [[buffer(0)]],
    device const bfloat* k [[buffer(1)]],
    device const float* cos [[buffer(2)]],
    device const float* sin [[buffer(3)]],
    device bfloat* q_out [[buffer(4)]],
    device bfloat* k_out [[buffer(5)]],
    constant uint& batch [[buffer(6)]],
    constant uint& seq_len [[buffer(7)]],
    constant uint& q_heads [[buffer(8)]],
    constant uint& k_heads [[buffer(9)]],
    constant uint& head_dim [[buffer(10)]],
    constant uint& rotary_dim [[buffer(11)]],
    constant uint& total_q [[buffer(12)]],
    constant uint& total [[buffer(13)]],
    constant uint& table_batch_stride [[buffer(14)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= total) {
        return;
    }

    const bool is_q = gid < total_q;
    const uint local = is_q ? gid : gid - total_q;
    const uint heads = is_q ? q_heads : k_heads;
    device const bfloat* src = is_q ? q : k;
    device bfloat* dst = is_q ? q_out : k_out;

    const uint d = local % head_dim;
    const uint h = (local / head_dim) % heads;
    const uint t = (local / (head_dim * heads)) % seq_len;
    const uint b = local / (head_dim * heads * seq_len);
    if (b >= batch) {
        return;
    }

    if (d >= rotary_dim) {
        dst[local] = src[local];
        return;
    }

    const uint half_rotary = rotary_dim / 2;
    const bool first_half = d < half_rotary;
    const uint pair_d = first_half ? d + half_rotary : d - half_rotary;
    const uint pair_idx = ((b * seq_len + t) * heads + h) * head_dim + pair_d;
    const uint table_t = table_batch_stride == 0 ? t : b * table_batch_stride + t;
    const uint table_idx = table_t * half_rotary + (first_half ? d : pair_d);
    const float x = static_cast<float>(src[local]);
    const float y = static_cast<float>(src[pair_idx]);
    const float c = cos[table_idx];
    const float s = sin[table_idx];
    const float rotated = first_half ? (x * c - y * s) : (y * s + x * c);
    dst[local] = static_cast<bfloat>(rotated);
}
"#;
pub(super) const METAL_GDN_QK_NORM_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kiln_gdn_qk_norm_f32_bf16(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device bfloat* q_out [[buffer(2)]],
    device bfloat* k_out [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& hidden [[buffer(5)]],
    constant float& q_scale [[buffer(6)]],
    constant float& eps [[buffer(7)]],
    constant uint& threadgroup_width [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float q_scratch[1024];
    threadgroup float k_scratch[1024];

    const uint row = gid.y;
    if (row >= rows) {
        return;
    }

    const uint base = row * hidden;
    float q_sum_sq = 0.0f;
    float k_sum_sq = 0.0f;
    for (uint col = tid; col < hidden; col += threadgroup_width) {
        const float qv = q[base + col];
        const float kv = k[base + col];
        q_sum_sq += qv * qv;
        k_sum_sq += kv * kv;
    }
    q_scratch[tid] = q_sum_sq;
    k_scratch[tid] = k_sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threadgroup_width / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            q_scratch[tid] += q_scratch[tid + stride];
            k_scratch[tid] += k_scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float q_inv = rsqrt(q_scratch[0] + eps);
    const float k_inv = rsqrt(k_scratch[0] + eps);
    for (uint col = tid; col < hidden; col += threadgroup_width) {
        const uint idx = base + col;
        q_out[idx] = static_cast<bfloat>(q[idx] * q_inv * q_scale);
        k_out[idx] = static_cast<bfloat>(k[idx] * k_inv);
    }
}

kernel void kiln_gdn_qk_norm_gqa_f32_bf16(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device bfloat* q_out [[buffer(2)]],
    device bfloat* k_out [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& nk [[buffer(5)]],
    constant uint& nv [[buffer(6)]],
    constant uint& hidden [[buffer(7)]],
    constant uint& gqa_ratio [[buffer(8)]],
    constant float& q_scale [[buffer(9)]],
    constant float& eps [[buffer(10)]],
    constant uint& threadgroup_width [[buffer(11)]],
    uint2 gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float q_scratch[1024];
    threadgroup float k_scratch[1024];

    const uint row = gid.y;
    if (row >= rows) {
        return;
    }

    const uint src_head = row % nk;
    const uint bt = row / nk;
    const uint base = row * hidden;
    float q_sum_sq = 0.0f;
    float k_sum_sq = 0.0f;
    for (uint col = tid; col < hidden; col += threadgroup_width) {
        const float qv = q[base + col];
        const float kv = k[base + col];
        q_sum_sq += qv * qv;
        k_sum_sq += kv * kv;
    }
    q_scratch[tid] = q_sum_sq;
    k_scratch[tid] = k_sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threadgroup_width / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            q_scratch[tid] += q_scratch[tid + stride];
            k_scratch[tid] += k_scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float q_inv = rsqrt(q_scratch[0] + eps);
    const float k_inv = rsqrt(k_scratch[0] + eps);
    const uint dst_head_base = src_head * gqa_ratio;
    for (uint col = tid; col < hidden; col += threadgroup_width) {
        const float q_norm = q[base + col] * q_inv * q_scale;
        const float k_norm = k[base + col] * k_inv;
        for (uint rep = 0; rep < gqa_ratio; ++rep) {
            const uint dst_head = dst_head_base + rep;
            const uint dst_idx = ((bt * nv + dst_head) * hidden) + col;
            q_out[dst_idx] = static_cast<bfloat>(q_norm);
            k_out[dst_idx] = static_cast<bfloat>(k_norm);
        }
    }
}
"#;
pub(super) const METAL_GDN_DECODE_QKV_CONV_NORM_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kiln_gdn_decode_qkv_conv_norm_bf16(
    device const bfloat* mixed_qkv [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device float* conv_state [[buffer(2)]],
    device bfloat* q_out [[buffer(3)]],
    device bfloat* k_out [[buffer(4)]],
    device bfloat* v_out [[buffer(5)]],
    constant uint& nk [[buffer(6)]],
    constant uint& nv [[buffer(7)]],
    constant float& q_scale [[buffer(8)]],
    constant float& eps [[buffer(9)]],
    uint2 tgroup [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    constexpr uint D = 128;
    threadgroup float values[D];
    threadgroup float sum_scratch[D];

    const uint row = tgroup.x;
    const uint batch_idx = tgroup.y;
    const uint local_row = row;
    const uint qk_dim = nk * D;
    const uint channels = qk_dim + qk_dim + nv * D;

    uint channel = 0;
    bool is_q = false;
    bool is_k = false;
    bool is_v = false;
    uint src_head = 0;
    uint v_head = 0;

    if (local_row < nk) {
        is_q = true;
        src_head = local_row;
        channel = src_head * D + tid;
    } else if (local_row < nk + nk) {
        is_k = true;
        src_head = local_row - nk;
        channel = qk_dim + src_head * D + tid;
    } else {
        is_v = true;
        v_head = local_row - nk - nk;
        channel = qk_dim + qk_dim + v_head * D + tid;
    }

    const uint token_idx = batch_idx * channels + channel;
    const uint state_base = (batch_idx * channels + channel) * 3;
    const uint weight_base = channel * 4;

    const float s0 = conv_state[state_base + 0];
    const float s1 = conv_state[state_base + 1];
    const float s2 = conv_state[state_base + 2];
    const float x0 = static_cast<float>(mixed_qkv[token_idx]);
    const float acc =
        s0 * static_cast<float>(weight[weight_base + 0]) +
        s1 * static_cast<float>(weight[weight_base + 1]) +
        s2 * static_cast<float>(weight[weight_base + 2]) +
        x0 * static_cast<float>(weight[weight_base + 3]);
    const float y = acc / (1.0f + exp(-acc));

    conv_state[state_base + 0] = s1;
    conv_state[state_base + 1] = s2;
    conv_state[state_base + 2] = x0;

    if (is_v) {
        const uint out_idx = (batch_idx * nv + v_head) * D + tid;
        v_out[out_idx] = static_cast<bfloat>(y);
        return;
    }

    values[tid] = y;
    sum_scratch[tid] = y * y;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = D / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sum_scratch[tid] += sum_scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inv = rsqrt(sum_scratch[0] + eps);
    const float norm = values[tid] * inv * (is_q ? q_scale : 1.0f);
    const uint dst_idx = (batch_idx * nk + src_head) * D + tid;
    if (is_q) {
        q_out[dst_idx] = static_cast<bfloat>(norm);
    } else if (is_k) {
        k_out[dst_idx] = static_cast<bfloat>(norm);
    }
}
"#;
pub(super) const METAL_LM_HEAD_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kiln_lm_head_bf16(
    device const bfloat* x [[buffer(0)]],
    device const bfloat* weight_t [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    constant uint& hidden [[buffer(3)]],
    constant uint& vocab [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= vocab) {
        return;
    }

    float acc = 0.0f;
    for (uint i = 0; i < hidden; ++i) {
        acc += static_cast<float>(x[i]) * static_cast<float>(weight_t[i * vocab + gid]);
    }
    out[gid] = static_cast<bfloat>(acc);
}

kernel void kiln_lm_head_argmax_chunks_bf16(
    device const bfloat* x [[buffer(0)]],
    device const bfloat* weight_t [[buffer(1)]],
    device float* partial_scores [[buffer(2)]],
    device float* partial_indices [[buffer(3)]],
    constant uint& hidden [[buffer(4)]],
    constant uint& vocab [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]
) {
    threadgroup float scores[256];
    threadgroup float indices[256];

    const uint col = group * 256 + tid;
    float score = -INFINITY;
    float index = 0.0f;
    if (col < vocab) {
        float acc = 0.0f;
        for (uint i = 0; i < hidden; ++i) {
            acc += static_cast<float>(x[i]) * static_cast<float>(weight_t[i * vocab + col]);
        }
        score = static_cast<float>(static_cast<bfloat>(acc));
        index = static_cast<float>(col);
    }
    scores[tid] = score;
    indices[tid] = index;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float other_score = scores[tid + stride];
            const float other_index = indices[tid + stride];
            if (other_score > scores[tid] ||
                (other_score == scores[tid] && other_index < indices[tid])) {
                scores[tid] = other_score;
                indices[tid] = other_index;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        partial_scores[group] = scores[0];
        partial_indices[group] = indices[0];
    }
}

kernel void kiln_lm_head_argmax_reduce_f32(
    device const float* partial_scores [[buffer(0)]],
    device const float* partial_indices [[buffer(1)]],
    device float* final_index [[buffer(2)]],
    constant uint& num_groups [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float scores[1024];
    threadgroup float indices[1024];

    float score = -INFINITY;
    float index = 0.0f;
    if (tid < num_groups) {
        score = partial_scores[tid];
        index = partial_indices[tid];
    }
    scores[tid] = score;
    indices[tid] = index;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 512; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float other_score = scores[tid + stride];
            const float other_index = indices[tid + stride];
            if (other_score > scores[tid] ||
                (other_score == scores[tid] && other_index < indices[tid])) {
                scores[tid] = other_score;
                indices[tid] = other_index;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        final_index[0] = indices[0];
    }
}

kernel void kiln_lm_head_argmax_chunks_batch_bf16(
    device const bfloat* x [[buffer(0)]],
    device const bfloat* weight_t [[buffer(1)]],
    device float* partial_scores [[buffer(2)]],
    device float* partial_indices [[buffer(3)]],
    constant uint& hidden [[buffer(4)]],
    constant uint& vocab [[buffer(5)]],
    constant uint& num_groups [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 group_pos [[threadgroup_position_in_grid]]
) {
    threadgroup float scores[256];
    threadgroup float indices[256];

    const uint group = group_pos.x;
    const uint row = group_pos.y;
    const uint col = group * 256 + tid;
    float score = -INFINITY;
    float index = 0.0f;
    if (col < vocab) {
        float acc = 0.0f;
        const device bfloat* row_x = x + row * hidden;
        for (uint i = 0; i < hidden; ++i) {
            acc += static_cast<float>(row_x[i]) * static_cast<float>(weight_t[i * vocab + col]);
        }
        score = static_cast<float>(static_cast<bfloat>(acc));
        index = static_cast<float>(col);
    }
    scores[tid] = score;
    indices[tid] = index;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float other_score = scores[tid + stride];
            const float other_index = indices[tid + stride];
            if (other_score > scores[tid] ||
                (other_score == scores[tid] && other_index < indices[tid])) {
                scores[tid] = other_score;
                indices[tid] = other_index;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        const uint offset = row * num_groups + group;
        partial_scores[offset] = scores[0];
        partial_indices[offset] = indices[0];
    }
}

kernel void kiln_lm_head_argmax_reduce_batch_f32(
    device const float* partial_scores [[buffer(0)]],
    device const float* partial_indices [[buffer(1)]],
    device float* final_indices [[buffer(2)]],
    constant uint& num_groups [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]]
) {
    threadgroup float scores[1024];
    threadgroup float indices[1024];

    float score = -INFINITY;
    float index = 0.0f;
    if (tid < num_groups) {
        const uint offset = row * num_groups + tid;
        score = partial_scores[offset];
        index = partial_indices[offset];
    }
    scores[tid] = score;
    indices[tid] = index;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 512; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float other_score = scores[tid + stride];
            const float other_index = indices[tid + stride];
            if (other_score > scores[tid] ||
                (other_score == scores[tid] && other_index < indices[tid])) {
                scores[tid] = other_score;
                indices[tid] = other_index;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        final_indices[row] = indices[0];
    }
}

#define KILN_SAMPLE_TOPK_MAX 64

inline bool kiln_score_better(float score, float index, float best_score, float best_index) {
    return score > best_score || (score == best_score && index < best_index);
}

inline bool kiln_history_count_for_token(
    device const uint* history_indices,
    device const uint* history_counts,
    uint history_len,
    uint token,
    thread uint& count
) {
    uint lo = 0;
    uint hi = history_len;
    while (lo < hi) {
        const uint mid = lo + ((hi - lo) >> 1);
        const uint value = history_indices[mid];
        if (value < token) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if (lo < history_len && history_indices[lo] == token) {
        count = history_counts[lo];
        return true;
    }
    count = 0;
    return false;
}

inline float kiln_apply_sample_penalties(
    float score,
    uint token,
    device const uint* history_indices,
    device const uint* history_counts,
    uint history_len,
    float repetition_penalty,
    float presence_penalty,
    float frequency_penalty
) {
    uint count = 0;
    if (!kiln_history_count_for_token(history_indices, history_counts, history_len, token, count)) {
        return score;
    }
    if (isfinite(repetition_penalty) && repetition_penalty > 0.0f &&
        fabs(repetition_penalty - 1.0f) > 0.00000011920929f) {
        score = score > 0.0f ? score / repetition_penalty : score * repetition_penalty;
    }
    if (isfinite(presence_penalty) && presence_penalty != 0.0f) {
        score -= presence_penalty;
    }
    if (isfinite(frequency_penalty) && frequency_penalty != 0.0f) {
        score -= frequency_penalty * static_cast<float>(count);
    }
    return score;
}

inline ulong kiln_splitmix64_next(thread ulong& state) {
    state += 0x9E3779B97F4A7C15ul;
    ulong z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ul;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBul;
    return z ^ (z >> 31);
}

inline float kiln_uniform01_from_seed(uint seed_lo, uint seed_hi) {
    ulong state = (static_cast<ulong>(seed_hi) << 32) | static_cast<ulong>(seed_lo);
    const ulong bits = kiln_splitmix64_next(state);
    const uint mantissa = static_cast<uint>((bits >> 40) & 0xFFFFFFul);
    return static_cast<float>(mantissa) / 16777216.0f;
}

kernel void kiln_lm_head_sample_topk_chunks_bf16(
    device const bfloat* x [[buffer(0)]],
    device const bfloat* weight_t [[buffer(1)]],
    device const uint* history_indices [[buffer(2)]],
    device const uint* history_counts [[buffer(3)]],
    device float* partial_scores [[buffer(4)]],
    device float* partial_indices [[buffer(5)]],
    constant uint& hidden [[buffer(6)]],
    constant uint& vocab [[buffer(7)]],
    constant uint& history_len [[buffer(8)]],
    constant float& repetition_penalty [[buffer(9)]],
    constant float& presence_penalty [[buffer(10)]],
    constant float& frequency_penalty [[buffer(11)]],
    constant float& inv_temperature [[buffer(12)]],
    constant uint& top_k [[buffer(13)]],
    uint tid [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]
) {
    threadgroup float scores[256];
    threadgroup float indices[256];

    const uint col = group * 256 + tid;
    float score = -INFINITY;
    float index = 0.0f;
    if (col < vocab) {
        float acc = 0.0f;
        for (uint i = 0; i < hidden; ++i) {
            acc += static_cast<float>(x[i]) * static_cast<float>(weight_t[i * vocab + col]);
        }
        score = static_cast<float>(static_cast<bfloat>(acc));
        score = kiln_apply_sample_penalties(
            score,
            col,
            history_indices,
            history_counts,
            history_len,
            repetition_penalty,
            presence_penalty,
            frequency_penalty
        );
        score *= inv_temperature;
        index = static_cast<float>(col);
    }
    scores[tid] = score;
    indices[tid] = index;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        const uint out_base = group * top_k;
        const uint k_limit = min(top_k, static_cast<uint>(KILN_SAMPLE_TOPK_MAX));
        for (uint k = 0; k < k_limit; ++k) {
            float best_score = -INFINITY;
            float best_index = 0.0f;
            uint best_pos = 0;
            for (uint i = 0; i < 256; ++i) {
                const float candidate_score = scores[i];
                const float candidate_index = indices[i];
                if (kiln_score_better(candidate_score, candidate_index, best_score, best_index)) {
                    best_score = candidate_score;
                    best_index = candidate_index;
                    best_pos = i;
                }
            }
            partial_scores[out_base + k] = best_score;
            partial_indices[out_base + k] = best_index;
            scores[best_pos] = -INFINITY;
        }
    }
}

kernel void kiln_lm_head_sample_reduce_f32(
    device const float* partial_scores [[buffer(0)]],
    device const float* partial_indices [[buffer(1)]],
    device float* final_index [[buffer(2)]],
    constant uint& num_groups [[buffer(3)]],
    constant uint& top_k [[buffer(4)]],
    constant float& top_p [[buffer(5)]],
    constant float& min_p [[buffer(6)]],
    constant uint& seed_lo [[buffer(7)]],
    constant uint& seed_hi [[buffer(8)]],
    uint tid [[thread_index_in_threadgroup]]
) {
    if (tid != 0) {
        return;
    }

    float top_scores[KILN_SAMPLE_TOPK_MAX];
    float top_indices[KILN_SAMPLE_TOPK_MAX];
    float probs[KILN_SAMPLE_TOPK_MAX];
    const uint k_limit = min(top_k, static_cast<uint>(KILN_SAMPLE_TOPK_MAX));
    for (uint i = 0; i < KILN_SAMPLE_TOPK_MAX; ++i) {
        top_scores[i] = -INFINITY;
        top_indices[i] = 0.0f;
        probs[i] = 0.0f;
    }

    const uint candidate_count = num_groups * k_limit;
    for (uint c = 0; c < candidate_count; ++c) {
        const float score = partial_scores[c];
        const float index = partial_indices[c];
        if (!isfinite(score)) {
            continue;
        }
        for (uint pos = 0; pos < k_limit; ++pos) {
            if (kiln_score_better(score, index, top_scores[pos], top_indices[pos])) {
                for (uint shift = k_limit - 1; shift > pos; --shift) {
                    top_scores[shift] = top_scores[shift - 1];
                    top_indices[shift] = top_indices[shift - 1];
                }
                top_scores[pos] = score;
                top_indices[pos] = index;
                break;
            }
        }
    }

    if (k_limit == 0 || !isfinite(top_scores[0])) {
        final_index[0] = 0.0f;
        return;
    }
    if (k_limit == 1) {
        final_index[0] = top_indices[0];
        return;
    }

    const float max_score = top_scores[0];
    float sum = 0.0f;
    for (uint i = 0; i < k_limit; ++i) {
        if (isfinite(top_scores[i])) {
            const float p = exp(top_scores[i] - max_score);
            probs[i] = p;
            sum += p;
        }
    }
    if (!isfinite(sum) || sum <= 0.0f) {
        final_index[0] = top_indices[0];
        return;
    }
    for (uint i = 0; i < k_limit; ++i) {
        probs[i] /= sum;
    }

    if (isfinite(min_p) && min_p > 0.0f) {
        const float threshold = min_p * probs[0];
        float filtered_sum = 0.0f;
        for (uint i = 0; i < k_limit; ++i) {
            if (probs[i] < threshold) {
                probs[i] = 0.0f;
            }
            filtered_sum += probs[i];
        }
        if (filtered_sum <= 0.0f || !isfinite(filtered_sum)) {
            final_index[0] = top_indices[0];
            return;
        }
        for (uint i = 0; i < k_limit; ++i) {
            probs[i] /= filtered_sum;
        }
    }

    if (top_p > 0.0f && top_p < 1.0f) {
        float cumsum = 0.0f;
        uint cutoff = k_limit;
        for (uint i = 0; i < k_limit; ++i) {
            cumsum += probs[i];
            if (cumsum >= top_p) {
                cutoff = i + 1;
                break;
            }
        }
        float filtered_sum = 0.0f;
        for (uint i = 0; i < k_limit; ++i) {
            if (i >= cutoff) {
                probs[i] = 0.0f;
            }
            filtered_sum += probs[i];
        }
        if (filtered_sum <= 0.0f || !isfinite(filtered_sum)) {
            final_index[0] = top_indices[0];
            return;
        }
        for (uint i = 0; i < k_limit; ++i) {
            probs[i] /= filtered_sum;
        }
    }

    const float r = kiln_uniform01_from_seed(seed_lo, seed_hi);
    float cumsum = 0.0f;
    for (uint i = 0; i < k_limit; ++i) {
        cumsum += probs[i];
        if (r < cumsum) {
            final_index[0] = top_indices[i];
            return;
        }
    }
    for (uint i = k_limit; i > 0; --i) {
        if (probs[i - 1] > 0.0f) {
            final_index[0] = top_indices[i - 1];
            return;
        }
    }
    final_index[0] = top_indices[0];
}
"#;
pub(super) const METAL_MLP_GATE_UP_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kiln_mlp_gate_up_bf16(
    device const bfloat* x [[buffer(0)]],
    device const bfloat* gate_t [[buffer(1)]],
    device const bfloat* up_t [[buffer(2)]],
    device bfloat* out [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& hidden [[buffer(5)]],
    constant uint& intermediate [[buffer(6)]],
    constant uint& row_pair_mode [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint cols2 = (intermediate + 1) >> 1;
    if (row_pair_mode == 0 || row_pair_mode == 6) {
        const uint total = rows * cols2;
        if (gid >= total) {
            return;
        }

        const uint row = gid / cols2;
        const uint col0 = (gid - row * cols2) << 1;
        const uint col1 = col0 + 1;
        const bool has_col1 = col1 < intermediate;
        const uint x_base = row * hidden;
        float gate_acc0 = 0.0f;
        float up_acc0 = 0.0f;
        float gate_acc1 = 0.0f;
        float up_acc1 = 0.0f;
        if (row_pair_mode == 6) {
            for (uint i = 0; i < hidden; ++i) {
                const float xv = static_cast<float>(x[x_base + i]);
                const uint w_idx0 = i * intermediate + col0;
                const bfloat2 gate_w = *(device const bfloat2*)(gate_t + w_idx0);
                const bfloat2 up_w = *(device const bfloat2*)(up_t + w_idx0);
                gate_acc0 += xv * static_cast<float>(gate_w[0]);
                up_acc0 += xv * static_cast<float>(up_w[0]);
                gate_acc1 += xv * static_cast<float>(gate_w[1]);
                up_acc1 += xv * static_cast<float>(up_w[1]);
            }
        } else {
            for (uint i = 0; i < hidden; ++i) {
                const float xv = static_cast<float>(x[x_base + i]);
                const uint w_idx0 = i * intermediate + col0;
                gate_acc0 += xv * static_cast<float>(gate_t[w_idx0]);
                up_acc0 += xv * static_cast<float>(up_t[w_idx0]);
                if (has_col1) {
                    const uint w_idx1 = w_idx0 + 1;
                    gate_acc1 += xv * static_cast<float>(gate_t[w_idx1]);
                    up_acc1 += xv * static_cast<float>(up_t[w_idx1]);
                }
            }
        }

        const uint out_base = row * intermediate;
        const float gate_sigmoid0 = 1.0f / (1.0f + exp(-gate_acc0));
        out[out_base + col0] = static_cast<bfloat>((gate_acc0 * gate_sigmoid0) * up_acc0);
        if (has_col1) {
            const float gate_sigmoid1 = 1.0f / (1.0f + exp(-gate_acc1));
            out[out_base + col1] = static_cast<bfloat>((gate_acc1 * gate_sigmoid1) * up_acc1);
        }
        return;
    }

    if (row_pair_mode == 3 || row_pair_mode == 7) {
        const uint total = cols2;
        if (gid >= total) {
            return;
        }

        const uint col0 = gid << 1;
        const uint col1 = col0 + 1;
        const bool has_col1 = col1 < intermediate;
        const uint x_base1 = hidden;
        const uint x_base2 = hidden << 1;
        float gate_acc00 = 0.0f;
        float up_acc00 = 0.0f;
        float gate_acc01 = 0.0f;
        float up_acc01 = 0.0f;
        float gate_acc10 = 0.0f;
        float up_acc10 = 0.0f;
        float gate_acc11 = 0.0f;
        float up_acc11 = 0.0f;
        float gate_acc20 = 0.0f;
        float up_acc20 = 0.0f;
        float gate_acc21 = 0.0f;
        float up_acc21 = 0.0f;
        if (row_pair_mode == 7) {
            for (uint i = 0; i < hidden; ++i) {
                const uint w_idx0 = i * intermediate + col0;
                const bfloat2 gate_w = *(device const bfloat2*)(gate_t + w_idx0);
                const bfloat2 up_w = *(device const bfloat2*)(up_t + w_idx0);
                const float gate_w0 = static_cast<float>(gate_w[0]);
                const float gate_w1 = static_cast<float>(gate_w[1]);
                const float up_w0 = static_cast<float>(up_w[0]);
                const float up_w1 = static_cast<float>(up_w[1]);

                const float xv0 = static_cast<float>(x[i]);
                gate_acc00 += xv0 * gate_w0;
                up_acc00 += xv0 * up_w0;
                gate_acc01 += xv0 * gate_w1;
                up_acc01 += xv0 * up_w1;

                const float xv1 = static_cast<float>(x[x_base1 + i]);
                gate_acc10 += xv1 * gate_w0;
                up_acc10 += xv1 * up_w0;
                gate_acc11 += xv1 * gate_w1;
                up_acc11 += xv1 * up_w1;

                const float xv2 = static_cast<float>(x[x_base2 + i]);
                gate_acc20 += xv2 * gate_w0;
                up_acc20 += xv2 * up_w0;
                gate_acc21 += xv2 * gate_w1;
                up_acc21 += xv2 * up_w1;
            }
        } else {
            for (uint i = 0; i < hidden; ++i) {
                const uint w_idx0 = i * intermediate + col0;
                const float gate_w0 = static_cast<float>(gate_t[w_idx0]);
                const float up_w0 = static_cast<float>(up_t[w_idx0]);
                float gate_w1 = 0.0f;
                float up_w1 = 0.0f;
                if (has_col1) {
                    const uint w_idx1 = w_idx0 + 1;
                    gate_w1 = static_cast<float>(gate_t[w_idx1]);
                    up_w1 = static_cast<float>(up_t[w_idx1]);
                }

                const float xv0 = static_cast<float>(x[i]);
                gate_acc00 += xv0 * gate_w0;
                up_acc00 += xv0 * up_w0;
                if (has_col1) {
                    gate_acc01 += xv0 * gate_w1;
                    up_acc01 += xv0 * up_w1;
                }

                const float xv1 = static_cast<float>(x[x_base1 + i]);
                gate_acc10 += xv1 * gate_w0;
                up_acc10 += xv1 * up_w0;
                if (has_col1) {
                    gate_acc11 += xv1 * gate_w1;
                    up_acc11 += xv1 * up_w1;
                }

                const float xv2 = static_cast<float>(x[x_base2 + i]);
                gate_acc20 += xv2 * gate_w0;
                up_acc20 += xv2 * up_w0;
                if (has_col1) {
                    gate_acc21 += xv2 * gate_w1;
                    up_acc21 += xv2 * up_w1;
                }
            }
        }

        const float gate_sigmoid00 = 1.0f / (1.0f + exp(-gate_acc00));
        out[col0] = static_cast<bfloat>((gate_acc00 * gate_sigmoid00) * up_acc00);
        if (has_col1) {
            const float gate_sigmoid01 = 1.0f / (1.0f + exp(-gate_acc01));
            out[col1] = static_cast<bfloat>((gate_acc01 * gate_sigmoid01) * up_acc01);
        }

        const uint out_base1 = intermediate;
        const float gate_sigmoid10 = 1.0f / (1.0f + exp(-gate_acc10));
        out[out_base1 + col0] = static_cast<bfloat>((gate_acc10 * gate_sigmoid10) * up_acc10);
        if (has_col1) {
            const float gate_sigmoid11 = 1.0f / (1.0f + exp(-gate_acc11));
            out[out_base1 + col1] = static_cast<bfloat>((gate_acc11 * gate_sigmoid11) * up_acc11);
        }

        const uint out_base2 = intermediate << 1;
        const float gate_sigmoid20 = 1.0f / (1.0f + exp(-gate_acc20));
        out[out_base2 + col0] = static_cast<bfloat>((gate_acc20 * gate_sigmoid20) * up_acc20);
        if (has_col1) {
            const float gate_sigmoid21 = 1.0f / (1.0f + exp(-gate_acc21));
            out[out_base2 + col1] = static_cast<bfloat>((gate_acc21 * gate_sigmoid21) * up_acc21);
        }
        return;
    }

    if (row_pair_mode == 4 || row_pair_mode == 5) {
        const uint row_quads = (rows + 3) >> 2;
        const uint total = row_quads * cols2;
        if (gid >= total) {
            return;
        }

        const uint row_quad = gid / cols2;
        const uint row0 = row_quad << 2;
        const uint row1 = row0 + 1;
        const uint row2 = row0 + 2;
        const uint row3 = row0 + 3;
        const bool has_row1 = row1 < rows;
        const bool has_row2 = row2 < rows;
        const bool has_row3 = row3 < rows;
        const uint col0 = (gid - row_quad * cols2) << 1;
        const uint col1 = col0 + 1;
        const bool has_col1 = col1 < intermediate;
        const uint x_base0 = row0 * hidden;
        const uint x_base1 = row1 * hidden;
        const uint x_base2 = row2 * hidden;
        const uint x_base3 = row3 * hidden;
        float gate_acc00 = 0.0f;
        float up_acc00 = 0.0f;
        float gate_acc01 = 0.0f;
        float up_acc01 = 0.0f;
        float gate_acc10 = 0.0f;
        float up_acc10 = 0.0f;
        float gate_acc11 = 0.0f;
        float up_acc11 = 0.0f;
        float gate_acc20 = 0.0f;
        float up_acc20 = 0.0f;
        float gate_acc21 = 0.0f;
        float up_acc21 = 0.0f;
        float gate_acc30 = 0.0f;
        float up_acc30 = 0.0f;
        float gate_acc31 = 0.0f;
        float up_acc31 = 0.0f;
        if (row_pair_mode == 5) {
            for (uint i = 0; i < hidden; ++i) {
                const uint w_idx0 = i * intermediate + col0;
                const bfloat2 gate_w = *(device const bfloat2*)(gate_t + w_idx0);
                const bfloat2 up_w = *(device const bfloat2*)(up_t + w_idx0);
                const float gate_w0 = static_cast<float>(gate_w[0]);
                const float gate_w1 = static_cast<float>(gate_w[1]);
                const float up_w0 = static_cast<float>(up_w[0]);
                const float up_w1 = static_cast<float>(up_w[1]);

                const float xv0 = static_cast<float>(x[x_base0 + i]);
                gate_acc00 += xv0 * gate_w0;
                up_acc00 += xv0 * up_w0;
                gate_acc01 += xv0 * gate_w1;
                up_acc01 += xv0 * up_w1;

                if (has_row1) {
                    const float xv1 = static_cast<float>(x[x_base1 + i]);
                    gate_acc10 += xv1 * gate_w0;
                    up_acc10 += xv1 * up_w0;
                    gate_acc11 += xv1 * gate_w1;
                    up_acc11 += xv1 * up_w1;
                }
                if (has_row2) {
                    const float xv2 = static_cast<float>(x[x_base2 + i]);
                    gate_acc20 += xv2 * gate_w0;
                    up_acc20 += xv2 * up_w0;
                    gate_acc21 += xv2 * gate_w1;
                    up_acc21 += xv2 * up_w1;
                }
                if (has_row3) {
                    const float xv3 = static_cast<float>(x[x_base3 + i]);
                    gate_acc30 += xv3 * gate_w0;
                    up_acc30 += xv3 * up_w0;
                    gate_acc31 += xv3 * gate_w1;
                    up_acc31 += xv3 * up_w1;
                }
            }
        } else {
            for (uint i = 0; i < hidden; ++i) {
                const uint w_idx0 = i * intermediate + col0;
                const float gate_w0 = static_cast<float>(gate_t[w_idx0]);
                const float up_w0 = static_cast<float>(up_t[w_idx0]);
                float gate_w1 = 0.0f;
                float up_w1 = 0.0f;
                if (has_col1) {
                    const uint w_idx1 = w_idx0 + 1;
                    gate_w1 = static_cast<float>(gate_t[w_idx1]);
                    up_w1 = static_cast<float>(up_t[w_idx1]);
                }

                const float xv0 = static_cast<float>(x[x_base0 + i]);
                gate_acc00 += xv0 * gate_w0;
                up_acc00 += xv0 * up_w0;
                if (has_col1) {
                    gate_acc01 += xv0 * gate_w1;
                    up_acc01 += xv0 * up_w1;
                }

                if (has_row1) {
                    const float xv1 = static_cast<float>(x[x_base1 + i]);
                    gate_acc10 += xv1 * gate_w0;
                    up_acc10 += xv1 * up_w0;
                    if (has_col1) {
                        gate_acc11 += xv1 * gate_w1;
                        up_acc11 += xv1 * up_w1;
                    }
                }
                if (has_row2) {
                    const float xv2 = static_cast<float>(x[x_base2 + i]);
                    gate_acc20 += xv2 * gate_w0;
                    up_acc20 += xv2 * up_w0;
                    if (has_col1) {
                        gate_acc21 += xv2 * gate_w1;
                        up_acc21 += xv2 * up_w1;
                    }
                }
                if (has_row3) {
                    const float xv3 = static_cast<float>(x[x_base3 + i]);
                    gate_acc30 += xv3 * gate_w0;
                    up_acc30 += xv3 * up_w0;
                    if (has_col1) {
                        gate_acc31 += xv3 * gate_w1;
                        up_acc31 += xv3 * up_w1;
                    }
                }
            }
        }

        const uint out_base0 = row0 * intermediate;
        const float gate_sigmoid00 = 1.0f / (1.0f + exp(-gate_acc00));
        out[out_base0 + col0] = static_cast<bfloat>((gate_acc00 * gate_sigmoid00) * up_acc00);
        if (has_col1) {
            const float gate_sigmoid01 = 1.0f / (1.0f + exp(-gate_acc01));
            out[out_base0 + col1] = static_cast<bfloat>((gate_acc01 * gate_sigmoid01) * up_acc01);
        }
        if (has_row1) {
            const uint out_base1 = row1 * intermediate;
            const float gate_sigmoid10 = 1.0f / (1.0f + exp(-gate_acc10));
            out[out_base1 + col0] = static_cast<bfloat>((gate_acc10 * gate_sigmoid10) * up_acc10);
            if (has_col1) {
                const float gate_sigmoid11 = 1.0f / (1.0f + exp(-gate_acc11));
                out[out_base1 + col1] = static_cast<bfloat>((gate_acc11 * gate_sigmoid11) * up_acc11);
            }
        }
        if (has_row2) {
            const uint out_base2 = row2 * intermediate;
            const float gate_sigmoid20 = 1.0f / (1.0f + exp(-gate_acc20));
            out[out_base2 + col0] = static_cast<bfloat>((gate_acc20 * gate_sigmoid20) * up_acc20);
            if (has_col1) {
                const float gate_sigmoid21 = 1.0f / (1.0f + exp(-gate_acc21));
                out[out_base2 + col1] = static_cast<bfloat>((gate_acc21 * gate_sigmoid21) * up_acc21);
            }
        }
        if (has_row3) {
            const uint out_base3 = row3 * intermediate;
            const float gate_sigmoid30 = 1.0f / (1.0f + exp(-gate_acc30));
            out[out_base3 + col0] = static_cast<bfloat>((gate_acc30 * gate_sigmoid30) * up_acc30);
            if (has_col1) {
                const float gate_sigmoid31 = 1.0f / (1.0f + exp(-gate_acc31));
                out[out_base3 + col1] = static_cast<bfloat>((gate_acc31 * gate_sigmoid31) * up_acc31);
            }
        }
        return;
    }

    const uint row_pairs = (rows + 1) >> 1;
    const uint total = row_pairs * cols2;
    if (gid >= total) {
        return;
    }

    const uint row_pair = gid / cols2;
    const uint row0 = row_pair << 1;
    const uint row1 = row0 + 1;
    const bool has_row1 = row1 < rows;
    const uint col0 = (gid - row_pair * cols2) << 1;
    const uint col1 = col0 + 1;
    const bool has_col1 = col1 < intermediate;
    const uint x_base0 = row0 * hidden;
    const uint x_base1 = row1 * hidden;
    float gate_acc00 = 0.0f;
    float up_acc00 = 0.0f;
    float gate_acc01 = 0.0f;
    float up_acc01 = 0.0f;
    float gate_acc10 = 0.0f;
    float up_acc10 = 0.0f;
    float gate_acc11 = 0.0f;
    float up_acc11 = 0.0f;
    for (uint i = 0; i < hidden; ++i) {
        const uint w_idx0 = i * intermediate + col0;
        const float gate_w0 = static_cast<float>(gate_t[w_idx0]);
        const float up_w0 = static_cast<float>(up_t[w_idx0]);
        const float xv0 = static_cast<float>(x[x_base0 + i]);
        gate_acc00 += xv0 * gate_w0;
        up_acc00 += xv0 * up_w0;
        if (has_row1) {
            const float xv1 = static_cast<float>(x[x_base1 + i]);
            gate_acc10 += xv1 * gate_w0;
            up_acc10 += xv1 * up_w0;
            if (has_col1) {
                const uint w_idx1 = w_idx0 + 1;
                const float gate_w1 = static_cast<float>(gate_t[w_idx1]);
                const float up_w1 = static_cast<float>(up_t[w_idx1]);
                gate_acc11 += xv1 * gate_w1;
                up_acc11 += xv1 * up_w1;
                gate_acc01 += xv0 * gate_w1;
                up_acc01 += xv0 * up_w1;
            }
        } else if (has_col1) {
            const uint w_idx1 = w_idx0 + 1;
            const float xv0_col1 = xv0;
            gate_acc01 += xv0_col1 * static_cast<float>(gate_t[w_idx1]);
            up_acc01 += xv0_col1 * static_cast<float>(up_t[w_idx1]);
        }
    }

    const uint out_base0 = row0 * intermediate;
    const float gate_sigmoid00 = 1.0f / (1.0f + exp(-gate_acc00));
    out[out_base0 + col0] = static_cast<bfloat>((gate_acc00 * gate_sigmoid00) * up_acc00);
    if (has_col1) {
        const float gate_sigmoid01 = 1.0f / (1.0f + exp(-gate_acc01));
        out[out_base0 + col1] = static_cast<bfloat>((gate_acc01 * gate_sigmoid01) * up_acc01);
    }
    if (has_row1) {
        const uint out_base1 = row1 * intermediate;
        const float gate_sigmoid10 = 1.0f / (1.0f + exp(-gate_acc10));
        out[out_base1 + col0] = static_cast<bfloat>((gate_acc10 * gate_sigmoid10) * up_acc10);
        if (has_col1) {
            const float gate_sigmoid11 = 1.0f / (1.0f + exp(-gate_acc11));
            out[out_base1 + col1] = static_cast<bfloat>((gate_acc11 * gate_sigmoid11) * up_acc11);
        }
    }
}

kernel void kiln_mlp_gate_up_serial_bf16(
    device const bfloat* x [[buffer(0)]],
    device const bfloat* gate_t [[buffer(1)]],
    device const bfloat* up_t [[buffer(2)]],
    device bfloat* out [[buffer(3)]],
    constant uint& hidden [[buffer(4)]],
    constant uint& intermediate [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint cols2 = intermediate >> 1;
    if (gid >= cols2) {
        return;
    }

    const uint col0 = gid << 1;
    float gate_acc0 = 0.0f;
    float up_acc0 = 0.0f;
    float gate_acc1 = 0.0f;
    float up_acc1 = 0.0f;
    for (uint i = 0; i < hidden; ++i) {
        const float xv = static_cast<float>(x[i]);
        const uint w_idx0 = i * intermediate + col0;
        const bfloat2 gate_w = *(device const bfloat2*)(gate_t + w_idx0);
        const bfloat2 up_w = *(device const bfloat2*)(up_t + w_idx0);
        gate_acc0 += xv * static_cast<float>(gate_w[0]);
        up_acc0 += xv * static_cast<float>(up_w[0]);
        gate_acc1 += xv * static_cast<float>(gate_w[1]);
        up_acc1 += xv * static_cast<float>(up_w[1]);
    }

    const float gate_sigmoid0 = 1.0f / (1.0f + exp(-gate_acc0));
    const float gate_sigmoid1 = 1.0f / (1.0f + exp(-gate_acc1));
    out[col0] = static_cast<bfloat>((gate_acc0 * gate_sigmoid0) * up_acc0);
    out[col0 + 1] = static_cast<bfloat>((gate_acc1 * gate_sigmoid1) * up_acc1);
}

kernel void kiln_mlp_silu_mul_bf16(
    device const bfloat* gate [[buffer(0)]],
    device const bfloat* up [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    constant uint& total [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= total) {
        return;
    }

    const float gate_val = static_cast<float>(gate[gid]);
    const float up_val = static_cast<float>(up[gid]);
    const float sigmoid = 1.0f / (1.0f + exp(-gate_val));
    out[gid] = static_cast<bfloat>((gate_val * sigmoid) * up_val);
}
"#;
pub(super) const METAL_ATTN_GATE_SIGMOID_MUL_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kiln_attn_gate_sigmoid_mul_bf16(
    device const bfloat* x [[buffer(0)]],
    device const bfloat* gate [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    constant uint& total [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= total) {
        return;
    }

    const float gate_sigmoid = 1.0f / (1.0f + exp(-static_cast<float>(gate[gid])));
    out[gid] = static_cast<bfloat>(
        static_cast<float>(x[gid]) * static_cast<float>(static_cast<bfloat>(gate_sigmoid))
    );
}
"#;
pub(super) const METAL_TRANSPOSED_COOP_GEMV_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kiln_transposed_coop_gemv4_bf16(
    device const bfloat* x [[buffer(0)]],
    device const bfloat* weight_t [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    constant uint& input_dim [[buffer(3)]],
    constant uint& output_dim [[buffer(4)]],
    uint tgroup [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint col_base = (tgroup * 4 + simd_group) * 4;
    if (col_base >= output_dim) {
        return;
    }

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    const bool full_tile = col_base + 3 < output_dim;
    const bool vector_load_safe = full_tile && (output_dim % 4 == 0);

    for (uint row = lane; row < input_dim; row += 32) {
        const float xv = static_cast<float>(x[row]);
        const uint weight_base = row * output_dim + col_base;
        if (vector_load_safe) {
            device const bfloat4* w4_ptr = (device const bfloat4*)(weight_t + weight_base);
            const bfloat4 w = *w4_ptr;
            acc0 += xv * static_cast<float>(w[0]);
            acc1 += xv * static_cast<float>(w[1]);
            acc2 += xv * static_cast<float>(w[2]);
            acc3 += xv * static_cast<float>(w[3]);
        } else {
            acc0 += xv * static_cast<float>(weight_t[weight_base + 0]);
            if (col_base + 1 < output_dim) {
                acc1 += xv * static_cast<float>(weight_t[weight_base + 1]);
            }
            if (col_base + 2 < output_dim) {
                acc2 += xv * static_cast<float>(weight_t[weight_base + 2]);
            }
            if (col_base + 3 < output_dim) {
                acc3 += xv * static_cast<float>(weight_t[weight_base + 3]);
            }
        }
    }

    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    acc2 = simd_sum(acc2);
    acc3 = simd_sum(acc3);

    if (lane == 0) {
        out[col_base + 0] = static_cast<bfloat>(acc0);
        if (col_base + 1 < output_dim) {
            out[col_base + 1] = static_cast<bfloat>(acc1);
        }
        if (col_base + 2 < output_dim) {
            out[col_base + 2] = static_cast<bfloat>(acc2);
        }
        if (col_base + 3 < output_dim) {
            out[col_base + 3] = static_cast<bfloat>(acc3);
        }
    }
}

kernel void kiln_transposed_coop_gemv8_bf16(
    device const bfloat* x [[buffer(0)]],
    device const bfloat* weight_t [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    constant uint& input_dim [[buffer(3)]],
    constant uint& output_dim [[buffer(4)]],
    uint tgroup [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint col_base = (tgroup * 4 + simd_group) * 8;
    if (col_base >= output_dim) {
        return;
    }

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    float acc4 = 0.0f;
    float acc5 = 0.0f;
    float acc6 = 0.0f;
    float acc7 = 0.0f;
    const bool full_tile = col_base + 7 < output_dim;
    const bool vector_load_safe = full_tile && (output_dim % 4 == 0);

    for (uint row = lane; row < input_dim; row += 32) {
        const float xv = static_cast<float>(x[row]);
        const uint weight_base = row * output_dim + col_base;
        if (vector_load_safe) {
            device const bfloat4* w4_ptr = (device const bfloat4*)(weight_t + weight_base);
            const bfloat4 w0 = w4_ptr[0];
            const bfloat4 w1 = w4_ptr[1];
            acc0 += xv * static_cast<float>(w0[0]);
            acc1 += xv * static_cast<float>(w0[1]);
            acc2 += xv * static_cast<float>(w0[2]);
            acc3 += xv * static_cast<float>(w0[3]);
            acc4 += xv * static_cast<float>(w1[0]);
            acc5 += xv * static_cast<float>(w1[1]);
            acc6 += xv * static_cast<float>(w1[2]);
            acc7 += xv * static_cast<float>(w1[3]);
        } else {
            acc0 += xv * static_cast<float>(weight_t[weight_base + 0]);
            if (col_base + 1 < output_dim) {
                acc1 += xv * static_cast<float>(weight_t[weight_base + 1]);
            }
            if (col_base + 2 < output_dim) {
                acc2 += xv * static_cast<float>(weight_t[weight_base + 2]);
            }
            if (col_base + 3 < output_dim) {
                acc3 += xv * static_cast<float>(weight_t[weight_base + 3]);
            }
            if (col_base + 4 < output_dim) {
                acc4 += xv * static_cast<float>(weight_t[weight_base + 4]);
            }
            if (col_base + 5 < output_dim) {
                acc5 += xv * static_cast<float>(weight_t[weight_base + 5]);
            }
            if (col_base + 6 < output_dim) {
                acc6 += xv * static_cast<float>(weight_t[weight_base + 6]);
            }
            if (col_base + 7 < output_dim) {
                acc7 += xv * static_cast<float>(weight_t[weight_base + 7]);
            }
        }
    }

    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    acc2 = simd_sum(acc2);
    acc3 = simd_sum(acc3);
    acc4 = simd_sum(acc4);
    acc5 = simd_sum(acc5);
    acc6 = simd_sum(acc6);
    acc7 = simd_sum(acc7);

    if (lane == 0) {
        out[col_base + 0] = static_cast<bfloat>(acc0);
        if (col_base + 1 < output_dim) {
            out[col_base + 1] = static_cast<bfloat>(acc1);
        }
        if (col_base + 2 < output_dim) {
            out[col_base + 2] = static_cast<bfloat>(acc2);
        }
        if (col_base + 3 < output_dim) {
            out[col_base + 3] = static_cast<bfloat>(acc3);
        }
        if (col_base + 4 < output_dim) {
            out[col_base + 4] = static_cast<bfloat>(acc4);
        }
        if (col_base + 5 < output_dim) {
            out[col_base + 5] = static_cast<bfloat>(acc5);
        }
        if (col_base + 6 < output_dim) {
            out[col_base + 6] = static_cast<bfloat>(acc6);
        }
        if (col_base + 7 < output_dim) {
            out[col_base + 7] = static_cast<bfloat>(acc7);
        }
    }
}

kernel void kiln_transposed_coop_gemv16_bf16(
    device const bfloat* x [[buffer(0)]],
    device const bfloat* weight_t [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    constant uint& input_dim [[buffer(3)]],
    constant uint& output_dim [[buffer(4)]],
    uint tgroup [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint col_base = (tgroup * 4 + simd_group) * 16;
    if (col_base >= output_dim) {
        return;
    }

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    float acc4 = 0.0f;
    float acc5 = 0.0f;
    float acc6 = 0.0f;
    float acc7 = 0.0f;
    float acc8 = 0.0f;
    float acc9 = 0.0f;
    float acc10 = 0.0f;
    float acc11 = 0.0f;
    float acc12 = 0.0f;
    float acc13 = 0.0f;
    float acc14 = 0.0f;
    float acc15 = 0.0f;
    const bool full_tile = col_base + 15 < output_dim;
    const bool vector_load_safe = full_tile && (output_dim % 4 == 0);

    for (uint row = lane; row < input_dim; row += 32) {
        const float xv = static_cast<float>(x[row]);
        const uint weight_base = row * output_dim + col_base;
        if (vector_load_safe) {
            device const bfloat4* w4_ptr = (device const bfloat4*)(weight_t + weight_base);
            const bfloat4 w0 = w4_ptr[0];
            const bfloat4 w1 = w4_ptr[1];
            const bfloat4 w2 = w4_ptr[2];
            const bfloat4 w3 = w4_ptr[3];
            acc0 += xv * static_cast<float>(w0[0]);
            acc1 += xv * static_cast<float>(w0[1]);
            acc2 += xv * static_cast<float>(w0[2]);
            acc3 += xv * static_cast<float>(w0[3]);
            acc4 += xv * static_cast<float>(w1[0]);
            acc5 += xv * static_cast<float>(w1[1]);
            acc6 += xv * static_cast<float>(w1[2]);
            acc7 += xv * static_cast<float>(w1[3]);
            acc8 += xv * static_cast<float>(w2[0]);
            acc9 += xv * static_cast<float>(w2[1]);
            acc10 += xv * static_cast<float>(w2[2]);
            acc11 += xv * static_cast<float>(w2[3]);
            acc12 += xv * static_cast<float>(w3[0]);
            acc13 += xv * static_cast<float>(w3[1]);
            acc14 += xv * static_cast<float>(w3[2]);
            acc15 += xv * static_cast<float>(w3[3]);
        } else {
            acc0 += xv * static_cast<float>(weight_t[weight_base + 0]);
            if (col_base + 1 < output_dim) {
                acc1 += xv * static_cast<float>(weight_t[weight_base + 1]);
            }
            if (col_base + 2 < output_dim) {
                acc2 += xv * static_cast<float>(weight_t[weight_base + 2]);
            }
            if (col_base + 3 < output_dim) {
                acc3 += xv * static_cast<float>(weight_t[weight_base + 3]);
            }
            if (col_base + 4 < output_dim) {
                acc4 += xv * static_cast<float>(weight_t[weight_base + 4]);
            }
            if (col_base + 5 < output_dim) {
                acc5 += xv * static_cast<float>(weight_t[weight_base + 5]);
            }
            if (col_base + 6 < output_dim) {
                acc6 += xv * static_cast<float>(weight_t[weight_base + 6]);
            }
            if (col_base + 7 < output_dim) {
                acc7 += xv * static_cast<float>(weight_t[weight_base + 7]);
            }
            if (col_base + 8 < output_dim) {
                acc8 += xv * static_cast<float>(weight_t[weight_base + 8]);
            }
            if (col_base + 9 < output_dim) {
                acc9 += xv * static_cast<float>(weight_t[weight_base + 9]);
            }
            if (col_base + 10 < output_dim) {
                acc10 += xv * static_cast<float>(weight_t[weight_base + 10]);
            }
            if (col_base + 11 < output_dim) {
                acc11 += xv * static_cast<float>(weight_t[weight_base + 11]);
            }
            if (col_base + 12 < output_dim) {
                acc12 += xv * static_cast<float>(weight_t[weight_base + 12]);
            }
            if (col_base + 13 < output_dim) {
                acc13 += xv * static_cast<float>(weight_t[weight_base + 13]);
            }
            if (col_base + 14 < output_dim) {
                acc14 += xv * static_cast<float>(weight_t[weight_base + 14]);
            }
            if (col_base + 15 < output_dim) {
                acc15 += xv * static_cast<float>(weight_t[weight_base + 15]);
            }
        }
    }

    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    acc2 = simd_sum(acc2);
    acc3 = simd_sum(acc3);
    acc4 = simd_sum(acc4);
    acc5 = simd_sum(acc5);
    acc6 = simd_sum(acc6);
    acc7 = simd_sum(acc7);
    acc8 = simd_sum(acc8);
    acc9 = simd_sum(acc9);
    acc10 = simd_sum(acc10);
    acc11 = simd_sum(acc11);
    acc12 = simd_sum(acc12);
    acc13 = simd_sum(acc13);
    acc14 = simd_sum(acc14);
    acc15 = simd_sum(acc15);

    if (lane == 0) {
        out[col_base + 0] = static_cast<bfloat>(acc0);
        if (col_base + 1 < output_dim) {
            out[col_base + 1] = static_cast<bfloat>(acc1);
        }
        if (col_base + 2 < output_dim) {
            out[col_base + 2] = static_cast<bfloat>(acc2);
        }
        if (col_base + 3 < output_dim) {
            out[col_base + 3] = static_cast<bfloat>(acc3);
        }
        if (col_base + 4 < output_dim) {
            out[col_base + 4] = static_cast<bfloat>(acc4);
        }
        if (col_base + 5 < output_dim) {
            out[col_base + 5] = static_cast<bfloat>(acc5);
        }
        if (col_base + 6 < output_dim) {
            out[col_base + 6] = static_cast<bfloat>(acc6);
        }
        if (col_base + 7 < output_dim) {
            out[col_base + 7] = static_cast<bfloat>(acc7);
        }
        if (col_base + 8 < output_dim) {
            out[col_base + 8] = static_cast<bfloat>(acc8);
        }
        if (col_base + 9 < output_dim) {
            out[col_base + 9] = static_cast<bfloat>(acc9);
        }
        if (col_base + 10 < output_dim) {
            out[col_base + 10] = static_cast<bfloat>(acc10);
        }
        if (col_base + 11 < output_dim) {
            out[col_base + 11] = static_cast<bfloat>(acc11);
        }
        if (col_base + 12 < output_dim) {
            out[col_base + 12] = static_cast<bfloat>(acc12);
        }
        if (col_base + 13 < output_dim) {
            out[col_base + 13] = static_cast<bfloat>(acc13);
        }
        if (col_base + 14 < output_dim) {
            out[col_base + 14] = static_cast<bfloat>(acc14);
        }
        if (col_base + 15 < output_dim) {
            out[col_base + 15] = static_cast<bfloat>(acc15);
        }
    }
}

kernel void kiln_transposed_coop_gemv8_batch_bf16(
    device const bfloat* x [[buffer(0)]],
    device const bfloat* weight_t [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    constant uint& input_dim [[buffer(3)]],
    constant uint& output_dim [[buffer(4)]],
    constant uint& row_pair_mode [[buffer(5)]],
    constant uint& row_group_size_arg [[buffer(6)]],
    uint2 tgroup [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const bool grouped_mode = row_pair_mode != 0;
    const uint row_group_size = grouped_mode ? row_group_size_arg : 1;
    const bool row_quad_mode = grouped_mode && row_group_size == 4;
    const uint tile_cols = row_quad_mode ? 4 : 8;
    const uint col_base = (tgroup.x * 4 + simd_group) * tile_cols;
    if (col_base >= output_dim) {
        return;
    }

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    float acc4 = 0.0f;
    float acc5 = 0.0f;
    float acc6 = 0.0f;
    float acc7 = 0.0f;
    const bool full_tile = col_base + tile_cols - 1 < output_dim;
    const bool vector_load_safe = full_tile && (output_dim % 4 == 0);
    const uint batch_idx = grouped_mode ? tgroup.y * row_group_size : tgroup.y;
    const uint x_base = batch_idx * input_dim;

    if (!grouped_mode) {
        for (uint row = lane; row < input_dim; row += 32) {
            const float xv = static_cast<float>(x[x_base + row]);
            const uint weight_base = row * output_dim + col_base;
            if (vector_load_safe) {
                device const bfloat4* w4_ptr = (device const bfloat4*)(weight_t + weight_base);
                const bfloat4 w0 = w4_ptr[0];
                const bfloat4 w1 = w4_ptr[1];
                acc0 += xv * static_cast<float>(w0[0]);
                acc1 += xv * static_cast<float>(w0[1]);
                acc2 += xv * static_cast<float>(w0[2]);
                acc3 += xv * static_cast<float>(w0[3]);
                acc4 += xv * static_cast<float>(w1[0]);
                acc5 += xv * static_cast<float>(w1[1]);
                acc6 += xv * static_cast<float>(w1[2]);
                acc7 += xv * static_cast<float>(w1[3]);
            } else {
                acc0 += xv * static_cast<float>(weight_t[weight_base + 0]);
                if (col_base + 1 < output_dim) {
                    acc1 += xv * static_cast<float>(weight_t[weight_base + 1]);
                }
                if (col_base + 2 < output_dim) {
                    acc2 += xv * static_cast<float>(weight_t[weight_base + 2]);
                }
                if (col_base + 3 < output_dim) {
                    acc3 += xv * static_cast<float>(weight_t[weight_base + 3]);
                }
                if (col_base + 4 < output_dim) {
                    acc4 += xv * static_cast<float>(weight_t[weight_base + 4]);
                }
                if (col_base + 5 < output_dim) {
                    acc5 += xv * static_cast<float>(weight_t[weight_base + 5]);
                }
                if (col_base + 6 < output_dim) {
                    acc6 += xv * static_cast<float>(weight_t[weight_base + 6]);
                }
                if (col_base + 7 < output_dim) {
                    acc7 += xv * static_cast<float>(weight_t[weight_base + 7]);
                }
            }
        }

        acc0 = simd_sum(acc0);
        acc1 = simd_sum(acc1);
        acc2 = simd_sum(acc2);
        acc3 = simd_sum(acc3);
        acc4 = simd_sum(acc4);
        acc5 = simd_sum(acc5);
        acc6 = simd_sum(acc6);
        acc7 = simd_sum(acc7);

        if (lane == 0) {
            const uint out_base = batch_idx * output_dim;
            out[out_base + col_base + 0] = static_cast<bfloat>(acc0);
            if (col_base + 1 < output_dim) {
                out[out_base + col_base + 1] = static_cast<bfloat>(acc1);
            }
            if (col_base + 2 < output_dim) {
                out[out_base + col_base + 2] = static_cast<bfloat>(acc2);
            }
            if (col_base + 3 < output_dim) {
                out[out_base + col_base + 3] = static_cast<bfloat>(acc3);
            }
            if (col_base + 4 < output_dim) {
                out[out_base + col_base + 4] = static_cast<bfloat>(acc4);
            }
            if (col_base + 5 < output_dim) {
                out[out_base + col_base + 5] = static_cast<bfloat>(acc5);
            }
            if (col_base + 6 < output_dim) {
                out[out_base + col_base + 6] = static_cast<bfloat>(acc6);
            }
            if (col_base + 7 < output_dim) {
                out[out_base + col_base + 7] = static_cast<bfloat>(acc7);
            }
        }
        return;
    }

    if (row_quad_mode) {
        const uint batch1 = batch_idx + 1;
        const uint batch2 = batch_idx + 2;
        const uint batch3 = batch_idx + 3;
        const bool has_batch1 = batch1 < row_pair_mode;
        const bool has_batch2 = batch2 < row_pair_mode;
        const bool has_batch3 = batch3 < row_pair_mode;
        const uint x_base1 = batch1 * input_dim;
        const uint x_base2 = batch2 * input_dim;
        const uint x_base3 = batch3 * input_dim;
        float acc10 = 0.0f;
        float acc11 = 0.0f;
        float acc12 = 0.0f;
        float acc13 = 0.0f;
        float acc20 = 0.0f;
        float acc21 = 0.0f;
        float acc22 = 0.0f;
        float acc23 = 0.0f;
        float acc30 = 0.0f;
        float acc31 = 0.0f;
        float acc32 = 0.0f;
        float acc33 = 0.0f;

        for (uint row = lane; row < input_dim; row += 32) {
            const float xv0 = static_cast<float>(x[x_base + row]);
            const float xv1 = has_batch1 ? static_cast<float>(x[x_base1 + row]) : 0.0f;
            const float xv2 = has_batch2 ? static_cast<float>(x[x_base2 + row]) : 0.0f;
            const float xv3 = has_batch3 ? static_cast<float>(x[x_base3 + row]) : 0.0f;
            const uint weight_base = row * output_dim + col_base;
            if (vector_load_safe) {
                device const bfloat4* w4_ptr = (device const bfloat4*)(weight_t + weight_base);
                const bfloat4 w = *w4_ptr;
                const float w0 = static_cast<float>(w[0]);
                const float w1 = static_cast<float>(w[1]);
                const float w2 = static_cast<float>(w[2]);
                const float w3 = static_cast<float>(w[3]);
                acc0 += xv0 * w0;
                acc1 += xv0 * w1;
                acc2 += xv0 * w2;
                acc3 += xv0 * w3;
                acc10 += xv1 * w0;
                acc11 += xv1 * w1;
                acc12 += xv1 * w2;
                acc13 += xv1 * w3;
                acc20 += xv2 * w0;
                acc21 += xv2 * w1;
                acc22 += xv2 * w2;
                acc23 += xv2 * w3;
                acc30 += xv3 * w0;
                acc31 += xv3 * w1;
                acc32 += xv3 * w2;
                acc33 += xv3 * w3;
            } else {
                const float w0 = static_cast<float>(weight_t[weight_base + 0]);
                acc0 += xv0 * w0;
                acc10 += xv1 * w0;
                acc20 += xv2 * w0;
                acc30 += xv3 * w0;
                if (col_base + 1 < output_dim) {
                    const float w1 = static_cast<float>(weight_t[weight_base + 1]);
                    acc1 += xv0 * w1;
                    acc11 += xv1 * w1;
                    acc21 += xv2 * w1;
                    acc31 += xv3 * w1;
                }
                if (col_base + 2 < output_dim) {
                    const float w2 = static_cast<float>(weight_t[weight_base + 2]);
                    acc2 += xv0 * w2;
                    acc12 += xv1 * w2;
                    acc22 += xv2 * w2;
                    acc32 += xv3 * w2;
                }
                if (col_base + 3 < output_dim) {
                    const float w3 = static_cast<float>(weight_t[weight_base + 3]);
                    acc3 += xv0 * w3;
                    acc13 += xv1 * w3;
                    acc23 += xv2 * w3;
                    acc33 += xv3 * w3;
                }
            }
        }

        acc0 = simd_sum(acc0);
        acc1 = simd_sum(acc1);
        acc2 = simd_sum(acc2);
        acc3 = simd_sum(acc3);
        acc10 = simd_sum(acc10);
        acc11 = simd_sum(acc11);
        acc12 = simd_sum(acc12);
        acc13 = simd_sum(acc13);
        acc20 = simd_sum(acc20);
        acc21 = simd_sum(acc21);
        acc22 = simd_sum(acc22);
        acc23 = simd_sum(acc23);
        acc30 = simd_sum(acc30);
        acc31 = simd_sum(acc31);
        acc32 = simd_sum(acc32);
        acc33 = simd_sum(acc33);

        if (lane == 0) {
            const uint out_base = batch_idx * output_dim;
            out[out_base + col_base + 0] = static_cast<bfloat>(acc0);
            if (col_base + 1 < output_dim) {
                out[out_base + col_base + 1] = static_cast<bfloat>(acc1);
            }
            if (col_base + 2 < output_dim) {
                out[out_base + col_base + 2] = static_cast<bfloat>(acc2);
            }
            if (col_base + 3 < output_dim) {
                out[out_base + col_base + 3] = static_cast<bfloat>(acc3);
            }
            if (has_batch1) {
                const uint out_base1 = batch1 * output_dim;
                out[out_base1 + col_base + 0] = static_cast<bfloat>(acc10);
                if (col_base + 1 < output_dim) {
                    out[out_base1 + col_base + 1] = static_cast<bfloat>(acc11);
                }
                if (col_base + 2 < output_dim) {
                    out[out_base1 + col_base + 2] = static_cast<bfloat>(acc12);
                }
                if (col_base + 3 < output_dim) {
                    out[out_base1 + col_base + 3] = static_cast<bfloat>(acc13);
                }
            }
            if (has_batch2) {
                const uint out_base2 = batch2 * output_dim;
                out[out_base2 + col_base + 0] = static_cast<bfloat>(acc20);
                if (col_base + 1 < output_dim) {
                    out[out_base2 + col_base + 1] = static_cast<bfloat>(acc21);
                }
                if (col_base + 2 < output_dim) {
                    out[out_base2 + col_base + 2] = static_cast<bfloat>(acc22);
                }
                if (col_base + 3 < output_dim) {
                    out[out_base2 + col_base + 3] = static_cast<bfloat>(acc23);
                }
            }
            if (has_batch3) {
                const uint out_base3 = batch3 * output_dim;
                out[out_base3 + col_base + 0] = static_cast<bfloat>(acc30);
                if (col_base + 1 < output_dim) {
                    out[out_base3 + col_base + 1] = static_cast<bfloat>(acc31);
                }
                if (col_base + 2 < output_dim) {
                    out[out_base3 + col_base + 2] = static_cast<bfloat>(acc32);
                }
                if (col_base + 3 < output_dim) {
                    out[out_base3 + col_base + 3] = static_cast<bfloat>(acc33);
                }
            }
        }
        return;
    }

    const uint batch1 = batch_idx + 1;
    const bool has_batch1 = batch1 < row_pair_mode;
    const uint x_base1 = batch1 * input_dim;
    float acc10 = 0.0f;
    float acc11 = 0.0f;
    float acc12 = 0.0f;
    float acc13 = 0.0f;
    float acc14 = 0.0f;
    float acc15 = 0.0f;
    float acc16 = 0.0f;
    float acc17 = 0.0f;

    for (uint row = lane; row < input_dim; row += 32) {
        const float xv0 = static_cast<float>(x[x_base + row]);
        const float xv1 = has_batch1 ? static_cast<float>(x[x_base1 + row]) : 0.0f;
        const uint weight_base = row * output_dim + col_base;
        if (vector_load_safe) {
            device const bfloat4* w4_ptr = (device const bfloat4*)(weight_t + weight_base);
            const bfloat4 w0 = w4_ptr[0];
            const bfloat4 w1 = w4_ptr[1];
            const float w00 = static_cast<float>(w0[0]);
            const float w01 = static_cast<float>(w0[1]);
            const float w02 = static_cast<float>(w0[2]);
            const float w03 = static_cast<float>(w0[3]);
            const float w04 = static_cast<float>(w1[0]);
            const float w05 = static_cast<float>(w1[1]);
            const float w06 = static_cast<float>(w1[2]);
            const float w07 = static_cast<float>(w1[3]);
            acc0 += xv0 * w00;
            acc1 += xv0 * w01;
            acc2 += xv0 * w02;
            acc3 += xv0 * w03;
            acc4 += xv0 * w04;
            acc5 += xv0 * w05;
            acc6 += xv0 * w06;
            acc7 += xv0 * w07;
            acc10 += xv1 * w00;
            acc11 += xv1 * w01;
            acc12 += xv1 * w02;
            acc13 += xv1 * w03;
            acc14 += xv1 * w04;
            acc15 += xv1 * w05;
            acc16 += xv1 * w06;
            acc17 += xv1 * w07;
        } else {
            const float w00 = static_cast<float>(weight_t[weight_base + 0]);
            acc0 += xv0 * w00;
            acc10 += xv1 * w00;
            if (col_base + 1 < output_dim) {
                const float w01 = static_cast<float>(weight_t[weight_base + 1]);
                acc1 += xv0 * w01;
                acc11 += xv1 * w01;
            }
            if (col_base + 2 < output_dim) {
                const float w02 = static_cast<float>(weight_t[weight_base + 2]);
                acc2 += xv0 * w02;
                acc12 += xv1 * w02;
            }
            if (col_base + 3 < output_dim) {
                const float w03 = static_cast<float>(weight_t[weight_base + 3]);
                acc3 += xv0 * w03;
                acc13 += xv1 * w03;
            }
            if (col_base + 4 < output_dim) {
                const float w04 = static_cast<float>(weight_t[weight_base + 4]);
                acc4 += xv0 * w04;
                acc14 += xv1 * w04;
            }
            if (col_base + 5 < output_dim) {
                const float w05 = static_cast<float>(weight_t[weight_base + 5]);
                acc5 += xv0 * w05;
                acc15 += xv1 * w05;
            }
            if (col_base + 6 < output_dim) {
                const float w06 = static_cast<float>(weight_t[weight_base + 6]);
                acc6 += xv0 * w06;
                acc16 += xv1 * w06;
            }
            if (col_base + 7 < output_dim) {
                const float w07 = static_cast<float>(weight_t[weight_base + 7]);
                acc7 += xv0 * w07;
                acc17 += xv1 * w07;
            }
        }
    }

    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    acc2 = simd_sum(acc2);
    acc3 = simd_sum(acc3);
    acc4 = simd_sum(acc4);
    acc5 = simd_sum(acc5);
    acc6 = simd_sum(acc6);
    acc7 = simd_sum(acc7);
    acc10 = simd_sum(acc10);
    acc11 = simd_sum(acc11);
    acc12 = simd_sum(acc12);
    acc13 = simd_sum(acc13);
    acc14 = simd_sum(acc14);
    acc15 = simd_sum(acc15);
    acc16 = simd_sum(acc16);
    acc17 = simd_sum(acc17);

    if (lane == 0) {
        const uint out_base = batch_idx * output_dim;
        out[out_base + col_base + 0] = static_cast<bfloat>(acc0);
        if (col_base + 1 < output_dim) {
            out[out_base + col_base + 1] = static_cast<bfloat>(acc1);
        }
        if (col_base + 2 < output_dim) {
            out[out_base + col_base + 2] = static_cast<bfloat>(acc2);
        }
        if (col_base + 3 < output_dim) {
            out[out_base + col_base + 3] = static_cast<bfloat>(acc3);
        }
        if (col_base + 4 < output_dim) {
            out[out_base + col_base + 4] = static_cast<bfloat>(acc4);
        }
        if (col_base + 5 < output_dim) {
            out[out_base + col_base + 5] = static_cast<bfloat>(acc5);
        }
        if (col_base + 6 < output_dim) {
            out[out_base + col_base + 6] = static_cast<bfloat>(acc6);
        }
        if (col_base + 7 < output_dim) {
            out[out_base + col_base + 7] = static_cast<bfloat>(acc7);
        }
        if (has_batch1) {
            const uint out_base1 = batch1 * output_dim;
            out[out_base1 + col_base + 0] = static_cast<bfloat>(acc10);
            if (col_base + 1 < output_dim) {
                out[out_base1 + col_base + 1] = static_cast<bfloat>(acc11);
            }
            if (col_base + 2 < output_dim) {
                out[out_base1 + col_base + 2] = static_cast<bfloat>(acc12);
            }
            if (col_base + 3 < output_dim) {
                out[out_base1 + col_base + 3] = static_cast<bfloat>(acc13);
            }
            if (col_base + 4 < output_dim) {
                out[out_base1 + col_base + 4] = static_cast<bfloat>(acc14);
            }
            if (col_base + 5 < output_dim) {
                out[out_base1 + col_base + 5] = static_cast<bfloat>(acc15);
            }
            if (col_base + 6 < output_dim) {
                out[out_base1 + col_base + 6] = static_cast<bfloat>(acc16);
            }
            if (col_base + 7 < output_dim) {
                out[out_base1 + col_base + 7] = static_cast<bfloat>(acc17);
            }
        }
    }
}

kernel void kiln_transposed_coop_gemv8_batch_row_triple_tile8_bf16(
    device const bfloat* x [[buffer(0)]],
    device const bfloat* weight_t [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    constant uint& input_dim [[buffer(3)]],
    constant uint& output_dim [[buffer(4)]],
    uint2 tgroup [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint TILE_COLS = 8;
    constexpr uint ROW_GROUP_SIZE = 3;
    const uint col_base = (tgroup.x * 4 + simd_group) * TILE_COLS;
    if (col_base >= output_dim) {
        return;
    }

    const uint batch_idx = tgroup.y * ROW_GROUP_SIZE;
    const uint batch1 = batch_idx + 1;
    const uint batch2 = batch_idx + 2;
    const uint x_base0 = batch_idx * input_dim;
    const uint x_base1 = batch1 * input_dim;
    const uint x_base2 = batch2 * input_dim;
    const bool full_tile = col_base + TILE_COLS - 1 < output_dim;
    const bool vector_load_safe = full_tile && (output_dim % 4 == 0);

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    float acc4 = 0.0f;
    float acc5 = 0.0f;
    float acc6 = 0.0f;
    float acc7 = 0.0f;
    float acc10 = 0.0f;
    float acc11 = 0.0f;
    float acc12 = 0.0f;
    float acc13 = 0.0f;
    float acc14 = 0.0f;
    float acc15 = 0.0f;
    float acc16 = 0.0f;
    float acc17 = 0.0f;
    float acc20 = 0.0f;
    float acc21 = 0.0f;
    float acc22 = 0.0f;
    float acc23 = 0.0f;
    float acc24 = 0.0f;
    float acc25 = 0.0f;
    float acc26 = 0.0f;
    float acc27 = 0.0f;

    for (uint row = lane; row < input_dim; row += 32) {
        const float xv0 = static_cast<float>(x[x_base0 + row]);
        const float xv1 = static_cast<float>(x[x_base1 + row]);
        const float xv2 = static_cast<float>(x[x_base2 + row]);
        const uint weight_base = row * output_dim + col_base;
        if (vector_load_safe) {
            device const bfloat4* w4_ptr = (device const bfloat4*)(weight_t + weight_base);
            const bfloat4 w_lo = w4_ptr[0];
            const bfloat4 w_hi = w4_ptr[1];
            const float w0 = static_cast<float>(w_lo[0]);
            const float w1 = static_cast<float>(w_lo[1]);
            const float w2 = static_cast<float>(w_lo[2]);
            const float w3 = static_cast<float>(w_lo[3]);
            const float w4 = static_cast<float>(w_hi[0]);
            const float w5 = static_cast<float>(w_hi[1]);
            const float w6 = static_cast<float>(w_hi[2]);
            const float w7 = static_cast<float>(w_hi[3]);
            acc0 += xv0 * w0;
            acc1 += xv0 * w1;
            acc2 += xv0 * w2;
            acc3 += xv0 * w3;
            acc4 += xv0 * w4;
            acc5 += xv0 * w5;
            acc6 += xv0 * w6;
            acc7 += xv0 * w7;
            acc10 += xv1 * w0;
            acc11 += xv1 * w1;
            acc12 += xv1 * w2;
            acc13 += xv1 * w3;
            acc14 += xv1 * w4;
            acc15 += xv1 * w5;
            acc16 += xv1 * w6;
            acc17 += xv1 * w7;
            acc20 += xv2 * w0;
            acc21 += xv2 * w1;
            acc22 += xv2 * w2;
            acc23 += xv2 * w3;
            acc24 += xv2 * w4;
            acc25 += xv2 * w5;
            acc26 += xv2 * w6;
            acc27 += xv2 * w7;
        } else {
            const float w0 = static_cast<float>(weight_t[weight_base + 0]);
            acc0 += xv0 * w0;
            acc10 += xv1 * w0;
            acc20 += xv2 * w0;
            if (col_base + 1 < output_dim) {
                const float w1 = static_cast<float>(weight_t[weight_base + 1]);
                acc1 += xv0 * w1;
                acc11 += xv1 * w1;
                acc21 += xv2 * w1;
            }
            if (col_base + 2 < output_dim) {
                const float w2 = static_cast<float>(weight_t[weight_base + 2]);
                acc2 += xv0 * w2;
                acc12 += xv1 * w2;
                acc22 += xv2 * w2;
            }
            if (col_base + 3 < output_dim) {
                const float w3 = static_cast<float>(weight_t[weight_base + 3]);
                acc3 += xv0 * w3;
                acc13 += xv1 * w3;
                acc23 += xv2 * w3;
            }
            if (col_base + 4 < output_dim) {
                const float w4 = static_cast<float>(weight_t[weight_base + 4]);
                acc4 += xv0 * w4;
                acc14 += xv1 * w4;
                acc24 += xv2 * w4;
            }
            if (col_base + 5 < output_dim) {
                const float w5 = static_cast<float>(weight_t[weight_base + 5]);
                acc5 += xv0 * w5;
                acc15 += xv1 * w5;
                acc25 += xv2 * w5;
            }
            if (col_base + 6 < output_dim) {
                const float w6 = static_cast<float>(weight_t[weight_base + 6]);
                acc6 += xv0 * w6;
                acc16 += xv1 * w6;
                acc26 += xv2 * w6;
            }
            if (col_base + 7 < output_dim) {
                const float w7 = static_cast<float>(weight_t[weight_base + 7]);
                acc7 += xv0 * w7;
                acc17 += xv1 * w7;
                acc27 += xv2 * w7;
            }
        }
    }

    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    acc2 = simd_sum(acc2);
    acc3 = simd_sum(acc3);
    acc4 = simd_sum(acc4);
    acc5 = simd_sum(acc5);
    acc6 = simd_sum(acc6);
    acc7 = simd_sum(acc7);
    acc10 = simd_sum(acc10);
    acc11 = simd_sum(acc11);
    acc12 = simd_sum(acc12);
    acc13 = simd_sum(acc13);
    acc14 = simd_sum(acc14);
    acc15 = simd_sum(acc15);
    acc16 = simd_sum(acc16);
    acc17 = simd_sum(acc17);
    acc20 = simd_sum(acc20);
    acc21 = simd_sum(acc21);
    acc22 = simd_sum(acc22);
    acc23 = simd_sum(acc23);
    acc24 = simd_sum(acc24);
    acc25 = simd_sum(acc25);
    acc26 = simd_sum(acc26);
    acc27 = simd_sum(acc27);

    if (lane == 0) {
        const uint out_base0 = batch_idx * output_dim;
        out[out_base0 + col_base + 0] = static_cast<bfloat>(acc0);
        if (col_base + 1 < output_dim) {
            out[out_base0 + col_base + 1] = static_cast<bfloat>(acc1);
        }
        if (col_base + 2 < output_dim) {
            out[out_base0 + col_base + 2] = static_cast<bfloat>(acc2);
        }
        if (col_base + 3 < output_dim) {
            out[out_base0 + col_base + 3] = static_cast<bfloat>(acc3);
        }
        if (col_base + 4 < output_dim) {
            out[out_base0 + col_base + 4] = static_cast<bfloat>(acc4);
        }
        if (col_base + 5 < output_dim) {
            out[out_base0 + col_base + 5] = static_cast<bfloat>(acc5);
        }
        if (col_base + 6 < output_dim) {
            out[out_base0 + col_base + 6] = static_cast<bfloat>(acc6);
        }
        if (col_base + 7 < output_dim) {
            out[out_base0 + col_base + 7] = static_cast<bfloat>(acc7);
        }

        const uint out_base1 = batch1 * output_dim;
        out[out_base1 + col_base + 0] = static_cast<bfloat>(acc10);
        if (col_base + 1 < output_dim) {
            out[out_base1 + col_base + 1] = static_cast<bfloat>(acc11);
        }
        if (col_base + 2 < output_dim) {
            out[out_base1 + col_base + 2] = static_cast<bfloat>(acc12);
        }
        if (col_base + 3 < output_dim) {
            out[out_base1 + col_base + 3] = static_cast<bfloat>(acc13);
        }
        if (col_base + 4 < output_dim) {
            out[out_base1 + col_base + 4] = static_cast<bfloat>(acc14);
        }
        if (col_base + 5 < output_dim) {
            out[out_base1 + col_base + 5] = static_cast<bfloat>(acc15);
        }
        if (col_base + 6 < output_dim) {
            out[out_base1 + col_base + 6] = static_cast<bfloat>(acc16);
        }
        if (col_base + 7 < output_dim) {
            out[out_base1 + col_base + 7] = static_cast<bfloat>(acc17);
        }

        const uint out_base2 = batch2 * output_dim;
        out[out_base2 + col_base + 0] = static_cast<bfloat>(acc20);
        if (col_base + 1 < output_dim) {
            out[out_base2 + col_base + 1] = static_cast<bfloat>(acc21);
        }
        if (col_base + 2 < output_dim) {
            out[out_base2 + col_base + 2] = static_cast<bfloat>(acc22);
        }
        if (col_base + 3 < output_dim) {
            out[out_base2 + col_base + 3] = static_cast<bfloat>(acc23);
        }
        if (col_base + 4 < output_dim) {
            out[out_base2 + col_base + 4] = static_cast<bfloat>(acc24);
        }
        if (col_base + 5 < output_dim) {
            out[out_base2 + col_base + 5] = static_cast<bfloat>(acc25);
        }
        if (col_base + 6 < output_dim) {
            out[out_base2 + col_base + 6] = static_cast<bfloat>(acc26);
        }
        if (col_base + 7 < output_dim) {
            out[out_base2 + col_base + 7] = static_cast<bfloat>(acc27);
        }
    }
}

kernel void kiln_transposed_coop_gemv8_batch_row_quad_tile8_bf16(
    device const bfloat* x [[buffer(0)]],
    device const bfloat* weight_t [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    constant uint& input_dim [[buffer(3)]],
    constant uint& output_dim [[buffer(4)]],
    constant uint& batch [[buffer(5)]],
    uint2 tgroup [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint TILE_COLS = 8;
    constexpr uint ROW_GROUP_SIZE = 4;
    const uint col_base = (tgroup.x * 4 + simd_group) * TILE_COLS;
    if (col_base >= output_dim) {
        return;
    }

    const uint batch_idx = tgroup.y * ROW_GROUP_SIZE;
    const uint batch1 = batch_idx + 1;
    const uint batch2 = batch_idx + 2;
    const uint batch3 = batch_idx + 3;
    const bool has_batch1 = batch1 < batch;
    const bool has_batch2 = batch2 < batch;
    const bool has_batch3 = batch3 < batch;
    const uint x_base0 = batch_idx * input_dim;
    const uint x_base1 = batch1 * input_dim;
    const uint x_base2 = batch2 * input_dim;
    const uint x_base3 = batch3 * input_dim;
    const bool full_tile = col_base + TILE_COLS - 1 < output_dim;
    const bool vector_load_safe = full_tile && (output_dim % 4 == 0);

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    float acc4 = 0.0f;
    float acc5 = 0.0f;
    float acc6 = 0.0f;
    float acc7 = 0.0f;
    float acc10 = 0.0f;
    float acc11 = 0.0f;
    float acc12 = 0.0f;
    float acc13 = 0.0f;
    float acc14 = 0.0f;
    float acc15 = 0.0f;
    float acc16 = 0.0f;
    float acc17 = 0.0f;
    float acc20 = 0.0f;
    float acc21 = 0.0f;
    float acc22 = 0.0f;
    float acc23 = 0.0f;
    float acc24 = 0.0f;
    float acc25 = 0.0f;
    float acc26 = 0.0f;
    float acc27 = 0.0f;
    float acc30 = 0.0f;
    float acc31 = 0.0f;
    float acc32 = 0.0f;
    float acc33 = 0.0f;
    float acc34 = 0.0f;
    float acc35 = 0.0f;
    float acc36 = 0.0f;
    float acc37 = 0.0f;

    for (uint row = lane; row < input_dim; row += 32) {
        const float xv0 = static_cast<float>(x[x_base0 + row]);
        const float xv1 = has_batch1 ? static_cast<float>(x[x_base1 + row]) : 0.0f;
        const float xv2 = has_batch2 ? static_cast<float>(x[x_base2 + row]) : 0.0f;
        const float xv3 = has_batch3 ? static_cast<float>(x[x_base3 + row]) : 0.0f;
        const uint weight_base = row * output_dim + col_base;
        if (vector_load_safe) {
            device const bfloat4* w4_ptr = (device const bfloat4*)(weight_t + weight_base);
            const bfloat4 w_lo = w4_ptr[0];
            const bfloat4 w_hi = w4_ptr[1];
            const float w0 = static_cast<float>(w_lo[0]);
            const float w1 = static_cast<float>(w_lo[1]);
            const float w2 = static_cast<float>(w_lo[2]);
            const float w3 = static_cast<float>(w_lo[3]);
            const float w4 = static_cast<float>(w_hi[0]);
            const float w5 = static_cast<float>(w_hi[1]);
            const float w6 = static_cast<float>(w_hi[2]);
            const float w7 = static_cast<float>(w_hi[3]);
            acc0 += xv0 * w0;
            acc1 += xv0 * w1;
            acc2 += xv0 * w2;
            acc3 += xv0 * w3;
            acc4 += xv0 * w4;
            acc5 += xv0 * w5;
            acc6 += xv0 * w6;
            acc7 += xv0 * w7;
            acc10 += xv1 * w0;
            acc11 += xv1 * w1;
            acc12 += xv1 * w2;
            acc13 += xv1 * w3;
            acc14 += xv1 * w4;
            acc15 += xv1 * w5;
            acc16 += xv1 * w6;
            acc17 += xv1 * w7;
            acc20 += xv2 * w0;
            acc21 += xv2 * w1;
            acc22 += xv2 * w2;
            acc23 += xv2 * w3;
            acc24 += xv2 * w4;
            acc25 += xv2 * w5;
            acc26 += xv2 * w6;
            acc27 += xv2 * w7;
            acc30 += xv3 * w0;
            acc31 += xv3 * w1;
            acc32 += xv3 * w2;
            acc33 += xv3 * w3;
            acc34 += xv3 * w4;
            acc35 += xv3 * w5;
            acc36 += xv3 * w6;
            acc37 += xv3 * w7;
        } else {
            const float w0 = static_cast<float>(weight_t[weight_base + 0]);
            acc0 += xv0 * w0;
            acc10 += xv1 * w0;
            acc20 += xv2 * w0;
            acc30 += xv3 * w0;
            if (col_base + 1 < output_dim) {
                const float w1 = static_cast<float>(weight_t[weight_base + 1]);
                acc1 += xv0 * w1;
                acc11 += xv1 * w1;
                acc21 += xv2 * w1;
                acc31 += xv3 * w1;
            }
            if (col_base + 2 < output_dim) {
                const float w2 = static_cast<float>(weight_t[weight_base + 2]);
                acc2 += xv0 * w2;
                acc12 += xv1 * w2;
                acc22 += xv2 * w2;
                acc32 += xv3 * w2;
            }
            if (col_base + 3 < output_dim) {
                const float w3 = static_cast<float>(weight_t[weight_base + 3]);
                acc3 += xv0 * w3;
                acc13 += xv1 * w3;
                acc23 += xv2 * w3;
                acc33 += xv3 * w3;
            }
            if (col_base + 4 < output_dim) {
                const float w4 = static_cast<float>(weight_t[weight_base + 4]);
                acc4 += xv0 * w4;
                acc14 += xv1 * w4;
                acc24 += xv2 * w4;
                acc34 += xv3 * w4;
            }
            if (col_base + 5 < output_dim) {
                const float w5 = static_cast<float>(weight_t[weight_base + 5]);
                acc5 += xv0 * w5;
                acc15 += xv1 * w5;
                acc25 += xv2 * w5;
                acc35 += xv3 * w5;
            }
            if (col_base + 6 < output_dim) {
                const float w6 = static_cast<float>(weight_t[weight_base + 6]);
                acc6 += xv0 * w6;
                acc16 += xv1 * w6;
                acc26 += xv2 * w6;
                acc36 += xv3 * w6;
            }
            if (col_base + 7 < output_dim) {
                const float w7 = static_cast<float>(weight_t[weight_base + 7]);
                acc7 += xv0 * w7;
                acc17 += xv1 * w7;
                acc27 += xv2 * w7;
                acc37 += xv3 * w7;
            }
        }
    }

    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    acc2 = simd_sum(acc2);
    acc3 = simd_sum(acc3);
    acc4 = simd_sum(acc4);
    acc5 = simd_sum(acc5);
    acc6 = simd_sum(acc6);
    acc7 = simd_sum(acc7);
    acc10 = simd_sum(acc10);
    acc11 = simd_sum(acc11);
    acc12 = simd_sum(acc12);
    acc13 = simd_sum(acc13);
    acc14 = simd_sum(acc14);
    acc15 = simd_sum(acc15);
    acc16 = simd_sum(acc16);
    acc17 = simd_sum(acc17);
    acc20 = simd_sum(acc20);
    acc21 = simd_sum(acc21);
    acc22 = simd_sum(acc22);
    acc23 = simd_sum(acc23);
    acc24 = simd_sum(acc24);
    acc25 = simd_sum(acc25);
    acc26 = simd_sum(acc26);
    acc27 = simd_sum(acc27);
    acc30 = simd_sum(acc30);
    acc31 = simd_sum(acc31);
    acc32 = simd_sum(acc32);
    acc33 = simd_sum(acc33);
    acc34 = simd_sum(acc34);
    acc35 = simd_sum(acc35);
    acc36 = simd_sum(acc36);
    acc37 = simd_sum(acc37);

    if (lane == 0) {
        const uint out_base0 = batch_idx * output_dim;
        out[out_base0 + col_base + 0] = static_cast<bfloat>(acc0);
        if (col_base + 1 < output_dim) {
            out[out_base0 + col_base + 1] = static_cast<bfloat>(acc1);
        }
        if (col_base + 2 < output_dim) {
            out[out_base0 + col_base + 2] = static_cast<bfloat>(acc2);
        }
        if (col_base + 3 < output_dim) {
            out[out_base0 + col_base + 3] = static_cast<bfloat>(acc3);
        }
        if (col_base + 4 < output_dim) {
            out[out_base0 + col_base + 4] = static_cast<bfloat>(acc4);
        }
        if (col_base + 5 < output_dim) {
            out[out_base0 + col_base + 5] = static_cast<bfloat>(acc5);
        }
        if (col_base + 6 < output_dim) {
            out[out_base0 + col_base + 6] = static_cast<bfloat>(acc6);
        }
        if (col_base + 7 < output_dim) {
            out[out_base0 + col_base + 7] = static_cast<bfloat>(acc7);
        }
        if (has_batch1) {
            const uint out_base1 = batch1 * output_dim;
            out[out_base1 + col_base + 0] = static_cast<bfloat>(acc10);
            if (col_base + 1 < output_dim) {
                out[out_base1 + col_base + 1] = static_cast<bfloat>(acc11);
            }
            if (col_base + 2 < output_dim) {
                out[out_base1 + col_base + 2] = static_cast<bfloat>(acc12);
            }
            if (col_base + 3 < output_dim) {
                out[out_base1 + col_base + 3] = static_cast<bfloat>(acc13);
            }
            if (col_base + 4 < output_dim) {
                out[out_base1 + col_base + 4] = static_cast<bfloat>(acc14);
            }
            if (col_base + 5 < output_dim) {
                out[out_base1 + col_base + 5] = static_cast<bfloat>(acc15);
            }
            if (col_base + 6 < output_dim) {
                out[out_base1 + col_base + 6] = static_cast<bfloat>(acc16);
            }
            if (col_base + 7 < output_dim) {
                out[out_base1 + col_base + 7] = static_cast<bfloat>(acc17);
            }
        }
        if (has_batch2) {
            const uint out_base2 = batch2 * output_dim;
            out[out_base2 + col_base + 0] = static_cast<bfloat>(acc20);
            if (col_base + 1 < output_dim) {
                out[out_base2 + col_base + 1] = static_cast<bfloat>(acc21);
            }
            if (col_base + 2 < output_dim) {
                out[out_base2 + col_base + 2] = static_cast<bfloat>(acc22);
            }
            if (col_base + 3 < output_dim) {
                out[out_base2 + col_base + 3] = static_cast<bfloat>(acc23);
            }
            if (col_base + 4 < output_dim) {
                out[out_base2 + col_base + 4] = static_cast<bfloat>(acc24);
            }
            if (col_base + 5 < output_dim) {
                out[out_base2 + col_base + 5] = static_cast<bfloat>(acc25);
            }
            if (col_base + 6 < output_dim) {
                out[out_base2 + col_base + 6] = static_cast<bfloat>(acc26);
            }
            if (col_base + 7 < output_dim) {
                out[out_base2 + col_base + 7] = static_cast<bfloat>(acc27);
            }
        }
        if (has_batch3) {
            const uint out_base3 = batch3 * output_dim;
            out[out_base3 + col_base + 0] = static_cast<bfloat>(acc30);
            if (col_base + 1 < output_dim) {
                out[out_base3 + col_base + 1] = static_cast<bfloat>(acc31);
            }
            if (col_base + 2 < output_dim) {
                out[out_base3 + col_base + 2] = static_cast<bfloat>(acc32);
            }
            if (col_base + 3 < output_dim) {
                out[out_base3 + col_base + 3] = static_cast<bfloat>(acc33);
            }
            if (col_base + 4 < output_dim) {
                out[out_base3 + col_base + 4] = static_cast<bfloat>(acc34);
            }
            if (col_base + 5 < output_dim) {
                out[out_base3 + col_base + 5] = static_cast<bfloat>(acc35);
            }
            if (col_base + 6 < output_dim) {
                out[out_base3 + col_base + 6] = static_cast<bfloat>(acc36);
            }
            if (col_base + 7 < output_dim) {
                out[out_base3 + col_base + 7] = static_cast<bfloat>(acc37);
            }
        }
    }
}
"#;
pub(super) const METAL_FUSED_QKV_TRANSPOSED_COOP_GEMV_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kiln_fused_qkv_transposed_coop_gemv8_bf16(
    device const bfloat* x [[buffer(0)]],
    device const bfloat* q_t [[buffer(1)]],
    device const bfloat* k_t [[buffer(2)]],
    device const bfloat* v_t [[buffer(3)]],
    device bfloat* q_out [[buffer(4)]],
    device bfloat* k_out [[buffer(5)]],
    device bfloat* v_out [[buffer(6)]],
    constant uint& input_dim [[buffer(7)]],
    constant uint& q_output_dim [[buffer(8)]],
    constant uint& k_output_dim [[buffer(9)]],
    constant uint& v_output_dim [[buffer(10)]],
    uint2 tgroup [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint TILE_COLS = 8;
    constexpr uint SIMD_GROUPS = 4;
    constexpr uint COLS_PER_TGROUP = TILE_COLS * SIMD_GROUPS;

    const uint q_groups = (q_output_dim + COLS_PER_TGROUP - 1) / COLS_PER_TGROUP;
    const uint k_groups = (k_output_dim + COLS_PER_TGROUP - 1) / COLS_PER_TGROUP;
    const uint v_groups = (v_output_dim + COLS_PER_TGROUP - 1) / COLS_PER_TGROUP;
    const uint group = tgroup.x;

    device const bfloat* weight_t = q_t;
    device bfloat* out = q_out;
    uint output_dim = q_output_dim;
    uint group_in_proj = group;
    if (group < q_groups) {
        weight_t = q_t;
        out = q_out;
        output_dim = q_output_dim;
    } else if (group < q_groups + k_groups) {
        weight_t = k_t;
        out = k_out;
        output_dim = k_output_dim;
        group_in_proj = group - q_groups;
    } else if (group < q_groups + k_groups + v_groups) {
        weight_t = v_t;
        out = v_out;
        output_dim = v_output_dim;
        group_in_proj = group - q_groups - k_groups;
    } else {
        return;
    }

    const uint col_base = group_in_proj * COLS_PER_TGROUP + simd_group * TILE_COLS;
    if (col_base >= output_dim) {
        return;
    }

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    float acc4 = 0.0f;
    float acc5 = 0.0f;
    float acc6 = 0.0f;
    float acc7 = 0.0f;
    const bool full_tile = col_base + 7 < output_dim;
    const bool vector_load_safe = full_tile && (output_dim % 4 == 0);

    for (uint row = lane; row < input_dim; row += 32) {
        const float xv = static_cast<float>(x[row]);
        const uint weight_base = row * output_dim + col_base;
        if (vector_load_safe) {
            device const bfloat4* w4_ptr = (device const bfloat4*)(weight_t + weight_base);
            const bfloat4 w0 = w4_ptr[0];
            const bfloat4 w1 = w4_ptr[1];
            acc0 += xv * static_cast<float>(w0[0]);
            acc1 += xv * static_cast<float>(w0[1]);
            acc2 += xv * static_cast<float>(w0[2]);
            acc3 += xv * static_cast<float>(w0[3]);
            acc4 += xv * static_cast<float>(w1[0]);
            acc5 += xv * static_cast<float>(w1[1]);
            acc6 += xv * static_cast<float>(w1[2]);
            acc7 += xv * static_cast<float>(w1[3]);
        } else {
            acc0 += xv * static_cast<float>(weight_t[weight_base + 0]);
            if (col_base + 1 < output_dim) {
                acc1 += xv * static_cast<float>(weight_t[weight_base + 1]);
            }
            if (col_base + 2 < output_dim) {
                acc2 += xv * static_cast<float>(weight_t[weight_base + 2]);
            }
            if (col_base + 3 < output_dim) {
                acc3 += xv * static_cast<float>(weight_t[weight_base + 3]);
            }
            if (col_base + 4 < output_dim) {
                acc4 += xv * static_cast<float>(weight_t[weight_base + 4]);
            }
            if (col_base + 5 < output_dim) {
                acc5 += xv * static_cast<float>(weight_t[weight_base + 5]);
            }
            if (col_base + 6 < output_dim) {
                acc6 += xv * static_cast<float>(weight_t[weight_base + 6]);
            }
            if (col_base + 7 < output_dim) {
                acc7 += xv * static_cast<float>(weight_t[weight_base + 7]);
            }
        }
    }

    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    acc2 = simd_sum(acc2);
    acc3 = simd_sum(acc3);
    acc4 = simd_sum(acc4);
    acc5 = simd_sum(acc5);
    acc6 = simd_sum(acc6);
    acc7 = simd_sum(acc7);

    if (lane == 0) {
        out[col_base + 0] = static_cast<bfloat>(acc0);
        if (col_base + 1 < output_dim) {
            out[col_base + 1] = static_cast<bfloat>(acc1);
        }
        if (col_base + 2 < output_dim) {
            out[col_base + 2] = static_cast<bfloat>(acc2);
        }
        if (col_base + 3 < output_dim) {
            out[col_base + 3] = static_cast<bfloat>(acc3);
        }
        if (col_base + 4 < output_dim) {
            out[col_base + 4] = static_cast<bfloat>(acc4);
        }
        if (col_base + 5 < output_dim) {
            out[col_base + 5] = static_cast<bfloat>(acc5);
        }
        if (col_base + 6 < output_dim) {
            out[col_base + 6] = static_cast<bfloat>(acc6);
        }
        if (col_base + 7 < output_dim) {
            out[col_base + 7] = static_cast<bfloat>(acc7);
        }
    }
}
"#;
pub(super) const METAL_LORA_DELTA_DECODE_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kiln_lora_hidden_decode_bf16(
    device const bfloat* x [[buffer(0)]],
    device const bfloat* a [[buffer(1)]],
    device bfloat* hidden [[buffer(2)]],
    constant uint& batch [[buffer(3)]],
    constant uint& input_dim [[buffer(4)]],
    constant uint& rank [[buffer(5)]],
    uint2 tgroup [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint rank_idx = tgroup.x;
    const uint batch_idx = tgroup.y;
    if (batch_idx >= batch || rank_idx >= rank) {
        return;
    }

    float acc = 0.0f;
    const uint x_base = batch_idx * input_dim;
    const uint a_base = rank_idx * input_dim;
    for (uint col = lane; col < input_dim; col += 32) {
        acc += static_cast<float>(x[x_base + col]) * static_cast<float>(a[a_base + col]);
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        hidden[batch_idx * rank + rank_idx] = static_cast<bfloat>(acc);
    }
}

kernel void kiln_lora_add_decode_bf16(
    device const bfloat* hidden [[buffer(0)]],
    device const bfloat* b [[buffer(1)]],
    device const bfloat* base [[buffer(2)]],
    device bfloat* out [[buffer(3)]],
    constant float& scale [[buffer(4)]],
    constant uint& batch [[buffer(5)]],
    constant uint& output_dim [[buffer(6)]],
    constant uint& rank [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint total = batch * output_dim;
    if (gid >= total) {
        return;
    }
    const uint batch_idx = gid / output_dim;
    const uint output_idx = gid - batch_idx * output_dim;

    float delta = 0.0f;
    const uint hidden_base = batch_idx * rank;
    const uint b_base = output_idx * rank;
    for (uint r = 0; r < rank; ++r) {
        delta += static_cast<float>(hidden[hidden_base + r]) * static_cast<float>(b[b_base + r]);
    }
    const bfloat delta_bf16 = static_cast<bfloat>(scale * delta);
    out[gid] = static_cast<bfloat>(static_cast<float>(base[gid]) + static_cast<float>(delta_bf16));
}
"#;
pub(super) const METAL_GDN_IN_PROJ_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kiln_gdn_in_proj_decode_bf16(
    device const bfloat* x [[buffer(0)]],
    device const bfloat* qkv_t [[buffer(1)]],
    device const bfloat* z_t [[buffer(2)]],
    device const bfloat* a_t [[buffer(3)]],
    device const bfloat* b_t [[buffer(4)]],
    device bfloat* qkv_out [[buffer(5)]],
    device bfloat* z_out [[buffer(6)]],
    device bfloat* a_out [[buffer(7)]],
    device bfloat* b_out [[buffer(8)]],
    constant uint& hidden [[buffer(9)]],
    constant uint& qkv_dim [[buffer(10)]],
    constant uint& z_dim [[buffer(11)]],
    constant uint& nv [[buffer(12)]],
    constant uint& batch [[buffer(13)]],
    constant uint& row_pair_mode [[buffer(14)]],
    uint gid [[thread_position_in_grid]]
) {
    if (batch == 1) {
        if (row_pair_mode == 6 || row_pair_mode == 7) {
            const bool x2_mode = row_pair_mode == 7;
            const uint qkv_pairs = qkv_dim >> 1;
            const uint z_pairs = z_dim >> 1;
            const uint total = qkv_pairs + z_pairs + (nv * 2);
            if (gid >= total) {
                return;
            }

            if (gid < qkv_pairs) {
                const uint col0 = gid << 1;
                float acc0 = 0.0f;
                float acc1 = 0.0f;
                if (x2_mode) {
                    for (uint i = 0; i < hidden; i += 2) {
                        const bfloat2 xv = *(device const bfloat2*)(x + i);
                        const float x0 = static_cast<float>(xv[0]);
                        const float x1 = static_cast<float>(xv[1]);
                        const uint w_idx0 = i * qkv_dim + col0;
                        const uint w_idx1 = w_idx0 + qkv_dim;
                        const bfloat2 w0 = *(device const bfloat2*)(qkv_t + w_idx0);
                        const bfloat2 w1 = *(device const bfloat2*)(qkv_t + w_idx1);
                        acc0 += x0 * static_cast<float>(w0[0]) + x1 * static_cast<float>(w1[0]);
                        acc1 += x0 * static_cast<float>(w0[1]) + x1 * static_cast<float>(w1[1]);
                    }
                } else {
                    for (uint i = 0; i < hidden; ++i) {
                        const float xv = static_cast<float>(x[i]);
                        const uint w_idx = i * qkv_dim + col0;
                        const bfloat2 w = *(device const bfloat2*)(qkv_t + w_idx);
                        acc0 += xv * static_cast<float>(w[0]);
                        acc1 += xv * static_cast<float>(w[1]);
                    }
                }
                qkv_out[col0] = static_cast<bfloat>(acc0);
                qkv_out[col0 + 1] = static_cast<bfloat>(acc1);
            } else if (gid < qkv_pairs + z_pairs) {
                const uint col0 = (gid - qkv_pairs) << 1;
                float acc0 = 0.0f;
                float acc1 = 0.0f;
                if (x2_mode) {
                    for (uint i = 0; i < hidden; i += 2) {
                        const bfloat2 xv = *(device const bfloat2*)(x + i);
                        const float x0 = static_cast<float>(xv[0]);
                        const float x1 = static_cast<float>(xv[1]);
                        const uint w_idx0 = i * z_dim + col0;
                        const uint w_idx1 = w_idx0 + z_dim;
                        const bfloat2 w0 = *(device const bfloat2*)(z_t + w_idx0);
                        const bfloat2 w1 = *(device const bfloat2*)(z_t + w_idx1);
                        acc0 += x0 * static_cast<float>(w0[0]) + x1 * static_cast<float>(w1[0]);
                        acc1 += x0 * static_cast<float>(w0[1]) + x1 * static_cast<float>(w1[1]);
                    }
                } else {
                    for (uint i = 0; i < hidden; ++i) {
                        const float xv = static_cast<float>(x[i]);
                        const uint w_idx = i * z_dim + col0;
                        const bfloat2 w = *(device const bfloat2*)(z_t + w_idx);
                        acc0 += xv * static_cast<float>(w[0]);
                        acc1 += xv * static_cast<float>(w[1]);
                    }
                }
                z_out[col0] = static_cast<bfloat>(acc0);
                z_out[col0 + 1] = static_cast<bfloat>(acc1);
            } else if (gid < qkv_pairs + z_pairs + nv) {
                const uint col = gid - qkv_pairs - z_pairs;
                float acc = 0.0f;
                if (x2_mode) {
                    for (uint i = 0; i < hidden; i += 2) {
                        const bfloat2 xv = *(device const bfloat2*)(x + i);
                        acc += static_cast<float>(xv[0]) * static_cast<float>(a_t[i * nv + col]);
                        acc += static_cast<float>(xv[1]) * static_cast<float>(a_t[(i + 1) * nv + col]);
                    }
                } else {
                    for (uint i = 0; i < hidden; ++i) {
                        acc += static_cast<float>(x[i]) * static_cast<float>(a_t[i * nv + col]);
                    }
                }
                a_out[col] = static_cast<bfloat>(acc);
            } else {
                const uint col = gid - qkv_pairs - z_pairs - nv;
                float acc = 0.0f;
                if (x2_mode) {
                    for (uint i = 0; i < hidden; i += 2) {
                        const bfloat2 xv = *(device const bfloat2*)(x + i);
                        acc += static_cast<float>(xv[0]) * static_cast<float>(b_t[i * nv + col]);
                        acc += static_cast<float>(xv[1]) * static_cast<float>(b_t[(i + 1) * nv + col]);
                    }
                } else {
                    for (uint i = 0; i < hidden; ++i) {
                        acc += static_cast<float>(x[i]) * static_cast<float>(b_t[i * nv + col]);
                    }
                }
                b_out[col] = static_cast<bfloat>(acc);
            }
        } else {
            const uint total = qkv_dim + z_dim + (nv * 2);
            if (gid >= total) {
                return;
            }

            float acc = 0.0f;
            if (gid < qkv_dim) {
                const uint col = gid;
                for (uint i = 0; i < hidden; ++i) {
                    acc += static_cast<float>(x[i]) * static_cast<float>(qkv_t[i * qkv_dim + col]);
                }
                qkv_out[col] = static_cast<bfloat>(acc);
            } else if (gid < qkv_dim + z_dim) {
                const uint col = gid - qkv_dim;
                for (uint i = 0; i < hidden; ++i) {
                    acc += static_cast<float>(x[i]) * static_cast<float>(z_t[i * z_dim + col]);
                }
                z_out[col] = static_cast<bfloat>(acc);
            } else if (gid < qkv_dim + z_dim + nv) {
                const uint col = gid - qkv_dim - z_dim;
                for (uint i = 0; i < hidden; ++i) {
                    acc += static_cast<float>(x[i]) * static_cast<float>(a_t[i * nv + col]);
                }
                a_out[col] = static_cast<bfloat>(acc);
            } else {
                const uint col = gid - qkv_dim - z_dim - nv;
                for (uint i = 0; i < hidden; ++i) {
                    acc += static_cast<float>(x[i]) * static_cast<float>(b_t[i * nv + col]);
                }
                b_out[col] = static_cast<bfloat>(acc);
            }
        }
        return;
    }

    if (row_pair_mode == 0) {
        const uint qkv_pairs = (qkv_dim + 1) >> 1;
        const uint z_pairs = (z_dim + 1) >> 1;
        const uint total = qkv_pairs + z_pairs + (nv * 2);
        if (gid >= total * batch) {
            return;
        }
        const uint batch_idx = gid / total;
        const uint local_gid = gid - batch_idx * total;
        const uint x_base = batch_idx * hidden;

        if (local_gid < qkv_pairs) {
            const uint col0 = local_gid << 1;
            const uint col1 = col0 + 1;
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            for (uint i = 0; i < hidden; ++i) {
                const float xv = static_cast<float>(x[x_base + i]);
                const uint w_idx = i * qkv_dim + col0;
                acc0 += xv * static_cast<float>(qkv_t[w_idx]);
                if (col1 < qkv_dim) {
                    acc1 += xv * static_cast<float>(qkv_t[w_idx + 1]);
                }
            }
            const uint out_base = batch_idx * qkv_dim;
            qkv_out[out_base + col0] = static_cast<bfloat>(acc0);
            if (col1 < qkv_dim) {
                qkv_out[out_base + col1] = static_cast<bfloat>(acc1);
            }
        } else if (local_gid < qkv_pairs + z_pairs) {
            const uint local_z = local_gid - qkv_pairs;
            const uint col0 = local_z << 1;
            const uint col1 = col0 + 1;
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            for (uint i = 0; i < hidden; ++i) {
                const float xv = static_cast<float>(x[x_base + i]);
                const uint w_idx = i * z_dim + col0;
                acc0 += xv * static_cast<float>(z_t[w_idx]);
                if (col1 < z_dim) {
                    acc1 += xv * static_cast<float>(z_t[w_idx + 1]);
                }
            }
            const uint out_base = batch_idx * z_dim;
            z_out[out_base + col0] = static_cast<bfloat>(acc0);
            if (col1 < z_dim) {
                z_out[out_base + col1] = static_cast<bfloat>(acc1);
            }
        } else if (local_gid < qkv_pairs + z_pairs + nv) {
            const uint col = local_gid - qkv_pairs - z_pairs;
            float acc = 0.0f;
            for (uint i = 0; i < hidden; ++i) {
                acc += static_cast<float>(x[x_base + i]) * static_cast<float>(a_t[i * nv + col]);
            }
            a_out[batch_idx * nv + col] = static_cast<bfloat>(acc);
        } else {
            const uint col = local_gid - qkv_pairs - z_pairs - nv;
            float acc = 0.0f;
            for (uint i = 0; i < hidden; ++i) {
                acc += static_cast<float>(x[x_base + i]) * static_cast<float>(b_t[i * nv + col]);
            }
            b_out[batch_idx * nv + col] = static_cast<bfloat>(acc);
        }
        return;
    }

    const uint qkv_pairs = (qkv_dim + 1) >> 1;
    const uint z_pairs = (z_dim + 1) >> 1;
    const uint total = qkv_pairs + z_pairs + (nv * 2);
    if (row_pair_mode == 3) {
        if (gid >= total) {
            return;
        }
        const uint local_gid = gid;
        const uint x_base1 = hidden;
        const uint x_base2 = hidden << 1;

        if (local_gid < qkv_pairs) {
            const uint col0 = local_gid << 1;
            const uint col1 = col0 + 1;
            const bool has_col1 = col1 < qkv_dim;
            float acc00 = 0.0f;
            float acc01 = 0.0f;
            float acc10 = 0.0f;
            float acc11 = 0.0f;
            float acc20 = 0.0f;
            float acc21 = 0.0f;
            for (uint i = 0; i < hidden; ++i) {
                const uint w_idx = i * qkv_dim + col0;
                const float w0 = static_cast<float>(qkv_t[w_idx]);
                const float w1 = has_col1 ? static_cast<float>(qkv_t[w_idx + 1]) : 0.0f;
                const float xv0 = static_cast<float>(x[i]);
                acc00 += xv0 * w0;
                acc01 += xv0 * w1;
                const float xv1 = static_cast<float>(x[x_base1 + i]);
                acc10 += xv1 * w0;
                acc11 += xv1 * w1;
                const float xv2 = static_cast<float>(x[x_base2 + i]);
                acc20 += xv2 * w0;
                acc21 += xv2 * w1;
            }
            qkv_out[col0] = static_cast<bfloat>(acc00);
            if (has_col1) {
                qkv_out[col1] = static_cast<bfloat>(acc01);
            }
            const uint out_base1 = qkv_dim;
            qkv_out[out_base1 + col0] = static_cast<bfloat>(acc10);
            if (has_col1) {
                qkv_out[out_base1 + col1] = static_cast<bfloat>(acc11);
            }
            const uint out_base2 = qkv_dim << 1;
            qkv_out[out_base2 + col0] = static_cast<bfloat>(acc20);
            if (has_col1) {
                qkv_out[out_base2 + col1] = static_cast<bfloat>(acc21);
            }
        } else if (local_gid < qkv_pairs + z_pairs) {
            const uint local_z = local_gid - qkv_pairs;
            const uint col0 = local_z << 1;
            const uint col1 = col0 + 1;
            const bool has_col1 = col1 < z_dim;
            float acc00 = 0.0f;
            float acc01 = 0.0f;
            float acc10 = 0.0f;
            float acc11 = 0.0f;
            float acc20 = 0.0f;
            float acc21 = 0.0f;
            for (uint i = 0; i < hidden; ++i) {
                const uint w_idx = i * z_dim + col0;
                const float w0 = static_cast<float>(z_t[w_idx]);
                const float w1 = has_col1 ? static_cast<float>(z_t[w_idx + 1]) : 0.0f;
                const float xv0 = static_cast<float>(x[i]);
                acc00 += xv0 * w0;
                acc01 += xv0 * w1;
                const float xv1 = static_cast<float>(x[x_base1 + i]);
                acc10 += xv1 * w0;
                acc11 += xv1 * w1;
                const float xv2 = static_cast<float>(x[x_base2 + i]);
                acc20 += xv2 * w0;
                acc21 += xv2 * w1;
            }
            z_out[col0] = static_cast<bfloat>(acc00);
            if (has_col1) {
                z_out[col1] = static_cast<bfloat>(acc01);
            }
            const uint out_base1 = z_dim;
            z_out[out_base1 + col0] = static_cast<bfloat>(acc10);
            if (has_col1) {
                z_out[out_base1 + col1] = static_cast<bfloat>(acc11);
            }
            const uint out_base2 = z_dim << 1;
            z_out[out_base2 + col0] = static_cast<bfloat>(acc20);
            if (has_col1) {
                z_out[out_base2 + col1] = static_cast<bfloat>(acc21);
            }
        } else if (local_gid < qkv_pairs + z_pairs + nv) {
            const uint col = local_gid - qkv_pairs - z_pairs;
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            float acc2 = 0.0f;
            for (uint i = 0; i < hidden; ++i) {
                const float w = static_cast<float>(a_t[i * nv + col]);
                acc0 += static_cast<float>(x[i]) * w;
                acc1 += static_cast<float>(x[x_base1 + i]) * w;
                acc2 += static_cast<float>(x[x_base2 + i]) * w;
            }
            a_out[col] = static_cast<bfloat>(acc0);
            a_out[nv + col] = static_cast<bfloat>(acc1);
            a_out[(nv << 1) + col] = static_cast<bfloat>(acc2);
        } else {
            const uint col = local_gid - qkv_pairs - z_pairs - nv;
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            float acc2 = 0.0f;
            for (uint i = 0; i < hidden; ++i) {
                const float w = static_cast<float>(b_t[i * nv + col]);
                acc0 += static_cast<float>(x[i]) * w;
                acc1 += static_cast<float>(x[x_base1 + i]) * w;
                acc2 += static_cast<float>(x[x_base2 + i]) * w;
            }
            b_out[col] = static_cast<bfloat>(acc0);
            b_out[nv + col] = static_cast<bfloat>(acc1);
            b_out[(nv << 1) + col] = static_cast<bfloat>(acc2);
        }
        return;
    }

    if (row_pair_mode == 4) {
        const uint row_quads = (batch + 3) >> 2;
        if (gid >= total * row_quads) {
            return;
        }
        const uint row_quad = gid / total;
        const uint local_gid = gid - row_quad * total;
        const uint row0 = row_quad << 2;
        const uint row1 = row0 + 1;
        const uint row2 = row0 + 2;
        const uint row3 = row0 + 3;
        const bool has_row1 = row1 < batch;
        const bool has_row2 = row2 < batch;
        const bool has_row3 = row3 < batch;
        const uint x_base0 = row0 * hidden;
        const uint x_base1 = row1 * hidden;
        const uint x_base2 = row2 * hidden;
        const uint x_base3 = row3 * hidden;

        if (local_gid < qkv_pairs) {
            const uint col0 = local_gid << 1;
            const uint col1 = col0 + 1;
            const bool has_col1 = col1 < qkv_dim;
            float acc00 = 0.0f;
            float acc01 = 0.0f;
            float acc10 = 0.0f;
            float acc11 = 0.0f;
            float acc20 = 0.0f;
            float acc21 = 0.0f;
            float acc30 = 0.0f;
            float acc31 = 0.0f;
            for (uint i = 0; i < hidden; ++i) {
                const uint w_idx = i * qkv_dim + col0;
                const float w0 = static_cast<float>(qkv_t[w_idx]);
                const float w1 = has_col1 ? static_cast<float>(qkv_t[w_idx + 1]) : 0.0f;
                const float xv0 = static_cast<float>(x[x_base0 + i]);
                acc00 += xv0 * w0;
                acc01 += xv0 * w1;
                if (has_row1) {
                    const float xv1 = static_cast<float>(x[x_base1 + i]);
                    acc10 += xv1 * w0;
                    acc11 += xv1 * w1;
                }
                if (has_row2) {
                    const float xv2 = static_cast<float>(x[x_base2 + i]);
                    acc20 += xv2 * w0;
                    acc21 += xv2 * w1;
                }
                if (has_row3) {
                    const float xv3 = static_cast<float>(x[x_base3 + i]);
                    acc30 += xv3 * w0;
                    acc31 += xv3 * w1;
                }
            }
            const uint out_base0 = row0 * qkv_dim;
            qkv_out[out_base0 + col0] = static_cast<bfloat>(acc00);
            if (has_col1) {
                qkv_out[out_base0 + col1] = static_cast<bfloat>(acc01);
            }
            if (has_row1) {
                const uint out_base1 = row1 * qkv_dim;
                qkv_out[out_base1 + col0] = static_cast<bfloat>(acc10);
                if (has_col1) {
                    qkv_out[out_base1 + col1] = static_cast<bfloat>(acc11);
                }
            }
            if (has_row2) {
                const uint out_base2 = row2 * qkv_dim;
                qkv_out[out_base2 + col0] = static_cast<bfloat>(acc20);
                if (has_col1) {
                    qkv_out[out_base2 + col1] = static_cast<bfloat>(acc21);
                }
            }
            if (has_row3) {
                const uint out_base3 = row3 * qkv_dim;
                qkv_out[out_base3 + col0] = static_cast<bfloat>(acc30);
                if (has_col1) {
                    qkv_out[out_base3 + col1] = static_cast<bfloat>(acc31);
                }
            }
        } else if (local_gid < qkv_pairs + z_pairs) {
            const uint local_z = local_gid - qkv_pairs;
            const uint col0 = local_z << 1;
            const uint col1 = col0 + 1;
            const bool has_col1 = col1 < z_dim;
            float acc00 = 0.0f;
            float acc01 = 0.0f;
            float acc10 = 0.0f;
            float acc11 = 0.0f;
            float acc20 = 0.0f;
            float acc21 = 0.0f;
            float acc30 = 0.0f;
            float acc31 = 0.0f;
            for (uint i = 0; i < hidden; ++i) {
                const uint w_idx = i * z_dim + col0;
                const float w0 = static_cast<float>(z_t[w_idx]);
                const float w1 = has_col1 ? static_cast<float>(z_t[w_idx + 1]) : 0.0f;
                const float xv0 = static_cast<float>(x[x_base0 + i]);
                acc00 += xv0 * w0;
                acc01 += xv0 * w1;
                if (has_row1) {
                    const float xv1 = static_cast<float>(x[x_base1 + i]);
                    acc10 += xv1 * w0;
                    acc11 += xv1 * w1;
                }
                if (has_row2) {
                    const float xv2 = static_cast<float>(x[x_base2 + i]);
                    acc20 += xv2 * w0;
                    acc21 += xv2 * w1;
                }
                if (has_row3) {
                    const float xv3 = static_cast<float>(x[x_base3 + i]);
                    acc30 += xv3 * w0;
                    acc31 += xv3 * w1;
                }
            }
            const uint out_base0 = row0 * z_dim;
            z_out[out_base0 + col0] = static_cast<bfloat>(acc00);
            if (has_col1) {
                z_out[out_base0 + col1] = static_cast<bfloat>(acc01);
            }
            if (has_row1) {
                const uint out_base1 = row1 * z_dim;
                z_out[out_base1 + col0] = static_cast<bfloat>(acc10);
                if (has_col1) {
                    z_out[out_base1 + col1] = static_cast<bfloat>(acc11);
                }
            }
            if (has_row2) {
                const uint out_base2 = row2 * z_dim;
                z_out[out_base2 + col0] = static_cast<bfloat>(acc20);
                if (has_col1) {
                    z_out[out_base2 + col1] = static_cast<bfloat>(acc21);
                }
            }
            if (has_row3) {
                const uint out_base3 = row3 * z_dim;
                z_out[out_base3 + col0] = static_cast<bfloat>(acc30);
                if (has_col1) {
                    z_out[out_base3 + col1] = static_cast<bfloat>(acc31);
                }
            }
        } else if (local_gid < qkv_pairs + z_pairs + nv) {
            const uint col = local_gid - qkv_pairs - z_pairs;
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            float acc2 = 0.0f;
            float acc3 = 0.0f;
            for (uint i = 0; i < hidden; ++i) {
                const float w = static_cast<float>(a_t[i * nv + col]);
                acc0 += static_cast<float>(x[x_base0 + i]) * w;
                if (has_row1) {
                    acc1 += static_cast<float>(x[x_base1 + i]) * w;
                }
                if (has_row2) {
                    acc2 += static_cast<float>(x[x_base2 + i]) * w;
                }
                if (has_row3) {
                    acc3 += static_cast<float>(x[x_base3 + i]) * w;
                }
            }
            a_out[row0 * nv + col] = static_cast<bfloat>(acc0);
            if (has_row1) {
                a_out[row1 * nv + col] = static_cast<bfloat>(acc1);
            }
            if (has_row2) {
                a_out[row2 * nv + col] = static_cast<bfloat>(acc2);
            }
            if (has_row3) {
                a_out[row3 * nv + col] = static_cast<bfloat>(acc3);
            }
        } else {
            const uint col = local_gid - qkv_pairs - z_pairs - nv;
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            float acc2 = 0.0f;
            float acc3 = 0.0f;
            for (uint i = 0; i < hidden; ++i) {
                const float w = static_cast<float>(b_t[i * nv + col]);
                acc0 += static_cast<float>(x[x_base0 + i]) * w;
                if (has_row1) {
                    acc1 += static_cast<float>(x[x_base1 + i]) * w;
                }
                if (has_row2) {
                    acc2 += static_cast<float>(x[x_base2 + i]) * w;
                }
                if (has_row3) {
                    acc3 += static_cast<float>(x[x_base3 + i]) * w;
                }
            }
            b_out[row0 * nv + col] = static_cast<bfloat>(acc0);
            if (has_row1) {
                b_out[row1 * nv + col] = static_cast<bfloat>(acc1);
            }
            if (has_row2) {
                b_out[row2 * nv + col] = static_cast<bfloat>(acc2);
            }
            if (has_row3) {
                b_out[row3 * nv + col] = static_cast<bfloat>(acc3);
            }
        }
        return;
    }

    const uint row_pairs = (batch + 1) >> 1;
    if (gid >= total * row_pairs) {
        return;
    }
    const uint row_pair = gid / total;
    const uint local_gid = gid - row_pair * total;
    const uint row0 = row_pair << 1;
    const uint row1 = row0 + 1;
    const bool has_row1 = row1 < batch;
    const uint x_base0 = row0 * hidden;
    const uint x_base1 = row1 * hidden;

    if (local_gid < qkv_pairs) {
        const uint col0 = local_gid << 1;
        const uint col1 = col0 + 1;
        const bool has_col1 = col1 < qkv_dim;
        float acc00 = 0.0f;
        float acc01 = 0.0f;
        float acc10 = 0.0f;
        float acc11 = 0.0f;
        for (uint i = 0; i < hidden; ++i) {
            const float xv0 = static_cast<float>(x[x_base0 + i]);
            const uint w_idx = i * qkv_dim + col0;
            const float w0 = static_cast<float>(qkv_t[w_idx]);
            acc00 += xv0 * w0;
            if (has_col1) {
                const float w1 = static_cast<float>(qkv_t[w_idx + 1]);
                acc01 += xv0 * w1;
                if (has_row1) {
                    const float xv1 = static_cast<float>(x[x_base1 + i]);
                    acc10 += xv1 * w0;
                    acc11 += xv1 * w1;
                }
            } else if (has_row1) {
                const float xv1 = static_cast<float>(x[x_base1 + i]);
                acc10 += xv1 * w0;
            }
        }
        const uint out_base0 = row0 * qkv_dim;
        qkv_out[out_base0 + col0] = static_cast<bfloat>(acc00);
        if (has_col1) {
            qkv_out[out_base0 + col1] = static_cast<bfloat>(acc01);
        }
        if (has_row1) {
            const uint out_base1 = row1 * qkv_dim;
            qkv_out[out_base1 + col0] = static_cast<bfloat>(acc10);
            if (has_col1) {
                qkv_out[out_base1 + col1] = static_cast<bfloat>(acc11);
            }
        }
    } else if (local_gid < qkv_pairs + z_pairs) {
        const uint local_z = local_gid - qkv_pairs;
        const uint col0 = local_z << 1;
        const uint col1 = col0 + 1;
        const bool has_col1 = col1 < z_dim;
        float acc00 = 0.0f;
        float acc01 = 0.0f;
        float acc10 = 0.0f;
        float acc11 = 0.0f;
        for (uint i = 0; i < hidden; ++i) {
            const float xv0 = static_cast<float>(x[x_base0 + i]);
            const uint w_idx = i * z_dim + col0;
            const float w0 = static_cast<float>(z_t[w_idx]);
            acc00 += xv0 * w0;
            if (has_col1) {
                const float w1 = static_cast<float>(z_t[w_idx + 1]);
                acc01 += xv0 * w1;
                if (has_row1) {
                    const float xv1 = static_cast<float>(x[x_base1 + i]);
                    acc10 += xv1 * w0;
                    acc11 += xv1 * w1;
                }
            } else if (has_row1) {
                const float xv1 = static_cast<float>(x[x_base1 + i]);
                acc10 += xv1 * w0;
            }
        }
        const uint out_base0 = row0 * z_dim;
        z_out[out_base0 + col0] = static_cast<bfloat>(acc00);
        if (has_col1) {
            z_out[out_base0 + col1] = static_cast<bfloat>(acc01);
        }
        if (has_row1) {
            const uint out_base1 = row1 * z_dim;
            z_out[out_base1 + col0] = static_cast<bfloat>(acc10);
            if (has_col1) {
                z_out[out_base1 + col1] = static_cast<bfloat>(acc11);
            }
        }
    } else if (local_gid < qkv_pairs + z_pairs + nv) {
        const uint col = local_gid - qkv_pairs - z_pairs;
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        for (uint i = 0; i < hidden; ++i) {
            const float w = static_cast<float>(a_t[i * nv + col]);
            acc0 += static_cast<float>(x[x_base0 + i]) * w;
            if (has_row1) {
                acc1 += static_cast<float>(x[x_base1 + i]) * w;
            }
        }
        a_out[row0 * nv + col] = static_cast<bfloat>(acc0);
        if (has_row1) {
            a_out[row1 * nv + col] = static_cast<bfloat>(acc1);
        }
    } else {
        const uint col = local_gid - qkv_pairs - z_pairs - nv;
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        for (uint i = 0; i < hidden; ++i) {
            const float w = static_cast<float>(b_t[i * nv + col]);
            acc0 += static_cast<float>(x[x_base0 + i]) * w;
            if (has_row1) {
                acc1 += static_cast<float>(x[x_base1 + i]) * w;
            }
        }
        b_out[row0 * nv + col] = static_cast<bfloat>(acc0);
        if (has_row1) {
            b_out[row1 * nv + col] = static_cast<bfloat>(acc1);
        }
    }
}
"#;
pub(super) const METAL_PAGED_KV_HEAD_MAJOR_READ_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kiln_paged_kv_head_major_read_bf16(
    device const bfloat* k_pool [[buffer(0)]],
    device const bfloat* v_pool [[buffer(1)]],
    device bfloat* k_out [[buffer(2)]],
    device bfloat* v_out [[buffer(3)]],
    constant uint& start_slot [[buffer(4)]],
    constant uint& seq_len [[buffer(5)]],
    constant uint& heads [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint total = seq_len * heads * head_dim;
    if (gid >= total) {
        return;
    }

    const uint d = gid % head_dim;
    const uint h = (gid / head_dim) % heads;
    const uint t = gid / (head_dim * heads);
    const uint pool_idx = ((start_slot + t) * heads + h) * head_dim + d;
    const uint out_idx = (h * seq_len + t) * head_dim + d;

    k_out[out_idx] = k_pool[pool_idx];
    v_out[out_idx] = v_pool[pool_idx];
}
"#;
pub(super) const METAL_PAGED_KV_HEAD_MAJOR_READ_APPEND_TOKEN_MAJOR_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kiln_paged_kv_head_major_read_append_token_major_bf16(
    device const bfloat* k_pool [[buffer(0)]],
    device const bfloat* v_pool [[buffer(1)]],
    device const bfloat* k_tail [[buffer(2)]],
    device const bfloat* v_tail [[buffer(3)]],
    device bfloat* k_out [[buffer(4)]],
    device bfloat* v_out [[buffer(5)]],
    constant uint& start_slot [[buffer(6)]],
    constant uint& prefix_len [[buffer(7)]],
    constant uint& tail_len [[buffer(8)]],
    constant uint& heads [[buffer(9)]],
    constant uint& head_dim [[buffer(10)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint total_len = prefix_len + tail_len;
    const uint total = total_len * heads * head_dim;
    if (gid >= total) {
        return;
    }

    const uint d = gid % head_dim;
    const uint h = (gid / head_dim) % heads;
    const uint t = gid / (head_dim * heads);
    const uint out_idx = (h * total_len + t) * head_dim + d;

    if (t < prefix_len) {
        const uint pool_idx = ((start_slot + t) * heads + h) * head_dim + d;
        k_out[out_idx] = k_pool[pool_idx];
        v_out[out_idx] = v_pool[pool_idx];
    } else {
        const uint tail_t = t - prefix_len;
        const uint tail_idx = (tail_t * heads + h) * head_dim + d;
        k_out[out_idx] = k_tail[tail_idx];
        v_out[out_idx] = v_tail[tail_idx];
    }
}
"#;
pub(super) const METAL_PAGED_ATTN_DECODE_CONTIGUOUS_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kiln_paged_attn_decode_contiguous_bf16_d256(
    device const bfloat* q [[buffer(0)]],
    device const bfloat* k_pool [[buffer(1)]],
    device const bfloat* v_pool [[buffer(2)]],
    device bfloat* out [[buffer(3)]],
    constant uint& start_slot [[buffer(4)]],
    constant uint& seq_len [[buffer(5)]],
    constant uint& q_heads [[buffer(6)]],
    constant uint& kv_heads [[buffer(7)]],
    constant float& scale [[buffer(8)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
    constexpr uint D = 256;
    constexpr uint BN = 32;
    constexpr uint BD = 32;
    constexpr uint EPT = D / BD;
    constexpr uint QWEN_HEADS_PER_KV = 4;

    const uint head_idx = tid.y;
    if (head_idx >= q_heads) {
        return;
    }

    const uint kv_head_idx = head_idx / QWEN_HEADS_PER_KV;
    if (kv_head_idx >= kv_heads) {
        return;
    }

    thread float q_frag[EPT];
    thread float k_frag[EPT];
    thread float o_frag[EPT];
    threadgroup float outputs[BN * BD];
    threadgroup float max_scores[BN];
    threadgroup float sum_exp_scores[BN];

    device const bfloat* q_ptr = q + head_idx * D + simd_lid * EPT;
    device const bfloat* k_ptr =
        k_pool + ((start_slot + simd_gid) * kv_heads + kv_head_idx) * D + simd_lid * EPT;
    device const bfloat* v_ptr =
        v_pool + ((start_slot + simd_gid) * kv_heads + kv_head_idx) * D + simd_lid * EPT;
    device bfloat* out_ptr = out + head_idx * D + simd_gid * EPT;

    for (uint i = 0; i < EPT; ++i) {
        q_frag[i] = scale * static_cast<float>(q_ptr[i]);
        o_frag[i] = 0.0f;
    }

    float max_score = -INFINITY;
    float sum_exp_score = 0.0f;

    for (uint t = simd_gid; t < seq_len; t += BN) {
        for (uint i = 0; i < EPT; ++i) {
            k_frag[i] = static_cast<float>(k_ptr[i]);
        }

        float score = 0.0f;
        for (uint i = 0; i < EPT; ++i) {
            score += q_frag[i] * k_frag[i];
        }
        score = simd_sum(score);

        const float new_max = max(max_score, score);
        const float factor = fast::exp(max_score - new_max);
        const float exp_score = fast::exp(score - new_max);

        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;

        for (uint i = 0; i < EPT; ++i) {
            o_frag[i] = o_frag[i] * factor + exp_score * static_cast<float>(v_ptr[i]);
        }

        k_ptr += BN * kv_heads * D;
        v_ptr += BN * kv_heads * D;
    }

    if (simd_lid == 0) {
        max_scores[simd_gid] = max_score;
        sum_exp_scores[simd_gid] = sum_exp_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float partial_max = max_scores[simd_lid];
    const float global_max = simd_max(partial_max);
    const float partial_factor = fast::exp(partial_max - global_max);
    const float denom = simd_sum(sum_exp_scores[simd_lid] * partial_factor);

    for (uint i = 0; i < EPT; ++i) {
        outputs[simd_lid * BD + simd_gid] = o_frag[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o_frag[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * partial_factor) / denom;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (simd_lid == 0) {
        for (uint i = 0; i < EPT; ++i) {
            out_ptr[i] = static_cast<bfloat>(o_frag[i]);
        }
    }
}

kernel void kiln_paged_attn_decode_contiguous_batch_bf16_d256(
    device const bfloat* q [[buffer(0)]],
    device const bfloat* k_pool [[buffer(1)]],
    device const bfloat* v_pool [[buffer(2)]],
    device bfloat* out [[buffer(3)]],
    device const uint* start_slots [[buffer(4)]],
    constant uint& batch [[buffer(5)]],
    constant uint& seq_len [[buffer(6)]],
    constant uint& q_heads [[buffer(7)]],
    constant uint& kv_heads [[buffer(8)]],
    constant float& scale [[buffer(9)]],
    constant uint& total_slots [[buffer(10)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
    constexpr uint D = 256;
    constexpr uint BN = 32;
    constexpr uint BD = 32;
    constexpr uint EPT = D / BD;
    constexpr uint QWEN_HEADS_PER_KV = 4;

    const uint batch_idx = tid.x;
    const uint head_idx = tid.y;
    if (batch_idx >= batch || head_idx >= q_heads) {
        return;
    }

    const uint kv_head_idx = head_idx / QWEN_HEADS_PER_KV;
    if (kv_head_idx >= kv_heads) {
        return;
    }

    device bfloat* out_ptr =
        out + (batch_idx * q_heads + head_idx) * D + simd_gid * EPT;
    const uint start_slot = start_slots[batch_idx];
    if (start_slot >= total_slots || seq_len == 0 || seq_len > total_slots - start_slot) {
        if (simd_lid == 0) {
            for (uint i = 0; i < EPT; ++i) {
                out_ptr[i] = static_cast<bfloat>(0.0f);
            }
        }
        return;
    }

    thread float q_frag[EPT];
    thread float k_frag[EPT];
    thread float o_frag[EPT];
    threadgroup float outputs[BN * BD];
    threadgroup float max_scores[BN];
    threadgroup float sum_exp_scores[BN];

    device const bfloat* q_ptr =
        q + (batch_idx * q_heads + head_idx) * D + simd_lid * EPT;
    device const bfloat* k_ptr =
        k_pool + ((start_slot + simd_gid) * kv_heads + kv_head_idx) * D + simd_lid * EPT;
    device const bfloat* v_ptr =
        v_pool + ((start_slot + simd_gid) * kv_heads + kv_head_idx) * D + simd_lid * EPT;

    for (uint i = 0; i < EPT; ++i) {
        q_frag[i] = scale * static_cast<float>(q_ptr[i]);
        o_frag[i] = 0.0f;
    }

    float max_score = -INFINITY;
    float sum_exp_score = 0.0f;

    for (uint t = simd_gid; t < seq_len; t += BN) {
        for (uint i = 0; i < EPT; ++i) {
            k_frag[i] = static_cast<float>(k_ptr[i]);
        }

        float score = 0.0f;
        for (uint i = 0; i < EPT; ++i) {
            score += q_frag[i] * k_frag[i];
        }
        score = simd_sum(score);

        const float new_max = max(max_score, score);
        const float factor = fast::exp(max_score - new_max);
        const float exp_score = fast::exp(score - new_max);

        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;

        for (uint i = 0; i < EPT; ++i) {
            o_frag[i] = o_frag[i] * factor + exp_score * static_cast<float>(v_ptr[i]);
        }

        k_ptr += BN * kv_heads * D;
        v_ptr += BN * kv_heads * D;
    }

    if (simd_lid == 0) {
        max_scores[simd_gid] = max_score;
        sum_exp_scores[simd_gid] = sum_exp_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float partial_max = max_scores[simd_lid];
    const float global_max = simd_max(partial_max);
    const float partial_factor = fast::exp(partial_max - global_max);
    const float denom = simd_sum(sum_exp_scores[simd_lid] * partial_factor);

    for (uint i = 0; i < EPT; ++i) {
        outputs[simd_lid * BD + simd_gid] = o_frag[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o_frag[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * partial_factor) / denom;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (simd_lid == 0) {
        for (uint i = 0; i < EPT; ++i) {
            out_ptr[i] = static_cast<bfloat>(o_frag[i]);
        }
    }
}

kernel void kiln_paged_attn_decode_contiguous_batch_dyn_seqlen_bf16_d256(
    device const bfloat* q [[buffer(0)]],
    device const bfloat* k_pool [[buffer(1)]],
    device const bfloat* v_pool [[buffer(2)]],
    device bfloat* out [[buffer(3)]],
    device const uint* block_table [[buffer(4)]],
    device const int* seqused_k [[buffer(5)]],
    constant uint& batch [[buffer(6)]],
    constant uint& max_blocks_per_seq [[buffer(7)]],
    constant uint& max_seqlen_k [[buffer(8)]],
    constant uint& page_block_size [[buffer(9)]],
    constant uint& q_heads [[buffer(10)]],
    constant uint& kv_heads [[buffer(11)]],
    constant float& scale [[buffer(12)]],
    constant uint& total_slots [[buffer(13)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
    constexpr uint D = 256;
    constexpr uint BN = 32;
    constexpr uint BD = 32;
    constexpr uint EPT = D / BD;
    constexpr uint QWEN_HEADS_PER_KV = 4;

    const uint batch_idx = tid.x;
    const uint head_idx = tid.y;
    if (batch_idx >= batch || head_idx >= q_heads) {
        return;
    }

    const uint kv_head_idx = head_idx / QWEN_HEADS_PER_KV;
    if (kv_head_idx >= kv_heads) {
        return;
    }

    device bfloat* out_ptr =
        out + (batch_idx * q_heads + head_idx) * D + simd_gid * EPT;
    const int row_len_i = seqused_k[batch_idx];
    if (row_len_i <= 0 || page_block_size == 0 || max_blocks_per_seq == 0) {
        if (simd_lid == 0) {
            for (uint i = 0; i < EPT; ++i) {
                out_ptr[i] = static_cast<bfloat>(0.0f);
            }
        }
        return;
    }
    const uint row_len = min(static_cast<uint>(row_len_i), max_seqlen_k);

    thread float q_frag[EPT];
    thread float k_frag[EPT];
    thread float o_frag[EPT];
    threadgroup float outputs[BN * BD];
    threadgroup float max_scores[BN];
    threadgroup float sum_exp_scores[BN];

    device const bfloat* q_ptr =
        q + (batch_idx * q_heads + head_idx) * D + simd_lid * EPT;

    for (uint i = 0; i < EPT; ++i) {
        q_frag[i] = scale * static_cast<float>(q_ptr[i]);
        o_frag[i] = 0.0f;
    }

    float max_score = -INFINITY;
    float sum_exp_score = 0.0f;

    for (uint t = simd_gid; t < row_len; t += BN) {
        const uint block_idx = t / page_block_size;
        const uint block_offset = t - block_idx * page_block_size;
        if (block_idx >= max_blocks_per_seq) {
            continue;
        }
        const uint physical_block = block_table[batch_idx * max_blocks_per_seq + block_idx];
        const uint pool_slot = physical_block * page_block_size + block_offset;
        if (pool_slot >= total_slots) {
            continue;
        }
        device const bfloat* k_ptr =
            k_pool + (pool_slot * kv_heads + kv_head_idx) * D + simd_lid * EPT;
        device const bfloat* v_ptr =
            v_pool + (pool_slot * kv_heads + kv_head_idx) * D + simd_lid * EPT;

        for (uint i = 0; i < EPT; ++i) {
            k_frag[i] = static_cast<float>(k_ptr[i]);
        }

        float score = 0.0f;
        for (uint i = 0; i < EPT; ++i) {
            score += q_frag[i] * k_frag[i];
        }
        score = simd_sum(score);

        const float new_max = max(max_score, score);
        const float factor = fast::exp(max_score - new_max);
        const float exp_score = fast::exp(score - new_max);

        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;

        for (uint i = 0; i < EPT; ++i) {
            o_frag[i] = o_frag[i] * factor + exp_score * static_cast<float>(v_ptr[i]);
        }
    }

    if (simd_lid == 0) {
        max_scores[simd_gid] = max_score;
        sum_exp_scores[simd_gid] = sum_exp_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float partial_max = max_scores[simd_lid];
    const float global_max = simd_max(partial_max);
    const float partial_factor = fast::exp(partial_max - global_max);
    const float denom = simd_sum(sum_exp_scores[simd_lid] * partial_factor);

    for (uint i = 0; i < EPT; ++i) {
        outputs[simd_lid * BD + simd_gid] = o_frag[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o_frag[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * partial_factor) / denom;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (simd_lid == 0) {
        for (uint i = 0; i < EPT; ++i) {
            out_ptr[i] = static_cast<bfloat>(o_frag[i]);
        }
    }
}
"#;
pub(super) const METAL_PAGED_KV_WRITE_TOKEN_MAJOR_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kiln_paged_kv_write_token_major_bf16(
    device const bfloat* k_src [[buffer(0)]],
    device const bfloat* v_src [[buffer(1)]],
    device bfloat* k_pool [[buffer(2)]],
    device bfloat* v_pool [[buffer(3)]],
    constant uint& slot [[buffer(4)]],
    constant uint& heads [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint total = heads * head_dim;
    if (gid >= total) {
        return;
    }

    const uint pool_idx = slot * total + gid;
    k_pool[pool_idx] = k_src[gid];
    v_pool[pool_idx] = v_src[gid];
}

kernel void kiln_paged_kv_write_token_major_batch_bf16(
    device const bfloat* k_src [[buffer(0)]],
    device const bfloat* v_src [[buffer(1)]],
    device bfloat* k_pool [[buffer(2)]],
    device bfloat* v_pool [[buffer(3)]],
    device const uint* slots [[buffer(4)]],
    constant uint& batch [[buffer(5)]],
    constant uint& heads [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    constant uint& total_slots [[buffer(8)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint row_stride = heads * head_dim;
    const uint total = batch * row_stride;
    if (gid >= total) {
        return;
    }

    const uint batch_idx = gid / row_stride;
    const uint local = gid - batch_idx * row_stride;
    const uint slot = slots[batch_idx];
    if (slot >= total_slots) {
        return;
    }
    const uint pool_idx = slot * row_stride + local;
    k_pool[pool_idx] = k_src[gid];
    v_pool[pool_idx] = v_src[gid];
}
"#;
pub(super) const METAL_GDN_GATES_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

inline float kiln_stable_sigmoid(float x) {
    if (x >= 0.0f) {
        return 1.0f / (1.0f + exp(-x));
    }
    const float e = exp(x);
    return e / (1.0f + e);
}

inline float kiln_stable_softplus(float x) {
    if (x > 20.0f) {
        return x;
    }
    if (x < -20.0f) {
        return exp(x);
    }
    return log(1.0f + exp(x));
}

kernel void kiln_gdn_gates_bf16(
    device const bfloat* a [[buffer(0)]],
    device const bfloat* b [[buffer(1)]],
    device const float* a_log [[buffer(2)]],
    device const bfloat* dt_bias [[buffer(3)]],
    device bfloat* beta_out [[buffer(4)]],
    device bfloat* g_out [[buffer(5)]],
    constant uint& nv [[buffer(6)]],
    constant uint& total [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= total) {
        return;
    }

    const uint h = gid % nv;
    const float a_val = static_cast<float>(a[gid]);
    const float b_val = static_cast<float>(b[gid]);
    const float a_log_val = static_cast<float>(a_log[h]);
    const float dt_bias_val = static_cast<float>(dt_bias[h]);

    const float beta = kiln_stable_sigmoid(b_val);
    const float sp = kiln_stable_softplus(a_val + dt_bias_val);
    const float g = sp * -exp(a_log_val);

    beta_out[gid] = static_cast<bfloat>(beta);
    g_out[gid] = static_cast<bfloat>(g);
}

kernel void kiln_gdn_gates_decay_bf16(
    device const bfloat* a [[buffer(0)]],
    device const bfloat* b [[buffer(1)]],
    device const float* a_log [[buffer(2)]],
    device const bfloat* dt_bias [[buffer(3)]],
    device bfloat* beta_out [[buffer(4)]],
    device bfloat* decay_out [[buffer(5)]],
    constant uint& nv [[buffer(6)]],
    constant uint& total [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= total) {
        return;
    }

    const uint h = gid % nv;
    const float a_val = static_cast<float>(a[gid]);
    const float b_val = static_cast<float>(b[gid]);
    const float a_log_val = static_cast<float>(a_log[h]);
    const float dt_bias_val = static_cast<float>(dt_bias[h]);

    const float beta = kiln_stable_sigmoid(b_val);
    const float sp = kiln_stable_softplus(a_val + dt_bias_val);
    const float g = sp * -exp(a_log_val);
    const bfloat g_bf = static_cast<bfloat>(g);

    beta_out[gid] = static_cast<bfloat>(beta);
    decay_out[gid] = static_cast<bfloat>(exp(static_cast<float>(g_bf)));
}

kernel void kiln_gdn_gates_decay_ab_bf16(
    device const bfloat* ab [[buffer(0)]],
    device const float* a_log [[buffer(1)]],
    device const bfloat* dt_bias [[buffer(2)]],
    device bfloat* beta_out [[buffer(3)]],
    device bfloat* decay_out [[buffer(4)]],
    constant uint& nv [[buffer(5)]],
    constant uint& total [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= total) {
        return;
    }

    const uint h = gid % nv;
    const uint row = gid / nv;
    const uint ab_base = row * (nv * 2);
    const float a_val = static_cast<float>(ab[ab_base + h]);
    const float b_val = static_cast<float>(ab[ab_base + nv + h]);
    const float a_log_val = static_cast<float>(a_log[h]);
    const float dt_bias_val = static_cast<float>(dt_bias[h]);

    const float beta = kiln_stable_sigmoid(b_val);
    const float sp = kiln_stable_softplus(a_val + dt_bias_val);
    const float g = sp * -exp(a_log_val);
    const bfloat g_bf = static_cast<bfloat>(g);

    beta_out[gid] = static_cast<bfloat>(beta);
    decay_out[gid] = static_cast<bfloat>(exp(static_cast<float>(g_bf)));
}
"#;
pub(super) const METAL_GDN_DECODE_GATES_RECURRENT_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kiln_gdn_decode_gates_recurrent_bf16(
    device const bfloat* q [[buffer(0)]],
    device const bfloat* k [[buffer(1)]],
    device const bfloat* v [[buffer(2)]],
    device const bfloat* a [[buffer(3)]],
    device const bfloat* b [[buffer(4)]],
    device const float* a_log [[buffer(5)]],
    device const bfloat* dt_bias [[buffer(6)]],
    device bfloat* state [[buffer(7)]],
    device bfloat* out [[buffer(8)]],
    constant uint& batch_heads [[buffer(9)]],
    constant uint& dk [[buffer(10)]],
    constant uint& dv [[buffer(11)]],
    constant uint& value_heads [[buffer(12)]],
    constant uint& q_heads [[buffer(13)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    constexpr uint NSG = 4;
    constexpr uint LANES = 32;
    if (gid >= batch_heads * dv || tid >= LANES || dk != 128 || dv != 128) {
        return;
    }

    const uint bh = gid / dv;
    const uint d = gid - bh * dv;
    const uint batch_idx = bh / value_heads;
    const uint head_idx = bh - batch_idx * value_heads;
    const uint q_group = value_heads / q_heads;
    const uint q_head_idx = head_idx / q_group;
    const uint qk_base = (batch_idx * q_heads + q_head_idx) * dk;
    const uint v_base = (batch_idx * value_heads + head_idx) * dv;
    const uint gate_idx = batch_idx * value_heads + head_idx;
    const uint state_base = bh * dk * dv;

    const float beta = static_cast<float>(static_cast<bfloat>(
        kiln_stable_sigmoid(static_cast<float>(b[gate_idx]))
    ));
    const float g = static_cast<float>(static_cast<bfloat>(
        kiln_stable_softplus(static_cast<float>(a[gate_idx]) + static_cast<float>(dt_bias[head_idx])) *
        -exp(static_cast<float>(a_log[head_idx]))
    ));
    const float decay = static_cast<float>(static_cast<bfloat>(exp(g)));

    float ls[NSG];
    for (uint j = 0; j < NSG; ++j) {
        const uint is = tid * NSG + j;
        ls[j] = static_cast<float>(state[state_base + is * dv + d]);
    }

    float s_k = 0.0f;
    for (uint j = 0; j < NSG; ++j) {
        const uint is = tid * NSG + j;
        const float decayed = static_cast<float>(static_cast<bfloat>(ls[j] * decay));
        ls[j] = decayed;
        s_k += decayed * static_cast<float>(k[qk_base + is]);
    }
    s_k = simd_sum(s_k);

    const float delta = static_cast<float>(static_cast<bfloat>(
        (static_cast<float>(v[v_base + d]) - s_k) * beta
    ));

    float y = 0.0f;
    for (uint j = 0; j < NSG; ++j) {
        const uint is = tid * NSG + j;
        const float new_s = static_cast<float>(static_cast<bfloat>(
            ls[j] + static_cast<float>(k[qk_base + is]) * delta
        ));
        ls[j] = new_s;
        y += new_s * static_cast<float>(q[qk_base + is]);
        state[state_base + is * dv + d] = static_cast<bfloat>(new_s);
    }
    y = simd_sum(y);

    if (tid == 0) {
        out[v_base + d] = static_cast<bfloat>(y);
    }
}
"#;
pub(super) const METAL_GDN_DECODE_GATES_RECURRENT_RMSNORM_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kiln_gdn_decode_gates_recurrent_rmsnorm_bf16(
    device const bfloat* q [[buffer(0)]],
    device const bfloat* k [[buffer(1)]],
    device const bfloat* v [[buffer(2)]],
    device const bfloat* a [[buffer(3)]],
    device const bfloat* b [[buffer(4)]],
    device const float* a_log [[buffer(5)]],
    device const bfloat* dt_bias [[buffer(6)]],
    device bfloat* state [[buffer(7)]],
    device const bfloat* z [[buffer(8)]],
    device const float* weight [[buffer(9)]],
    device bfloat* out [[buffer(10)]],
    constant uint& batch_heads [[buffer(11)]],
    constant uint& dk [[buffer(12)]],
    constant uint& dv [[buffer(13)]],
    constant uint& value_heads [[buffer(14)]],
    constant uint& q_heads [[buffer(15)]],
    constant float& eps [[buffer(16)]],
    uint bh [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float scratch[128];
    if (bh >= batch_heads || tid >= dv || dk != 128 || dv != 128) {
        return;
    }

    const uint d = tid;
    const uint batch_idx = bh / value_heads;
    const uint head_idx = bh - batch_idx * value_heads;
    const uint q_group = value_heads / q_heads;
    const uint q_head_idx = head_idx / q_group;
    const uint qk_base = (batch_idx * q_heads + q_head_idx) * dk;
    const uint v_base = (batch_idx * value_heads + head_idx) * dv;
    const uint gate_idx = batch_idx * value_heads + head_idx;
    const uint state_base = bh * dk * dv;

    const float beta = static_cast<float>(static_cast<bfloat>(
        kiln_stable_sigmoid(static_cast<float>(b[gate_idx]))
    ));
    const float g = static_cast<float>(static_cast<bfloat>(
        kiln_stable_softplus(static_cast<float>(a[gate_idx]) + static_cast<float>(dt_bias[head_idx])) *
        -exp(static_cast<float>(a_log[head_idx]))
    ));
    const float decay = static_cast<float>(static_cast<bfloat>(exp(g)));

    float s_k = 0.0f;
    for (uint i = 0; i < dk; ++i) {
        const uint state_idx = state_base + i * dv + d;
        const float decayed = static_cast<float>(static_cast<bfloat>(
            static_cast<float>(state[state_idx]) * decay
        ));
        state[state_idx] = static_cast<bfloat>(decayed);
        s_k += decayed * static_cast<float>(k[qk_base + i]);
    }
    const float delta = static_cast<float>(static_cast<bfloat>(
        (static_cast<float>(v[v_base + d]) - s_k) * beta
    ));

    float y = 0.0f;
    for (uint i = 0; i < dk; ++i) {
        const uint state_idx = state_base + i * dv + d;
        const float new_s = static_cast<float>(static_cast<bfloat>(
            static_cast<float>(state[state_idx]) + static_cast<float>(k[qk_base + i]) * delta
        ));
        state[state_idx] = static_cast<bfloat>(new_s);
        y += new_s * static_cast<float>(q[qk_base + i]);
    }

    scratch[tid] = y * y;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 64; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float rms_inv = rsqrt((scratch[0] / static_cast<float>(dv)) + eps);
    const float zv = static_cast<float>(z[v_base + d]);
    const float gate = zv / (1.0f + exp(-zv));
    out[v_base + d] = static_cast<bfloat>(
        y * rms_inv * static_cast<float>(weight[d]) * gate
    );
}
"#;
pub(super) const METAL_GATED_RMSNORM_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kiln_gated_rmsnorm_bf16(
    device const bfloat* x [[buffer(0)]],
    device const bfloat* z [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    device bfloat* out [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& hidden [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    constant uint& threadgroup_width [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float scratch[1024];

    const uint row = gid.y;
    if (row >= rows) {
        return;
    }

    const uint base = row * hidden;
    float sum_sq = 0.0f;
    if (tid < hidden) {
        const float xv = static_cast<float>(x[base + tid]);
        sum_sq = xv * xv;
    }
    scratch[tid] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threadgroup_width / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid < hidden) {
        const float rms_inv = rsqrt((scratch[0] / static_cast<float>(hidden)) + eps);
        const float xv = static_cast<float>(x[base + tid]);
        const float zv = static_cast<float>(z[base + tid]);
        const float gate = zv / (1.0f + exp(-zv));
        out[base + tid] = static_cast<bfloat>(xv * rms_inv * static_cast<float>(weight[tid]) * gate);
    }
}
"#;
pub(super) const METAL_GDN_RECURRENT_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kiln_gdn_recurrent_bf16(
    device const bfloat* q [[buffer(0)]],
    device const bfloat* k [[buffer(1)]],
    device const bfloat* v [[buffer(2)]],
    device const bfloat* beta [[buffer(3)]],
    device const bfloat* g [[buffer(4)]],
    device bfloat* state [[buffer(5)]],
    device bfloat* out [[buffer(6)]],
    constant uint& batch_heads [[buffer(7)]],
    constant uint& dk [[buffer(8)]],
    constant uint& dv [[buffer(9)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint total = batch_heads * dv;
    if (gid >= total) {
        return;
    }

    const uint bh = gid / dv;
    const uint col = gid - bh * dv;
    const uint qk_base = bh * dk;
    const uint v_base = bh * dv;
    const uint state_base = bh * dk * dv;

    const float decay = exp(static_cast<float>(g[bh]));
    const float beta_t = static_cast<float>(beta[bh]);

    float v_pred = 0.0f;
    for (uint i = 0; i < dk; ++i) {
        const float k_i = static_cast<float>(k[qk_base + i]);
        const float s_i = static_cast<float>(state[state_base + i * dv + col]);
        v_pred += k_i * (decay * s_i);
    }

    const float v_t = static_cast<float>(v[v_base + col]);
    const float delta = beta_t * (v_t - v_pred);

    float out_acc = 0.0f;
    for (uint i = 0; i < dk; ++i) {
        const float q_i = static_cast<float>(q[qk_base + i]);
        const float k_i = static_cast<float>(k[qk_base + i]);
        const uint state_idx = state_base + i * dv + col;
        const float old_s = static_cast<float>(state[state_idx]);
        const float new_s = decay * old_s + k_i * delta;
        state[state_idx] = static_cast<bfloat>(new_s);
        out_acc += q_i * new_s;
    }

    out[v_base + col] = static_cast<bfloat>(out_acc);
}

kernel void kiln_gdn_forward_substitution_bf16(
    device const bfloat* a_strict [[buffer(0)]],
    device const bfloat* v_prime [[buffer(1)]],
    device const bfloat* beta [[buffer(2)]],
    device bfloat* out [[buffer(3)]],
    constant uint& batch_heads [[buffer(4)]],
    constant uint& chunk_size [[buffer(5)]],
    constant uint& dv [[buffer(6)]],
    uint bh [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    if (bh >= batch_heads) {
        return;
    }

    // Conservative Qwen3.5 envelope: C <= 64, dv <= 128. Static threadgroup
    // storage keeps the kernel simple and under Apple Silicon's common 32 KiB
    // per-threadgroup memory budget: (64*64 + 64*128) bf16 = 24 KiB.
    threadgroup bfloat sA[4096];
    threadgroup bfloat sW[8192];

    const uint a_base = bh * chunk_size * chunk_size;
    const uint v_base = bh * chunk_size * dv;
    const uint beta_base = bh * chunk_size;
    const uint total_a = chunk_size * chunk_size;

    for (uint i = tid; i < total_a; i += 128) {
        sA[i] = a_strict[a_base + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint t = 0; t < chunk_size; ++t) {
        const float beta_t = static_cast<float>(beta[beta_base + t]);

        for (uint d = tid; d < dv; d += 128) {
            float acc = 0.0f;
            for (uint i = 0; i < t; ++i) {
                const float a = static_cast<float>(sA[t * chunk_size + i]);
                const float w = static_cast<float>(sW[i * dv + d]);
                acc += a * w;
            }

            const uint row_col = t * dv + d;
            const float vp = static_cast<float>(v_prime[v_base + row_col]);
            const bfloat w = static_cast<bfloat>(beta_t * (vp - acc));
            sW[row_col] = w;
            out[v_base + row_col] = w;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void kiln_gdn_forward_substitution_f32(
    device const float* a_strict [[buffer(0)]],
    device const float* v_prime [[buffer(1)]],
    device const float* beta [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& batch_heads [[buffer(4)]],
    constant uint& chunk_size [[buffer(5)]],
    constant uint& dv [[buffer(6)]],
    uint bh [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    if (bh >= batch_heads) {
        return;
    }

    threadgroup float sW[8192];

    const uint a_base = bh * chunk_size * chunk_size;
    const uint v_base = bh * chunk_size * dv;
    const uint beta_base = bh * chunk_size;

    for (uint t = 0; t < chunk_size; ++t) {
        const float beta_t = beta[beta_base + t];

        for (uint d = tid; d < dv; d += 128) {
            float acc = 0.0f;
            for (uint i = 0; i < t; ++i) {
                acc += a_strict[a_base + t * chunk_size + i] * sW[i * dv + d];
            }

            const uint row_col = t * dv + d;
            const float w = beta_t * (v_prime[v_base + row_col] - acc);
            sW[row_col] = w;
            out[v_base + row_col] = w;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void kiln_gdn_chunk_prep_bf16(
    device const bfloat* g [[buffer(0)]],
    device const bfloat* v [[buffer(1)]],
    device const bfloat* kkt [[buffer(2)]],
    device const bfloat* qkt [[buffer(3)]],
    device const bfloat* ks_entry [[buffer(4)]],
    device const bfloat* q_s [[buffer(5)]],
    device bfloat* a_strict [[buffer(6)]],
    device bfloat* b_mask [[buffer(7)]],
    device bfloat* v_prime [[buffer(8)]],
    device bfloat* q_s_scaled [[buffer(9)]],
    device bfloat* decay_last_col [[buffer(10)]],
    device bfloat* p_last [[buffer(11)]],
    constant uint& batch_heads [[buffer(12)]],
    constant uint& chunk_size [[buffer(13)]],
    constant uint& dv [[buffer(14)]],
    uint bh [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    if (bh >= batch_heads) {
        return;
    }

    threadgroup float sBigG[64];
    threadgroup bfloat sP[64];

    const uint g_base = bh * chunk_size;
    const uint cdv_base = bh * chunk_size * dv;
    const uint cc_base = bh * chunk_size * chunk_size;

    if (tid < chunk_size) {
        sBigG[tid] = static_cast<float>(g[g_base + tid]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float acc = 0.0f;
        for (uint i = 0; i < 64; ++i) {
            acc += sBigG[i];
            sBigG[i] = acc;
        }
        for (uint i = 0; i < 64; ++i) {
            sP[i] = static_cast<bfloat>(exp(sBigG[i]));
        }
        p_last[bh] = sP[chunk_size - 1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint cc = chunk_size * chunk_size;
    for (uint idx = tid; idx < cc; idx += 128) {
        const uint t = idx / chunk_size;
        const uint i = idx - t * chunk_size;
        const bfloat decay_bf = static_cast<bfloat>(exp(sBigG[t] - sBigG[i]));
        const float decay = static_cast<float>(decay_bf);
        const float kkt_val = static_cast<float>(kkt[cc_base + idx]);
        const float qkt_val = static_cast<float>(qkt[cc_base + idx]);
        a_strict[cc_base + idx] =
            (i < t) ? static_cast<bfloat>(kkt_val * decay) : static_cast<bfloat>(0.0f);
        b_mask[cc_base + idx] =
            (i <= t) ? static_cast<bfloat>(qkt_val * decay) : static_cast<bfloat>(0.0f);
    }

    const uint cdv = chunk_size * dv;
    for (uint idx = tid; idx < cdv; idx += 128) {
        const uint t = idx / dv;
        const float p = static_cast<float>(sP[t]);
        const float v_val = static_cast<float>(v[cdv_base + idx]);
        const float ks_val = static_cast<float>(ks_entry[cdv_base + idx]);
        const float qs_val = static_cast<float>(q_s[cdv_base + idx]);
        v_prime[cdv_base + idx] = static_cast<bfloat>(v_val - ks_val * p);
        q_s_scaled[cdv_base + idx] = static_cast<bfloat>(qs_val * p);
    }

    if (tid < chunk_size) {
        const float decay = exp(sBigG[chunk_size - 1] - sBigG[tid]);
        decay_last_col[g_base + tid] = static_cast<bfloat>(decay);
    }
}
"#;
pub(super) const METAL_GDN_RECURRENT_PREFILL_HEAD_LAST_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kiln_gdn_recurrent_prefill_head_last_bf16(
    device const bfloat* q [[buffer(0)]],
    device const bfloat* k [[buffer(1)]],
    device const bfloat* v [[buffer(2)]],
    device const bfloat* beta [[buffer(3)]],
    device const bfloat* g [[buffer(4)]],
    device bfloat* state [[buffer(5)]],
    device bfloat* out [[buffer(6)]],
    constant uint& batch_heads [[buffer(7)]],
    constant uint& seq_len [[buffer(8)]],
    constant uint& dk [[buffer(9)]],
    constant uint& dv [[buffer(10)]],
    constant uint& value_heads [[buffer(11)]],
    constant uint& q_heads [[buffer(12)]],
    constant uint& input_mode [[buffer(13)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    constexpr uint NSG = 4;
    constexpr uint LANES = 32;
    if (gid >= batch_heads * dv || tid >= LANES || dk != 128) {
        return;
    }

    const uint bh = gid / dv;
    const uint d = gid - bh * dv;
    const uint batch_idx = bh / value_heads;
    const uint head_idx = bh - batch_idx * value_heads;
    const uint q_group = value_heads / q_heads;
    const uint q_head_idx = head_idx / q_group;
    const uint qk_base = (batch_idx * q_heads + q_head_idx) * seq_len * dk;
    const uint v_base = bh * seq_len * dv;
    const uint gate_base = bh * seq_len;
    const uint state_base = bh * dk * dv;

    float ls[NSG];
    for (uint j = 0; j < NSG; ++j) {
        const uint is = tid * NSG + j;
        ls[j] = static_cast<float>(state[state_base + is * dv + d]);
    }

    for (uint t = 0; t < seq_len; ++t) {
        const uint qk_t = (input_mode == 0)
            ? qk_base + t * dk
            : ((batch_idx * seq_len + t) * q_heads + q_head_idx) * dk;
        const uint v_t = (input_mode == 0)
            ? v_base + t * dv
            : ((batch_idx * seq_len + t) * value_heads + head_idx) * dv;
        const uint gate_t = (input_mode == 0)
            ? gate_base + t
            : (batch_idx * seq_len + t) * value_heads + head_idx;
        const float decay = static_cast<float>(static_cast<bfloat>(
            exp(static_cast<float>(g[gate_t]))
        ));

        float s_k = 0.0f;
        for (uint j = 0; j < NSG; ++j) {
            const uint is = tid * NSG + j;
            const float decayed = static_cast<float>(static_cast<bfloat>(ls[j] * decay));
            ls[j] = decayed;
            s_k += decayed * static_cast<float>(k[qk_t + is]);
        }
        s_k = simd_sum(s_k);

        const float delta = static_cast<float>(static_cast<bfloat>(
            (static_cast<float>(v[v_t + d]) - s_k) *
            static_cast<float>(beta[gate_t])
        ));

        float y = 0.0f;
        for (uint j = 0; j < NSG; ++j) {
            const uint is = tid * NSG + j;
            const float new_s = static_cast<float>(static_cast<bfloat>(
                ls[j] + static_cast<float>(k[qk_t + is]) * delta
            ));
            ls[j] = new_s;
            y += new_s * static_cast<float>(q[qk_t + is]);
        }
        y = simd_sum(y);

        if (tid == 0) {
            const uint out_idx = ((batch_idx * seq_len + t) * value_heads + head_idx) * dv + d;
            out[out_idx] = static_cast<bfloat>(y);
        }
    }

    for (uint j = 0; j < NSG; ++j) {
        const uint is = tid * NSG + j;
        state[state_base + is * dv + d] = static_cast<bfloat>(ls[j]);
    }
}
"#;
pub(super) const METAL_GDN_RECURRENT_PREFILL_HEAD_LAST_DECAY_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kiln_gdn_recurrent_prefill_head_last_decay_bf16(
    device const bfloat* q [[buffer(0)]],
    device const bfloat* k [[buffer(1)]],
    device const bfloat* v [[buffer(2)]],
    device const bfloat* beta [[buffer(3)]],
    device const bfloat* decay [[buffer(4)]],
    device bfloat* state [[buffer(5)]],
    device bfloat* out [[buffer(6)]],
    constant uint& batch_heads [[buffer(7)]],
    constant uint& seq_len [[buffer(8)]],
    constant uint& dk [[buffer(9)]],
    constant uint& dv [[buffer(10)]],
    constant uint& value_heads [[buffer(11)]],
    constant uint& q_heads [[buffer(12)]],
    constant uint& input_mode [[buffer(13)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    constexpr uint NSG = 4;
    constexpr uint LANES = 32;
    if (gid >= batch_heads * dv || tid >= LANES || dk != 128) {
        return;
    }

    const uint bh = gid / dv;
    const uint d = gid - bh * dv;
    const uint batch_idx = bh / value_heads;
    const uint head_idx = bh - batch_idx * value_heads;
    const uint q_group = value_heads / q_heads;
    const uint q_head_idx = head_idx / q_group;
    const uint qk_base = (batch_idx * q_heads + q_head_idx) * seq_len * dk;
    const uint v_base = bh * seq_len * dv;
    const uint gate_base = bh * seq_len;
    const uint state_base = bh * dk * dv;

    float ls[NSG];
    for (uint j = 0; j < NSG; ++j) {
        const uint is = tid * NSG + j;
        ls[j] = static_cast<float>(state[state_base + is * dv + d]);
    }

    for (uint t = 0; t < seq_len; ++t) {
        const uint qk_t = (input_mode == 0)
            ? qk_base + t * dk
            : ((batch_idx * seq_len + t) * q_heads + q_head_idx) * dk;
        const uint v_t = (input_mode == 0)
            ? v_base + t * dv
            : ((batch_idx * seq_len + t) * value_heads + head_idx) * dv;
        const uint gate_t = (input_mode == 0)
            ? gate_base + t
            : (batch_idx * seq_len + t) * value_heads + head_idx;
        const float decay_t = static_cast<float>(decay[gate_t]);

        float s_k = 0.0f;
        for (uint j = 0; j < NSG; ++j) {
            const uint is = tid * NSG + j;
            const float decayed = static_cast<float>(static_cast<bfloat>(ls[j] * decay_t));
            ls[j] = decayed;
            s_k += decayed * static_cast<float>(k[qk_t + is]);
        }
        s_k = simd_sum(s_k);

        const float delta = static_cast<float>(static_cast<bfloat>(
            (static_cast<float>(v[v_t + d]) - s_k) *
            static_cast<float>(beta[gate_t])
        ));

        float y = 0.0f;
        for (uint j = 0; j < NSG; ++j) {
            const uint is = tid * NSG + j;
            const float new_s = static_cast<float>(static_cast<bfloat>(
                ls[j] + static_cast<float>(k[qk_t + is]) * delta
            ));
            ls[j] = new_s;
            y += new_s * static_cast<float>(q[qk_t + is]);
        }
        y = simd_sum(y);

        if (tid == 0) {
            const uint out_idx = ((batch_idx * seq_len + t) * value_heads + head_idx) * dv + d;
            out[out_idx] = static_cast<bfloat>(y);
        }
    }

    for (uint j = 0; j < NSG; ++j) {
        const uint is = tid * NSG + j;
        state[state_base + is * dv + d] = static_cast<bfloat>(ls[j]);
    }
}
"#;
pub(super) const METAL_GDN_FULL_CHUNK_FORWARD_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kiln_gdn_full_chunk_forward_bf16(
    device const bfloat* g [[buffer(0)]],
    device const bfloat* v [[buffer(1)]],
    device const bfloat* kkt [[buffer(2)]],
    device const bfloat* qkt [[buffer(3)]],
    device const bfloat* ks_entry [[buffer(4)]],
    device const bfloat* q_s [[buffer(5)]],
    device const bfloat* beta [[buffer(6)]],
    device const bfloat* k_t [[buffer(7)]],
    device bfloat* state [[buffer(8)]],
    device bfloat* out [[buffer(9)]],
    constant uint& batch_heads [[buffer(10)]],
    constant uint& dk [[buffer(11)]],
    constant uint& dv [[buffer(12)]],
    constant uint& output_mode [[buffer(13)]],
    constant uint& t_start [[buffer(14)]],
    constant uint& seq_len [[buffer(15)]],
    constant uint& heads [[buffer(16)]],
    constant uint& g_bh_stride [[buffer(17)]],
    constant uint& g_t_stride [[buffer(18)]],
    constant uint& v_bh_stride [[buffer(19)]],
    constant uint& v_t_stride [[buffer(20)]],
    constant uint& v_d_stride [[buffer(21)]],
    constant uint& beta_bh_stride [[buffer(22)]],
    constant uint& beta_t_stride [[buffer(23)]],
    constant uint& kt_bh_stride [[buffer(24)]],
    constant uint& kt_k_stride [[buffer(25)]],
    constant uint& kt_t_stride [[buffer(26)]],
    uint bh [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    constexpr uint C = 64;
    constexpr uint MAX_DV = 128;
    if (bh >= batch_heads) {
        return;
    }

    threadgroup bfloat sArow[64];
    threadgroup bfloat sBrow[64];
    threadgroup bfloat sW[8192];
    threadgroup float sBigG[64];
    threadgroup float sP[64];
    threadgroup float sDecayLast[64];
    threadgroup float sPLast;

    const uint g_strided_base = bh * g_bh_stride;
    const uint v_strided_base = bh * v_bh_stride;
    const uint beta_base = bh * beta_bh_stride;
    const uint cdv_base = bh * C * dv;
    const uint cc_base = bh * C * C;
    const uint kt_strided_base = bh * kt_bh_stride;
    const uint state_base = bh * dk * dv;

    for (uint i = tid; i < C; i += 128) {
        sBigG[i] = static_cast<float>(g[g_strided_base + i * g_t_stride]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float acc = 0.0f;
        for (uint i = 0; i < C; ++i) {
            acc += sBigG[i];
            sBigG[i] = acc;
        }
        sPLast = exp(sBigG[C - 1]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < C; i += 128) {
        sP[i] = exp(sBigG[i]);
        sDecayLast[i] = exp(sBigG[C - 1] - sBigG[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint t = 0; t < C; ++t) {
        for (uint i = tid; i < C; i += 128) {
            const uint ti = t * C + i;
            const float decay = exp(sBigG[t] - sBigG[i]);
            const float a_val = (i < t) ? static_cast<float>(kkt[cc_base + ti]) * decay : 0.0f;
            const float b_val = (i <= t) ? static_cast<float>(qkt[cc_base + ti]) * decay : 0.0f;
            sArow[i] = static_cast<bfloat>(a_val);
            sBrow[i] = static_cast<bfloat>(b_val);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float beta_t = static_cast<float>(beta[beta_base + t * beta_t_stride]);
        const float p_t = static_cast<float>(static_cast<bfloat>(sP[t]));

        if (tid < dv) {
            float acc_a = 0.0f;
            for (uint i = 0; i < t; ++i) {
                acc_a += static_cast<float>(sArow[i]) *
                         static_cast<float>(sW[i * MAX_DV + tid]);
            }

            const uint td = t * dv + tid;
            const float vp = static_cast<float>(static_cast<bfloat>(
                static_cast<float>(v[v_strided_base + t * v_t_stride + tid * v_d_stride]) -
                static_cast<float>(ks_entry[cdv_base + td]) * p_t
            ));
            const float w_val = beta_t * (vp - acc_a);
            sW[t * MAX_DV + tid] = static_cast<bfloat>(w_val);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < dv) {
            float acc_b = 0.0f;
            for (uint i = 0; i <= t; ++i) {
                acc_b += static_cast<float>(sBrow[i]) *
                         static_cast<float>(sW[i * MAX_DV + tid]);
            }

            const uint td = t * dv + tid;
            const float qss = static_cast<float>(static_cast<bfloat>(
                static_cast<float>(q_s[cdv_base + td]) * p_t
            ));
            const bfloat out_val = static_cast<bfloat>(qss + acc_b);
            if (output_mode == 0) {
                out[cdv_base + td] = out_val;
            } else {
                const uint batch_idx = bh / heads;
                const uint head_idx = bh - batch_idx * heads;
                const uint out_t = t_start + t;
                const uint out_idx = ((batch_idx * seq_len + out_t) * heads + head_idx) * dv + tid;
                out[out_idx] = out_val;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid < dv) {
        const float p_last = static_cast<float>(static_cast<bfloat>(sPLast));
        for (uint k_idx = 0; k_idx < dk; ++k_idx) {
            float delta = 0.0f;
            for (uint t = 0; t < C; ++t) {
                const float kt = static_cast<float>(
                    k_t[kt_strided_base + k_idx * kt_k_stride + t * kt_t_stride]
                );
                const float w = static_cast<float>(sW[t * MAX_DV + tid]);
                const float decay_last = static_cast<float>(static_cast<bfloat>(sDecayLast[t]));
                const float w_weighted = static_cast<float>(static_cast<bfloat>(w * decay_last));
                delta += kt * w_weighted;
            }
            const uint state_idx = state_base + k_idx * dv + tid;
            const float prev = static_cast<float>(state[state_idx]);
            state[state_idx] = static_cast<bfloat>(prev * p_last + delta);
        }
    }
}
"#;
pub(super) const METAL_CONV1D_PREFILL_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kiln_causal_conv1d_prefill_bf16_f32_k4(
    device const bfloat* x [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device float* conv_state [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& batch [[buffer(4)]],
    constant uint& channels [[buffer(5)]],
    constant uint& seq_len [[buffer(6)]],
    constant uint& threadgroup_width [[buffer(7)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    const uint total_channels = batch * channels;
    if (gid >= total_channels) {
        return;
    }

    const uint b = gid / channels;
    const uint c = gid - b * channels;
    const uint x_base = (b * channels + c) * seq_len;
    const uint state_base = (b * channels + c) * 3;
    const uint weight_base = c * 4;

    threadgroup float s_state[3];
    if (tid < 3) {
        s_state[tid] = conv_state[state_base + tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint t = tid; t < seq_len; t += threadgroup_width) {
        float acc = 0.0f;
        for (uint j = 0; j < 4; ++j) {
            const uint padded_idx = t + j;
            float v;
            if (padded_idx < 3) {
                v = s_state[padded_idx];
            } else {
                v = static_cast<float>(x[x_base + padded_idx - 3]);
            }
            acc += v * static_cast<float>(weight[weight_base + j]);
        }
        out[x_base + t] = acc / (1.0f + exp(-acc));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        if (seq_len >= 3) {
            conv_state[state_base + 0] = static_cast<float>(x[x_base + seq_len - 3]);
            conv_state[state_base + 1] = static_cast<float>(x[x_base + seq_len - 2]);
            conv_state[state_base + 2] = static_cast<float>(x[x_base + seq_len - 1]);
        } else if (seq_len == 2) {
            conv_state[state_base + 0] = s_state[2];
            conv_state[state_base + 1] = static_cast<float>(x[x_base + 0]);
            conv_state[state_base + 2] = static_cast<float>(x[x_base + 1]);
        } else if (seq_len == 1) {
            conv_state[state_base + 0] = s_state[1];
            conv_state[state_base + 1] = s_state[2];
            conv_state[state_base + 2] = static_cast<float>(x[x_base]);
        }
    }
}

kernel void kiln_gdn_prefill_qkv_conv_split_bf16_f32_k4(
    device const bfloat* x [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device float* conv_state [[buffer(2)]],
    device float* q_out [[buffer(3)]],
    device float* k_out [[buffer(4)]],
    device bfloat* v_out [[buffer(5)]],
    constant uint& batch [[buffer(6)]],
    constant uint& seq_len [[buffer(7)]],
    constant uint& channels [[buffer(8)]],
    constant uint& qk_dim [[buffer(9)]],
    constant uint& v_dim [[buffer(10)]],
    constant uint& threadgroup_width [[buffer(11)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    const uint total_channels = batch * channels;
    if (gid >= total_channels) {
        return;
    }

    const uint b = gid / channels;
    const uint c = gid - b * channels;
    const uint state_base = (b * channels + c) * 3;
    const uint weight_base = c * 4;

    threadgroup float s_state[3];
    if (tid < 3) {
        s_state[tid] = conv_state[state_base + tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint t = tid; t < seq_len; t += threadgroup_width) {
        float acc = 0.0f;
        for (uint j = 0; j < 4; ++j) {
            const uint padded_idx = t + j;
            float xv;
            if (padded_idx < 3) {
                xv = s_state[padded_idx];
            } else {
                const uint x_idx = (b * seq_len + (padded_idx - 3)) * channels + c;
                xv = static_cast<float>(x[x_idx]);
            }
            acc += xv * static_cast<float>(weight[weight_base + j]);
        }

        const float y = acc / (1.0f + exp(-acc));
        if (c < qk_dim) {
            q_out[(b * seq_len + t) * qk_dim + c] = y;
        } else if (c < qk_dim * 2) {
            k_out[(b * seq_len + t) * qk_dim + (c - qk_dim)] = y;
        } else {
            v_out[(b * seq_len + t) * v_dim + (c - qk_dim * 2)] = static_cast<bfloat>(y);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        if (seq_len >= 3) {
            conv_state[state_base + 0] =
                static_cast<float>(x[(b * seq_len + (seq_len - 3)) * channels + c]);
            conv_state[state_base + 1] =
                static_cast<float>(x[(b * seq_len + (seq_len - 2)) * channels + c]);
            conv_state[state_base + 2] =
                static_cast<float>(x[(b * seq_len + (seq_len - 1)) * channels + c]);
        } else if (seq_len == 2) {
            conv_state[state_base + 0] = s_state[2];
            conv_state[state_base + 1] = static_cast<float>(x[(b * seq_len) * channels + c]);
            conv_state[state_base + 2] = static_cast<float>(x[(b * seq_len + 1) * channels + c]);
        } else if (seq_len == 1) {
            conv_state[state_base + 0] = s_state[1];
            conv_state[state_base + 1] = s_state[2];
            conv_state[state_base + 2] = static_cast<float>(x[(b * seq_len) * channels + c]);
        }
    }
}
"#;
pub(super) const METAL_CONV1D_UPDATE_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void kiln_causal_conv1d_update_bf16_f32_k4(
    device const bfloat* x [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device float* conv_state [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& batch [[buffer(4)]],
    constant uint& channels [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint total = batch * channels;
    if (gid >= total) {
        return;
    }

    const uint c = gid % channels;
    const uint state_base = gid * 3;
    const uint weight_base = c * 4;

    const float s0 = conv_state[state_base + 0];
    const float s1 = conv_state[state_base + 1];
    const float s2 = conv_state[state_base + 2];
    const float x0 = static_cast<float>(x[gid]);

    const float acc =
        s0 * static_cast<float>(weight[weight_base + 0]) +
        s1 * static_cast<float>(weight[weight_base + 1]) +
        s2 * static_cast<float>(weight[weight_base + 2]) +
        x0 * static_cast<float>(weight[weight_base + 3]);

    out[gid] = acc / (1.0f + exp(-acc));
    conv_state[state_base + 0] = s1;
    conv_state[state_base + 1] = s2;
    conv_state[state_base + 2] = x0;
}
"#;
