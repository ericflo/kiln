// Kiln GDN recurrent (single-token) forward kernel.
//
// One CUDA block per (batch, head). One thread per dv column (so blockDim.x
// = dv). Each thread:
//   1. Loads its column S[:, my_d] from global state into per-thread
//      registers/local memory as F32 (dk floats).
//   2. Cooperatively pulls k_t, q_t into shared memory; thread 0 loads the
//      scalar decay = exp(g_t) and beta_t.
//   3. __syncthreads() once.
//   4. Phase A: per-thread loop over dk computes
//        decayed[i] = decay * s_local[i]
//        v_pred    += k_smem[i] * decayed[i]
//      (v_pred contributes to one element of the dv-wide v_pred vector,
//       the element this thread owns).
//   5. Computes delta = beta * (v_t[my_d] - v_pred).
//   6. Phase B: per-thread loop over dk computes
//        new_s = decayed[i] + k_smem[i] * delta
//        s_local[i] = new_s             (still in registers)
//        out_acc += q_smem[i] * new_s
//      and writes new_s back to global state.
//   7. Writes bf16(out_acc) to out[my_d].
//
// kiln configures dk = dv = 128, so the per-thread state column is 128 F32
// values = 512 bytes ≈ 128 32-bit registers — comfortably under sm_86's 255
// reg/thread cap. Shared memory is tiny: 2 * dk * 4 (q, k as F32) + a few
// scalars ≈ 1 KiB. The win over the candle-op path is eliminating the
// chunkwise machinery (preshape, decay matrix, KKT, forward sub, B_mask,
// matmul into state) when there is only a single token to advance.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

#include "recurrent_gdn_fwd.h"

namespace {

__device__ __forceinline__ float bf16_to_f32(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

__device__ __forceinline__ __nv_bfloat16 f32_to_bf16(float v) {
    return __float2bfloat16(v);
}

// MAX_DK caps the per-thread register array size. kiln uses dk = 128.
template <int MAX_DK>
__global__ void recurrent_gdn_fwd_kernel(
    const __nv_bfloat16 *__restrict__ q,     // [B*H, dk]
    const __nv_bfloat16 *__restrict__ k,     // [B*H, dk]
    const __nv_bfloat16 *__restrict__ v,     // [B*H, dv]
    const __nv_bfloat16 *__restrict__ beta,  // [B*H]
    const __nv_bfloat16 *__restrict__ g,     // [B*H]
    __nv_bfloat16 *__restrict__ state,       // [B*H, dk, dv]
    __nv_bfloat16 *__restrict__ out,         // [B*H, dv]
    int dk,
    int dv
) {
    const int bh  = blockIdx.x;
    const int tid = threadIdx.x;

    if (tid >= dv) return;

    const __nv_bfloat16 *q_base    = q     + (size_t)bh * dk;
    const __nv_bfloat16 *k_base    = k     + (size_t)bh * dk;
    const __nv_bfloat16 *v_base    = v     + (size_t)bh * dv;
    __nv_bfloat16       *s_base    = state + (size_t)bh * dk * dv;
    __nv_bfloat16       *out_base  = out   + (size_t)bh * dv;

    extern __shared__ float smem[];
    float *q_smem = smem;            // dk F32 floats
    float *k_smem = smem + dk;       // dk F32 floats
    float *scalars = smem + 2 * dk;  // [decay, beta]

    // Cooperative load of q_t / k_t into shared memory as F32.
    for (int i = tid; i < dk; i += blockDim.x) {
        q_smem[i] = bf16_to_f32(q_base[i]);
        k_smem[i] = bf16_to_f32(k_base[i]);
    }

    // Thread 0 loads the per-(batch,head) scalars.
    if (tid == 0) {
        float g_val   = bf16_to_f32(g[bh]);
        float b_val   = bf16_to_f32(beta[bh]);
        scalars[0]    = expf(g_val);   // decay
        scalars[1]    = b_val;          // beta
    }
    __syncthreads();

    const float decay  = scalars[0];
    const float beta_t = scalars[1];

    // Per-thread state column [dk] in registers (or local memory for large dk).
    float s_local[MAX_DK];
    #pragma unroll
    for (int i = 0; i < MAX_DK; ++i) {
        if (i < dk) {
            s_local[i] = bf16_to_f32(s_base[(size_t)i * dv + tid]);
        }
    }

    // Phase A: compute decayed state and v_pred for this column.
    //   decayed[i] = decay * s_local[i]
    //   v_pred    += k_smem[i] * decayed[i]
    float v_pred = 0.0f;
    #pragma unroll
    for (int i = 0; i < MAX_DK; ++i) {
        if (i < dk) {
            float d = decay * s_local[i];
            s_local[i] = d;  // overwrite with decayed value
            v_pred += k_smem[i] * d;
        }
    }

    float v_t   = bf16_to_f32(v_base[tid]);
    float delta = beta_t * (v_t - v_pred);

    // Phase B: update state and accumulate output.
    //   new_s = decayed[i] + k_smem[i] * delta
    //   s_local[i] = new_s
    //   out_acc += q_smem[i] * new_s
    float out_acc = 0.0f;
    #pragma unroll
    for (int i = 0; i < MAX_DK; ++i) {
        if (i < dk) {
            float new_s = s_local[i] + k_smem[i] * delta;
            s_local[i] = new_s;
            out_acc += q_smem[i] * new_s;
            s_base[(size_t)i * dv + tid] = f32_to_bf16(new_s);
        }
    }

    out_base[tid] = f32_to_bf16(out_acc);
}

__device__ __forceinline__ float stable_sigmoid(float x) {
    if (x >= 0.0f) {
        float z = expf(-x);
        return 1.0f / (1.0f + z);
    }
    float z = expf(x);
    return z / (1.0f + z);
}

__device__ __forceinline__ float stable_softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float to_f32(float x) {
    return x;
}

__device__ __forceinline__ float to_f32(__nv_bfloat16 x) {
    return bf16_to_f32(x);
}

template <typename VType>
__global__ void gdn_decode_gates_recurrent_rmsnorm_bf16_kernel(
    const __nv_bfloat16 *__restrict__ q,        // [B, 1, q_heads, dk]
    const __nv_bfloat16 *__restrict__ k,        // [B, 1, q_heads, dk]
    const VType *__restrict__ v,                  // [B, 1, value_heads, dv]
    const __nv_bfloat16 *__restrict__ a,        // [B, 1, value_heads]
    const __nv_bfloat16 *__restrict__ b,        // [B, 1, value_heads]
    const __nv_bfloat16 *__restrict__ a_log,    // [value_heads]
    const __nv_bfloat16 *__restrict__ dt_bias,  // [value_heads]
    __nv_bfloat16 *__restrict__ state,          // [B, value_heads, dk, dv]
    const __nv_bfloat16 *__restrict__ z,        // [B, 1, value_heads, dv]
    const __nv_bfloat16 *__restrict__ weight,   // [dv]
    __nv_bfloat16 *__restrict__ out,            // [B, 1, value_heads, dv]
    int q_heads,
    int value_heads,
    float eps
) {
    __shared__ float k_smem[128];
    __shared__ float sq_smem[128];
    __shared__ float scalars[2];

    const int bh = blockIdx.x;
    const int tid = threadIdx.x;
    if (tid >= 128) return;

    const int batch_idx = bh / value_heads;
    const int head_idx = bh - batch_idx * value_heads;
    const int q_group = value_heads / q_heads;
    const int q_head_idx = head_idx / q_group;

    const size_t qk_base = ((size_t)batch_idx * q_heads + q_head_idx) * 128;
    const size_t v_base = ((size_t)batch_idx * value_heads + head_idx) * 128;
    const size_t gate_idx = (size_t)batch_idx * value_heads + head_idx;
    const size_t state_base = (size_t)bh * 128 * 128;

    k_smem[tid] = bf16_to_f32(k[qk_base + tid]);

    if (tid == 0) {
        const float beta = stable_sigmoid(bf16_to_f32(b[gate_idx]));
        const float g = -expf(bf16_to_f32(a_log[head_idx]))
            * stable_softplus(bf16_to_f32(a[gate_idx]) + bf16_to_f32(dt_bias[head_idx]));
        scalars[0] = expf(bf16_to_f32(f32_to_bf16(g)));
        scalars[1] = bf16_to_f32(f32_to_bf16(beta));
    }
    __syncthreads();

    const float decay = scalars[0];
    const float beta_t = scalars[1];

    float s_local[128];
    float v_pred = 0.0f;
    #pragma unroll
    for (int i = 0; i < 128; ++i) {
        const float d = decay * bf16_to_f32(state[state_base + (size_t)i * 128 + tid]);
        s_local[i] = d;
        v_pred += k_smem[i] * d;
    }

    const float delta = beta_t * (to_f32(v[v_base + tid]) - v_pred);
    float y = 0.0f;
    #pragma unroll
    for (int i = 0; i < 128; ++i) {
        const float new_s = s_local[i] + k_smem[i] * delta;
        state[state_base + (size_t)i * 128 + tid] = f32_to_bf16(new_s);
        y += bf16_to_f32(q[qk_base + i]) * new_s;
    }

    out[v_base + tid] = f32_to_bf16(y);

}

} // namespace

extern "C" kiln_gdn_recurrent_status_t kiln_gdn_recurrent_forward(
    const void *q,
    const void *k,
    const void *v,
    const void *beta,
    const void *g,
    void *state,
    void *out,
    int batch_heads,
    int dk,
    int dv,
    void *stream
) {
    if (batch_heads <= 0 || dk <= 0 || dv <= 0) {
        return -1;
    }
    if (dv > 1024) {
        // blockDim.x must equal dv and is capped by CUDA at 1024.
        return -2;
    }

    int threads = dv;
    int blocks  = batch_heads;

    // Shared mem: q (dk F32) + k (dk F32) + 2 scalars.
    size_t smem_bytes = (size_t)(2 * dk + 2) * sizeof(float);

    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);

    if (dk <= 128) {
        recurrent_gdn_fwd_kernel<128><<<blocks, threads, smem_bytes, s>>>(
            reinterpret_cast<const __nv_bfloat16 *>(q),
            reinterpret_cast<const __nv_bfloat16 *>(k),
            reinterpret_cast<const __nv_bfloat16 *>(v),
            reinterpret_cast<const __nv_bfloat16 *>(beta),
            reinterpret_cast<const __nv_bfloat16 *>(g),
            reinterpret_cast<__nv_bfloat16 *>(state),
            reinterpret_cast<__nv_bfloat16 *>(out),
            dk,
            dv
        );
    } else if (dk <= 256) {
        recurrent_gdn_fwd_kernel<256><<<blocks, threads, smem_bytes, s>>>(
            reinterpret_cast<const __nv_bfloat16 *>(q),
            reinterpret_cast<const __nv_bfloat16 *>(k),
            reinterpret_cast<const __nv_bfloat16 *>(v),
            reinterpret_cast<const __nv_bfloat16 *>(beta),
            reinterpret_cast<const __nv_bfloat16 *>(g),
            reinterpret_cast<__nv_bfloat16 *>(state),
            reinterpret_cast<__nv_bfloat16 *>(out),
            dk,
            dv
        );
    } else {
        return -3;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return (int)err;
    }
    return 0;
}


extern "C" kiln_gdn_recurrent_status_t kiln_gdn_decode_gates_recurrent_vf32_bf16(
    const void *q,
    const void *k,
    const void *v,
    const void *a,
    const void *b,
    const void *a_log,
    const void *dt_bias,
    void *state,
    const void *z,
    const void *weight,
    void *out,
    int batch,
    int q_heads,
    int value_heads,
    int dk,
    int dv,
    float eps,
    void *stream
) {
    if (batch <= 0 || q_heads <= 0 || value_heads <= 0) return -1;
    if (dk != 128 || dv != 128) return -2;
    if (value_heads < q_heads || (value_heads % q_heads) != 0) return -3;

    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    gdn_decode_gates_recurrent_rmsnorm_bf16_kernel<float><<<batch * value_heads, 128, 0, s>>>(
        reinterpret_cast<const __nv_bfloat16 *>(q),
        reinterpret_cast<const __nv_bfloat16 *>(k),
        reinterpret_cast<const float *>(v),
        reinterpret_cast<const __nv_bfloat16 *>(a),
        reinterpret_cast<const __nv_bfloat16 *>(b),
        reinterpret_cast<const __nv_bfloat16 *>(a_log),
        reinterpret_cast<const __nv_bfloat16 *>(dt_bias),
        reinterpret_cast<__nv_bfloat16 *>(state),
        reinterpret_cast<const __nv_bfloat16 *>(z),
        reinterpret_cast<const __nv_bfloat16 *>(weight),
        reinterpret_cast<__nv_bfloat16 *>(out),
        q_heads,
        value_heads,
        eps
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return (int)err;
    return 0;
}

extern "C" kiln_gdn_recurrent_status_t kiln_gdn_decode_gates_recurrent_bf16(
    const void *q,
    const void *k,
    const void *v,
    const void *a,
    const void *b,
    const void *a_log,
    const void *dt_bias,
    void *state,
    const void *z,
    const void *weight,
    void *out,
    int batch,
    int q_heads,
    int value_heads,
    int dk,
    int dv,
    float eps,
    void *stream
) {
    if (batch <= 0 || q_heads <= 0 || value_heads <= 0) return -1;
    if (dk != 128 || dv != 128) return -2;
    if (value_heads < q_heads || (value_heads % q_heads) != 0) return -3;

    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    gdn_decode_gates_recurrent_rmsnorm_bf16_kernel<__nv_bfloat16><<<batch * value_heads, 128, 0, s>>>(
        reinterpret_cast<const __nv_bfloat16 *>(q),
        reinterpret_cast<const __nv_bfloat16 *>(k),
        reinterpret_cast<const __nv_bfloat16 *>(v),
        reinterpret_cast<const __nv_bfloat16 *>(a),
        reinterpret_cast<const __nv_bfloat16 *>(b),
        reinterpret_cast<const __nv_bfloat16 *>(a_log),
        reinterpret_cast<const __nv_bfloat16 *>(dt_bias),
        reinterpret_cast<__nv_bfloat16 *>(state),
        reinterpret_cast<const __nv_bfloat16 *>(z),
        reinterpret_cast<const __nv_bfloat16 *>(weight),
        reinterpret_cast<__nv_bfloat16 *>(out),
        q_heads,
        value_heads,
        eps
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return (int)err;
    return 0;
}
