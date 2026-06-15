#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <type_traits>
#include <utility>

#if defined(KILN_ENABLE_CK_TILE_FMHA) && __has_include(<ck_tile/ops/fmha_bwd.hpp>)
#include <ck_tile/ops/fmha_bwd.hpp>
#include <ck_tile/ops/epilogue.hpp>
#include <ck_tile/ops/fmha.hpp>
#include <ck_tile/ops/mask.hpp>
#define KILN_HAS_CK_TILE_FMHA 1
#else
#define KILN_HAS_CK_TILE_FMHA 0
#endif

#if KILN_HAS_CK_TILE_FMHA && defined(KILN_ENABLE_CK_TILE_FMHA_FWD) && \
    __has_include(<ck_tile/ops/fmha_fwd_v3_impl.hpp>)
#include <ck_tile/ops/fmha_fwd_v3_impl.hpp>
#define KILN_HAS_CK_TILE_FMHA_FWD 1
#else
#define KILN_HAS_CK_TILE_FMHA_FWD 0
#endif

namespace {

constexpr int KilnLogicalWarpSize = 32;

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || \
    defined(__gfx1103__) || defined(__gfx1151__) || defined(__gfx11_generic__)
#define KILN_ROCM_HAS_GFX11_WMMA 1
#else
#define KILN_ROCM_HAS_GFX11_WMMA 0
#endif

bool env_truthy(const char* name)
{
    const char* value = std::getenv(name);
    if(value == nullptr || value[0] == '\0')
    {
        return false;
    }
    return !(value[0] == '0' && value[1] == '\0');
}

int env_i32_or(const char* name, int fallback)
{
    const char* value = std::getenv(name);
    if(value == nullptr || value[0] == '\0')
    {
        return fallback;
    }
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if(end == value || parsed <= 0 || parsed > INT32_MAX)
    {
        return fallback;
    }
    return static_cast<int>(parsed);
}

bool native_keysplit_enabled(int seqlen_q, int seqlen_k)
{
    if(env_truthy("KILN_ROCM_FLASH_NATIVE_KEYSPLIT"))
    {
        return true;
    }
    const int threshold = env_i32_or("KILN_ROCM_FLASH_NATIVE_KEYSPLIT_MIN_SEQ", INT32_MAX);
    return std::max(seqlen_q, seqlen_k) >= threshold;
}

bool native_wmma_qblock_enabled(int head_dim, int seqlen_q, int seqlen_k)
{
    if(head_dim != 256)
    {
        return false;
    }
    if(env_truthy("KILN_ROCM_FLASH_NATIVE_WMMA_QBLOCK"))
    {
        return true;
    }
    const int threshold = env_i32_or("KILN_ROCM_FLASH_NATIVE_WMMA_QBLOCK_MIN_SEQ", INT32_MAX);
    return std::max(seqlen_q, seqlen_k) >= threshold;
}

bool native_gqa_qblock_enabled(int head_dim,
                               int seqlen_q,
                               int seqlen_k,
                               int num_heads,
                               int num_heads_k)
{
    if(head_dim != 256 || num_heads_k <= 0 || num_heads % num_heads_k != 0)
    {
        return false;
    }
    const int groups_per_kv_head = num_heads / num_heads_k;
    if(groups_per_kv_head != 4)
    {
        return false;
    }
    if(env_truthy("KILN_ROCM_FLASH_NATIVE_GQA_QBLOCK"))
    {
        return true;
    }
    const int threshold = env_i32_or("KILN_ROCM_FLASH_NATIVE_GQA_QBLOCK_MIN_SEQ", 256);
    return std::max(seqlen_q, seqlen_k) >= threshold;
}

bool wmma_gqa_r64k32_enabled(int seqlen_q, int seqlen_k)
{
    if(env_truthy("KILN_ROCM_FLASH_DISABLE_WMMA_GQA_R64K32") ||
       env_truthy("KILN_ROCM_FLASH_DISABLE_WMMA_GQA_R64K16"))
    {
        return false;
    }
    if(env_truthy("KILN_ROCM_FLASH_USE_WMMA_GQA_R64K32") ||
       env_truthy("KILN_ROCM_FLASH_USE_WMMA_GQA_R64K16"))
    {
        return true;
    }
    const int legacy_threshold = env_i32_or("KILN_ROCM_FLASH_WMMA_GQA_R64K16_MIN_SEQ", 256);
    const int threshold = env_i32_or("KILN_ROCM_FLASH_WMMA_GQA_R64K32_MIN_SEQ", legacy_threshold);
    return std::max(seqlen_q, seqlen_k) >= threshold;
}

bool current_device_supports_gfx11_wmma()
{
    int device = 0;
    hipError_t err = hipGetDevice(&device);
    if(err != hipSuccess)
    {
        return false;
    }
    hipDeviceProp_t props;
    err = hipGetDeviceProperties(&props, device);
    if(err != hipSuccess)
    {
        return false;
    }
    return std::strstr(props.gcnArchName, "gfx11") != nullptr;
}

__device__ __forceinline__ __bf16 kiln_to_native_bf16(hip_bfloat16 value)
{
    return static_cast<__bf16>(static_cast<float>(value));
}

template <int Rows, int XDim>
__device__ float row_sum_rows(float v)
{
    __shared__ float scratch[Rows][XDim];
    const int x = static_cast<int>(threadIdx.x);
    const int y = static_cast<int>(threadIdx.y);
    scratch[y][x] = v;
    __syncthreads();

    for(int stride = XDim / 2; stride > 0; stride >>= 1)
    {
        if(x < stride)
        {
            scratch[y][x] += scratch[y][x + stride];
        }
        __syncthreads();
    }
    return scratch[y][0];
}

__device__ int causal_key_limit(int q_idx, int seqlen_q, int seqlen_k)
{
    // Bottom-right causal alignment, matching flash-attention decode/prefill:
    // query row q attends through key position q + (sk - sq).
    const int aligned_pos = q_idx + (seqlen_k - seqlen_q);
    const int limit = aligned_pos + 1;
    if(limit < 0)
    {
        return 0;
    }
    return limit < seqlen_k ? limit : seqlen_k;
}

__device__ __forceinline__ float kiln_wave_reduce_sum(float v)
{
    for(int offset = KilnLogicalWarpSize >> 1; offset > 0; offset >>= 1)
    {
        v += __shfl_xor(v, offset, KilnLogicalWarpSize);
    }
    return v;
}

__device__ __forceinline__ float kiln_wave_reduce_max(float v)
{
    for(int offset = KilnLogicalWarpSize >> 1; offset > 0; offset >>= 1)
    {
        v = fmaxf(v, __shfl_xor(v, offset, KilnLogicalWarpSize));
    }
    return v;
}

__device__ __forceinline__ float kiln_quarter_wave_reduce_sum(float v)
{
    for(int offset = KilnLogicalWarpSize >> 3; offset > 0; offset >>= 1)
    {
        v += __shfl_xor(v, offset, KilnLogicalWarpSize / 4);
    }
    return v;
}

__device__ __forceinline__ float kiln_quarter_wave_reduce_max(float v)
{
    for(int offset = KilnLogicalWarpSize >> 3; offset > 0; offset >>= 1)
    {
        v = fmaxf(v, __shfl_xor(v, offset, KilnLogicalWarpSize / 4));
    }
    return v;
}

__device__ __forceinline__ void kiln_bf16_pair_to_f32(uint32_t packed, float& lo, float& hi)
{
    lo = __uint_as_float((packed & 0xffffu) << 16);
    hi = __uint_as_float((packed >> 16) << 16);
}

template <int HeadDim, int Rows>
__device__ __forceinline__ float row_sum_x(float v)
{
    static_assert(HeadDim % KilnLogicalWarpSize == 0, "HeadDim must be a multiple of wave size");
    constexpr int WavesPerRow = HeadDim / KilnLogicalWarpSize;
    __shared__ float partial[Rows][WavesPerRow];
    const int x = static_cast<int>(threadIdx.x);
    const int y = static_cast<int>(threadIdx.y);
    const int lane = x & (KilnLogicalWarpSize - 1);
    const int wave = x / KilnLogicalWarpSize;

    const float wave_sum = kiln_wave_reduce_sum(v);
    if(lane == 0)
    {
        partial[y][wave] = wave_sum;
    }
    __syncthreads();

    float total = 0.0f;
    if(wave == 0)
    {
        total = lane < WavesPerRow ? partial[y][lane] : 0.0f;
        total = kiln_wave_reduce_sum(total);
        if(lane == 0)
        {
            partial[y][0] = total;
        }
    }
    __syncthreads();
    return partial[y][0];
}

using kiln_bf16x16 = __bf16 __attribute__((ext_vector_type(16)));
using kiln_f32x8 = float __attribute__((ext_vector_type(8)));

__global__ void kiln_rocm_flash_wmma_qk16_bf16_kernel(const hip_bfloat16* __restrict__ a,
                                                      const hip_bfloat16* __restrict__ b,
                                                      float* __restrict__ out)
{
    const int lane = static_cast<int>(threadIdx.x) & 31;
#if KILN_ROCM_HAS_GFX11_WMMA
    constexpr int Tile = 16;
    kiln_bf16x16 a_vec;
    kiln_bf16x16 b_vec;
    const int ab_lane = lane & 15;
    for(int k_idx = 0; k_idx < Tile; ++k_idx)
    {
        a_vec[k_idx] = kiln_to_native_bf16(a[ab_lane * Tile + k_idx]);
        // The attention kernel needs Q * K^T. WMMA consumes B as KxN, so a
        // contiguous K row supplies one logical B column.
        b_vec[k_idx] = kiln_to_native_bf16(b[ab_lane * Tile + k_idx]);
    }
    kiln_f32x8 c_vec = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    c_vec = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a_vec, b_vec, c_vec);

    const int out_col = lane & 15;
    const int out_row_parity = lane >> 4;
    for(int i = 0; i < 8; ++i)
    {
        const int out_row = i * 2 + out_row_parity;
        out[out_row * Tile + out_col] = c_vec[i];
    }
#else
    if(lane == 0)
    {
        out[0] = NAN;
    }
#endif
}

__global__ void kiln_rocm_flash_fwd_wmma_qblock256_bf16_kernel(
    const hip_bfloat16* __restrict__ q,
    const hip_bfloat16* __restrict__ k,
    const hip_bfloat16* __restrict__ v,
    hip_bfloat16* __restrict__ out,
    float* __restrict__ lse,
    int batch_size,
    int seqlen_q,
    int seqlen_k,
    int num_heads,
    int num_heads_k,
    float softmax_scale,
    int is_causal)
{
    constexpr int HeadDim = 256;
    constexpr int Rows = 16;
    constexpr int KeyBlock = 32;
    constexpr int WmmaRows = 16;
    constexpr int WmmaCols = 16;
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int row = tid / KilnLogicalWarpSize;
    const int q_block = static_cast<int>(blockIdx.x) * Rows;
    const int q_idx = q_block + row;
    const int head = static_cast<int>(blockIdx.y);
    const int batch = static_cast<int>(blockIdx.z);

    if(batch >= batch_size || head >= num_heads || row >= Rows)
    {
        return;
    }

    __shared__ hip_bfloat16 q_tile[Rows][HeadDim];
    __shared__ hip_bfloat16 k_tile[KeyBlock][HeadDim];
    __shared__ hip_bfloat16 v_tile[KeyBlock][HeadDim];
    __shared__ float scores[Rows][KeyBlock];
    __shared__ float row_alpha[Rows];
    __shared__ float row_inv_l[Rows];

    constexpr int MaxLaneDims = (HeadDim + 31) / 32;
    float acc_vals[MaxLaneDims];
    int dim_vals[MaxLaneDims];
    int lane_dims = 0;

    const int groups_per_kv_head = num_heads / num_heads_k;
    const int kv_head = head / groups_per_kv_head;
    const bool valid_q = q_idx < seqlen_q;
    const size_t q_base =
        ((static_cast<size_t>(batch) * seqlen_q + (valid_q ? q_idx : 0)) * num_heads + head) *
        HeadDim;

    for(int dim = lane; dim < HeadDim; dim += KilnLogicalWarpSize)
    {
        dim_vals[lane_dims] = dim;
        acc_vals[lane_dims] = 0.0f;
        ++lane_dims;
    }

    for(int idx = tid; idx < Rows * HeadDim; idx += static_cast<int>(blockDim.x))
    {
        const int q_row = idx / HeadDim;
        const int dim = idx - q_row * HeadDim;
        const int load_q = q_block + q_row;
        if(load_q < seqlen_q)
        {
            const size_t load_q_base =
                ((static_cast<size_t>(batch) * seqlen_q + load_q) * num_heads + head) * HeadDim;
            q_tile[q_row][dim] = q[load_q_base + dim];
        }
        else
        {
            q_tile[q_row][dim] = hip_bfloat16(0.0f);
        }
    }
    __syncthreads();

    float m = -INFINITY;
    float l = 0.0f;
    const int key_limit = valid_q ? (is_causal ? causal_key_limit(q_idx, seqlen_q, seqlen_k)
                                               : seqlen_k)
                                  : 0;
    const int last_q = min(seqlen_q - 1, q_block + Rows - 1);
    const int max_key_limit =
        last_q >= q_block ? (is_causal ? causal_key_limit(last_q, seqlen_q, seqlen_k) : seqlen_k)
                          : 0;

    for(int key_base = 0; key_base < max_key_limit; key_base += KeyBlock)
    {
        const int tile_count = min(KeyBlock, max_key_limit - key_base);
        for(int idx = tid; idx < tile_count * HeadDim; idx += static_cast<int>(blockDim.x))
        {
            const int key_offset = idx / HeadDim;
            const int dim = idx - key_offset * HeadDim;
            const int key_idx = key_base + key_offset;
            const size_t kv_base =
                ((static_cast<size_t>(batch) * seqlen_k + key_idx) * num_heads_k + kv_head) *
                HeadDim;
            k_tile[key_offset][dim] = k[kv_base + dim];
            v_tile[key_offset][dim] = v[kv_base + dim];
        }
        __syncthreads();

#if KILN_ROCM_HAS_GFX11_WMMA
        if(tid < KilnLogicalWarpSize)
        {
            const int ab_lane = lane & 15;
            for(int key_sub = 0; key_sub < KeyBlock; key_sub += WmmaCols)
            {
                kiln_f32x8 c_vec = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
                for(int dim_base = 0; dim_base < HeadDim; dim_base += WmmaCols)
                {
                    kiln_bf16x16 q_vec;
                    kiln_bf16x16 k_vec;
                    for(int k_inner = 0; k_inner < WmmaCols; ++k_inner)
                    {
                        const int dim = dim_base + k_inner;
                        q_vec[k_inner] =
                            ab_lane < Rows ? kiln_to_native_bf16(q_tile[ab_lane][dim])
                                           : static_cast<__bf16>(0.0f);
                        k_vec[k_inner] =
                            key_sub + ab_lane < tile_count
                                ? kiln_to_native_bf16(k_tile[key_sub + ab_lane][dim])
                                : static_cast<__bf16>(0.0f);
                    }
                    c_vec = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(q_vec, k_vec, c_vec);
                }

                const int score_col = key_sub + (lane & 15);
                const int score_row_parity = lane >> 4;
                for(int i = 0; i < 8; ++i)
                {
                    const int score_row = i * 2 + score_row_parity;
                    if(score_row < Rows && score_col < KeyBlock)
                    {
                        scores[score_row][score_col] = c_vec[i] * softmax_scale;
                    }
                }
            }
        }
#else
        if(tid < Rows * KeyBlock)
        {
            scores[tid / KeyBlock][tid % KeyBlock] = -INFINITY;
        }
#endif
        __syncthreads();

        if(lane == 0)
        {
            float tile_m = -INFINITY;
            for(int col = 0; col < KeyBlock; ++col)
            {
                const int key_idx = key_base + col;
                if(key_idx < key_limit)
                {
                    tile_m = fmaxf(tile_m, scores[row][col]);
                }
            }

            const float new_m = fmaxf(m, tile_m);
            float beta_sum = 0.0f;
            if(tile_m > -INFINITY)
            {
                for(int col = 0; col < KeyBlock; ++col)
                {
                    const int key_idx = key_base + col;
                    if(key_idx < key_limit)
                    {
                        const float beta = expf(scores[row][col] - new_m);
                        scores[row][col] = beta;
                        beta_sum += beta;
                    }
                    else
                    {
                        scores[row][col] = 0.0f;
                    }
                }
            }
            else
            {
                for(int col = 0; col < KeyBlock; ++col)
                {
                    scores[row][col] = 0.0f;
                }
            }
            const float alpha = l > 0.0f ? expf(m - new_m) : 0.0f;
            row_alpha[row] = alpha;
            l = l * alpha + beta_sum;
            m = new_m;
        }
        __syncthreads();

        const float alpha = row_alpha[row];
        for(int i = 0; i < lane_dims; ++i)
        {
            const int dim = dim_vals[i];
            float weighted = 0.0f;
            for(int col = 0; col < tile_count; ++col)
            {
                const float beta = scores[row][col];
                weighted += beta * static_cast<float>(v_tile[col][dim]);
            }
            acc_vals[i] = acc_vals[i] * alpha + weighted;
        }
        __syncthreads();
    }

    if(lane == 0)
    {
        row_inv_l[row] = l > 0.0f ? 1.0f / l : 0.0f;
        if(valid_q && lse != nullptr)
        {
            lse[(static_cast<size_t>(batch) * num_heads + head) * seqlen_q + q_idx] =
                l > 0.0f ? m + logf(l) : -INFINITY;
        }
    }
    __syncthreads();

    if(valid_q)
    {
        const float inv_l = row_inv_l[row];
        for(int i = 0; i < lane_dims; ++i)
        {
            out[q_base + dim_vals[i]] = hip_bfloat16(acc_vals[i] * inv_l);
        }
    }
}

__global__ void kiln_rocm_flash_fwd_wmma_gqa_r32h1k32_bf16_kernel(
    const hip_bfloat16* __restrict__ q,
    const hip_bfloat16* __restrict__ k,
    const hip_bfloat16* __restrict__ v,
    hip_bfloat16* __restrict__ out,
    float* __restrict__ lse,
    int batch_size,
    int seqlen_q,
    int seqlen_k,
    int num_heads,
    int num_heads_k,
    float softmax_scale,
    int is_causal)
{
    constexpr int HeadDim = 256;
    constexpr int Rows = 32;
    constexpr int KeyBlock = 32;
    constexpr int WmmaTile = 16;
    constexpr int WmmaRowTiles = Rows / WmmaTile;
    constexpr int WmmaColTiles = KeyBlock / WmmaTile;
    constexpr int WmmaTasks = WmmaRowTiles * WmmaColTiles;

    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int row = tid / KilnLogicalWarpSize;
    const int q_block = static_cast<int>(blockIdx.x) * Rows;
    const int q_idx = q_block + row;
    const int head = static_cast<int>(blockIdx.y);
    const int batch = static_cast<int>(blockIdx.z);

    if(batch >= batch_size || head >= num_heads || row >= Rows)
    {
        return;
    }

    __shared__ hip_bfloat16 q_tile[Rows][HeadDim];
    __shared__ hip_bfloat16 k_tile[KeyBlock][HeadDim];
    __shared__ hip_bfloat16 v_tile[KeyBlock][HeadDim];
    __shared__ float scores[Rows][KeyBlock];
    __shared__ float row_alpha[Rows];
    __shared__ float row_inv_l[Rows];

    constexpr int MaxLaneDims = (HeadDim + 31) / 32;
    float acc_vals[MaxLaneDims];
    int dim_vals[MaxLaneDims];
    int lane_dims = 0;

    const int groups_per_kv_head = num_heads / num_heads_k;
    const int kv_head = head / groups_per_kv_head;
    const bool valid_q = q_idx < seqlen_q;
    const size_t q_base =
        ((static_cast<size_t>(batch) * seqlen_q + (valid_q ? q_idx : 0)) * num_heads + head) *
        HeadDim;

    for(int dim = lane; dim < HeadDim; dim += KilnLogicalWarpSize)
    {
        dim_vals[lane_dims] = dim;
        acc_vals[lane_dims] = 0.0f;
        ++lane_dims;
    }

    for(int idx = tid; idx < Rows * HeadDim; idx += static_cast<int>(blockDim.x))
    {
        const int q_row = idx / HeadDim;
        const int dim = idx - q_row * HeadDim;
        const int load_q = q_block + q_row;
        if(load_q < seqlen_q)
        {
            const size_t load_q_base =
                ((static_cast<size_t>(batch) * seqlen_q + load_q) * num_heads + head) * HeadDim;
            q_tile[q_row][dim] = q[load_q_base + dim];
        }
        else
        {
            q_tile[q_row][dim] = hip_bfloat16(0.0f);
        }
    }
    __syncthreads();

    float m = -INFINITY;
    float l = 0.0f;
    const int key_limit = valid_q ? (is_causal ? causal_key_limit(q_idx, seqlen_q, seqlen_k)
                                               : seqlen_k)
                                  : 0;
    const int last_q = min(seqlen_q - 1, q_block + Rows - 1);
    const int max_key_limit =
        last_q >= q_block ? (is_causal ? causal_key_limit(last_q, seqlen_q, seqlen_k) : seqlen_k)
                          : 0;

    for(int key_base = 0; key_base < max_key_limit; key_base += KeyBlock)
    {
        const int tile_count = min(KeyBlock, max_key_limit - key_base);
        for(int idx = tid; idx < tile_count * HeadDim; idx += static_cast<int>(blockDim.x))
        {
            const int key_offset = idx / HeadDim;
            const int dim = idx - key_offset * HeadDim;
            const int key_idx = key_base + key_offset;
            const size_t kv_base =
                ((static_cast<size_t>(batch) * seqlen_k + key_idx) * num_heads_k + kv_head) *
                HeadDim;
            k_tile[key_offset][dim] = k[kv_base + dim];
            v_tile[key_offset][dim] = v[kv_base + dim];
        }
        __syncthreads();

#if KILN_ROCM_HAS_GFX11_WMMA
        if(tid < WmmaTasks * KilnLogicalWarpSize)
        {
            const int task = tid / KilnLogicalWarpSize;
            const int task_lane = tid & 31;
            const int row_tile = task / WmmaColTiles;
            const int col_tile = task - row_tile * WmmaColTiles;
            const int row_start = row_tile * WmmaTile;
            const int key_sub = col_tile * WmmaTile;
            const int ab_lane = task_lane & 15;

            kiln_f32x8 c_vec = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            for(int dim_base = 0; dim_base < HeadDim; dim_base += WmmaTile)
            {
                kiln_bf16x16 q_vec;
                kiln_bf16x16 k_vec;
                for(int k_inner = 0; k_inner < WmmaTile; ++k_inner)
                {
                    const int dim = dim_base + k_inner;
                    q_vec[k_inner] = kiln_to_native_bf16(q_tile[row_start + ab_lane][dim]);
                    k_vec[k_inner] =
                        key_sub + ab_lane < tile_count
                            ? kiln_to_native_bf16(k_tile[key_sub + ab_lane][dim])
                            : static_cast<__bf16>(0.0f);
                }
                c_vec = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(q_vec, k_vec, c_vec);
            }

            const int score_col = key_sub + (task_lane & 15);
            const int score_row_parity = task_lane >> 4;
            for(int i = 0; i < 8; ++i)
            {
                const int score_row = row_start + i * 2 + score_row_parity;
                if(score_row < Rows && score_col < KeyBlock)
                {
                    scores[score_row][score_col] = c_vec[i] * softmax_scale;
                }
            }
        }
#else
        if(tid < Rows * KeyBlock)
        {
            scores[tid / KeyBlock][tid % KeyBlock] = -INFINITY;
        }
#endif
        __syncthreads();

        if(lane == 0)
        {
            float tile_m = -INFINITY;
            for(int col = 0; col < KeyBlock; ++col)
            {
                const int key_idx = key_base + col;
                if(key_idx < key_limit)
                {
                    tile_m = fmaxf(tile_m, scores[row][col]);
                }
            }

            const float new_m = fmaxf(m, tile_m);
            float beta_sum = 0.0f;
            if(tile_m > -INFINITY)
            {
                for(int col = 0; col < KeyBlock; ++col)
                {
                    const int key_idx = key_base + col;
                    if(key_idx < key_limit)
                    {
                        const float beta = expf(scores[row][col] - new_m);
                        scores[row][col] = beta;
                        beta_sum += beta;
                    }
                    else
                    {
                        scores[row][col] = 0.0f;
                    }
                }
            }
            else
            {
                for(int col = 0; col < KeyBlock; ++col)
                {
                    scores[row][col] = 0.0f;
                }
            }
            const float alpha = l > 0.0f ? expf(m - new_m) : 0.0f;
            row_alpha[row] = alpha;
            l = l * alpha + beta_sum;
            m = new_m;
        }
        __syncthreads();

        const float alpha = row_alpha[row];
        for(int i = 0; i < lane_dims; ++i)
        {
            const int dim = dim_vals[i];
            float weighted = 0.0f;
            for(int col = 0; col < tile_count; ++col)
            {
                const float beta = scores[row][col];
                weighted += beta * static_cast<float>(v_tile[col][dim]);
            }
            acc_vals[i] = acc_vals[i] * alpha + weighted;
        }
        __syncthreads();
    }

    if(lane == 0)
    {
        row_inv_l[row] = l > 0.0f ? 1.0f / l : 0.0f;
        if(valid_q && lse != nullptr)
        {
            lse[(static_cast<size_t>(batch) * num_heads + head) * seqlen_q + q_idx] =
                l > 0.0f ? m + logf(l) : -INFINITY;
        }
    }
    __syncthreads();

    if(valid_q)
    {
        const float inv_l = row_inv_l[row];
        for(int i = 0; i < lane_dims; ++i)
        {
            out[q_base + dim_vals[i]] = hip_bfloat16(acc_vals[i] * inv_l);
        }
    }
}

__global__ void kiln_rocm_flash_fwd_wmma_gqa_r64h1k32_bf16_kernel(
    const hip_bfloat16* __restrict__ q,
    const hip_bfloat16* __restrict__ k,
    const hip_bfloat16* __restrict__ v,
    hip_bfloat16* __restrict__ out,
    float* __restrict__ lse,
    int batch_size,
    int seqlen_q,
    int seqlen_k,
    int num_heads,
    int num_heads_k,
    float softmax_scale,
    int is_causal)
{
    constexpr int HeadDim = 256;
    constexpr int Rows = 64;
    constexpr int QuarterRows = Rows / 4;
    constexpr int KeyBlock = 32;
    constexpr int WmmaTile = 16;
    constexpr int WmmaRowTiles = Rows / WmmaTile;
    constexpr int WmmaColTiles = KeyBlock / WmmaTile;
    constexpr int WmmaTasks = WmmaRowTiles * WmmaColTiles;
    constexpr int Bf16PerU32 = 2;
    constexpr int HeadDimPairs = HeadDim / Bf16PerU32;
    static_assert(HeadDim % Bf16PerU32 == 0, "HeadDim must be even for packed bf16 staging");
    static_assert(Rows % 4 == 0, "Rows must be divisible by four");

    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp_row = tid / KilnLogicalWarpSize;
    const int row0 = warp_row;
    const int row1 = warp_row + QuarterRows;
    const int row2 = warp_row + 2 * QuarterRows;
    const int row3 = warp_row + 3 * QuarterRows;
    const int q_block = static_cast<int>(blockIdx.x) * Rows;
    const int q_idx0 = q_block + row0;
    const int q_idx1 = q_block + row1;
    const int q_idx2 = q_block + row2;
    const int q_idx3 = q_block + row3;
    const int head = static_cast<int>(blockIdx.y);
    const int batch = static_cast<int>(blockIdx.z);

    if(batch >= batch_size || head >= num_heads || warp_row >= QuarterRows)
    {
        return;
    }

    __shared__ hip_bfloat16 q_tile[Rows][HeadDim];
    __shared__ hip_bfloat16 k_tile[WmmaTile][HeadDim];
    __shared__ hip_bfloat16 v_tile[KeyBlock][HeadDim];
    __shared__ float scores[Rows][KeyBlock];

    constexpr int LanePairs = HeadDimPairs / KilnLogicalWarpSize;
    static_assert(HeadDimPairs % KilnLogicalWarpSize == 0,
                  "HeadDimPairs must be evenly distributed across a wave");
    float acc_vals0_lo[LanePairs];
    float acc_vals0_hi[LanePairs];
    float acc_vals1_lo[LanePairs];
    float acc_vals1_hi[LanePairs];
    float acc_vals2_lo[LanePairs];
    float acc_vals2_hi[LanePairs];
    float acc_vals3_lo[LanePairs];
    float acc_vals3_hi[LanePairs];
    int dim_pair_vals[LanePairs];

    const int groups_per_kv_head = num_heads / num_heads_k;
    const int kv_head = head / groups_per_kv_head;
    const size_t q_head_pair_base =
        (static_cast<size_t>(batch) * seqlen_q * num_heads + head) * HeadDimPairs;
    const size_t q_seq_stride_pairs = static_cast<size_t>(num_heads) * HeadDimPairs;
    const size_t kv_head_pair_base =
        (static_cast<size_t>(batch) * seqlen_k * num_heads_k + kv_head) * HeadDimPairs;
    const size_t kv_seq_stride_pairs = static_cast<size_t>(num_heads_k) * HeadDimPairs;
    const bool valid_q0 = q_idx0 < seqlen_q;
    const bool valid_q1 = q_idx1 < seqlen_q;
    const bool valid_q2 = q_idx2 < seqlen_q;
    const bool valid_q3 = q_idx3 < seqlen_q;
    const size_t q_base0 =
        ((static_cast<size_t>(batch) * seqlen_q + (valid_q0 ? q_idx0 : 0)) * num_heads + head) *
        HeadDim;
    const size_t q_base1 =
        ((static_cast<size_t>(batch) * seqlen_q + (valid_q1 ? q_idx1 : 0)) * num_heads + head) *
        HeadDim;
    const size_t q_base2 =
        ((static_cast<size_t>(batch) * seqlen_q + (valid_q2 ? q_idx2 : 0)) * num_heads + head) *
        HeadDim;
    const size_t q_base3 =
        ((static_cast<size_t>(batch) * seqlen_q + (valid_q3 ? q_idx3 : 0)) * num_heads + head) *
        HeadDim;

    for(int i = 0; i < LanePairs; ++i)
    {
        dim_pair_vals[i] = lane + i * KilnLogicalWarpSize;
        acc_vals0_lo[i] = 0.0f;
        acc_vals0_hi[i] = 0.0f;
        acc_vals1_lo[i] = 0.0f;
        acc_vals1_hi[i] = 0.0f;
        acc_vals2_lo[i] = 0.0f;
        acc_vals2_hi[i] = 0.0f;
        acc_vals3_lo[i] = 0.0f;
        acc_vals3_hi[i] = 0.0f;
    }

    uint32_t* q_tile_u32 = reinterpret_cast<uint32_t*>(&q_tile[0][0]);
    uint32_t* k_tile_u32 = reinterpret_cast<uint32_t*>(&k_tile[0][0]);
    uint32_t* v_tile_u32 = reinterpret_cast<uint32_t*>(&v_tile[0][0]);
    const uint32_t* q_u32 = reinterpret_cast<const uint32_t*>(q);
    const uint32_t* k_u32 = reinterpret_cast<const uint32_t*>(k);
    const uint32_t* v_u32 = reinterpret_cast<const uint32_t*>(v);

    for(int pair = tid; pair < Rows * HeadDimPairs; pair += static_cast<int>(blockDim.x))
    {
        const int q_row = pair / HeadDimPairs;
        const int dim_pair = pair - q_row * HeadDimPairs;
        const int load_q = q_block + q_row;
        if(load_q < seqlen_q)
        {
            q_tile_u32[pair] =
                q_u32[q_head_pair_base + static_cast<size_t>(load_q) * q_seq_stride_pairs +
                      dim_pair];
        }
        else
        {
            q_tile_u32[pair] = 0;
        }
    }
    __syncthreads();

    float m0 = -INFINITY;
    float l0 = 0.0f;
    float m1 = -INFINITY;
    float l1 = 0.0f;
    float m2 = -INFINITY;
    float l2 = 0.0f;
    float m3 = -INFINITY;
    float l3 = 0.0f;
    const int key_limit0 = valid_q0 ? (is_causal ? causal_key_limit(q_idx0, seqlen_q, seqlen_k)
                                                 : seqlen_k)
                                    : 0;
    const int key_limit1 = valid_q1 ? (is_causal ? causal_key_limit(q_idx1, seqlen_q, seqlen_k)
                                                 : seqlen_k)
                                    : 0;
    const int key_limit2 = valid_q2 ? (is_causal ? causal_key_limit(q_idx2, seqlen_q, seqlen_k)
                                                 : seqlen_k)
                                    : 0;
    const int key_limit3 = valid_q3 ? (is_causal ? causal_key_limit(q_idx3, seqlen_q, seqlen_k)
                                                 : seqlen_k)
                                    : 0;
    const int last_q = min(seqlen_q - 1, q_block + Rows - 1);
    const int max_key_limit =
        last_q >= q_block ? (is_causal ? causal_key_limit(last_q, seqlen_q, seqlen_k) : seqlen_k)
                          : 0;

    for(int key_base = 0; key_base < max_key_limit; key_base += KeyBlock)
    {
        const int tile_count = min(KeyBlock, max_key_limit - key_base);
        for(int pair = tid; pair < tile_count * HeadDimPairs; pair += static_cast<int>(blockDim.x))
        {
            const int key_offset = pair / HeadDimPairs;
            const int dim_pair = pair - key_offset * HeadDimPairs;
            const int key_idx = key_base + key_offset;
            v_tile_u32[pair] =
                v_u32[kv_head_pair_base + static_cast<size_t>(key_idx) * kv_seq_stride_pairs +
                      dim_pair];
        }
        __syncthreads();

#if KILN_ROCM_HAS_GFX11_WMMA
        for(int col_tile = 0; col_tile < WmmaColTiles; ++col_tile)
        {
            const int key_col_start = col_tile * WmmaTile;
            for(int pair = tid; pair < WmmaTile * HeadDimPairs; pair += static_cast<int>(blockDim.x))
            {
                const int key_offset = key_col_start + pair / HeadDimPairs;
                const int dim_pair = pair % HeadDimPairs;
                if(key_offset < tile_count)
                {
                    const int key_idx = key_base + key_offset;
                    k_tile_u32[pair] =
                        k_u32[kv_head_pair_base +
                              static_cast<size_t>(key_idx) * kv_seq_stride_pairs + dim_pair];
                }
                else
                {
                    k_tile_u32[pair] = 0;
                }
            }
            __syncthreads();

            if(tid < WmmaRowTiles * KilnLogicalWarpSize)
            {
                const int task = tid / KilnLogicalWarpSize;
                const int task_lane = tid & 31;
                const int row_tile = task;
                const int row_start = row_tile * WmmaTile;
                const int ab_lane = task_lane & 15;

                kiln_f32x8 c_vec = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
                for(int dim_base = 0; dim_base < HeadDim; dim_base += WmmaTile)
                {
                    kiln_bf16x16 q_vec;
                    kiln_bf16x16 k_vec;
                    for(int k_inner = 0; k_inner < WmmaTile; ++k_inner)
                    {
                        const int dim = dim_base + k_inner;
                        q_vec[k_inner] = kiln_to_native_bf16(q_tile[row_start + ab_lane][dim]);
                        k_vec[k_inner] = kiln_to_native_bf16(k_tile[ab_lane][dim]);
                    }
                    c_vec = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(q_vec, k_vec, c_vec);
                }

                const int score_col = key_col_start + (task_lane & 15);
                const int score_row_parity = task_lane >> 4;
                for(int i = 0; i < 8; ++i)
                {
                    const int score_row = row_start + i * 2 + score_row_parity;
                    if(score_row < Rows && score_col < KeyBlock)
                    {
                        scores[score_row][score_col] = c_vec[i] * softmax_scale;
                    }
                }
            }
            __syncthreads();
        }
#else
        if(tid < Rows * KeyBlock)
        {
            scores[tid / KeyBlock][tid % KeyBlock] = -INFINITY;
        }
#endif
        __syncthreads();

        float alpha0 = 0.0f;
        float alpha1 = 0.0f;
        float alpha2 = 0.0f;
        float alpha3 = 0.0f;
        const bool full_valid_tile = key_base + KeyBlock <= key_limit0;
        {
            const int softmax_group = lane >> 3;
            const int col0 = lane & 7;
            const int col1 = col0 + 8;
            const int col2 = col0 + 16;
            const int col3 = col0 + 24;
            const int softmax_row = softmax_group == 0
                                        ? row0
                                        : (softmax_group == 1
                                               ? row1
                                               : (softmax_group == 2 ? row2 : row3));
            float beta0;
            float beta1;
            float new_m0;
            float new_m1;
            float new_m2;
            float new_m3;
            if(full_valid_tile)
            {
                const float score0 = scores[softmax_row][col0];
                const float score1 = scores[softmax_row][col1];
                const float score2 = scores[softmax_row][col2];
                const float score3 = scores[softmax_row][col3];
                const float tile_m_group =
                    kiln_quarter_wave_reduce_max(fmaxf(fmaxf(score0, score1), fmaxf(score2, score3)));
                const float tile_m0 = __shfl(tile_m_group, 0, KilnLogicalWarpSize);
                const float tile_m1 = __shfl(tile_m_group, 8, KilnLogicalWarpSize);
                const float tile_m2 = __shfl(tile_m_group, 16, KilnLogicalWarpSize);
                const float tile_m3 = __shfl(tile_m_group, 24, KilnLogicalWarpSize);
                new_m0 = fmaxf(m0, tile_m0);
                new_m1 = fmaxf(m1, tile_m1);
                new_m2 = fmaxf(m2, tile_m2);
                new_m3 = fmaxf(m3, tile_m3);
                const float new_m = softmax_group == 0
                                        ? new_m0
                                        : (softmax_group == 1
                                               ? new_m1
                                               : (softmax_group == 2 ? new_m2 : new_m3));
                beta0 = expf(score0 - new_m);
                beta1 = expf(score1 - new_m);
                const float beta2 = expf(score2 - new_m);
                const float beta3 = expf(score3 - new_m);
                scores[softmax_row][col0] = beta0;
                scores[softmax_row][col1] = beta1;
                scores[softmax_row][col2] = beta2;
                scores[softmax_row][col3] = beta3;
                beta0 += beta2;
                beta1 += beta3;
            }
            else
            {
                const int key_idx0 = key_base + col0;
                const int key_idx1 = key_base + col1;
                const int key_idx2 = key_base + col2;
                const int key_idx3 = key_base + col3;
                const int key_limit = softmax_group == 0
                                          ? key_limit0
                                          : (softmax_group == 1
                                                 ? key_limit1
                                                 : (softmax_group == 2 ? key_limit2 : key_limit3));
                const bool valid_key0 = key_idx0 < key_limit;
                const bool valid_key1 = key_idx1 < key_limit;
                const bool valid_key2 = key_idx2 < key_limit;
                const bool valid_key3 = key_idx3 < key_limit;
                const float score0 = valid_key0 ? scores[softmax_row][col0] : -INFINITY;
                const float score1 = valid_key1 ? scores[softmax_row][col1] : -INFINITY;
                const float score2 = valid_key2 ? scores[softmax_row][col2] : -INFINITY;
                const float score3 = valid_key3 ? scores[softmax_row][col3] : -INFINITY;
                const float local_tile_m = fmaxf(fmaxf(score0, score1), fmaxf(score2, score3));
                const float tile_m_group = kiln_quarter_wave_reduce_max(local_tile_m);
                const float tile_m0 = __shfl(tile_m_group, 0, KilnLogicalWarpSize);
                const float tile_m1 = __shfl(tile_m_group, 8, KilnLogicalWarpSize);
                const float tile_m2 = __shfl(tile_m_group, 16, KilnLogicalWarpSize);
                const float tile_m3 = __shfl(tile_m_group, 24, KilnLogicalWarpSize);
                new_m0 = fmaxf(m0, tile_m0);
                new_m1 = fmaxf(m1, tile_m1);
                new_m2 = fmaxf(m2, tile_m2);
                new_m3 = fmaxf(m3, tile_m3);
                const float new_m = softmax_group == 0
                                        ? new_m0
                                        : (softmax_group == 1
                                               ? new_m1
                                               : (softmax_group == 2 ? new_m2 : new_m3));
                const float tile_m = softmax_group == 0
                                         ? tile_m0
                                         : (softmax_group == 1
                                                ? tile_m1
                                                : (softmax_group == 2 ? tile_m2 : tile_m3));
                beta0 = valid_key0 && tile_m > -INFINITY ? expf(score0 - new_m) : 0.0f;
                beta1 = valid_key1 && tile_m > -INFINITY ? expf(score1 - new_m) : 0.0f;
                const float beta2 =
                    valid_key2 && tile_m > -INFINITY ? expf(score2 - new_m) : 0.0f;
                const float beta3 =
                    valid_key3 && tile_m > -INFINITY ? expf(score3 - new_m) : 0.0f;
                scores[softmax_row][col0] = beta0;
                scores[softmax_row][col1] = beta1;
                scores[softmax_row][col2] = beta2;
                scores[softmax_row][col3] = beta3;
                beta0 += beta2;
                beta1 += beta3;
            }

            const float beta_sum_group = kiln_quarter_wave_reduce_sum(beta0 + beta1);
            const float beta_sum0 = __shfl(beta_sum_group, 0, KilnLogicalWarpSize);
            const float beta_sum1 = __shfl(beta_sum_group, 8, KilnLogicalWarpSize);
            const float beta_sum2 = __shfl(beta_sum_group, 16, KilnLogicalWarpSize);
            const float beta_sum3 = __shfl(beta_sum_group, 24, KilnLogicalWarpSize);
            alpha0 = l0 > 0.0f ? expf(m0 - new_m0) : 0.0f;
            alpha1 = l1 > 0.0f ? expf(m1 - new_m1) : 0.0f;
            alpha2 = l2 > 0.0f ? expf(m2 - new_m2) : 0.0f;
            alpha3 = l3 > 0.0f ? expf(m3 - new_m3) : 0.0f;

            l0 = l0 * alpha0 + beta_sum0;
            l1 = l1 * alpha1 + beta_sum1;
            l2 = l2 * alpha2 + beta_sum2;
            l3 = l3 * alpha3 + beta_sum3;
            m0 = new_m0;
            m1 = new_m1;
            m2 = new_m2;
            m3 = new_m3;
        }
        __syncwarp();

#pragma unroll
        for(int i = 0; i < LanePairs; ++i)
        {
            acc_vals0_lo[i] *= alpha0;
            acc_vals0_hi[i] *= alpha0;
            acc_vals1_lo[i] *= alpha1;
            acc_vals1_hi[i] *= alpha1;
            acc_vals2_lo[i] *= alpha2;
            acc_vals2_hi[i] *= alpha2;
            acc_vals3_lo[i] *= alpha3;
            acc_vals3_hi[i] *= alpha3;
        }
        for(int col = 0; col < tile_count; ++col)
        {
            const float score0 = scores[row0][col];
            const float score1 = scores[row1][col];
            const float score2 = scores[row2][col];
            const float score3 = scores[row3][col];
#pragma unroll
            for(int i = 0; i < LanePairs; ++i)
            {
                float vv_lo;
                float vv_hi;
                const int dim_pair = dim_pair_vals[i];
                kiln_bf16_pair_to_f32(v_tile_u32[col * HeadDimPairs + dim_pair], vv_lo, vv_hi);
                acc_vals0_lo[i] += score0 * vv_lo;
                acc_vals0_hi[i] += score0 * vv_hi;
                acc_vals1_lo[i] += score1 * vv_lo;
                acc_vals1_hi[i] += score1 * vv_hi;
                acc_vals2_lo[i] += score2 * vv_lo;
                acc_vals2_hi[i] += score2 * vv_hi;
                acc_vals3_lo[i] += score3 * vv_lo;
                acc_vals3_hi[i] += score3 * vv_hi;
            }
        }
        __syncthreads();
    }

    if(lane == 0)
    {
        if(valid_q0 && lse != nullptr)
        {
            lse[(static_cast<size_t>(batch) * num_heads + head) * seqlen_q + q_idx0] =
                l0 > 0.0f ? m0 + logf(l0) : -INFINITY;
        }
        if(valid_q1 && lse != nullptr)
        {
            lse[(static_cast<size_t>(batch) * num_heads + head) * seqlen_q + q_idx1] =
                l1 > 0.0f ? m1 + logf(l1) : -INFINITY;
        }
        if(valid_q2 && lse != nullptr)
        {
            lse[(static_cast<size_t>(batch) * num_heads + head) * seqlen_q + q_idx2] =
                l2 > 0.0f ? m2 + logf(l2) : -INFINITY;
        }
        if(valid_q3 && lse != nullptr)
        {
            lse[(static_cast<size_t>(batch) * num_heads + head) * seqlen_q + q_idx3] =
                l3 > 0.0f ? m3 + logf(l3) : -INFINITY;
        }
    }

    if(valid_q0)
    {
        const float inv_l = l0 > 0.0f ? 1.0f / l0 : 0.0f;
        for(int i = 0; i < LanePairs; ++i)
        {
            const int dim = dim_pair_vals[i] * Bf16PerU32;
            out[q_base0 + dim] = hip_bfloat16(acc_vals0_lo[i] * inv_l);
            out[q_base0 + dim + 1] = hip_bfloat16(acc_vals0_hi[i] * inv_l);
        }
    }
    if(valid_q1)
    {
        const float inv_l = l1 > 0.0f ? 1.0f / l1 : 0.0f;
        for(int i = 0; i < LanePairs; ++i)
        {
            const int dim = dim_pair_vals[i] * Bf16PerU32;
            out[q_base1 + dim] = hip_bfloat16(acc_vals1_lo[i] * inv_l);
            out[q_base1 + dim + 1] = hip_bfloat16(acc_vals1_hi[i] * inv_l);
        }
    }
    if(valid_q2)
    {
        const float inv_l = l2 > 0.0f ? 1.0f / l2 : 0.0f;
        for(int i = 0; i < LanePairs; ++i)
        {
            const int dim = dim_pair_vals[i] * Bf16PerU32;
            out[q_base2 + dim] = hip_bfloat16(acc_vals2_lo[i] * inv_l);
            out[q_base2 + dim + 1] = hip_bfloat16(acc_vals2_hi[i] * inv_l);
        }
    }
    if(valid_q3)
    {
        const float inv_l = l3 > 0.0f ? 1.0f / l3 : 0.0f;
        for(int i = 0; i < LanePairs; ++i)
        {
            const int dim = dim_pair_vals[i] * Bf16PerU32;
            out[q_base3 + dim] = hip_bfloat16(acc_vals3_lo[i] * inv_l);
            out[q_base3 + dim + 1] = hip_bfloat16(acc_vals3_hi[i] * inv_l);
        }
    }
}

#if KILN_HAS_CK_TILE_FMHA_FWD
struct KilnFmhaGfx11Policy : ck_tile::BlockFmhaV3PipelineDefaultPolicy
{
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto GetAlignmentK()
    {
        return 4;
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto GetAlignmentV()
    {
        return 4;
    }
};

template <int HeadDim, bool Causal>
int launch_ck_fwd_bf16(const hip_bfloat16* q,
                       const hip_bfloat16* k,
                       const hip_bfloat16* v,
                       hip_bfloat16* out,
                       float* lse,
                       int batch_size,
                       int seqlen_q,
                       int seqlen_k,
                       int num_heads,
                       int num_heads_k,
                       float softmax_scale,
                       void* stream)
{
    using data_type = ck_tile::fmha_fwd_v3_args::data_type_enum;
    using dtype = ck_tile::fmha_fwd_v3_problem_traits<data_type::bf16>;
    // CK Tile v3 asserts that QK head_dim is consumed in a single K0 loop.
    // gfx115x can satisfy that for d=128 with this policy's 4-element bf16
    // lane alignment. Larger head sizes are declined by the dispatcher below.
    using fmha_block_tile = ck_tile::sequence<256, 32, HeadDim, 128, 32, HeadDim>;
    using fmha_warp_gemm_shape = ck_tile::sequence<32, 32, 16>;
    using fmha_block_warps = ck_tile::sequence<8, 1, 1>;
    using fmha_shape = ck_tile::TileFmhaShape<fmha_block_tile,
                                             fmha_block_warps,
                                             fmha_warp_gemm_shape,
                                             fmha_block_warps,
                                             fmha_warp_gemm_shape,
                                             true>;
    using fmha_traits = ck_tile::TileFmhaFwdV3Traits<true, true, false, false, false, -1>;
    using fmha_mask = ck_tile::GenericAttentionMask<Causal, false>;
    using fmha_pipeline_problem =
        ck_tile::BlockFmhaFwdV3PipelineProblem<typename dtype::qkvp_dtype,
                                               typename dtype::qkvp_dtype,
                                               typename dtype::qkvp_dtype,
                                               typename dtype::acc_dtype,
                                               typename dtype::acc_dtype,
                                               typename dtype::lse_dtype,
                                               typename dtype::qkvp_dtype,
                                               typename dtype::acc_dtype,
                                               typename dtype::o_dtype,
                                               fmha_shape,
                                               false,
                                               fmha_mask,
                                               fmha_traits>;
    using fmha_pipeline =
        ck_tile::BlockFmhaFwdV3Pipeline<fmha_pipeline_problem, KilnFmhaGfx11Policy>;
    using fmha_epilogue = ck_tile::Default2DEpilogue<
        ck_tile::Default2DEpilogueProblem<typename dtype::acc_dtype,
                                          typename dtype::o_dtype,
                                          true,
                                          true,
                                          true>>;
    using fmha_kernel = ck_tile::FmhaFwdV3Kernel<fmha_pipeline, fmha_epilogue>;

    const ck_tile::index_t mask_type =
        Causal ? static_cast<ck_tile::index_t>(mask_enum::mask_bottom_right)
               : static_cast<ck_tile::index_t>(mask_enum::no_mask);
    int remap_opt = 2;
    if(Causal && ((num_heads % 8 != 0) || (16384 < seqlen_q)))
    {
        remap_opt = 65536 <= seqlen_q ? 0 : 1;
    }

    (void)lse;
    auto kargs = fmha_kernel::MakeKargs(q,
                                        k,
                                        v,
                                        nullptr,
                                        out,
                                        seqlen_q,
                                        seqlen_k,
                                        HeadDim,
                                        HeadDim,
                                        num_heads,
                                        num_heads / num_heads_k,
                                        softmax_scale,
                                        num_heads * HeadDim,
                                        num_heads_k * HeadDim,
                                        num_heads_k * HeadDim,
                                        num_heads * HeadDim,
                                        HeadDim,
                                        HeadDim,
                                        HeadDim,
                                        seqlen_q,
                                        HeadDim,
                                        seqlen_q * num_heads * HeadDim,
                                        seqlen_k * num_heads_k * HeadDim,
                                        seqlen_k * num_heads_k * HeadDim,
                                        num_heads * seqlen_q,
                                        seqlen_q * num_heads * HeadDim,
                                        Causal ? -1 : 0,
                                        0,
                                        mask_type,
                                        remap_opt,
                                        nullptr,
                                        nullptr);
    const dim3 grids = fmha_kernel::GridSize(batch_size, num_heads, seqlen_q, HeadDim);

    try
    {
        const ck_tile::stream_config stream_config{static_cast<hipStream_t>(stream), false};
        ck_tile::launch_kernel(stream_config,
                               ck_tile::make_kernel<fmha_kernel::kBlockPerCu>(
                                   fmha_kernel{}, grids, fmha_kernel::BlockSize(), 0, kargs));
    }
    catch(const std::exception&)
    {
        return -20;
    }

    const hipError_t err = hipPeekAtLastError();
    return err == hipSuccess ? 0 : -static_cast<int>(err);
}
#endif

template <int HeadDim, int Rows, int XDim>
__global__ void kiln_rocm_flash_fwd_qrows_bf16_kernel(const hip_bfloat16* __restrict__ q,
                                                const hip_bfloat16* __restrict__ k,
                                                const hip_bfloat16* __restrict__ v,
                                                hip_bfloat16* __restrict__ out,
                                                float* __restrict__ lse,
                                                int batch_size,
                                                int seqlen_q,
                                                int seqlen_k,
                                                int num_heads,
                                                int num_heads_k,
                                                float softmax_scale,
                                                int is_causal)
{
    const int q_idx = static_cast<int>(blockIdx.x) * Rows + static_cast<int>(threadIdx.y);
    const int head = static_cast<int>(blockIdx.y);
    const int batch = static_cast<int>(blockIdx.z);
    const int x = static_cast<int>(threadIdx.x);
    const int row = static_cast<int>(threadIdx.y);
    const int dim0 = x;
    const int dim1 = x + XDim;

    if(batch >= batch_size || head >= num_heads || x >= XDim || row >= Rows)
    {
        return;
    }

    const int groups_per_kv_head = num_heads / num_heads_k;
    const int kv_head = head / groups_per_kv_head;
    const bool valid_q = q_idx < seqlen_q;
    const size_t q_base =
        ((static_cast<size_t>(batch) * seqlen_q + (valid_q ? q_idx : 0)) * num_heads + head) *
        HeadDim;
    const float qv0 = (valid_q && dim0 < HeadDim) ? static_cast<float>(q[q_base + dim0]) : 0.0f;
    const float qv1 = (valid_q && dim1 < HeadDim) ? static_cast<float>(q[q_base + dim1]) : 0.0f;
    const int key_limit = valid_q ? (is_causal ? causal_key_limit(q_idx, seqlen_q, seqlen_k)
                                               : seqlen_k)
                                  : 0;
    const int last_q = min(seqlen_q - 1, static_cast<int>(blockIdx.x) * Rows + Rows - 1);
    const int max_key_limit =
        last_q >= static_cast<int>(blockIdx.x) * Rows
            ? (is_causal ? causal_key_limit(last_q, seqlen_q, seqlen_k) : seqlen_k)
            : 0;

    float m = -INFINITY;
    float l = 0.0f;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    __shared__ float scores[Rows];

    for(int key = 0; key < max_key_limit; ++key)
    {
        const bool valid_key = key < key_limit;
        const size_t kv_base =
            ((static_cast<size_t>(batch) * seqlen_k + key) * num_heads_k + kv_head) * HeadDim;
        const float kval0 =
            valid_key && dim0 < HeadDim ? static_cast<float>(k[kv_base + dim0]) : 0.0f;
        const float kval1 =
            valid_key && dim1 < HeadDim ? static_cast<float>(k[kv_base + dim1]) : 0.0f;
        const float partial = qv0 * kval0 + qv1 * kval1;
        const float score = valid_key ? row_sum_rows<Rows, XDim>(partial) * softmax_scale
                                      : row_sum_rows<Rows, XDim>(0.0f);
        if(x == 0)
        {
            scores[row] = score;
        }
        __syncthreads();

        if(valid_key)
        {
            const float score_row = scores[row];
            const float new_m = fmaxf(m, score_row);
            const float alpha = expf(m - new_m);
            const float beta = expf(score_row - new_m);
            const float vv0 =
                dim0 < HeadDim ? static_cast<float>(v[kv_base + dim0]) : 0.0f;
            const float vv1 =
                dim1 < HeadDim ? static_cast<float>(v[kv_base + dim1]) : 0.0f;

            acc0 = acc0 * alpha + beta * vv0;
            acc1 = acc1 * alpha + beta * vv1;
            l = l * alpha + beta;
            m = new_m;
        }
        __syncthreads();
    }

    if(valid_q)
    {
        const float inv_l = l > 0.0f ? 1.0f / l : 0.0f;
        if(dim0 < HeadDim)
        {
            out[q_base + dim0] = hip_bfloat16(acc0 * inv_l);
        }
        if(dim1 < HeadDim)
        {
            out[q_base + dim1] = hip_bfloat16(acc1 * inv_l);
        }
        if(x == 0 && lse != nullptr)
        {
            lse[(static_cast<size_t>(batch) * num_heads + head) * seqlen_q + q_idx] =
                l > 0.0f ? m + logf(l) : -INFINITY;
        }
    }
}

template <int HeadDim, int Rows>
__global__ void kiln_rocm_flash_fwd_warprows_bf16_kernel(const hip_bfloat16* __restrict__ q,
                                                         const hip_bfloat16* __restrict__ k,
                                                         const hip_bfloat16* __restrict__ v,
                                                         hip_bfloat16* __restrict__ out,
                                                         float* __restrict__ lse,
                                                         int batch_size,
                                                         int seqlen_q,
                                                         int seqlen_k,
                                                         int num_heads,
                                                         int num_heads_k,
                                                         float softmax_scale,
                                                         int is_causal)
{
    const int lane = static_cast<int>(threadIdx.x) % KilnLogicalWarpSize;
    const int row = static_cast<int>(threadIdx.x) / KilnLogicalWarpSize;
    const int q_idx = static_cast<int>(blockIdx.x) * Rows + row;
    const int head = static_cast<int>(blockIdx.y);
    const int batch = static_cast<int>(blockIdx.z);

    if(batch >= batch_size || head >= num_heads || row >= Rows)
    {
        return;
    }

    constexpr int MaxLaneDims = (HeadDim + 31) / 32;
    float q_vals[MaxLaneDims];
    float acc_vals[MaxLaneDims];
    int dim_vals[MaxLaneDims];
    int lane_dims = 0;

    const int groups_per_kv_head = num_heads / num_heads_k;
    const int kv_head = head / groups_per_kv_head;
    const bool valid_q = q_idx < seqlen_q;
    const size_t q_base =
        ((static_cast<size_t>(batch) * seqlen_q + (valid_q ? q_idx : 0)) * num_heads + head) *
        HeadDim;

    for(int dim = lane; dim < HeadDim; dim += KilnLogicalWarpSize)
    {
        dim_vals[lane_dims] = dim;
        q_vals[lane_dims] = valid_q ? static_cast<float>(q[q_base + dim]) : 0.0f;
        acc_vals[lane_dims] = 0.0f;
        ++lane_dims;
    }

    const int key_limit = valid_q ? (is_causal ? causal_key_limit(q_idx, seqlen_q, seqlen_k)
                                               : seqlen_k)
                                  : 0;
    const int last_q = min(seqlen_q - 1, static_cast<int>(blockIdx.x) * Rows + Rows - 1);
    const int max_key_limit =
        last_q >= static_cast<int>(blockIdx.x) * Rows
            ? (is_causal ? causal_key_limit(last_q, seqlen_q, seqlen_k) : seqlen_k)
            : 0;

    float m = -INFINITY;
    float l = 0.0f;

    for(int key = 0; key < max_key_limit; ++key)
    {
        const bool valid_key = key < key_limit;
        const size_t kv_base =
            ((static_cast<size_t>(batch) * seqlen_k + key) * num_heads_k + kv_head) * HeadDim;

        float partial = 0.0f;
        for(int i = 0; i < lane_dims; ++i)
        {
            partial += q_vals[i] *
                       (valid_key ? static_cast<float>(k[kv_base + dim_vals[i]]) : 0.0f);
        }
        const float score =
            valid_key ? kiln_wave_reduce_sum(partial) * softmax_scale : -INFINITY;

        if(valid_key)
        {
            const float new_m = fmaxf(m, score);
            const float alpha = expf(m - new_m);
            const float beta = expf(score - new_m);
            for(int i = 0; i < lane_dims; ++i)
            {
                const float vv = static_cast<float>(v[kv_base + dim_vals[i]]);
                acc_vals[i] = acc_vals[i] * alpha + beta * vv;
            }
            l = l * alpha + beta;
            m = new_m;
        }
    }

    if(valid_q)
    {
        const float inv_l = l > 0.0f ? 1.0f / l : 0.0f;
        for(int i = 0; i < lane_dims; ++i)
        {
            out[q_base + dim_vals[i]] = hip_bfloat16(acc_vals[i] * inv_l);
        }
        if(lane == 0 && lse != nullptr)
        {
            lse[(static_cast<size_t>(batch) * num_heads + head) * seqlen_q + q_idx] =
                l > 0.0f ? m + logf(l) : -INFINITY;
        }
    }
}

template <int HeadDim, int NumKeyWarps>
__global__ void kiln_rocm_flash_fwd_keysplit_bf16_kernel(const hip_bfloat16* __restrict__ q,
                                                         const hip_bfloat16* __restrict__ k,
                                                         const hip_bfloat16* __restrict__ v,
                                                         hip_bfloat16* __restrict__ out,
                                                         float* __restrict__ lse,
                                                         int batch_size,
                                                         int seqlen_q,
                                                         int seqlen_k,
                                                         int num_heads,
                                                         int num_heads_k,
                                                         float softmax_scale,
                                                         int is_causal)
{
    const int lane = static_cast<int>(threadIdx.x) % KilnLogicalWarpSize;
    const int key_warp = static_cast<int>(threadIdx.x) / KilnLogicalWarpSize;
    const int q_idx = static_cast<int>(blockIdx.x);
    const int head = static_cast<int>(blockIdx.y);
    const int batch = static_cast<int>(blockIdx.z);

    if(batch >= batch_size || q_idx >= seqlen_q || head >= num_heads ||
       key_warp >= NumKeyWarps)
    {
        return;
    }

    constexpr int MaxLaneDims = (HeadDim + 31) / 32;
    float q_vals[MaxLaneDims];
    float acc_vals[MaxLaneDims];
    int dim_vals[MaxLaneDims];
    int lane_dims = 0;

    const int groups_per_kv_head = num_heads / num_heads_k;
    const int kv_head = head / groups_per_kv_head;
    const size_t q_base =
        ((static_cast<size_t>(batch) * seqlen_q + q_idx) * num_heads + head) * HeadDim;

    for(int dim = lane; dim < HeadDim; dim += KilnLogicalWarpSize)
    {
        dim_vals[lane_dims] = dim;
        q_vals[lane_dims] = static_cast<float>(q[q_base + dim]);
        acc_vals[lane_dims] = 0.0f;
        ++lane_dims;
    }

    const int key_limit = is_causal ? causal_key_limit(q_idx, seqlen_q, seqlen_k) : seqlen_k;
    float m = -INFINITY;
    float l = 0.0f;

    for(int key = key_warp; key < key_limit; key += NumKeyWarps)
    {
        const size_t kv_base =
            ((static_cast<size_t>(batch) * seqlen_k + key) * num_heads_k + kv_head) * HeadDim;

        float partial = 0.0f;
        for(int i = 0; i < lane_dims; ++i)
        {
            partial += q_vals[i] * static_cast<float>(k[kv_base + dim_vals[i]]);
        }
        const float score = kiln_wave_reduce_sum(partial) * softmax_scale;
        const float new_m = fmaxf(m, score);
        const float alpha = expf(m - new_m);
        const float beta = expf(score - new_m);
        for(int i = 0; i < lane_dims; ++i)
        {
            const float vv = static_cast<float>(v[kv_base + dim_vals[i]]);
            acc_vals[i] = acc_vals[i] * alpha + beta * vv;
        }
        l = l * alpha + beta;
        m = new_m;
    }

    __shared__ float partial_m[NumKeyWarps];
    __shared__ float partial_l[NumKeyWarps];
    __shared__ float partial_acc[NumKeyWarps][HeadDim];
    if(lane == 0)
    {
        partial_m[key_warp] = m;
        partial_l[key_warp] = l;
    }
    for(int i = 0; i < lane_dims; ++i)
    {
        partial_acc[key_warp][dim_vals[i]] = acc_vals[i];
    }
    __syncthreads();

    if(key_warp == 0)
    {
        float global_m = -INFINITY;
        for(int w = 0; w < NumKeyWarps; ++w)
        {
            if(partial_l[w] > 0.0f)
            {
                global_m = fmaxf(global_m, partial_m[w]);
            }
        }

        float global_l = 0.0f;
        for(int w = 0; w < NumKeyWarps; ++w)
        {
            if(partial_l[w] > 0.0f)
            {
                global_l += partial_l[w] * expf(partial_m[w] - global_m);
            }
        }
        const float inv_l = global_l > 0.0f ? 1.0f / global_l : 0.0f;

        for(int dim = lane; dim < HeadDim; dim += KilnLogicalWarpSize)
        {
            float acc = 0.0f;
            for(int w = 0; w < NumKeyWarps; ++w)
            {
                if(partial_l[w] > 0.0f)
                {
                    acc += partial_acc[w][dim] * expf(partial_m[w] - global_m);
                }
            }
            out[q_base + dim] = hip_bfloat16(acc * inv_l);
        }
        if(lane == 0 && lse != nullptr)
        {
            lse[(static_cast<size_t>(batch) * num_heads + head) * seqlen_q + q_idx] =
                global_l > 0.0f ? global_m + logf(global_l) : -INFINITY;
        }
    }
}

template <int HeadDim, int Rows, int KeyBlock>
__global__ void kiln_rocm_flash_fwd_qblock_cached_bf16_kernel(
    const hip_bfloat16* __restrict__ q,
    const hip_bfloat16* __restrict__ k,
    const hip_bfloat16* __restrict__ v,
    hip_bfloat16* __restrict__ out,
    float* __restrict__ lse,
    int batch_size,
    int seqlen_q,
    int seqlen_k,
    int num_heads,
    int num_heads_k,
    float softmax_scale,
    int is_causal)
{
    const int lane = static_cast<int>(threadIdx.x) % KilnLogicalWarpSize;
    const int row = static_cast<int>(threadIdx.x) / KilnLogicalWarpSize;
    const int q_block = static_cast<int>(blockIdx.x) * Rows;
    const int q_idx = q_block + row;
    const int head = static_cast<int>(blockIdx.y);
    const int batch = static_cast<int>(blockIdx.z);

    if(batch >= batch_size || head >= num_heads || row >= Rows)
    {
        return;
    }

    __shared__ hip_bfloat16 k_tile[KeyBlock][HeadDim];
    __shared__ hip_bfloat16 v_tile[KeyBlock][HeadDim];

    constexpr int MaxLaneDims = (HeadDim + 31) / 32;
    float q_vals[MaxLaneDims];
    float acc_vals[MaxLaneDims];
    int dim_vals[MaxLaneDims];
    int lane_dims = 0;

    const int groups_per_kv_head = num_heads / num_heads_k;
    const int kv_head = head / groups_per_kv_head;
    const bool valid_q = q_idx < seqlen_q;
    const size_t q_base =
        ((static_cast<size_t>(batch) * seqlen_q + (valid_q ? q_idx : 0)) * num_heads + head) *
        HeadDim;

    for(int dim = lane; dim < HeadDim; dim += KilnLogicalWarpSize)
    {
        dim_vals[lane_dims] = dim;
        q_vals[lane_dims] = valid_q ? static_cast<float>(q[q_base + dim]) : 0.0f;
        acc_vals[lane_dims] = 0.0f;
        ++lane_dims;
    }

    const int key_limit = valid_q ? (is_causal ? causal_key_limit(q_idx, seqlen_q, seqlen_k)
                                               : seqlen_k)
                                  : 0;
    const int last_q = min(seqlen_q - 1, q_block + Rows - 1);
    const int max_key_limit =
        last_q >= q_block ? (is_causal ? causal_key_limit(last_q, seqlen_q, seqlen_k) : seqlen_k)
                          : 0;

    float m = -INFINITY;
    float l = 0.0f;

    for(int key_base = 0; key_base < max_key_limit; key_base += KeyBlock)
    {
        const int tile_count = min(KeyBlock, max_key_limit - key_base);
        const int linear_thread = static_cast<int>(threadIdx.x);
        const int thread_count = static_cast<int>(blockDim.x);
        for(int idx = linear_thread; idx < tile_count * HeadDim; idx += thread_count)
        {
            const int key_offset = idx / HeadDim;
            const int dim = idx - key_offset * HeadDim;
            const int key = key_base + key_offset;
            const size_t kv_base =
                ((static_cast<size_t>(batch) * seqlen_k + key) * num_heads_k + kv_head) *
                HeadDim;
            k_tile[key_offset][dim] = k[kv_base + dim];
            v_tile[key_offset][dim] = v[kv_base + dim];
        }
        __syncthreads();

        for(int key_offset = 0; key_offset < tile_count; ++key_offset)
        {
            const int key = key_base + key_offset;
            const bool valid_key = valid_q && key < key_limit;

            float partial = 0.0f;
            for(int i = 0; i < lane_dims; ++i)
            {
                partial += q_vals[i] * static_cast<float>(k_tile[key_offset][dim_vals[i]]);
            }
            const float score =
                valid_key ? kiln_wave_reduce_sum(partial) * softmax_scale : -INFINITY;

            if(valid_key)
            {
                const float new_m = fmaxf(m, score);
                const float alpha = expf(m - new_m);
                const float beta = expf(score - new_m);
                for(int i = 0; i < lane_dims; ++i)
                {
                    const float vv = static_cast<float>(v_tile[key_offset][dim_vals[i]]);
                    acc_vals[i] = acc_vals[i] * alpha + beta * vv;
                }
                l = l * alpha + beta;
                m = new_m;
            }
        }
        __syncthreads();
    }

    if(valid_q)
    {
        const float inv_l = l > 0.0f ? 1.0f / l : 0.0f;
        for(int i = 0; i < lane_dims; ++i)
        {
            out[q_base + dim_vals[i]] = hip_bfloat16(acc_vals[i] * inv_l);
        }
        if(lane == 0 && lse != nullptr)
        {
            lse[(static_cast<size_t>(batch) * num_heads + head) * seqlen_q + q_idx] =
                l > 0.0f ? m + logf(l) : -INFINITY;
        }
    }
}

template <int HeadDim, int Rows, int QueryHeadsPerBlock, int KeyBlock>
__global__ void kiln_rocm_flash_fwd_gqa_qblock_cached_bf16_kernel(
    const hip_bfloat16* __restrict__ q,
    const hip_bfloat16* __restrict__ k,
    const hip_bfloat16* __restrict__ v,
    hip_bfloat16* __restrict__ out,
    float* __restrict__ lse,
    int batch_size,
    int seqlen_q,
    int seqlen_k,
    int num_heads,
    int num_heads_k,
    float softmax_scale,
    int is_causal)
{
    const int lane = static_cast<int>(threadIdx.x) % KilnLogicalWarpSize;
    const int warp = static_cast<int>(threadIdx.x) / KilnLogicalWarpSize;
    const int row = warp % Rows;
    const int q_head_in_group = warp / Rows;
    const int q_block = static_cast<int>(blockIdx.x) * Rows;
    const int q_idx = q_block + row;
    const int head_group = static_cast<int>(blockIdx.y);
    const int head = head_group * QueryHeadsPerBlock + q_head_in_group;
    const int batch = static_cast<int>(blockIdx.z);

    if(batch >= batch_size || q_head_in_group >= QueryHeadsPerBlock ||
       head >= num_heads || row >= Rows)
    {
        return;
    }
    const int groups_per_kv_head = num_heads / num_heads_k;
    const int kv_head = head / groups_per_kv_head;
    if(kv_head >= num_heads_k)
    {
        return;
    }

    __shared__ hip_bfloat16 k_tile[KeyBlock][HeadDim];
    __shared__ hip_bfloat16 v_tile[KeyBlock][HeadDim];

    constexpr int MaxLaneDims = (HeadDim + 31) / 32;
    float q_vals[MaxLaneDims];
    float acc_vals[MaxLaneDims];
    int dim_vals[MaxLaneDims];
    int lane_dims = 0;

    const bool valid_q = q_idx < seqlen_q;
    const size_t q_base =
        ((static_cast<size_t>(batch) * seqlen_q + (valid_q ? q_idx : 0)) * num_heads + head) *
        HeadDim;

    for(int dim = lane; dim < HeadDim; dim += KilnLogicalWarpSize)
    {
        dim_vals[lane_dims] = dim;
        q_vals[lane_dims] = valid_q ? static_cast<float>(q[q_base + dim]) : 0.0f;
        acc_vals[lane_dims] = 0.0f;
        ++lane_dims;
    }

    const int key_limit = valid_q ? (is_causal ? causal_key_limit(q_idx, seqlen_q, seqlen_k)
                                               : seqlen_k)
                                  : 0;
    const int last_q = min(seqlen_q - 1, q_block + Rows - 1);
    const int max_key_limit =
        last_q >= q_block ? (is_causal ? causal_key_limit(last_q, seqlen_q, seqlen_k) : seqlen_k)
                          : 0;

    float m = -INFINITY;
    float l = 0.0f;

    for(int key_base = 0; key_base < max_key_limit; key_base += KeyBlock)
    {
        const int tile_count = min(KeyBlock, max_key_limit - key_base);
        const int linear_thread = static_cast<int>(threadIdx.x);
        const int thread_count = static_cast<int>(blockDim.x);
        for(int idx = linear_thread; idx < tile_count * HeadDim; idx += thread_count)
        {
            const int key_offset = idx / HeadDim;
            const int dim = idx - key_offset * HeadDim;
            const int key = key_base + key_offset;
            const size_t kv_base =
                ((static_cast<size_t>(batch) * seqlen_k + key) * num_heads_k + kv_head) *
                HeadDim;
            k_tile[key_offset][dim] = k[kv_base + dim];
            v_tile[key_offset][dim] = v[kv_base + dim];
        }
        __syncthreads();

        for(int key_offset = 0; key_offset < tile_count; ++key_offset)
        {
            const int key = key_base + key_offset;
            const bool valid_key = valid_q && key < key_limit;

            float partial = 0.0f;
            for(int i = 0; i < lane_dims; ++i)
            {
                partial += q_vals[i] * static_cast<float>(k_tile[key_offset][dim_vals[i]]);
            }
            const float score =
                valid_key ? kiln_wave_reduce_sum(partial) * softmax_scale : -INFINITY;

            if(valid_key)
            {
                const float new_m = fmaxf(m, score);
                const float alpha = expf(m - new_m);
                const float beta = expf(score - new_m);
                for(int i = 0; i < lane_dims; ++i)
                {
                    const float vv = static_cast<float>(v_tile[key_offset][dim_vals[i]]);
                    acc_vals[i] = acc_vals[i] * alpha + beta * vv;
                }
                l = l * alpha + beta;
                m = new_m;
            }
        }
        __syncthreads();
    }

    if(valid_q)
    {
        const float inv_l = l > 0.0f ? 1.0f / l : 0.0f;
        for(int i = 0; i < lane_dims; ++i)
        {
            out[q_base + dim_vals[i]] = hip_bfloat16(acc_vals[i] * inv_l);
        }
        if(lane == 0 && lse != nullptr)
        {
            lse[(static_cast<size_t>(batch) * num_heads + head) * seqlen_q + q_idx] =
                l > 0.0f ? m + logf(l) : -INFINITY;
        }
    }
}

template <int HeadDim, int Rows>
__global__ void kiln_rocm_flash_fwd_stream_update_bf16_kernel(
    const hip_bfloat16* __restrict__ q,
    const hip_bfloat16* __restrict__ k,
    const hip_bfloat16* __restrict__ v,
    float* __restrict__ row_m,
    float* __restrict__ row_l,
    float* __restrict__ acc,
    int batch_size,
    int seqlen_q_tile,
    int seqlen_k,
    int seqlen_q_total,
    int num_heads,
    int num_heads_k,
    float softmax_scale,
    int is_causal,
    int q_start,
    int key_start,
    int key_len)
{
    const int lane = static_cast<int>(threadIdx.x) % KilnLogicalWarpSize;
    const int row = static_cast<int>(threadIdx.x) / KilnLogicalWarpSize;
    const int q_local = static_cast<int>(blockIdx.x) * Rows + row;
    const int head = static_cast<int>(blockIdx.y);
    const int batch = static_cast<int>(blockIdx.z);

    if(batch >= batch_size || head >= num_heads || row >= Rows || q_local >= seqlen_q_tile)
    {
        return;
    }

    constexpr int MaxLaneDims = (HeadDim + 31) / 32;
    float q_vals[MaxLaneDims];
    float acc_vals[MaxLaneDims];
    int dim_vals[MaxLaneDims];
    int lane_dims = 0;

    const int groups_per_kv_head = num_heads / num_heads_k;
    const int kv_head = head / groups_per_kv_head;
    const int q_abs = q_start + q_local;
    const size_t q_base =
        ((static_cast<size_t>(batch) * seqlen_q_tile + q_local) * num_heads + head) * HeadDim;
    const size_t acc_base = q_base;
    const size_t state_idx =
        (static_cast<size_t>(batch) * num_heads + head) * seqlen_q_tile + q_local;

    for(int dim = lane; dim < HeadDim; dim += KilnLogicalWarpSize)
    {
        dim_vals[lane_dims] = dim;
        q_vals[lane_dims] = static_cast<float>(q[q_base + dim]);
        acc_vals[lane_dims] = acc[acc_base + dim];
        ++lane_dims;
    }

    float m = row_m[state_idx];
    float l = row_l[state_idx];

    const int key_limit = is_causal ? causal_key_limit(q_abs, seqlen_q_total, seqlen_k) : seqlen_k;
    const int key_end = min(key_start + key_len, key_limit);

    for(int key = key_start; key < key_end; ++key)
    {
        const size_t kv_base =
            ((static_cast<size_t>(batch) * seqlen_k + key) * num_heads_k + kv_head) * HeadDim;

        float partial = 0.0f;
        for(int i = 0; i < lane_dims; ++i)
        {
            partial += q_vals[i] * static_cast<float>(k[kv_base + dim_vals[i]]);
        }
        const float score = kiln_wave_reduce_sum(partial) * softmax_scale;
        const float new_m = fmaxf(m, score);
        const float alpha = expf(m - new_m);
        const float beta = expf(score - new_m);
        for(int i = 0; i < lane_dims; ++i)
        {
            const float vv = static_cast<float>(v[kv_base + dim_vals[i]]);
            acc_vals[i] = acc_vals[i] * alpha + beta * vv;
        }
        l = l * alpha + beta;
        m = new_m;
    }

    for(int i = 0; i < lane_dims; ++i)
    {
        acc[acc_base + dim_vals[i]] = acc_vals[i];
    }
    if(lane == 0)
    {
        row_m[state_idx] = m;
        row_l[state_idx] = l;
    }
}

template <int HeadDim, int Rows>
__global__ void kiln_rocm_flash_fwd_stream_finalize_bf16_kernel(
    const float* __restrict__ row_m,
    const float* __restrict__ row_l,
    const float* __restrict__ acc,
    hip_bfloat16* __restrict__ out,
    float* __restrict__ lse,
    int batch_size,
    int seqlen_q_tile,
    int num_heads)
{
    const int lane = static_cast<int>(threadIdx.x) % KilnLogicalWarpSize;
    const int row = static_cast<int>(threadIdx.x) / KilnLogicalWarpSize;
    const int q_local = static_cast<int>(blockIdx.x) * Rows + row;
    const int head = static_cast<int>(blockIdx.y);
    const int batch = static_cast<int>(blockIdx.z);

    if(batch >= batch_size || head >= num_heads || row >= Rows || q_local >= seqlen_q_tile)
    {
        return;
    }

    const size_t base =
        ((static_cast<size_t>(batch) * seqlen_q_tile + q_local) * num_heads + head) * HeadDim;
    const size_t state_idx =
        (static_cast<size_t>(batch) * num_heads + head) * seqlen_q_tile + q_local;
    const float l = row_l[state_idx];
    const float inv_l = l > 0.0f ? 1.0f / l : 0.0f;

    for(int dim = lane; dim < HeadDim; dim += KilnLogicalWarpSize)
    {
        out[base + dim] = hip_bfloat16(acc[base + dim] * inv_l);
    }
    if(lane == 0 && lse != nullptr)
    {
        const float m = row_m[state_idx];
        lse[state_idx] = l > 0.0f ? m + logf(l) : -INFINITY;
    }
}

template <int HeadDim, int KPar>
__global__ void kiln_rocm_flash_bwd_dq_bf16_kernel(const hip_bfloat16* __restrict__ dout,
                                                   const hip_bfloat16* __restrict__ q,
                                                   const hip_bfloat16* __restrict__ k,
                                                   const hip_bfloat16* __restrict__ v,
                                                   const hip_bfloat16* __restrict__ out,
                                                   const float* __restrict__ lse,
                                                   hip_bfloat16* __restrict__ dq,
                                                   int batch_size,
                                                   int seqlen_q,
                                                   int seqlen_k,
                                                   int num_heads,
                                                   int num_heads_k,
                                                   float softmax_scale,
                                                   int is_causal)
{
    const int q_idx = static_cast<int>(blockIdx.x);
    const int head = static_cast<int>(blockIdx.y);
    const int batch = static_cast<int>(blockIdx.z);
    const int dim = static_cast<int>(threadIdx.x);
    const int lane = static_cast<int>(threadIdx.y);

    if(batch >= batch_size || q_idx >= seqlen_q || head >= num_heads || dim >= HeadDim ||
       lane >= KPar)
    {
        return;
    }

    const int groups_per_kv_head = num_heads / num_heads_k;
    const int kv_head = head / groups_per_kv_head;
    const size_t q_base =
        ((static_cast<size_t>(batch) * seqlen_q + q_idx) * num_heads + head) * HeadDim;
    const float qv = static_cast<float>(q[q_base + dim]);
    const float dov = static_cast<float>(dout[q_base + dim]);
    const float outv = static_cast<float>(out[q_base + dim]);
    const float d_i = row_sum_x<HeadDim, KPar>(dov * outv);
    const float row_lse = lse[(static_cast<size_t>(batch) * num_heads + head) * seqlen_q + q_idx];
    const int key_limit = is_causal ? causal_key_limit(q_idx, seqlen_q, seqlen_k) : seqlen_k;

    float acc = 0.0f;
    __shared__ float scores[KPar];
    __shared__ float dps[KPar];
    for(int key_base = 0; key_base < key_limit; key_base += KPar)
    {
        const int key = key_base + lane;
        const bool valid_key = key < key_limit;
        const size_t kv_base =
            ((static_cast<size_t>(batch) * seqlen_k + key) * num_heads_k + kv_head) * HeadDim;
        const float kval = valid_key ? static_cast<float>(k[kv_base + dim]) : 0.0f;
        const float vv = valid_key ? static_cast<float>(v[kv_base + dim]) : 0.0f;
        const float score =
            valid_key ? row_sum_x<HeadDim, KPar>(qv * kval) * softmax_scale
                      : row_sum_x<HeadDim, KPar>(0.0f);
        const float dp = valid_key ? row_sum_x<HeadDim, KPar>(dov * vv)
                                   : row_sum_x<HeadDim, KPar>(0.0f);
        if(dim == 0)
        {
            scores[lane] = score;
            dps[lane] = dp;
        }
        __syncthreads();

        if(lane == 0)
        {
            for(int key_lane = 0; key_lane < KPar; ++key_lane)
            {
                const int update_key = key_base + key_lane;
                if(update_key >= key_limit)
                {
                    break;
                }
                const size_t update_kv_base =
                    ((static_cast<size_t>(batch) * seqlen_k + update_key) * num_heads_k +
                     kv_head) *
                    HeadDim;
                const float p = expf(scores[key_lane] - row_lse);
                const float ds = p * (dps[key_lane] - d_i) * softmax_scale;
                acc += ds * static_cast<float>(k[update_kv_base + dim]);
            }
        }
        __syncthreads();
    }

    if(lane == 0)
    {
        dq[q_base + dim] = hip_bfloat16(acc);
    }
}

template <int HeadDim>
__global__ void kiln_rocm_flash_bwd_delta_bf16_kernel(const hip_bfloat16* __restrict__ dout,
                                                      const hip_bfloat16* __restrict__ out,
                                                      float* __restrict__ delta,
                                                      int batch_size,
                                                      int seqlen_q,
                                                      int num_heads)
{
    const int q_idx = static_cast<int>(blockIdx.x);
    const int head = static_cast<int>(blockIdx.y);
    const int batch = static_cast<int>(blockIdx.z);
    const int dim = static_cast<int>(threadIdx.x);

    if(batch >= batch_size || q_idx >= seqlen_q || head >= num_heads || dim >= HeadDim)
    {
        return;
    }

    const size_t q_base =
        ((static_cast<size_t>(batch) * seqlen_q + q_idx) * num_heads + head) * HeadDim;
    const float dov = static_cast<float>(dout[q_base + dim]);
    const float outv = static_cast<float>(out[q_base + dim]);
    const float d_i = row_sum_x<HeadDim, 1>(dov * outv);

    if(dim == 0)
    {
        delta[(static_cast<size_t>(batch) * num_heads + head) * seqlen_q + q_idx] = d_i;
    }
}

template <int HeadDim, int KPar>
__global__ void kiln_rocm_flash_bwd_dq_delta_bf16_kernel(const hip_bfloat16* __restrict__ q,
                                                         const hip_bfloat16* __restrict__ k,
                                                         const hip_bfloat16* __restrict__ v,
                                                         const hip_bfloat16* __restrict__ dout,
                                                         const float* __restrict__ lse,
                                                         const float* __restrict__ delta,
                                                         hip_bfloat16* __restrict__ dq,
                                                         int batch_size,
                                                         int seqlen_q,
                                                         int seqlen_k,
                                                         int num_heads,
                                                         int num_heads_k,
                                                         float softmax_scale,
                                                         int is_causal)
{
    const int q_idx = static_cast<int>(blockIdx.x);
    const int head = static_cast<int>(blockIdx.y);
    const int batch = static_cast<int>(blockIdx.z);
    const int dim = static_cast<int>(threadIdx.x);
    const int lane = static_cast<int>(threadIdx.y);

    if(batch >= batch_size || q_idx >= seqlen_q || head >= num_heads || dim >= HeadDim ||
       lane >= KPar)
    {
        return;
    }

    const int groups_per_kv_head = num_heads / num_heads_k;
    const int kv_head = head / groups_per_kv_head;
    const size_t q_base =
        ((static_cast<size_t>(batch) * seqlen_q + q_idx) * num_heads + head) * HeadDim;
    const float qv = static_cast<float>(q[q_base + dim]);
    const float dov = static_cast<float>(dout[q_base + dim]);
    const size_t row_idx =
        (static_cast<size_t>(batch) * num_heads + head) * seqlen_q + q_idx;
    const float d_i = delta[row_idx];
    const float row_lse = lse[row_idx];
    const int key_limit = is_causal ? causal_key_limit(q_idx, seqlen_q, seqlen_k) : seqlen_k;

    float acc = 0.0f;
    __shared__ float scores[KPar];
    __shared__ float dps[KPar];
    for(int key_base = 0; key_base < key_limit; key_base += KPar)
    {
        const int key = key_base + lane;
        const bool valid_key = key < key_limit;
        const size_t kv_base =
            ((static_cast<size_t>(batch) * seqlen_k + key) * num_heads_k + kv_head) * HeadDim;
        const float kval = valid_key ? static_cast<float>(k[kv_base + dim]) : 0.0f;
        const float vv = valid_key ? static_cast<float>(v[kv_base + dim]) : 0.0f;
        const float score =
            valid_key ? row_sum_x<HeadDim, KPar>(qv * kval) * softmax_scale
                      : row_sum_x<HeadDim, KPar>(0.0f);
        const float dp = valid_key ? row_sum_x<HeadDim, KPar>(dov * vv)
                                   : row_sum_x<HeadDim, KPar>(0.0f);
        if(dim == 0)
        {
            scores[lane] = score;
            dps[lane] = dp;
        }
        __syncthreads();

        if(lane == 0)
        {
            for(int key_lane = 0; key_lane < KPar; ++key_lane)
            {
                const int update_key = key_base + key_lane;
                if(update_key >= key_limit)
                {
                    break;
                }
                const size_t update_kv_base =
                    ((static_cast<size_t>(batch) * seqlen_k + update_key) * num_heads_k +
                     kv_head) *
                    HeadDim;
                const float p = expf(scores[key_lane] - row_lse);
                const float ds = p * (dps[key_lane] - d_i) * softmax_scale;
                acc += ds * static_cast<float>(k[update_kv_base + dim]);
            }
        }
        __syncthreads();
    }

    if(lane == 0)
    {
        dq[q_base + dim] = hip_bfloat16(acc);
    }
}

template <int HeadDim, int QPar>
__global__ void kiln_rocm_flash_bwd_dkdv_bf16_kernel(const hip_bfloat16* __restrict__ dout,
                                                     const hip_bfloat16* __restrict__ q,
                                                     const hip_bfloat16* __restrict__ k,
                                                     const hip_bfloat16* __restrict__ v,
                                                     const hip_bfloat16* __restrict__ out,
                                                     const float* __restrict__ lse,
                                                     hip_bfloat16* __restrict__ dk,
                                                     hip_bfloat16* __restrict__ dv,
                                                     int batch_size,
                                                     int seqlen_q,
                                                     int seqlen_k,
                                                     int num_heads,
                                                     int num_heads_k,
                                                     float softmax_scale,
                                                     int is_causal)
{
    const int key = static_cast<int>(blockIdx.x);
    const int head = static_cast<int>(blockIdx.y);
    const int batch = static_cast<int>(blockIdx.z);
    const int dim = static_cast<int>(threadIdx.x);
    const int lane = static_cast<int>(threadIdx.y);

    if(batch >= batch_size || key >= seqlen_k || head >= num_heads || dim >= HeadDim ||
       lane >= QPar)
    {
        return;
    }

    const int groups_per_kv_head = num_heads / num_heads_k;
    const int kv_head = head / groups_per_kv_head;
    const size_t kv_base =
        ((static_cast<size_t>(batch) * seqlen_k + key) * num_heads_k + kv_head) * HeadDim;
    const float kval = static_cast<float>(k[kv_base + dim]);
    const float vv = static_cast<float>(v[kv_base + dim]);

    int q_start = 0;
    if(is_causal)
    {
        q_start = key - (seqlen_k - seqlen_q);
        if(q_start < 0)
        {
            q_start = 0;
        }
    }

    float dk_acc = 0.0f;
    float dv_acc = 0.0f;
    __shared__ float scores[QPar];
    __shared__ float dps[QPar];
    __shared__ float dis[QPar];
    for(int q_base_idx = q_start; q_base_idx < seqlen_q; q_base_idx += QPar)
    {
        const int q_idx = q_base_idx + lane;
        const bool valid_q = q_idx < seqlen_q;
        const size_t q_base =
            ((static_cast<size_t>(batch) * seqlen_q + q_idx) * num_heads + head) * HeadDim;
        const float qv = valid_q ? static_cast<float>(q[q_base + dim]) : 0.0f;
        const float dov = valid_q ? static_cast<float>(dout[q_base + dim]) : 0.0f;
        const float outv = valid_q ? static_cast<float>(out[q_base + dim]) : 0.0f;
        const float d_i = valid_q ? row_sum_x<HeadDim, QPar>(dov * outv)
                                  : row_sum_x<HeadDim, QPar>(0.0f);
        const float score = valid_q ? row_sum_x<HeadDim, QPar>(qv * kval) * softmax_scale
                                    : row_sum_x<HeadDim, QPar>(0.0f);
        const float dp =
            valid_q ? row_sum_x<HeadDim, QPar>(dov * vv) : row_sum_x<HeadDim, QPar>(0.0f);
        if(dim == 0)
        {
            scores[lane] = score;
            dps[lane] = dp;
            dis[lane] = d_i;
        }
        __syncthreads();

        if(lane == 0)
        {
            for(int q_lane = 0; q_lane < QPar; ++q_lane)
            {
                const int update_q = q_base_idx + q_lane;
                if(update_q >= seqlen_q)
                {
                    break;
                }
                const size_t update_q_base =
                    ((static_cast<size_t>(batch) * seqlen_q + update_q) * num_heads + head) *
                    HeadDim;
                const float row_lse =
                    lse[(static_cast<size_t>(batch) * num_heads + head) * seqlen_q + update_q];
                const float p = expf(scores[q_lane] - row_lse);
                const float ds = p * (dps[q_lane] - dis[q_lane]) * softmax_scale;
                dk_acc += ds * static_cast<float>(q[update_q_base + dim]);
                dv_acc += p * static_cast<float>(dout[update_q_base + dim]);
            }
        }
        __syncthreads();
    }

    const size_t expanded_base =
        ((static_cast<size_t>(batch) * seqlen_k + key) * num_heads + head) * HeadDim;
    if(lane == 0)
    {
        dk[expanded_base + dim] = hip_bfloat16(dk_acc);
        dv[expanded_base + dim] = hip_bfloat16(dv_acc);
    }
}

template <int HeadDim, int QPar>
__global__ void kiln_rocm_flash_bwd_dkdv_delta_bf16_kernel(const hip_bfloat16* __restrict__ dout,
                                                           const hip_bfloat16* __restrict__ q,
                                                           const hip_bfloat16* __restrict__ k,
                                                           const hip_bfloat16* __restrict__ v,
                                                           const float* __restrict__ lse,
                                                           const float* __restrict__ delta,
                                                           hip_bfloat16* __restrict__ dk,
                                                           hip_bfloat16* __restrict__ dv,
                                                           int batch_size,
                                                           int seqlen_q,
                                                           int seqlen_k,
                                                           int num_heads,
                                                           int num_heads_k,
                                                           float softmax_scale,
                                                           int is_causal)
{
    const int key = static_cast<int>(blockIdx.x);
    const int head = static_cast<int>(blockIdx.y);
    const int batch = static_cast<int>(blockIdx.z);
    const int dim = static_cast<int>(threadIdx.x);
    const int lane = static_cast<int>(threadIdx.y);

    if(batch >= batch_size || key >= seqlen_k || head >= num_heads || dim >= HeadDim ||
       lane >= QPar)
    {
        return;
    }

    const int groups_per_kv_head = num_heads / num_heads_k;
    const int kv_head = head / groups_per_kv_head;
    const size_t kv_base =
        ((static_cast<size_t>(batch) * seqlen_k + key) * num_heads_k + kv_head) * HeadDim;
    const float kval = static_cast<float>(k[kv_base + dim]);
    const float vv = static_cast<float>(v[kv_base + dim]);

    int q_start = 0;
    if(is_causal)
    {
        q_start = key - (seqlen_k - seqlen_q);
        if(q_start < 0)
        {
            q_start = 0;
        }
    }

    float dk_acc = 0.0f;
    float dv_acc = 0.0f;
    __shared__ float scores[QPar];
    __shared__ float dps[QPar];
    __shared__ float dis[QPar];
    for(int q_base_idx = q_start; q_base_idx < seqlen_q; q_base_idx += QPar)
    {
        const int q_idx = q_base_idx + lane;
        const bool valid_q = q_idx < seqlen_q;
        const size_t q_base =
            ((static_cast<size_t>(batch) * seqlen_q + q_idx) * num_heads + head) * HeadDim;
        const float qv = valid_q ? static_cast<float>(q[q_base + dim]) : 0.0f;
        const float dov = valid_q ? static_cast<float>(dout[q_base + dim]) : 0.0f;
        const float d_i =
            valid_q ? delta[(static_cast<size_t>(batch) * num_heads + head) * seqlen_q + q_idx]
                    : 0.0f;
        const float score = valid_q ? row_sum_x<HeadDim, QPar>(qv * kval) * softmax_scale
                                    : row_sum_x<HeadDim, QPar>(0.0f);
        const float dp =
            valid_q ? row_sum_x<HeadDim, QPar>(dov * vv) : row_sum_x<HeadDim, QPar>(0.0f);
        if(dim == 0)
        {
            scores[lane] = score;
            dps[lane] = dp;
            dis[lane] = d_i;
        }
        __syncthreads();

        if(lane == 0)
        {
            for(int q_lane = 0; q_lane < QPar; ++q_lane)
            {
                const int update_q = q_base_idx + q_lane;
                if(update_q >= seqlen_q)
                {
                    break;
                }
                const size_t update_q_base =
                    ((static_cast<size_t>(batch) * seqlen_q + update_q) * num_heads + head) *
                    HeadDim;
                const float row_lse =
                    lse[(static_cast<size_t>(batch) * num_heads + head) * seqlen_q + update_q];
                const float p = expf(scores[q_lane] - row_lse);
                const float ds = p * (dps[q_lane] - dis[q_lane]) * softmax_scale;
                dk_acc += ds * static_cast<float>(q[update_q_base + dim]);
                dv_acc += p * static_cast<float>(dout[update_q_base + dim]);
            }
        }
        __syncthreads();
    }

    const size_t expanded_base =
        ((static_cast<size_t>(batch) * seqlen_k + key) * num_heads + head) * HeadDim;
    if(lane == 0)
    {
        dk[expanded_base + dim] = hip_bfloat16(dk_acc);
        dv[expanded_base + dim] = hip_bfloat16(dv_acc);
    }
}

template <int HeadDim, int QPar>
__global__ void kiln_rocm_flash_bwd_dkdv_collapsed_gqa_bf16_kernel(
    const hip_bfloat16* __restrict__ dout,
    const hip_bfloat16* __restrict__ q,
    const hip_bfloat16* __restrict__ k,
    const hip_bfloat16* __restrict__ v,
    const hip_bfloat16* __restrict__ out,
    const float* __restrict__ lse,
    hip_bfloat16* __restrict__ dk,
    hip_bfloat16* __restrict__ dv,
    int batch_size,
    int seqlen_q,
    int seqlen_k,
    int num_heads,
    int num_heads_k,
    float softmax_scale,
    int is_causal)
{
    const int key = static_cast<int>(blockIdx.x);
    const int kv_head = static_cast<int>(blockIdx.y);
    const int batch = static_cast<int>(blockIdx.z);
    const int dim = static_cast<int>(threadIdx.x);
    const int lane = static_cast<int>(threadIdx.y);

    if(batch >= batch_size || key >= seqlen_k || kv_head >= num_heads_k || dim >= HeadDim ||
       lane >= QPar)
    {
        return;
    }

    const int groups_per_kv_head = num_heads / num_heads_k;
    const size_t kv_base =
        ((static_cast<size_t>(batch) * seqlen_k + key) * num_heads_k + kv_head) * HeadDim;
    const float kval = static_cast<float>(k[kv_base + dim]);
    const float vv = static_cast<float>(v[kv_base + dim]);

    int q_start = 0;
    if(is_causal)
    {
        q_start = key - (seqlen_k - seqlen_q);
        if(q_start < 0)
        {
            q_start = 0;
        }
    }

    float dk_acc = 0.0f;
    float dv_acc = 0.0f;
    __shared__ float scores[QPar];
    __shared__ float dps[QPar];
    __shared__ float dis[QPar];

    for(int group = 0; group < groups_per_kv_head; ++group)
    {
        const int head = kv_head * groups_per_kv_head + group;
        for(int q_base_idx = q_start; q_base_idx < seqlen_q; q_base_idx += QPar)
        {
            const int q_idx = q_base_idx + lane;
            const bool valid_q = q_idx < seqlen_q;
            const size_t q_base =
                ((static_cast<size_t>(batch) * seqlen_q + q_idx) * num_heads + head) * HeadDim;
            const float qv = valid_q ? static_cast<float>(q[q_base + dim]) : 0.0f;
            const float dov = valid_q ? static_cast<float>(dout[q_base + dim]) : 0.0f;
            const float outv = valid_q ? static_cast<float>(out[q_base + dim]) : 0.0f;
            const float d_i = valid_q ? row_sum_x<HeadDim, QPar>(dov * outv)
                                      : row_sum_x<HeadDim, QPar>(0.0f);
            const float score = valid_q ? row_sum_x<HeadDim, QPar>(qv * kval) * softmax_scale
                                        : row_sum_x<HeadDim, QPar>(0.0f);
            const float dp =
                valid_q ? row_sum_x<HeadDim, QPar>(dov * vv) : row_sum_x<HeadDim, QPar>(0.0f);
            if(dim == 0)
            {
                scores[lane] = score;
                dps[lane] = dp;
                dis[lane] = d_i;
            }
            __syncthreads();

            if(lane == 0)
            {
                for(int q_lane = 0; q_lane < QPar; ++q_lane)
                {
                    const int update_q = q_base_idx + q_lane;
                    if(update_q >= seqlen_q)
                    {
                        break;
                    }
                    const size_t update_q_base =
                        ((static_cast<size_t>(batch) * seqlen_q + update_q) * num_heads + head) *
                        HeadDim;
                    const float row_lse =
                        lse[(static_cast<size_t>(batch) * num_heads + head) * seqlen_q + update_q];
                    const float p = expf(scores[q_lane] - row_lse);
                    const float ds = p * (dps[q_lane] - dis[q_lane]) * softmax_scale;
                    dk_acc += ds * static_cast<float>(q[update_q_base + dim]);
                    dv_acc += p * static_cast<float>(dout[update_q_base + dim]);
                }
            }
            __syncthreads();
        }
    }

    if(lane == 0)
    {
        dk[kv_base + dim] = hip_bfloat16(dk_acc);
        dv[kv_base + dim] = hip_bfloat16(dv_acc);
    }
}

template <int HeadDim, int QPar>
__global__ void kiln_rocm_flash_bwd_dkdv_collapsed_gqa_delta_bf16_kernel(
    const hip_bfloat16* __restrict__ dout,
    const hip_bfloat16* __restrict__ q,
    const hip_bfloat16* __restrict__ k,
    const hip_bfloat16* __restrict__ v,
    const float* __restrict__ lse,
    const float* __restrict__ delta,
    hip_bfloat16* __restrict__ dk,
    hip_bfloat16* __restrict__ dv,
    int batch_size,
    int seqlen_q,
    int seqlen_k,
    int num_heads,
    int num_heads_k,
    float softmax_scale,
    int is_causal)
{
    const int key = static_cast<int>(blockIdx.x);
    const int kv_head = static_cast<int>(blockIdx.y);
    const int batch = static_cast<int>(blockIdx.z);
    const int dim = static_cast<int>(threadIdx.x);
    const int lane = static_cast<int>(threadIdx.y);

    if(batch >= batch_size || key >= seqlen_k || kv_head >= num_heads_k || dim >= HeadDim ||
       lane >= QPar)
    {
        return;
    }

    const int groups_per_kv_head = num_heads / num_heads_k;
    const size_t kv_base =
        ((static_cast<size_t>(batch) * seqlen_k + key) * num_heads_k + kv_head) * HeadDim;
    const float kval = static_cast<float>(k[kv_base + dim]);
    const float vv = static_cast<float>(v[kv_base + dim]);

    int q_start = 0;
    if(is_causal)
    {
        q_start = key - (seqlen_k - seqlen_q);
        if(q_start < 0)
        {
            q_start = 0;
        }
    }

    float dk_acc = 0.0f;
    float dv_acc = 0.0f;
    __shared__ float scores[QPar];
    __shared__ float dps[QPar];
    __shared__ float dis[QPar];

    for(int group = 0; group < groups_per_kv_head; ++group)
    {
        const int head = kv_head * groups_per_kv_head + group;
        for(int q_base_idx = q_start; q_base_idx < seqlen_q; q_base_idx += QPar)
        {
            const int q_idx = q_base_idx + lane;
            const bool valid_q = q_idx < seqlen_q;
            const size_t q_base =
                ((static_cast<size_t>(batch) * seqlen_q + q_idx) * num_heads + head) * HeadDim;
            const float qv = valid_q ? static_cast<float>(q[q_base + dim]) : 0.0f;
            const float dov = valid_q ? static_cast<float>(dout[q_base + dim]) : 0.0f;
            const float d_i =
                valid_q ? delta[(static_cast<size_t>(batch) * num_heads + head) * seqlen_q + q_idx]
                        : 0.0f;
            const float score = valid_q ? row_sum_x<HeadDim, QPar>(qv * kval) * softmax_scale
                                        : row_sum_x<HeadDim, QPar>(0.0f);
            const float dp =
                valid_q ? row_sum_x<HeadDim, QPar>(dov * vv) : row_sum_x<HeadDim, QPar>(0.0f);
            if(dim == 0)
            {
                scores[lane] = score;
                dps[lane] = dp;
                dis[lane] = d_i;
            }
            __syncthreads();

            if(lane == 0)
            {
                for(int q_lane = 0; q_lane < QPar; ++q_lane)
                {
                    const int update_q = q_base_idx + q_lane;
                    if(update_q >= seqlen_q)
                    {
                        break;
                    }
                    const size_t update_q_base =
                        ((static_cast<size_t>(batch) * seqlen_q + update_q) * num_heads + head) *
                        HeadDim;
                    const float row_lse =
                        lse[(static_cast<size_t>(batch) * num_heads + head) * seqlen_q + update_q];
                    const float p = expf(scores[q_lane] - row_lse);
                    const float ds = p * (dps[q_lane] - dis[q_lane]) * softmax_scale;
                    dk_acc += ds * static_cast<float>(q[update_q_base + dim]);
                    dv_acc += p * static_cast<float>(dout[update_q_base + dim]);
                }
            }
            __syncthreads();
        }
    }

    if(lane == 0)
    {
        dk[kv_base + dim] = hip_bfloat16(dk_acc);
        dv[kv_base + dim] = hip_bfloat16(dv_acc);
    }
}

template <int HeadDim>
__global__ void kiln_rocm_flash_bwd_dq_delta_warp_bf16_kernel(
    const hip_bfloat16* __restrict__ q,
    const hip_bfloat16* __restrict__ k,
    const hip_bfloat16* __restrict__ v,
    const hip_bfloat16* __restrict__ dout,
    const float* __restrict__ lse,
    const float* __restrict__ delta,
    hip_bfloat16* __restrict__ dq,
    int batch_size,
    int seqlen_q,
    int seqlen_k,
    int num_heads,
    int num_heads_k,
    float softmax_scale,
    int is_causal)
{
    static_assert(HeadDim % KilnLogicalWarpSize == 0,
                  "HeadDim must be a multiple of warp size");
    constexpr int ValuesPerLane = HeadDim / KilnLogicalWarpSize;
    const int q_idx = static_cast<int>(blockIdx.x);
    const int head = static_cast<int>(blockIdx.y);
    const int batch = static_cast<int>(blockIdx.z);
    const int lane = static_cast<int>(threadIdx.x);

    if(batch >= batch_size || q_idx >= seqlen_q || head >= num_heads ||
       lane >= KilnLogicalWarpSize)
    {
        return;
    }

    const int groups_per_kv_head = num_heads / num_heads_k;
    const int kv_head = head / groups_per_kv_head;
    const size_t q_base =
        ((static_cast<size_t>(batch) * seqlen_q + q_idx) * num_heads + head) * HeadDim;
    const size_t row_idx =
        (static_cast<size_t>(batch) * num_heads + head) * seqlen_q + q_idx;
    const float d_i = delta[row_idx];
    const float row_lse = lse[row_idx];
    const int key_limit = is_causal ? causal_key_limit(q_idx, seqlen_q, seqlen_k) : seqlen_k;

    float qv[ValuesPerLane];
    float dov[ValuesPerLane];
    float acc[ValuesPerLane];
#pragma unroll
    for(int i = 0; i < ValuesPerLane; ++i)
    {
        const int dim = lane + i * KilnLogicalWarpSize;
        qv[i] = static_cast<float>(q[q_base + dim]);
        dov[i] = static_cast<float>(dout[q_base + dim]);
        acc[i] = 0.0f;
    }

    for(int key = 0; key < key_limit; ++key)
    {
        const size_t kv_base =
            ((static_cast<size_t>(batch) * seqlen_k + key) * num_heads_k + kv_head) * HeadDim;
        float score_partial = 0.0f;
        float dp_partial = 0.0f;
#pragma unroll
        for(int i = 0; i < ValuesPerLane; ++i)
        {
            const int dim = lane + i * KilnLogicalWarpSize;
            const float kval = static_cast<float>(k[kv_base + dim]);
            const float vv = static_cast<float>(v[kv_base + dim]);
            score_partial += qv[i] * kval;
            dp_partial += dov[i] * vv;
        }
        const float score = kiln_wave_reduce_sum(score_partial) * softmax_scale;
        const float dp = kiln_wave_reduce_sum(dp_partial);
        const float p = expf(score - row_lse);
        const float ds = p * (dp - d_i) * softmax_scale;
#pragma unroll
        for(int i = 0; i < ValuesPerLane; ++i)
        {
            const int dim = lane + i * KilnLogicalWarpSize;
            acc[i] += ds * static_cast<float>(k[kv_base + dim]);
        }
    }

#pragma unroll
    for(int i = 0; i < ValuesPerLane; ++i)
    {
        const int dim = lane + i * KilnLogicalWarpSize;
        dq[q_base + dim] = hip_bfloat16(acc[i]);
    }
}

template <int HeadDim>
__global__ void kiln_rocm_flash_bwd_dkdv_collapsed_gqa_delta_warp_bf16_kernel(
    const hip_bfloat16* __restrict__ dout,
    const hip_bfloat16* __restrict__ q,
    const hip_bfloat16* __restrict__ k,
    const hip_bfloat16* __restrict__ v,
    const float* __restrict__ lse,
    const float* __restrict__ delta,
    hip_bfloat16* __restrict__ dk,
    hip_bfloat16* __restrict__ dv,
    int batch_size,
    int seqlen_q,
    int seqlen_k,
    int num_heads,
    int num_heads_k,
    float softmax_scale,
    int is_causal)
{
    static_assert(HeadDim % KilnLogicalWarpSize == 0,
                  "HeadDim must be a multiple of warp size");
    constexpr int ValuesPerLane = HeadDim / KilnLogicalWarpSize;
    const int key = static_cast<int>(blockIdx.x);
    const int kv_head = static_cast<int>(blockIdx.y);
    const int batch = static_cast<int>(blockIdx.z);
    const int lane = static_cast<int>(threadIdx.x);

    if(batch >= batch_size || key >= seqlen_k || kv_head >= num_heads_k ||
       lane >= KilnLogicalWarpSize)
    {
        return;
    }

    const int groups_per_kv_head = num_heads / num_heads_k;
    const size_t kv_base =
        ((static_cast<size_t>(batch) * seqlen_k + key) * num_heads_k + kv_head) * HeadDim;
    float kval[ValuesPerLane];
    float vv[ValuesPerLane];
    float dk_acc[ValuesPerLane];
    float dv_acc[ValuesPerLane];
#pragma unroll
    for(int i = 0; i < ValuesPerLane; ++i)
    {
        const int dim = lane + i * KilnLogicalWarpSize;
        kval[i] = static_cast<float>(k[kv_base + dim]);
        vv[i] = static_cast<float>(v[kv_base + dim]);
        dk_acc[i] = 0.0f;
        dv_acc[i] = 0.0f;
    }

    int q_start = 0;
    if(is_causal)
    {
        q_start = key - (seqlen_k - seqlen_q);
        if(q_start < 0)
        {
            q_start = 0;
        }
    }

    for(int group = 0; group < groups_per_kv_head; ++group)
    {
        const int head = kv_head * groups_per_kv_head + group;
        for(int q_idx = q_start; q_idx < seqlen_q; ++q_idx)
        {
            const size_t q_base =
                ((static_cast<size_t>(batch) * seqlen_q + q_idx) * num_heads + head) * HeadDim;
            float score_partial = 0.0f;
            float dp_partial = 0.0f;
#pragma unroll
            for(int i = 0; i < ValuesPerLane; ++i)
            {
                const int dim = lane + i * KilnLogicalWarpSize;
                const float qv = static_cast<float>(q[q_base + dim]);
                const float dov = static_cast<float>(dout[q_base + dim]);
                score_partial += qv * kval[i];
                dp_partial += dov * vv[i];
            }
            const size_t row_idx =
                (static_cast<size_t>(batch) * num_heads + head) * seqlen_q + q_idx;
            const float score = kiln_wave_reduce_sum(score_partial) * softmax_scale;
            const float dp = kiln_wave_reduce_sum(dp_partial);
            const float p = expf(score - lse[row_idx]);
            const float ds = p * (dp - delta[row_idx]) * softmax_scale;
#pragma unroll
            for(int i = 0; i < ValuesPerLane; ++i)
            {
                const int dim = lane + i * KilnLogicalWarpSize;
                const float qv = static_cast<float>(q[q_base + dim]);
                const float dov = static_cast<float>(dout[q_base + dim]);
                dk_acc[i] += ds * qv;
                dv_acc[i] += p * dov;
            }
        }
    }

#pragma unroll
    for(int i = 0; i < ValuesPerLane; ++i)
    {
        const int dim = lane + i * KilnLogicalWarpSize;
        dk[kv_base + dim] = hip_bfloat16(dk_acc[i]);
        dv[kv_base + dim] = hip_bfloat16(dv_acc[i]);
    }
}

__global__ void kiln_rocm_flash_f32_to_bf16_kernel(const float* __restrict__ src,
                                                   hip_bfloat16* __restrict__ dst,
                                                   long long n)
{
    const long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    if(idx < n)
    {
        dst[idx] = hip_bfloat16(src[idx]);
    }
}

template <int HeadDim>
__global__ void kiln_rocm_flash_reduce_gqa_grads_f32_to_bf16_kernel(
    const float* __restrict__ src,
    hip_bfloat16* __restrict__ dst,
    int batch_size,
    int seqlen_k,
    int num_heads,
    int num_heads_k)
{
    const long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    const long long n = static_cast<long long>(batch_size) * seqlen_k * num_heads_k * HeadDim;
    if(idx >= n)
    {
        return;
    }

    const int dim = static_cast<int>(idx % HeadDim);
    long long rem = idx / HeadDim;
    const int kv_head = static_cast<int>(rem % num_heads_k);
    rem /= num_heads_k;
    const int key = static_cast<int>(rem % seqlen_k);
    const int batch = static_cast<int>(rem / seqlen_k);
    const int groups_per_kv_head = num_heads / num_heads_k;

    float acc = 0.0f;
    for(int group = 0; group < groups_per_kv_head; ++group)
    {
        const int head = kv_head * groups_per_kv_head + group;
        const size_t src_idx =
            ((static_cast<size_t>(batch) * seqlen_k + key) * num_heads + head) * HeadDim + dim;
        acc += src[src_idx];
    }
    dst[idx] = hip_bfloat16(acc);
}

template <int HeadDim, int Rows, int XDim>
int launch_fwd(const void* q,
               const void* k,
               const void* v,
               void* out,
               void* softmax_lse_out,
               int batch_size,
               int seqlen_q,
               int seqlen_k,
               int num_heads,
               int num_heads_k,
               float softmax_scale,
               int is_causal,
               void* stream)
{
    dim3 grid(static_cast<unsigned>((seqlen_q + Rows - 1) / Rows),
              static_cast<unsigned>(num_heads),
              static_cast<unsigned>(batch_size));
    dim3 block(static_cast<unsigned>(Rows * KilnLogicalWarpSize));
    hipLaunchKernelGGL((kiln_rocm_flash_fwd_warprows_bf16_kernel<HeadDim, Rows>),
                       grid,
                       block,
                       0,
                       static_cast<hipStream_t>(stream),
                       static_cast<const hip_bfloat16*>(q),
                       static_cast<const hip_bfloat16*>(k),
                       static_cast<const hip_bfloat16*>(v),
                       static_cast<hip_bfloat16*>(out),
                       static_cast<float*>(softmax_lse_out),
                       batch_size,
                       seqlen_q,
                       seqlen_k,
                       num_heads,
                       num_heads_k,
                       softmax_scale,
                       is_causal != 0 ? 1 : 0);

    const hipError_t err = hipGetLastError();
    return err == hipSuccess ? 0 : static_cast<int>(err);
}

template <int HeadDim, int NumKeyWarps>
int launch_fwd_keysplit(const void* q,
                        const void* k,
                        const void* v,
                        void* out,
                        void* softmax_lse_out,
                        int batch_size,
                        int seqlen_q,
                        int seqlen_k,
                        int num_heads,
                        int num_heads_k,
                        float softmax_scale,
                        int is_causal,
                        void* stream)
{
    dim3 grid(static_cast<unsigned>(seqlen_q),
              static_cast<unsigned>(num_heads),
              static_cast<unsigned>(batch_size));
    dim3 block(static_cast<unsigned>(NumKeyWarps * KilnLogicalWarpSize));
    hipLaunchKernelGGL((kiln_rocm_flash_fwd_keysplit_bf16_kernel<HeadDim, NumKeyWarps>),
                       grid,
                       block,
                       0,
                       static_cast<hipStream_t>(stream),
                       static_cast<const hip_bfloat16*>(q),
                       static_cast<const hip_bfloat16*>(k),
                       static_cast<const hip_bfloat16*>(v),
                       static_cast<hip_bfloat16*>(out),
                       static_cast<float*>(softmax_lse_out),
                       batch_size,
                       seqlen_q,
                       seqlen_k,
                       num_heads,
                       num_heads_k,
                       softmax_scale,
                       is_causal != 0 ? 1 : 0);

    const hipError_t err = hipGetLastError();
    return err == hipSuccess ? 0 : static_cast<int>(err);
}

template <int HeadDim, int Rows, int KeyBlock>
int launch_fwd_qblock_cached(const void* q,
                             const void* k,
                             const void* v,
                             void* out,
                             void* softmax_lse_out,
                             int batch_size,
                             int seqlen_q,
                             int seqlen_k,
                             int num_heads,
                             int num_heads_k,
                             float softmax_scale,
                             int is_causal,
                             void* stream)
{
    dim3 grid(static_cast<unsigned>((seqlen_q + Rows - 1) / Rows),
              static_cast<unsigned>(num_heads),
              static_cast<unsigned>(batch_size));
    dim3 block(static_cast<unsigned>(Rows * KilnLogicalWarpSize));
    hipLaunchKernelGGL((kiln_rocm_flash_fwd_qblock_cached_bf16_kernel<HeadDim, Rows, KeyBlock>),
                       grid,
                       block,
                       0,
                       static_cast<hipStream_t>(stream),
                       static_cast<const hip_bfloat16*>(q),
                       static_cast<const hip_bfloat16*>(k),
                       static_cast<const hip_bfloat16*>(v),
                       static_cast<hip_bfloat16*>(out),
                       static_cast<float*>(softmax_lse_out),
                       batch_size,
                       seqlen_q,
                       seqlen_k,
                       num_heads,
                       num_heads_k,
                       softmax_scale,
                       is_causal != 0 ? 1 : 0);

    const hipError_t err = hipGetLastError();
    return err == hipSuccess ? 0 : static_cast<int>(err);
}

template <int HeadDim, int Rows, int QueryHeadsPerBlock, int KeyBlock>
int launch_fwd_gqa_qblock_cached(const void* q,
                                 const void* k,
                                 const void* v,
                                 void* out,
                                 void* softmax_lse_out,
                                 int batch_size,
                                 int seqlen_q,
                                 int seqlen_k,
                                 int num_heads,
                                 int num_heads_k,
                                 float softmax_scale,
                                 int is_causal,
                                 void* stream)
{
    dim3 grid(static_cast<unsigned>((seqlen_q + Rows - 1) / Rows),
              static_cast<unsigned>((num_heads + QueryHeadsPerBlock - 1) / QueryHeadsPerBlock),
              static_cast<unsigned>(batch_size));
    dim3 block(static_cast<unsigned>(Rows * QueryHeadsPerBlock * KilnLogicalWarpSize));
    hipLaunchKernelGGL((kiln_rocm_flash_fwd_gqa_qblock_cached_bf16_kernel<HeadDim,
                                                                           Rows,
                                                                           QueryHeadsPerBlock,
                                                                           KeyBlock>),
                       grid,
                       block,
                       0,
                       static_cast<hipStream_t>(stream),
                       static_cast<const hip_bfloat16*>(q),
                       static_cast<const hip_bfloat16*>(k),
                       static_cast<const hip_bfloat16*>(v),
                       static_cast<hip_bfloat16*>(out),
                       static_cast<float*>(softmax_lse_out),
                       batch_size,
                       seqlen_q,
                       seqlen_k,
                       num_heads,
                       num_heads_k,
                       softmax_scale,
                       is_causal != 0 ? 1 : 0);

    const hipError_t err = hipGetLastError();
    return err == hipSuccess ? 0 : static_cast<int>(err);
}

int launch_fwd_wmma_qblock256(const void* q,
                              const void* k,
                              const void* v,
                              void* out,
                              void* softmax_lse_out,
                              int batch_size,
                              int seqlen_q,
                              int seqlen_k,
                              int num_heads,
                              int num_heads_k,
                              float softmax_scale,
                              int is_causal,
                              void* stream)
{
    dim3 grid(static_cast<unsigned>((seqlen_q + 15) / 16),
              static_cast<unsigned>(num_heads),
              static_cast<unsigned>(batch_size));
    dim3 block(512);
    hipLaunchKernelGGL(kiln_rocm_flash_fwd_wmma_qblock256_bf16_kernel,
                       grid,
                       block,
                       0,
                       static_cast<hipStream_t>(stream),
                       static_cast<const hip_bfloat16*>(q),
                       static_cast<const hip_bfloat16*>(k),
                       static_cast<const hip_bfloat16*>(v),
                       static_cast<hip_bfloat16*>(out),
                       static_cast<float*>(softmax_lse_out),
                       batch_size,
                       seqlen_q,
                       seqlen_k,
                       num_heads,
                       num_heads_k,
                       softmax_scale,
                       is_causal != 0 ? 1 : 0);

    const hipError_t err = hipGetLastError();
    return err == hipSuccess ? 0 : static_cast<int>(err);
}

int launch_fwd_wmma_gqa_r32h1k32(const void* q,
                                  const void* k,
                                  const void* v,
                                  void* out,
                                  void* softmax_lse_out,
                                  int batch_size,
                                  int seqlen_q,
                                  int seqlen_k,
                                  int num_heads,
                                  int num_heads_k,
                                  float softmax_scale,
                                  int is_causal,
                                  void* stream)
{
    dim3 grid(static_cast<unsigned>((seqlen_q + 31) / 32),
              static_cast<unsigned>(num_heads),
              static_cast<unsigned>(batch_size));
    dim3 block(1024);
    hipLaunchKernelGGL(kiln_rocm_flash_fwd_wmma_gqa_r32h1k32_bf16_kernel,
                       grid,
                       block,
                       0,
                       static_cast<hipStream_t>(stream),
                       static_cast<const hip_bfloat16*>(q),
                       static_cast<const hip_bfloat16*>(k),
                       static_cast<const hip_bfloat16*>(v),
                       static_cast<hip_bfloat16*>(out),
                       static_cast<float*>(softmax_lse_out),
                       batch_size,
                       seqlen_q,
                       seqlen_k,
                       num_heads,
                       num_heads_k,
                       softmax_scale,
                       is_causal != 0 ? 1 : 0);

    const hipError_t err = hipGetLastError();
    return err == hipSuccess ? 0 : static_cast<int>(err);
}

int launch_fwd_wmma_gqa_r64h1k32(const void* q,
                                  const void* k,
                                  const void* v,
                                  void* out,
                                  void* softmax_lse_out,
                                  int batch_size,
                                  int seqlen_q,
                                  int seqlen_k,
                                  int num_heads,
                                  int num_heads_k,
                                  float softmax_scale,
                                  int is_causal,
                                  void* stream)
{
    dim3 grid(static_cast<unsigned>((seqlen_q + 63) / 64),
              static_cast<unsigned>(num_heads),
              static_cast<unsigned>(batch_size));
    dim3 block(512);
    hipLaunchKernelGGL(kiln_rocm_flash_fwd_wmma_gqa_r64h1k32_bf16_kernel,
                       grid,
                       block,
                       0,
                       static_cast<hipStream_t>(stream),
                       static_cast<const hip_bfloat16*>(q),
                       static_cast<const hip_bfloat16*>(k),
                       static_cast<const hip_bfloat16*>(v),
                       static_cast<hip_bfloat16*>(out),
                       static_cast<float*>(softmax_lse_out),
                       batch_size,
                       seqlen_q,
                       seqlen_k,
                       num_heads,
                       num_heads_k,
                       softmax_scale,
                       is_causal != 0 ? 1 : 0);

    const hipError_t err = hipGetLastError();
    return err == hipSuccess ? 0 : static_cast<int>(err);
}

template <int HeadDim, int Rows>
int launch_fwd_stream_update(const void* q,
                             const void* k,
                             const void* v,
                             void* row_m,
                             void* row_l,
                             void* acc,
                             int batch_size,
                             int seqlen_q_tile,
                             int seqlen_k,
                             int seqlen_q_total,
                             int num_heads,
                             int num_heads_k,
                             float softmax_scale,
                             int is_causal,
                             int q_start,
                             int key_start,
                             int key_len,
                             void* stream)
{
    dim3 grid(static_cast<unsigned>((seqlen_q_tile + Rows - 1) / Rows),
              static_cast<unsigned>(num_heads),
              static_cast<unsigned>(batch_size));
    dim3 block(static_cast<unsigned>(Rows * KilnLogicalWarpSize));
    hipLaunchKernelGGL((kiln_rocm_flash_fwd_stream_update_bf16_kernel<HeadDim, Rows>),
                       grid,
                       block,
                       0,
                       static_cast<hipStream_t>(stream),
                       static_cast<const hip_bfloat16*>(q),
                       static_cast<const hip_bfloat16*>(k),
                       static_cast<const hip_bfloat16*>(v),
                       static_cast<float*>(row_m),
                       static_cast<float*>(row_l),
                       static_cast<float*>(acc),
                       batch_size,
                       seqlen_q_tile,
                       seqlen_k,
                       seqlen_q_total,
                       num_heads,
                       num_heads_k,
                       softmax_scale,
                       is_causal != 0 ? 1 : 0,
                       q_start,
                       key_start,
                       key_len);

    const hipError_t err = hipGetLastError();
    return err == hipSuccess ? 0 : static_cast<int>(err);
}

template <int HeadDim, int Rows>
int launch_fwd_stream_finalize(const void* row_m,
                               const void* row_l,
                               const void* acc,
                               void* out,
                               void* softmax_lse_out,
                               int batch_size,
                               int seqlen_q_tile,
                               int num_heads,
                               void* stream)
{
    dim3 grid(static_cast<unsigned>((seqlen_q_tile + Rows - 1) / Rows),
              static_cast<unsigned>(num_heads),
              static_cast<unsigned>(batch_size));
    dim3 block(static_cast<unsigned>(Rows * KilnLogicalWarpSize));
    hipLaunchKernelGGL((kiln_rocm_flash_fwd_stream_finalize_bf16_kernel<HeadDim, Rows>),
                       grid,
                       block,
                       0,
                       static_cast<hipStream_t>(stream),
                       static_cast<const float*>(row_m),
                       static_cast<const float*>(row_l),
                       static_cast<const float*>(acc),
                       static_cast<hip_bfloat16*>(out),
                       static_cast<float*>(softmax_lse_out),
                       batch_size,
                       seqlen_q_tile,
                       num_heads);

    const hipError_t err = hipGetLastError();
    return err == hipSuccess ? 0 : static_cast<int>(err);
}

template <int HeadDim, int KPar>
int launch_bwd(const void* dout,
               const void* q,
               const void* k,
               const void* v,
               const void* out,
               const void* softmax_lse,
               void* dq,
               void* dk,
               void* dv,
               int batch_size,
               int seqlen_q,
               int seqlen_k,
               int num_heads,
               int num_heads_k,
               float softmax_scale,
               int is_causal,
               void* stream)
{
    dim3 block(HeadDim, KPar);
    dim3 dq_grid(static_cast<unsigned>(seqlen_q),
                 static_cast<unsigned>(num_heads),
                 static_cast<unsigned>(batch_size));
    const bool use_delta_bwd =
        env_truthy("KILN_ROCM_FLASH_BWD_PRECOMPUTE_DELTA") ||
        std::max(seqlen_q, seqlen_k) <=
            env_i32_or("KILN_ROCM_FLASH_BWD_PRECOMPUTE_DELTA_MAX_SEQ", 1024);
    if(use_delta_bwd)
    {
        float* delta = nullptr;
        const size_t delta_count =
            static_cast<size_t>(batch_size) * num_heads * seqlen_q;
        const hipError_t alloc_err =
            hipMallocAsync(reinterpret_cast<void**>(&delta),
                           delta_count * sizeof(float),
                           static_cast<hipStream_t>(stream));
        if(alloc_err == hipSuccess)
        {
            dim3 delta_grid(static_cast<unsigned>(seqlen_q),
                            static_cast<unsigned>(num_heads),
                            static_cast<unsigned>(batch_size));
            dim3 delta_block(static_cast<unsigned>(HeadDim), 1);
            hipLaunchKernelGGL((kiln_rocm_flash_bwd_delta_bf16_kernel<HeadDim>),
                               delta_grid,
                               delta_block,
                               0,
                               static_cast<hipStream_t>(stream),
                               static_cast<const hip_bfloat16*>(dout),
                               static_cast<const hip_bfloat16*>(out),
                               delta,
                               batch_size,
                               seqlen_q,
                               num_heads);
            hipError_t err = hipGetLastError();
            if(err != hipSuccess)
            {
                hipFreeAsync(delta, static_cast<hipStream_t>(stream));
                return static_cast<int>(err);
            }

            hipLaunchKernelGGL((kiln_rocm_flash_bwd_dq_delta_bf16_kernel<HeadDim, KPar>),
                               dq_grid,
                               block,
                               0,
                               static_cast<hipStream_t>(stream),
                               static_cast<const hip_bfloat16*>(q),
                               static_cast<const hip_bfloat16*>(k),
                               static_cast<const hip_bfloat16*>(v),
                               static_cast<const hip_bfloat16*>(dout),
                               static_cast<const float*>(softmax_lse),
                               delta,
                               static_cast<hip_bfloat16*>(dq),
                               batch_size,
                               seqlen_q,
                               seqlen_k,
                               num_heads,
                               num_heads_k,
                               softmax_scale,
                               is_causal != 0 ? 1 : 0);
            err = hipGetLastError();
            if(err != hipSuccess)
            {
                hipFreeAsync(delta, static_cast<hipStream_t>(stream));
                return static_cast<int>(err);
            }

            dim3 dkdv_grid(static_cast<unsigned>(seqlen_k),
                           static_cast<unsigned>(num_heads),
                           static_cast<unsigned>(batch_size));
            hipLaunchKernelGGL((kiln_rocm_flash_bwd_dkdv_delta_bf16_kernel<HeadDim, KPar>),
                               dkdv_grid,
                               block,
                               0,
                               static_cast<hipStream_t>(stream),
                               static_cast<const hip_bfloat16*>(dout),
                               static_cast<const hip_bfloat16*>(q),
                               static_cast<const hip_bfloat16*>(k),
                               static_cast<const hip_bfloat16*>(v),
                               static_cast<const float*>(softmax_lse),
                               delta,
                               static_cast<hip_bfloat16*>(dk),
                               static_cast<hip_bfloat16*>(dv),
                               batch_size,
                               seqlen_q,
                               seqlen_k,
                               num_heads,
                               num_heads_k,
                               softmax_scale,
                               is_causal != 0 ? 1 : 0);
            err = hipGetLastError();
            const hipError_t free_err = hipFreeAsync(delta, static_cast<hipStream_t>(stream));
            if(err != hipSuccess)
            {
                return static_cast<int>(err);
            }
            return free_err == hipSuccess ? 0 : static_cast<int>(free_err);
        }
    }

    hipLaunchKernelGGL((kiln_rocm_flash_bwd_dq_bf16_kernel<HeadDim, KPar>),
                       dq_grid,
                       block,
                       0,
                       static_cast<hipStream_t>(stream),
                       static_cast<const hip_bfloat16*>(dout),
                       static_cast<const hip_bfloat16*>(q),
                       static_cast<const hip_bfloat16*>(k),
                       static_cast<const hip_bfloat16*>(v),
                       static_cast<const hip_bfloat16*>(out),
                       static_cast<const float*>(softmax_lse),
                       static_cast<hip_bfloat16*>(dq),
                       batch_size,
                       seqlen_q,
                       seqlen_k,
                       num_heads,
                       num_heads_k,
                       softmax_scale,
                       is_causal != 0 ? 1 : 0);
    hipError_t err = hipGetLastError();
    if(err != hipSuccess)
    {
        return static_cast<int>(err);
    }

    dim3 dkdv_grid(static_cast<unsigned>(seqlen_k),
                   static_cast<unsigned>(num_heads),
                   static_cast<unsigned>(batch_size));
    hipLaunchKernelGGL((kiln_rocm_flash_bwd_dkdv_bf16_kernel<HeadDim, KPar>),
                       dkdv_grid,
                       block,
                       0,
                       static_cast<hipStream_t>(stream),
                       static_cast<const hip_bfloat16*>(dout),
                       static_cast<const hip_bfloat16*>(q),
                       static_cast<const hip_bfloat16*>(k),
                       static_cast<const hip_bfloat16*>(v),
                       static_cast<const hip_bfloat16*>(out),
                       static_cast<const float*>(softmax_lse),
                       static_cast<hip_bfloat16*>(dk),
                       static_cast<hip_bfloat16*>(dv),
                       batch_size,
                       seqlen_q,
                       seqlen_k,
                       num_heads,
                       num_heads_k,
                       softmax_scale,
                       is_causal != 0 ? 1 : 0);
    err = hipGetLastError();
    return err == hipSuccess ? 0 : static_cast<int>(err);
}

template <int HeadDim, int KPar>
int launch_bwd_collapsed_gqa(const void* dout,
                             const void* q,
                             const void* k,
                             const void* v,
                             const void* out,
                             const void* softmax_lse,
                             void* dq,
                             void* dk,
                             void* dv,
                             int batch_size,
                             int seqlen_q,
                             int seqlen_k,
                             int num_heads,
                             int num_heads_k,
                             float softmax_scale,
                             int is_causal,
                             void* stream)
{
    dim3 block(HeadDim, KPar);
    dim3 dq_grid(static_cast<unsigned>(seqlen_q),
                 static_cast<unsigned>(num_heads),
                 static_cast<unsigned>(batch_size));
    {
        float* delta = nullptr;
        const size_t delta_count =
            static_cast<size_t>(batch_size) * num_heads * seqlen_q;
        const hipError_t alloc_err =
            hipMallocAsync(reinterpret_cast<void**>(&delta),
                           delta_count * sizeof(float),
                           static_cast<hipStream_t>(stream));
        if(alloc_err == hipSuccess)
        {
            dim3 delta_grid(static_cast<unsigned>(seqlen_q),
                            static_cast<unsigned>(num_heads),
                            static_cast<unsigned>(batch_size));
            dim3 delta_block(static_cast<unsigned>(HeadDim), 1);
            hipLaunchKernelGGL((kiln_rocm_flash_bwd_delta_bf16_kernel<HeadDim>),
                               delta_grid,
                               delta_block,
                               0,
                               static_cast<hipStream_t>(stream),
                               static_cast<const hip_bfloat16*>(dout),
                               static_cast<const hip_bfloat16*>(out),
                               delta,
                               batch_size,
                               seqlen_q,
                               num_heads);
            hipError_t err = hipGetLastError();
            if(err != hipSuccess)
            {
                hipFreeAsync(delta, static_cast<hipStream_t>(stream));
                return static_cast<int>(err);
            }

            hipLaunchKernelGGL((kiln_rocm_flash_bwd_dq_delta_bf16_kernel<HeadDim, KPar>),
                               dq_grid,
                               block,
                               0,
                               static_cast<hipStream_t>(stream),
                               static_cast<const hip_bfloat16*>(q),
                               static_cast<const hip_bfloat16*>(k),
                               static_cast<const hip_bfloat16*>(v),
                               static_cast<const hip_bfloat16*>(dout),
                               static_cast<const float*>(softmax_lse),
                               delta,
                               static_cast<hip_bfloat16*>(dq),
                               batch_size,
                               seqlen_q,
                               seqlen_k,
                               num_heads,
                               num_heads_k,
                               softmax_scale,
                               is_causal != 0 ? 1 : 0);
            err = hipGetLastError();
            if(err != hipSuccess)
            {
                hipFreeAsync(delta, static_cast<hipStream_t>(stream));
                return static_cast<int>(err);
            }

            dim3 dkdv_grid(static_cast<unsigned>(seqlen_k),
                           static_cast<unsigned>(num_heads_k),
                           static_cast<unsigned>(batch_size));
            hipLaunchKernelGGL(
                (kiln_rocm_flash_bwd_dkdv_collapsed_gqa_delta_bf16_kernel<HeadDim, KPar>),
                dkdv_grid,
                block,
                0,
                static_cast<hipStream_t>(stream),
                static_cast<const hip_bfloat16*>(dout),
                static_cast<const hip_bfloat16*>(q),
                static_cast<const hip_bfloat16*>(k),
                static_cast<const hip_bfloat16*>(v),
                static_cast<const float*>(softmax_lse),
                delta,
                static_cast<hip_bfloat16*>(dk),
                static_cast<hip_bfloat16*>(dv),
                batch_size,
                seqlen_q,
                seqlen_k,
                num_heads,
                num_heads_k,
                softmax_scale,
                is_causal != 0 ? 1 : 0);
            err = hipGetLastError();
            const hipError_t free_err = hipFreeAsync(delta, static_cast<hipStream_t>(stream));
            if(err != hipSuccess)
            {
                return static_cast<int>(err);
            }
            return free_err == hipSuccess ? 0 : static_cast<int>(free_err);
        }
    }

    hipLaunchKernelGGL((kiln_rocm_flash_bwd_dq_bf16_kernel<HeadDim, KPar>),
                       dq_grid,
                       block,
                       0,
                       static_cast<hipStream_t>(stream),
                       static_cast<const hip_bfloat16*>(dout),
                       static_cast<const hip_bfloat16*>(q),
                       static_cast<const hip_bfloat16*>(k),
                       static_cast<const hip_bfloat16*>(v),
                       static_cast<const hip_bfloat16*>(out),
                       static_cast<const float*>(softmax_lse),
                       static_cast<hip_bfloat16*>(dq),
                       batch_size,
                       seqlen_q,
                       seqlen_k,
                       num_heads,
                       num_heads_k,
                       softmax_scale,
                       is_causal != 0 ? 1 : 0);
    hipError_t err = hipGetLastError();
    if(err != hipSuccess)
    {
        return static_cast<int>(err);
    }

    dim3 dkdv_grid(static_cast<unsigned>(seqlen_k),
                   static_cast<unsigned>(num_heads_k),
                   static_cast<unsigned>(batch_size));
    hipLaunchKernelGGL((kiln_rocm_flash_bwd_dkdv_collapsed_gqa_bf16_kernel<HeadDim, KPar>),
                       dkdv_grid,
                       block,
                       0,
                       static_cast<hipStream_t>(stream),
                       static_cast<const hip_bfloat16*>(dout),
                       static_cast<const hip_bfloat16*>(q),
                       static_cast<const hip_bfloat16*>(k),
                       static_cast<const hip_bfloat16*>(v),
                       static_cast<const hip_bfloat16*>(out),
                       static_cast<const float*>(softmax_lse),
                       static_cast<hip_bfloat16*>(dk),
                       static_cast<hip_bfloat16*>(dv),
                       batch_size,
                       seqlen_q,
                       seqlen_k,
                       num_heads,
                       num_heads_k,
                       softmax_scale,
                       is_causal != 0 ? 1 : 0);
    err = hipGetLastError();
    return err == hipSuccess ? 0 : static_cast<int>(err);
}

template <int HeadDim>
int launch_bwd_collapsed_gqa_warp(const void* dout,
                                  const void* q,
                                  const void* k,
                                  const void* v,
                                  const void* out,
                                  const void* softmax_lse,
                                  void* dq,
                                  void* dk,
                                  void* dv,
                                  int batch_size,
                                  int seqlen_q,
                                  int seqlen_k,
                                  int num_heads,
                                  int num_heads_k,
                                  float softmax_scale,
                                  int is_causal,
                                  void* stream)
{
    float* delta = nullptr;
    const size_t delta_count = static_cast<size_t>(batch_size) * num_heads * seqlen_q;
    const hipError_t alloc_err = hipMallocAsync(reinterpret_cast<void**>(&delta),
                                               delta_count * sizeof(float),
                                               static_cast<hipStream_t>(stream));
    if(alloc_err != hipSuccess)
    {
        return launch_bwd_collapsed_gqa<HeadDim, 1>(dout,
                                                    q,
                                                    k,
                                                    v,
                                                    out,
                                                    softmax_lse,
                                                    dq,
                                                    dk,
                                                    dv,
                                                    batch_size,
                                                    seqlen_q,
                                                    seqlen_k,
                                                    num_heads,
                                                    num_heads_k,
                                                    softmax_scale,
                                                    is_causal,
                                                    stream);
    }

    dim3 delta_grid(static_cast<unsigned>(seqlen_q),
                    static_cast<unsigned>(num_heads),
                    static_cast<unsigned>(batch_size));
    dim3 delta_block(static_cast<unsigned>(HeadDim), 1);
    hipLaunchKernelGGL((kiln_rocm_flash_bwd_delta_bf16_kernel<HeadDim>),
                       delta_grid,
                       delta_block,
                       0,
                       static_cast<hipStream_t>(stream),
                       static_cast<const hip_bfloat16*>(dout),
                       static_cast<const hip_bfloat16*>(out),
                       delta,
                       batch_size,
                       seqlen_q,
                       num_heads);
    hipError_t err = hipGetLastError();
    if(err != hipSuccess)
    {
        hipFreeAsync(delta, static_cast<hipStream_t>(stream));
        return static_cast<int>(err);
    }

    dim3 dq_grid(static_cast<unsigned>(seqlen_q),
                 static_cast<unsigned>(num_heads),
                 static_cast<unsigned>(batch_size));
    dim3 warp_block(static_cast<unsigned>(KilnLogicalWarpSize));
    hipLaunchKernelGGL((kiln_rocm_flash_bwd_dq_delta_warp_bf16_kernel<HeadDim>),
                       dq_grid,
                       warp_block,
                       0,
                       static_cast<hipStream_t>(stream),
                       static_cast<const hip_bfloat16*>(q),
                       static_cast<const hip_bfloat16*>(k),
                       static_cast<const hip_bfloat16*>(v),
                       static_cast<const hip_bfloat16*>(dout),
                       static_cast<const float*>(softmax_lse),
                       delta,
                       static_cast<hip_bfloat16*>(dq),
                       batch_size,
                       seqlen_q,
                       seqlen_k,
                       num_heads,
                       num_heads_k,
                       softmax_scale,
                       is_causal != 0 ? 1 : 0);
    err = hipGetLastError();
    if(err != hipSuccess)
    {
        hipFreeAsync(delta, static_cast<hipStream_t>(stream));
        return static_cast<int>(err);
    }

    dim3 dkdv_grid(static_cast<unsigned>(seqlen_k),
                   static_cast<unsigned>(num_heads_k),
                   static_cast<unsigned>(batch_size));
    hipLaunchKernelGGL((kiln_rocm_flash_bwd_dkdv_collapsed_gqa_delta_warp_bf16_kernel<HeadDim>),
                       dkdv_grid,
                       warp_block,
                       0,
                       static_cast<hipStream_t>(stream),
                       static_cast<const hip_bfloat16*>(dout),
                       static_cast<const hip_bfloat16*>(q),
                       static_cast<const hip_bfloat16*>(k),
                       static_cast<const hip_bfloat16*>(v),
                       static_cast<const float*>(softmax_lse),
                       delta,
                       static_cast<hip_bfloat16*>(dk),
                       static_cast<hip_bfloat16*>(dv),
                       batch_size,
                       seqlen_q,
                       seqlen_k,
                       num_heads,
                       num_heads_k,
                       softmax_scale,
                       is_causal != 0 ? 1 : 0);
    err = hipGetLastError();
    const hipError_t free_err = hipFreeAsync(delta, static_cast<hipStream_t>(stream));
    if(err != hipSuccess)
    {
        return static_cast<int>(err);
    }
    return free_err == hipSuccess ? 0 : static_cast<int>(free_err);
}

#if KILN_HAS_CK_TILE_FMHA
template <int HeadDim>
int launch_bwd_collapsed_gqa_ck_trload(const void* dout,
                                       const void* q,
                                       const void* k,
                                       const void* v,
                                       const void* out,
                                       const void* softmax_lse,
                                       void* dq,
                                       void* dk,
                                       void* dv,
                                       int batch_size,
                                       int seqlen_q,
                                       int seqlen_k,
                                       int num_heads,
                                       int num_heads_k,
                                       float softmax_scale,
                                       int is_causal,
                                       void* stream)
{
    if(!is_causal || HeadDim != 256)
    {
        return -4;
    }

    using dtype = FmhaBwdTypeConfig<FmhaBwdBf16>;
    using block_tile = ck_tile::sequence<32, 32, 16, 32, 16, 32, 16, HeadDim, HeadDim>;
    using warps = ck_tile::sequence<2, 1, 1>;
    using warp_tile = ck_tile::sequence<16, 16, 16>;
    using shape = ck_tile::TileFmhaBwdShape<block_tile,
                                            warps,
                                            warp_tile,
                                            warps,
                                            warp_tile,
                                            warps,
                                            warp_tile,
                                            warps,
                                            warp_tile,
                                            warps,
                                            warp_tile,
                                            0>;
    using traits =
        ck_tile::TileFmhaBwdTraits<0, 0, ck_tile::BlockAttentionBiasEnum::NO_BIAS, false, -1>;
    using mask = ck_tile::GenericAttentionMask<true, false>;
    using problem =
        ck_tile::BlockFmhaBwdPipelineProblem<typename dtype::QDataType,
                                             typename dtype::KDataType,
                                             typename dtype::VDataType,
                                             typename dtype::GemmDataType,
                                             typename dtype::LSEDataType,
                                             typename dtype::AccDataType,
                                             typename dtype::DDataType,
                                             typename dtype::BiasDataType,
                                             typename dtype::RandValOutputDataType,
                                             typename dtype::ODataType,
                                             typename dtype::OGradDataType,
                                             typename dtype::QGradDataType,
                                             typename dtype::AccDataType,
                                             typename dtype::AccDataType,
                                             typename dtype::BiasGradDataType,
                                             shape,
                                             false,
                                             false,
                                             mask,
                                             ck_tile::BlockDropoutBwd<false, false, false>,
                                             true,
                                             traits>;
    using pipeline = ck_tile::BlockFmhaBwdDQDKDVPipeline<problem>;
    using kgrad_epilogue = ck_tile::Default2DEpilogue<
        ck_tile::Default2DEpilogueProblem<typename dtype::AccDataType,
                                          typename dtype::AccDataType,
                                          false,
                                          false,
                                          true>>;
    using vgrad_epilogue = ck_tile::Default2DEpilogue<
        ck_tile::Default2DEpilogueProblem<typename dtype::AccDataType,
                                          typename dtype::AccDataType,
                                          false,
                                          false,
                                          true>>;
    using qgrad_epilogue = ck_tile::Default2DEpilogue<
        ck_tile::Default2DEpilogueProblem<typename dtype::AccDataType,
                                          typename dtype::QGradDataType,
                                          false,
                                          false,
                                          true>>;
    using kernel =
        ck_tile::FmhaBwdDQDKDVKernel<pipeline, kgrad_epilogue, vgrad_epilogue, qgrad_epilogue>;

    const hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    float* delta = nullptr;
    float* dq_acc = nullptr;
    float* dk_full = nullptr;
    float* dv_full = nullptr;
    const size_t delta_count = static_cast<size_t>(batch_size) * num_heads * seqlen_q;
    const size_t dq_count =
        static_cast<size_t>(batch_size) * seqlen_q * num_heads * HeadDim;
    const size_t dkdv_full_count =
        static_cast<size_t>(batch_size) * seqlen_k * num_heads * HeadDim;
    hipError_t err =
        hipMallocAsync(reinterpret_cast<void**>(&delta), delta_count * sizeof(float), hip_stream);
    if(err != hipSuccess)
    {
        return static_cast<int>(err);
    }
    err = hipMallocAsync(reinterpret_cast<void**>(&dq_acc), dq_count * sizeof(float), hip_stream);
    if(err != hipSuccess)
    {
        hipFreeAsync(delta, hip_stream);
        return static_cast<int>(err);
    }
    err = hipMallocAsync(reinterpret_cast<void**>(&dk_full),
                         dkdv_full_count * sizeof(float),
                         hip_stream);
    if(err != hipSuccess)
    {
        hipFreeAsync(dq_acc, hip_stream);
        hipFreeAsync(delta, hip_stream);
        return static_cast<int>(err);
    }
    err = hipMallocAsync(reinterpret_cast<void**>(&dv_full),
                         dkdv_full_count * sizeof(float),
                         hip_stream);
    if(err != hipSuccess)
    {
        hipFreeAsync(dk_full, hip_stream);
        hipFreeAsync(dq_acc, hip_stream);
        hipFreeAsync(delta, hip_stream);
        return static_cast<int>(err);
    }

    dim3 delta_grid(static_cast<unsigned>(seqlen_q),
                    static_cast<unsigned>(num_heads),
                    static_cast<unsigned>(batch_size));
    dim3 delta_block(static_cast<unsigned>(HeadDim), 1);
    hipLaunchKernelGGL((kiln_rocm_flash_bwd_delta_bf16_kernel<HeadDim>),
                       delta_grid,
                       delta_block,
                       0,
                       hip_stream,
                       static_cast<const hip_bfloat16*>(dout),
                       static_cast<const hip_bfloat16*>(out),
                       delta,
                       batch_size,
                       seqlen_q,
                       num_heads);
    err = hipGetLastError();
    if(err == hipSuccess)
    {
        err = hipMemsetAsync(dq_acc, 0, dq_count * sizeof(float), hip_stream);
    }
    if(err != hipSuccess)
    {
        hipFreeAsync(dv_full, hip_stream);
        hipFreeAsync(dk_full, hip_stream);
        hipFreeAsync(dq_acc, hip_stream);
        hipFreeAsync(delta, hip_stream);
        return static_cast<int>(err);
    }

    const int groups_per_kv_head = num_heads / num_heads_k;
    const ck_tile::index_t stride_q = num_heads * HeadDim;
    const ck_tile::index_t stride_k = num_heads_k * HeadDim;
    const ck_tile::index_t stride_v = num_heads_k * HeadDim;
    const ck_tile::index_t stride_do = num_heads * HeadDim;
    const ck_tile::index_t stride_dq_acc = num_heads * HeadDim;
    const ck_tile::index_t stride_dk = num_heads * HeadDim;
    const ck_tile::index_t stride_dv = num_heads * HeadDim;
    const ck_tile::index_t nhead_stride_q = HeadDim;
    const ck_tile::index_t nhead_stride_k = HeadDim;
    const ck_tile::index_t nhead_stride_v = HeadDim;
    const ck_tile::index_t nhead_stride_do = HeadDim;
    const ck_tile::index_t nhead_stride_lsed = seqlen_q;
    const ck_tile::index_t nhead_stride_dq_acc = HeadDim;
    const ck_tile::index_t nhead_stride_dk = HeadDim;
    const ck_tile::index_t nhead_stride_dv = HeadDim;
    const ck_tile::index_t batch_stride_q = seqlen_q * num_heads * HeadDim;
    const ck_tile::index_t batch_stride_k = seqlen_k * num_heads_k * HeadDim;
    const ck_tile::index_t batch_stride_v = seqlen_k * num_heads_k * HeadDim;
    const ck_tile::index_t batch_stride_do = seqlen_q * num_heads * HeadDim;
    const ck_tile::index_t batch_stride_lsed = num_heads * seqlen_q;
    const ck_tile::index_t batch_stride_dq_acc = seqlen_q * num_heads * HeadDim;
    const ck_tile::index_t batch_stride_dk = seqlen_k * num_heads * HeadDim;
    const ck_tile::index_t batch_stride_dv = seqlen_k * num_heads * HeadDim;

    auto kargs = kernel::MakeKargsImpl(q,
                                       k,
                                       v,
                                       nullptr,
                                       softmax_lse,
                                       dout,
                                       delta,
                                       nullptr,
                                       dk_full,
                                       dv_full,
                                       nullptr,
                                       dq_acc,
                                       seqlen_q,
                                       seqlen_k,
                                       HeadDim,
                                       HeadDim,
                                       num_heads,
                                       groups_per_kv_head,
                                       softmax_scale,
                                       stride_q,
                                       stride_k,
                                       stride_v,
                                       0,
                                       0,
                                       stride_do,
                                       stride_dq_acc,
                                       stride_dk,
                                       stride_dv,
                                       0,
                                       nhead_stride_q,
                                       nhead_stride_k,
                                       nhead_stride_v,
                                       0,
                                       0,
                                       nhead_stride_do,
                                       nhead_stride_lsed,
                                       nhead_stride_dq_acc,
                                       nhead_stride_dk,
                                       nhead_stride_dv,
                                       0,
                                       batch_stride_q,
                                       batch_stride_k,
                                       batch_stride_v,
                                       0,
                                       0,
                                       batch_stride_do,
                                       batch_stride_lsed,
                                       batch_stride_dq_acc,
                                       batch_stride_dk,
                                       batch_stride_dv,
                                       0,
                                       0,
                                       -1,
                                       0,
                                       static_cast<ck_tile::index_t>(mask_enum::mask_bottom_right),
                                       0.0f,
                                       std::pair<uint64_t, uint64_t>{0, 0});
    const dim3 ck_grid = kernel::GridSize(batch_size, num_heads, seqlen_k);
    const ck_tile::stream_config stream_config{hip_stream, false};
    try
    {
        ck_tile::launch_kernel(
            stream_config,
            ck_tile::make_kernel<kernel::kBlockPerCu>(
                kernel{}, ck_grid, kernel::BlockSize(), 0, kargs));
    }
    catch(const std::exception&)
    {
        hipFreeAsync(dv_full, hip_stream);
        hipFreeAsync(dk_full, hip_stream);
        hipFreeAsync(dq_acc, hip_stream);
        hipFreeAsync(delta, hip_stream);
        return -21;
    }
    err = hipGetLastError();
    if(err == hipSuccess)
    {
        constexpr int block = 256;
        const long long dq_n = static_cast<long long>(dq_count);
        const int dq_grid = static_cast<int>((dq_n + block - 1) / block);
        kiln_rocm_flash_f32_to_bf16_kernel<<<dq_grid, block, 0, hip_stream>>>(
            dq_acc, static_cast<hip_bfloat16*>(dq), dq_n);
        err = hipGetLastError();
    }
    if(err == hipSuccess)
    {
        constexpr int block = 256;
        const long long kv_n =
            static_cast<long long>(batch_size) * seqlen_k * num_heads_k * HeadDim;
        const int kv_grid = static_cast<int>((kv_n + block - 1) / block);
        kiln_rocm_flash_reduce_gqa_grads_f32_to_bf16_kernel<HeadDim><<<kv_grid,
                                                                        block,
                                                                        0,
                                                                        hip_stream>>>(
            dk_full,
            static_cast<hip_bfloat16*>(dk),
            batch_size,
            seqlen_k,
            num_heads,
            num_heads_k);
        err = hipGetLastError();
    }
    if(err == hipSuccess)
    {
        constexpr int block = 256;
        const long long kv_n =
            static_cast<long long>(batch_size) * seqlen_k * num_heads_k * HeadDim;
        const int kv_grid = static_cast<int>((kv_n + block - 1) / block);
        kiln_rocm_flash_reduce_gqa_grads_f32_to_bf16_kernel<HeadDim><<<kv_grid,
                                                                        block,
                                                                        0,
                                                                        hip_stream>>>(
            dv_full,
            static_cast<hip_bfloat16*>(dv),
            batch_size,
            seqlen_k,
            num_heads,
            num_heads_k);
        err = hipGetLastError();
    }

    const hipError_t free_dv_err = hipFreeAsync(dv_full, hip_stream);
    const hipError_t free_dk_err = hipFreeAsync(dk_full, hip_stream);
    const hipError_t free_dq_err = hipFreeAsync(dq_acc, hip_stream);
    const hipError_t free_delta_err = hipFreeAsync(delta, hip_stream);
    if(err != hipSuccess)
    {
        return static_cast<int>(err);
    }
    if(free_dv_err != hipSuccess)
    {
        return static_cast<int>(free_dv_err);
    }
    if(free_dk_err != hipSuccess)
    {
        return static_cast<int>(free_dk_err);
    }
    if(free_dq_err != hipSuccess)
    {
        return static_cast<int>(free_dq_err);
    }
    return free_delta_err == hipSuccess ? 0 : static_cast<int>(free_delta_err);
}
#endif

} // namespace

extern "C" int kiln_rocm_flash_attn_fwd_ck_bf16(const void* q,
                                                const void* k,
                                                const void* v,
                                                void* out,
                                                void* softmax_lse_out,
                                                int batch_size,
                                                int seqlen_q,
                                                int seqlen_k,
                                                int num_heads,
                                                int num_heads_k,
                                                int head_dim,
                                                float softmax_scale,
                                                int is_causal,
                                                void* stream)
{
    if(q == nullptr || k == nullptr || v == nullptr || out == nullptr)
    {
        return -1;
    }
    if(batch_size <= 0 || seqlen_q <= 0 || seqlen_k <= 0 || num_heads <= 0 || num_heads_k <= 0)
    {
        return -2;
    }
    if(num_heads % num_heads_k != 0)
    {
        return -3;
    }
    if(is_causal && seqlen_k < seqlen_q)
    {
        return -5;
    }

#if KILN_HAS_CK_TILE_FMHA_FWD
    if(!is_causal)
    {
        return -6;
    }
    switch(head_dim)
    {
    case 128:
        return launch_ck_fwd_bf16<128, true>(static_cast<const hip_bfloat16*>(q),
                                             static_cast<const hip_bfloat16*>(k),
                                             static_cast<const hip_bfloat16*>(v),
                                             static_cast<hip_bfloat16*>(out),
                                             static_cast<float*>(softmax_lse_out),
                                             batch_size,
                                             seqlen_q,
                                             seqlen_k,
                                             num_heads,
                                             num_heads_k,
                                             softmax_scale,
                                             stream);
    case 256:
        return -4;
    default:
        return -4;
    }
#else
    return -20;
#endif
}

extern "C" int kiln_rocm_flash_wmma_qk16_bf16(const void* a,
                                              const void* b,
                                              void* out,
                                              void* stream)
{
    if(a == nullptr || b == nullptr || out == nullptr)
    {
        return -1;
    }
    if(!current_device_supports_gfx11_wmma())
    {
        return -30;
    }
    dim3 grid(1);
    dim3 block(32);
    hipLaunchKernelGGL(kiln_rocm_flash_wmma_qk16_bf16_kernel,
                       grid,
                       block,
                       0,
                       static_cast<hipStream_t>(stream),
                       static_cast<const hip_bfloat16*>(a),
                       static_cast<const hip_bfloat16*>(b),
                       static_cast<float*>(out));

    const hipError_t err = hipGetLastError();
    return err == hipSuccess ? 0 : static_cast<int>(err);
}

extern "C" int kiln_rocm_flash_attn_fwd_bf16(const void* q,
                                             const void* k,
                                             const void* v,
                                             void* out,
                                             void* softmax_lse_out,
                                             int batch_size,
                                             int seqlen_q,
                                             int seqlen_k,
                                             int num_heads,
                                             int num_heads_k,
                                             int head_dim,
                                             float softmax_scale,
                                             int is_causal,
                                             void* stream)
{
    if(q == nullptr || k == nullptr || v == nullptr || out == nullptr)
    {
        return -1;
    }
    if(batch_size <= 0 || seqlen_q <= 0 || seqlen_k <= 0 || num_heads <= 0 || num_heads_k <= 0)
    {
        return -2;
    }
    if(num_heads % num_heads_k != 0)
    {
        return -3;
    }

    switch(head_dim)
    {
    case 128:
        if(native_keysplit_enabled(seqlen_q, seqlen_k))
        {
            return launch_fwd_keysplit<128, 8>(q,
                                               k,
                                               v,
                                               out,
                                               softmax_lse_out,
                                               batch_size,
                                               seqlen_q,
                                               seqlen_k,
                                               num_heads,
                                               num_heads_k,
                                               softmax_scale,
                                               is_causal,
                                               stream);
        }
        if(seqlen_k >= 512)
        {
            return launch_fwd_qblock_cached<128, 8, 32>(q,
                                                        k,
                                                        v,
                                                        out,
                                                        softmax_lse_out,
                                                        batch_size,
                                                        seqlen_q,
                                                        seqlen_k,
                                                        num_heads,
                                                        num_heads_k,
                                                        softmax_scale,
                                                        is_causal,
                                                        stream);
        }
        return launch_fwd<128, 8, 128>(q,
                                       k,
                                       v,
                                       out,
                                       softmax_lse_out,
                                       batch_size,
                                       seqlen_q,
                                       seqlen_k,
                                       num_heads,
                                       num_heads_k,
                                       softmax_scale,
                                       is_causal,
                                       stream);
    case 256:
        if(native_gqa_qblock_enabled(head_dim, seqlen_q, seqlen_k, num_heads, num_heads_k))
        {
            if(!env_truthy("KILN_ROCM_FLASH_DISABLE_WMMA_GQA_QBLOCK") &&
               current_device_supports_gfx11_wmma())
            {
                if(wmma_gqa_r64k32_enabled(seqlen_q, seqlen_k))
                {
                    return launch_fwd_wmma_gqa_r64h1k32(q,
                                                        k,
                                                        v,
                                                        out,
                                                        softmax_lse_out,
                                                        batch_size,
                                                        seqlen_q,
                                                        seqlen_k,
                                                        num_heads,
                                                        num_heads_k,
                                                        softmax_scale,
                                                        is_causal,
                                                        stream);
                }
                return launch_fwd_wmma_gqa_r32h1k32(q,
                                                    k,
                                                    v,
                                                    out,
                                                    softmax_lse_out,
                                                    batch_size,
                                                    seqlen_q,
                                                    seqlen_k,
                                                    num_heads,
                                                    num_heads_k,
                                                    softmax_scale,
                                                    is_causal,
                                                    stream);
            }
            return launch_fwd_gqa_qblock_cached<256, 8, 4, 64>(q,
                                                               k,
                                                               v,
                                                               out,
                                                               softmax_lse_out,
                                                               batch_size,
                                                               seqlen_q,
                                                               seqlen_k,
                                                               num_heads,
                                                               num_heads_k,
                                                               softmax_scale,
                                                               is_causal,
                                                               stream);
        }
        if(native_wmma_qblock_enabled(head_dim, seqlen_q, seqlen_k) &&
           current_device_supports_gfx11_wmma())
        {
            return launch_fwd_wmma_qblock256(q,
                                             k,
                                             v,
                                             out,
                                             softmax_lse_out,
                                             batch_size,
                                             seqlen_q,
                                             seqlen_k,
                                             num_heads,
                                             num_heads_k,
                                             softmax_scale,
                                             is_causal,
                                             stream);
        }
        if(native_keysplit_enabled(seqlen_q, seqlen_k))
        {
            return launch_fwd_keysplit<256, 8>(q,
                                               k,
                                               v,
                                               out,
                                               softmax_lse_out,
                                               batch_size,
                                               seqlen_q,
                                               seqlen_k,
                                               num_heads,
                                               num_heads_k,
                                               softmax_scale,
                                               is_causal,
                                               stream);
        }
        if(seqlen_k >= 512)
        {
            return launch_fwd_qblock_cached<256, 8, 32>(q,
                                                        k,
                                                        v,
                                                        out,
                                                        softmax_lse_out,
                                                        batch_size,
                                                        seqlen_q,
                                                        seqlen_k,
                                                        num_heads,
                                                        num_heads_k,
                                                        softmax_scale,
                                                        is_causal,
                                                        stream);
        }
        return launch_fwd<256, 8, 128>(q,
                                       k,
                                       v,
                                       out,
                                       softmax_lse_out,
                                       batch_size,
                                       seqlen_q,
                                       seqlen_k,
                                       num_heads,
                                       num_heads_k,
                                       softmax_scale,
                                       is_causal,
                                       stream);
    default:
        return -4;
    }
}

extern "C" int kiln_rocm_flash_attn_fwd_stream_update_bf16(const void* q,
                                                           const void* k,
                                                           const void* v,
                                                           void* row_m,
                                                           void* row_l,
                                                           void* acc,
                                                           int batch_size,
                                                           int seqlen_q_tile,
                                                           int seqlen_k,
                                                           int seqlen_q_total,
                                                           int num_heads,
                                                           int num_heads_k,
                                                           int head_dim,
                                                           float softmax_scale,
                                                           int is_causal,
                                                           int q_start,
                                                           int key_start,
                                                           int key_len,
                                                           void* stream)
{
    if(q == nullptr || k == nullptr || v == nullptr || row_m == nullptr || row_l == nullptr ||
       acc == nullptr)
    {
        return -1;
    }
    if(batch_size <= 0 || seqlen_q_tile <= 0 || seqlen_k <= 0 || seqlen_q_total <= 0 ||
       num_heads <= 0 || num_heads_k <= 0 || key_len <= 0)
    {
        return -2;
    }
    if(num_heads % num_heads_k != 0)
    {
        return -3;
    }
    if(q_start < 0 || key_start < 0 || q_start + seqlen_q_tile > seqlen_q_total ||
       key_start >= seqlen_k)
    {
        return -5;
    }

    switch(head_dim)
    {
    case 128:
        return launch_fwd_stream_update<128, 8>(q,
                                                k,
                                                v,
                                                row_m,
                                                row_l,
                                                acc,
                                                batch_size,
                                                seqlen_q_tile,
                                                seqlen_k,
                                                seqlen_q_total,
                                                num_heads,
                                                num_heads_k,
                                                softmax_scale,
                                                is_causal,
                                                q_start,
                                                key_start,
                                                key_len,
                                                stream);
    case 256:
        return launch_fwd_stream_update<256, 8>(q,
                                                k,
                                                v,
                                                row_m,
                                                row_l,
                                                acc,
                                                batch_size,
                                                seqlen_q_tile,
                                                seqlen_k,
                                                seqlen_q_total,
                                                num_heads,
                                                num_heads_k,
                                                softmax_scale,
                                                is_causal,
                                                q_start,
                                                key_start,
                                                key_len,
                                                stream);
    default:
        return -4;
    }
}

extern "C" int kiln_rocm_flash_attn_fwd_stream_finalize_bf16(const void* row_m,
                                                             const void* row_l,
                                                             const void* acc,
                                                             void* out,
                                                             void* softmax_lse_out,
                                                             int batch_size,
                                                             int seqlen_q_tile,
                                                             int num_heads,
                                                             int head_dim,
                                                             void* stream)
{
    if(row_m == nullptr || row_l == nullptr || acc == nullptr || out == nullptr ||
       softmax_lse_out == nullptr)
    {
        return -1;
    }
    if(batch_size <= 0 || seqlen_q_tile <= 0 || num_heads <= 0)
    {
        return -2;
    }

    switch(head_dim)
    {
    case 128:
        return launch_fwd_stream_finalize<128, 8>(row_m,
                                                  row_l,
                                                  acc,
                                                  out,
                                                  softmax_lse_out,
                                                  batch_size,
                                                  seqlen_q_tile,
                                                  num_heads,
                                                  stream);
    case 256:
        return launch_fwd_stream_finalize<256, 8>(row_m,
                                                  row_l,
                                                  acc,
                                                  out,
                                                  softmax_lse_out,
                                                  batch_size,
                                                  seqlen_q_tile,
                                                  num_heads,
                                                  stream);
    default:
        return -4;
    }
}

extern "C" int kiln_rocm_flash_attn_bwd_bf16(const void* dout,
                                             const void* q,
                                             const void* k,
                                             const void* v,
                                             const void* out,
                                             const void* softmax_lse,
                                             void* dq,
                                             void* dk,
                                             void* dv,
                                             int batch_size,
                                             int seqlen_q,
                                             int seqlen_k,
                                             int num_heads,
                                             int num_heads_k,
                                             int head_dim,
                                             float softmax_scale,
                                             int is_causal,
                                             void* stream)
{
    if(dout == nullptr || q == nullptr || k == nullptr || v == nullptr || out == nullptr ||
       softmax_lse == nullptr || dq == nullptr || dk == nullptr || dv == nullptr)
    {
        return -1;
    }
    if(batch_size <= 0 || seqlen_q <= 0 || seqlen_k <= 0 || num_heads <= 0 || num_heads_k <= 0)
    {
        return -2;
    }
    if(num_heads % num_heads_k != 0)
    {
        return -3;
    }

    switch(head_dim)
    {
    case 128:
        return launch_bwd<128, 8>(dout,
                                  q,
                                  k,
                                  v,
                                  out,
                                  softmax_lse,
                                  dq,
                                  dk,
                                  dv,
                                  batch_size,
                                  seqlen_q,
                                  seqlen_k,
                                  num_heads,
                                  num_heads_k,
                                  softmax_scale,
                                  is_causal,
                                  stream);
    case 256:
        return launch_bwd<256, 4>(dout,
                                  q,
                                  k,
                                  v,
                                  out,
                                  softmax_lse,
                                  dq,
                                  dk,
                                  dv,
                                  batch_size,
                                  seqlen_q,
                                  seqlen_k,
                                  num_heads,
                                  num_heads_k,
                                  softmax_scale,
                                  is_causal,
                                  stream);
    default:
        return -4;
    }
}

extern "C" int kiln_rocm_flash_attn_bwd_collapsed_gqa_bf16(const void* dout,
                                                           const void* q,
                                                           const void* k,
                                                           const void* v,
                                                           const void* out,
                                                           const void* softmax_lse,
                                                           void* dq,
                                                           void* dk,
                                                           void* dv,
                                                           int batch_size,
                                                           int seqlen_q,
                                                           int seqlen_k,
                                                           int num_heads,
                                                           int num_heads_k,
                                                           int head_dim,
                                                           float softmax_scale,
                                                           int is_causal,
                                                           void* stream)
{
    if(dout == nullptr || q == nullptr || k == nullptr || v == nullptr || out == nullptr ||
       softmax_lse == nullptr || dq == nullptr || dk == nullptr || dv == nullptr)
    {
        return -1;
    }
    if(batch_size <= 0 || seqlen_q <= 0 || seqlen_k <= 0 || num_heads <= 0 || num_heads_k <= 0)
    {
        return -2;
    }
    if(num_heads % num_heads_k != 0)
    {
        return -3;
    }

#if KILN_HAS_CK_TILE_FMHA
    if(!env_truthy("KILN_ROCM_FLASH_DISABLE_CK_BWD_COLLAPSED_GQA") && head_dim == 256 &&
       is_causal &&
       current_device_supports_gfx11_wmma())
    {
        return launch_bwd_collapsed_gqa_ck_trload<256>(dout,
                                                       q,
                                                       k,
                                                       v,
                                                       out,
                                                       softmax_lse,
                                                       dq,
                                                       dk,
                                                       dv,
                                                       batch_size,
                                                       seqlen_q,
                                                       seqlen_k,
                                                       num_heads,
                                                       num_heads_k,
                                                       softmax_scale,
                                                       is_causal,
                                                       stream);
    }
#endif

    switch(head_dim)
    {
    case 128:
    {
        const int qpar =
            env_i32_or("KILN_ROCM_FLASH_NATIVE_DIRECT_COLLAPSED_GQA_QPAR", 1);
        if(qpar <= 1)
        {
            return launch_bwd_collapsed_gqa_warp<128>(dout,
                                                      q,
                                                      k,
                                                      v,
                                                      out,
                                                      softmax_lse,
                                                      dq,
                                                      dk,
                                                      dv,
                                                      batch_size,
                                                      seqlen_q,
                                                      seqlen_k,
                                                      num_heads,
                                                      num_heads_k,
                                                      softmax_scale,
                                                      is_causal,
                                                      stream);
        }
        if(qpar <= 2)
        {
            return launch_bwd_collapsed_gqa<128, 2>(dout,
                                                    q,
                                                    k,
                                                    v,
                                                    out,
                                                    softmax_lse,
                                                    dq,
                                                    dk,
                                                    dv,
                                                    batch_size,
                                                    seqlen_q,
                                                    seqlen_k,
                                                    num_heads,
                                                    num_heads_k,
                                                    softmax_scale,
                                                    is_causal,
                                                    stream);
        }
        if(qpar <= 4)
        {
            return launch_bwd_collapsed_gqa<128, 4>(dout,
                                                    q,
                                                    k,
                                                    v,
                                                    out,
                                                    softmax_lse,
                                                    dq,
                                                    dk,
                                                    dv,
                                                    batch_size,
                                                    seqlen_q,
                                                    seqlen_k,
                                                    num_heads,
                                                    num_heads_k,
                                                    softmax_scale,
                                                    is_causal,
                                                    stream);
        }
        return launch_bwd_collapsed_gqa<128, 8>(dout,
                                                q,
                                                k,
                                                v,
                                                out,
                                                softmax_lse,
                                                dq,
                                                dk,
                                                dv,
                                                batch_size,
                                                seqlen_q,
                                                seqlen_k,
                                                num_heads,
                                                num_heads_k,
                                                softmax_scale,
                                                is_causal,
                                                stream);
    }
    case 256:
    {
        const int qpar =
            env_i32_or("KILN_ROCM_FLASH_NATIVE_DIRECT_COLLAPSED_GQA_QPAR", 1);
        if(qpar <= 1)
        {
            return launch_bwd_collapsed_gqa_warp<256>(dout,
                                                      q,
                                                      k,
                                                      v,
                                                      out,
                                                      softmax_lse,
                                                      dq,
                                                      dk,
                                                      dv,
                                                      batch_size,
                                                      seqlen_q,
                                                      seqlen_k,
                                                      num_heads,
                                                      num_heads_k,
                                                      softmax_scale,
                                                      is_causal,
                                                      stream);
        }
        if(qpar <= 2)
        {
            return launch_bwd_collapsed_gqa<256, 2>(dout,
                                                    q,
                                                    k,
                                                    v,
                                                    out,
                                                    softmax_lse,
                                                    dq,
                                                    dk,
                                                    dv,
                                                    batch_size,
                                                    seqlen_q,
                                                    seqlen_k,
                                                    num_heads,
                                                    num_heads_k,
                                                    softmax_scale,
                                                    is_causal,
                                                    stream);
        }
        return launch_bwd_collapsed_gqa<256, 4>(dout,
                                                q,
                                                k,
                                                v,
                                                out,
                                                softmax_lse,
                                                dq,
                                                dk,
                                                dv,
                                                batch_size,
                                                seqlen_q,
                                                seqlen_k,
                                                num_heads,
                                                num_heads_k,
                                                softmax_scale,
                                                is_causal,
                                                stream);
    }
    default:
        return -4;
    }
}

__global__ void kiln_rocm_rowsum_sub_last_axis_f32_kernel(const float* __restrict__ a,
                                                          const float* __restrict__ rowsum,
                                                          float* __restrict__ out,
                                                          long long n,
                                                          int sk)
{
    long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    if(idx >= n)
    {
        return;
    }
    out[idx] = a[idx] - rowsum[idx / sk];
}

__global__ void kiln_rocm_flash_collapse_gqa_bf16_kernel(const hip_bfloat16* __restrict__ expanded,
                                                         hip_bfloat16* __restrict__ collapsed,
                                                         long long n,
                                                         int seqlen,
                                                         int num_heads,
                                                         int num_heads_k,
                                                         int head_dim)
{
    const long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    if(idx >= n)
    {
        return;
    }

    const int dim = static_cast<int>(idx % head_dim);
    long long rem = idx / head_dim;
    const int kv_head = static_cast<int>(rem % num_heads_k);
    rem /= num_heads_k;
    const int token = static_cast<int>(rem % seqlen);
    const int batch = static_cast<int>(rem / seqlen);
    const int groups_per_kv_head = num_heads / num_heads_k;

    const size_t expanded_base =
        ((static_cast<size_t>(batch) * seqlen + token) * num_heads +
         kv_head * groups_per_kv_head) *
            head_dim +
        dim;
    float acc = 0.0f;
    for(int group = 0; group < groups_per_kv_head; ++group)
    {
        acc += static_cast<float>(expanded[expanded_base + static_cast<size_t>(group) * head_dim]);
    }
    collapsed[idx] = hip_bfloat16(acc);
}

extern "C" int kiln_rocm_flash_rowsum_sub_last_axis_f32(const void* a,
                                                        const void* rowsum,
                                                        void* out,
                                                        int bh,
                                                        int sq,
                                                        int sk,
                                                        void* stream)
{
    if(a == nullptr || rowsum == nullptr || out == nullptr)
    {
        return -1;
    }
    if(bh <= 0 || sq <= 0 || sk <= 0)
    {
        return -2;
    }
    long long n = static_cast<long long>(bh) * static_cast<long long>(sq) * static_cast<long long>(sk);
    constexpr int block = 256;
    int grid = static_cast<int>((n + block - 1) / block);
    kiln_rocm_rowsum_sub_last_axis_f32_kernel<<<grid,
                                                 block,
                                                 0,
                                                 static_cast<hipStream_t>(stream)>>>(
        static_cast<const float*>(a),
        static_cast<const float*>(rowsum),
        static_cast<float*>(out),
        n,
        sk);
    hipError_t err = hipGetLastError();
    return err == hipSuccess ? 0 : static_cast<int>(err);
}

extern "C" int kiln_rocm_flash_collapse_gqa_bf16(const void* expanded,
                                                 void* collapsed,
                                                 int batch_size,
                                                 int seqlen,
                                                 int num_heads,
                                                 int num_heads_k,
                                                 int head_dim,
                                                 void* stream)
{
    if(expanded == nullptr || collapsed == nullptr)
    {
        return -1;
    }
    if(batch_size <= 0 || seqlen <= 0 || num_heads <= 0 || num_heads_k <= 0 || head_dim <= 0)
    {
        return -2;
    }
    if(num_heads % num_heads_k != 0)
    {
        return -3;
    }

    const long long n = static_cast<long long>(batch_size) * seqlen * num_heads_k * head_dim;
    constexpr int block = 256;
    const int grid = static_cast<int>((n + block - 1) / block);
    kiln_rocm_flash_collapse_gqa_bf16_kernel<<<grid,
                                               block,
                                               0,
                                               static_cast<hipStream_t>(stream)>>>(
        static_cast<const hip_bfloat16*>(expanded),
        static_cast<hip_bfloat16*>(collapsed),
        n,
        seqlen,
        num_heads,
        num_heads_k,
        head_dim);
    const hipError_t err = hipGetLastError();
    return err == hipSuccess ? 0 : static_cast<int>(err);
}

__global__ void kiln_rocm_causal_mask_fill_offset_f32_kernel(float* __restrict__ scores,
                                                             long long n,
                                                             int sq,
                                                             int sk,
                                                             int q_start,
                                                             int causal_offset,
                                                             float fill)
{
    long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    if(idx >= n)
    {
        return;
    }
    int key_idx = static_cast<int>(idx % sk);
    int query_idx = static_cast<int>((idx / sk) % sq);
    int allowed_key = causal_offset + q_start + query_idx;
    if(key_idx > allowed_key)
    {
        scores[idx] = fill;
    }
}

extern "C" int kiln_rocm_flash_causal_mask_fill_offset_f32(void* scores,
                                                           int bh,
                                                           int sq,
                                                           int sk,
                                                           int q_start,
                                                           int causal_offset,
                                                           float fill,
                                                           void* stream)
{
    if(scores == nullptr)
    {
        return -1;
    }
    if(bh <= 0 || sq <= 0 || sk <= 0 || q_start < 0)
    {
        return -2;
    }
    long long n = static_cast<long long>(bh) * static_cast<long long>(sq) * static_cast<long long>(sk);
    constexpr int block = 256;
    int grid = static_cast<int>((n + block - 1) / block);
    kiln_rocm_causal_mask_fill_offset_f32_kernel<<<grid,
                                                    block,
                                                    0,
                                                    static_cast<hipStream_t>(stream)>>>(
        static_cast<float*>(scores),
        n,
        sq,
        sk,
        q_start,
        causal_offset,
        fill);
    hipError_t err = hipGetLastError();
    return err == hipSuccess ? 0 : static_cast<int>(err);
}

__global__ void kiln_rocm_flash_scale_mask_f32_kernel(float* __restrict__ scores,
                                                      long long n,
                                                      int sq,
                                                      int sk,
                                                      int q_start,
                                                      int k_start,
                                                      int causal_offset,
                                                      float scale,
                                                      float fill,
                                                      int is_causal)
{
    long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    if(idx >= n)
    {
        return;
    }
    const int key_local = static_cast<int>(idx % sk);
    const int query_local = static_cast<int>((idx / sk) % sq);
    if(is_causal)
    {
        const int key_global = k_start + key_local;
        const int allowed_key = causal_offset + q_start + query_local;
        if(key_global > allowed_key)
        {
            scores[idx] = fill;
            return;
        }
    }
    scores[idx] *= scale;
}

extern "C" int kiln_rocm_flash_scale_mask_f32(void* scores,
                                              int bh,
                                              int sq,
                                              int sk,
                                              int q_start,
                                              int k_start,
                                              int causal_offset,
                                              float scale,
                                              float fill,
                                              int is_causal,
                                              void* stream)
{
    if(scores == nullptr)
    {
        return -1;
    }
    if(bh <= 0 || sq <= 0 || sk <= 0 || q_start < 0 || k_start < 0)
    {
        return -2;
    }
    long long n = static_cast<long long>(bh) * static_cast<long long>(sq) *
                  static_cast<long long>(sk);
    constexpr int block = 256;
    int grid = static_cast<int>((n + block - 1) / block);
    kiln_rocm_flash_scale_mask_f32_kernel<<<grid,
                                             block,
                                             0,
                                             static_cast<hipStream_t>(stream)>>>(
        static_cast<float*>(scores),
        n,
        sq,
        sk,
        q_start,
        k_start,
        causal_offset,
        scale,
        fill,
        is_causal != 0 ? 1 : 0);
    hipError_t err = hipGetLastError();
    return err == hipSuccess ? 0 : static_cast<int>(err);
}

__global__ void kiln_rocm_flash_exp_mask_f32_kernel(float* __restrict__ scores,
                                                    const float* __restrict__ row_max,
                                                    long long n,
                                                    int sq,
                                                    int sk,
                                                    int q_start,
                                                    int k_start,
                                                    int causal_offset,
                                                    int is_causal)
{
    long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    if(idx >= n)
    {
        return;
    }
    const int key_local = static_cast<int>(idx % sk);
    const int query_local = static_cast<int>((idx / sk) % sq);
    if(is_causal)
    {
        const int key_global = k_start + key_local;
        const int allowed_key = causal_offset + q_start + query_local;
        if(key_global > allowed_key)
        {
            scores[idx] = 0.0f;
            return;
        }
    }
    const long long row = idx / sk;
    const float m = row_max[row];
    if(!isfinite(m))
    {
        scores[idx] = 0.0f;
        return;
    }
    scores[idx] = expf(scores[idx] - m);
}

extern "C" int kiln_rocm_flash_exp_mask_f32(void* scores,
                                            const void* row_max,
                                            int bh,
                                            int sq,
                                            int sk,
                                            int q_start,
                                            int k_start,
                                            int causal_offset,
                                            int is_causal,
                                            void* stream)
{
    if(scores == nullptr || row_max == nullptr)
    {
        return -1;
    }
    if(bh <= 0 || sq <= 0 || sk <= 0 || q_start < 0 || k_start < 0)
    {
        return -2;
    }
    long long n = static_cast<long long>(bh) * static_cast<long long>(sq) *
                  static_cast<long long>(sk);
    constexpr int block = 256;
    int grid = static_cast<int>((n + block - 1) / block);
    kiln_rocm_flash_exp_mask_f32_kernel<<<grid,
                                           block,
                                           0,
                                           static_cast<hipStream_t>(stream)>>>(
        static_cast<float*>(scores),
        static_cast<const float*>(row_max),
        n,
        sq,
        sk,
        q_start,
        k_start,
        causal_offset,
        is_causal != 0 ? 1 : 0);
    hipError_t err = hipGetLastError();
    return err == hipSuccess ? 0 : static_cast<int>(err);
}

__global__ void kiln_rocm_flash_prob_from_lse_f32_kernel(float* __restrict__ scores,
                                                         const float* __restrict__ lse,
                                                         long long n,
                                                         int sq,
                                                         int sk,
                                                         int total_q,
                                                         int q_start,
                                                         int k_start,
                                                         int causal_offset,
                                                         int is_causal)
{
    long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    if(idx >= n)
    {
        return;
    }

    const int key_local = static_cast<int>(idx % sk);
    const long long row = idx / sk;
    const int query_local = static_cast<int>(row % sq);
    const long long bh_row = row / sq;
    if(is_causal)
    {
        const int key_global = k_start + key_local;
        const int allowed_key = causal_offset + q_start + query_local;
        if(key_global > allowed_key)
        {
            scores[idx] = 0.0f;
            return;
        }
    }

    const float row_lse = lse[bh_row * static_cast<long long>(total_q) + q_start + query_local];
    if(!isfinite(row_lse))
    {
        scores[idx] = 0.0f;
        return;
    }
    scores[idx] = expf(scores[idx] - row_lse);
}

extern "C" int kiln_rocm_flash_prob_from_lse_f32(void* scores,
                                                 const void* lse,
                                                 int bh,
                                                 int sq,
                                                 int sk,
                                                 int total_q,
                                                 int q_start,
                                                 int k_start,
                                                 int causal_offset,
                                                 int is_causal,
                                                 void* stream)
{
    if(scores == nullptr || lse == nullptr)
    {
        return -1;
    }
    if(bh <= 0 || sq <= 0 || sk <= 0 || total_q <= 0 || q_start < 0 || k_start < 0)
    {
        return -2;
    }
    long long n = static_cast<long long>(bh) * static_cast<long long>(sq) *
                  static_cast<long long>(sk);
    constexpr int block = 256;
    int grid = static_cast<int>((n + block - 1) / block);
    kiln_rocm_flash_prob_from_lse_f32_kernel<<<grid,
                                                block,
                                                0,
                                                static_cast<hipStream_t>(stream)>>>(
        static_cast<float*>(scores),
        static_cast<const float*>(lse),
        n,
        sq,
        sk,
        total_q,
        q_start,
        k_start,
        causal_offset,
        is_causal != 0 ? 1 : 0);
    hipError_t err = hipGetLastError();
    return err == hipSuccess ? 0 : static_cast<int>(err);
}

__global__ void kiln_rocm_flash_softmax_bwd_f32_kernel(float* __restrict__ dp,
                                                       const float* __restrict__ p,
                                                       const float* __restrict__ d_rows,
                                                       long long n,
                                                       int sq,
                                                       int sk,
                                                       int total_q,
                                                       int q_start,
                                                       int k_start,
                                                       int causal_offset,
                                                       float scale,
                                                       int is_causal)
{
    long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    if(idx >= n)
    {
        return;
    }

    const int key_local = static_cast<int>(idx % sk);
    const long long row = idx / sk;
    const int query_local = static_cast<int>(row % sq);
    if(is_causal)
    {
        const int key_global = k_start + key_local;
        const int allowed_key = causal_offset + q_start + query_local;
        if(key_global > allowed_key)
        {
            dp[idx] = 0.0f;
            return;
        }
    }

    const float prob = p[idx];
    dp[idx] = prob * (dp[idx] - d_rows[row]) * scale;
    (void)total_q;
}

extern "C" int kiln_rocm_flash_softmax_bwd_f32(void* dp,
                                               const void* p,
                                               const void* d_rows,
                                               int bh,
                                               int sq,
                                               int sk,
                                               int total_q,
                                               int q_start,
                                               int k_start,
                                               int causal_offset,
                                               float scale,
                                               int is_causal,
                                               void* stream)
{
    if(dp == nullptr || p == nullptr || d_rows == nullptr)
    {
        return -1;
    }
    if(bh <= 0 || sq <= 0 || sk <= 0 || total_q <= 0 || q_start < 0 || k_start < 0)
    {
        return -2;
    }
    long long n = static_cast<long long>(bh) * static_cast<long long>(sq) *
                  static_cast<long long>(sk);
    constexpr int block = 256;
    int grid = static_cast<int>((n + block - 1) / block);
    kiln_rocm_flash_softmax_bwd_f32_kernel<<<grid,
                                              block,
                                              0,
                                              static_cast<hipStream_t>(stream)>>>(
        static_cast<float*>(dp),
        static_cast<const float*>(p),
        static_cast<const float*>(d_rows),
        n,
        sq,
        sk,
        total_q,
        q_start,
        k_start,
        causal_offset,
        scale,
        is_causal != 0 ? 1 : 0);
    hipError_t err = hipGetLastError();
    return err == hipSuccess ? 0 : static_cast<int>(err);
}

__global__ void kiln_rocm_flash_accum_axis1_f32_kernel(float* __restrict__ dst,
                                                       const float* __restrict__ src,
                                                       long long n,
                                                       int total_s,
                                                       int tile_s,
                                                       int head_dim,
                                                       int s_start)
{
    long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    if(idx >= n)
    {
        return;
    }

    const int dim = static_cast<int>(idx % head_dim);
    const long long row_col = idx / head_dim;
    const int s_local = static_cast<int>(row_col % tile_s);
    const long long bh_row = row_col / tile_s;
    const long long dst_idx =
        (bh_row * static_cast<long long>(total_s) + s_start + s_local) *
            static_cast<long long>(head_dim) +
        dim;
    dst[dst_idx] += src[idx];
}

extern "C" int kiln_rocm_flash_accum_axis1_f32(void* dst,
                                               const void* src,
                                               int bh,
                                               int total_s,
                                               int tile_s,
                                               int head_dim,
                                               int s_start,
                                               void* stream)
{
    if(dst == nullptr || src == nullptr)
    {
        return -1;
    }
    if(bh <= 0 || total_s <= 0 || tile_s <= 0 || head_dim <= 0 || s_start < 0 ||
       s_start + tile_s > total_s)
    {
        return -2;
    }
    long long n = static_cast<long long>(bh) * static_cast<long long>(tile_s) *
                  static_cast<long long>(head_dim);
    constexpr int block = 256;
    int grid = static_cast<int>((n + block - 1) / block);
    kiln_rocm_flash_accum_axis1_f32_kernel<<<grid,
                                              block,
                                              0,
                                              static_cast<hipStream_t>(stream)>>>(
        static_cast<float*>(dst),
        static_cast<const float*>(src),
        n,
        total_s,
        tile_s,
        head_dim,
        s_start);
    hipError_t err = hipGetLastError();
    return err == hipSuccess ? 0 : static_cast<int>(err);
}

__global__ void kiln_rocm_flash_online_update_state_f32_kernel(
    float* __restrict__ row_m,
    float* __restrict__ row_l,
    const float* __restrict__ block_m,
    const float* __restrict__ block_l,
    float* __restrict__ alpha,
    float* __restrict__ beta,
    int rows)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if(row >= rows)
    {
        return;
    }

    const float bm = block_m[row];
    const float bl = block_l[row];
    const float old_l = row_l[row];
    const float old_m = row_m[row];
    if(!(bl > 0.0f) || !isfinite(bm))
    {
        alpha[row] = 1.0f;
        beta[row] = 0.0f;
        return;
    }

    if(!(old_l > 0.0f) || !isfinite(old_m))
    {
        row_m[row] = bm;
        row_l[row] = bl;
        alpha[row] = 0.0f;
        beta[row] = 1.0f;
        return;
    }

    const float nm = old_m > bm ? old_m : bm;
    const float a = expf(old_m - nm);
    const float b = expf(bm - nm);
    row_m[row] = nm;
    row_l[row] = old_l * a + bl * b;
    alpha[row] = a;
    beta[row] = b;
}

extern "C" int kiln_rocm_flash_online_update_state_f32(void* row_m,
                                                       void* row_l,
                                                       const void* block_m,
                                                       const void* block_l,
                                                       void* alpha,
                                                       void* beta,
                                                       int rows,
                                                       void* stream)
{
    if(row_m == nullptr || row_l == nullptr || block_m == nullptr || block_l == nullptr ||
       alpha == nullptr || beta == nullptr)
    {
        return -1;
    }
    if(rows <= 0)
    {
        return -2;
    }
    constexpr int block = 256;
    int grid = (rows + block - 1) / block;
    kiln_rocm_flash_online_update_state_f32_kernel<<<grid,
                                                      block,
                                                      0,
                                                      static_cast<hipStream_t>(stream)>>>(
        static_cast<float*>(row_m),
        static_cast<float*>(row_l),
        static_cast<const float*>(block_m),
        static_cast<const float*>(block_l),
        static_cast<float*>(alpha),
        static_cast<float*>(beta),
        rows);
    hipError_t err = hipGetLastError();
    return err == hipSuccess ? 0 : static_cast<int>(err);
}

__global__ void kiln_rocm_flash_online_update_acc_f32_kernel(
    float* __restrict__ acc,
    const float* __restrict__ block_out,
    const float* __restrict__ alpha,
    const float* __restrict__ beta,
    long long n,
    int head_dim)
{
    long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    if(idx >= n)
    {
        return;
    }
    const long long row = idx / head_dim;
    acc[idx] = acc[idx] * alpha[row] + block_out[idx] * beta[row];
}

extern "C" int kiln_rocm_flash_online_update_acc_f32(void* acc,
                                                     const void* block_out,
                                                     const void* alpha,
                                                     const void* beta,
                                                     int rows,
                                                     int head_dim,
                                                     void* stream)
{
    if(acc == nullptr || block_out == nullptr || alpha == nullptr || beta == nullptr)
    {
        return -1;
    }
    if(rows <= 0 || head_dim <= 0)
    {
        return -2;
    }
    long long n = static_cast<long long>(rows) * static_cast<long long>(head_dim);
    constexpr int block = 256;
    int grid = static_cast<int>((n + block - 1) / block);
    kiln_rocm_flash_online_update_acc_f32_kernel<<<grid,
                                                    block,
                                                    0,
                                                    static_cast<hipStream_t>(stream)>>>(
        static_cast<float*>(acc),
        static_cast<const float*>(block_out),
        static_cast<const float*>(alpha),
        static_cast<const float*>(beta),
        n,
        head_dim);
    hipError_t err = hipGetLastError();
    return err == hipSuccess ? 0 : static_cast<int>(err);
}

__global__ void kiln_rocm_flash_online_finalize_f32_kernel(const float* __restrict__ acc,
                                                           const float* __restrict__ row_m,
                                                           const float* __restrict__ row_l,
                                                           float* __restrict__ out,
                                                           float* __restrict__ lse,
                                                           long long n,
                                                           int head_dim)
{
    long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    if(idx >= n)
    {
        return;
    }
    const long long row = idx / head_dim;
    const int dim = static_cast<int>(idx % head_dim);
    const float l = row_l[row];
    if(l > 0.0f)
    {
        out[idx] = acc[idx] / l;
        if(dim == 0)
        {
            lse[row] = row_m[row] + logf(l);
        }
    }
    else
    {
        out[idx] = 0.0f;
        if(dim == 0)
        {
            lse[row] = -INFINITY;
        }
    }
}

extern "C" int kiln_rocm_flash_online_finalize_f32(const void* acc,
                                                   const void* row_m,
                                                   const void* row_l,
                                                   void* out,
                                                   void* lse,
                                                   int rows,
                                                   int head_dim,
                                                   void* stream)
{
    if(acc == nullptr || row_m == nullptr || row_l == nullptr || out == nullptr || lse == nullptr)
    {
        return -1;
    }
    if(rows <= 0 || head_dim <= 0)
    {
        return -2;
    }
    long long n = static_cast<long long>(rows) * static_cast<long long>(head_dim);
    constexpr int block = 256;
    int grid = static_cast<int>((n + block - 1) / block);
    kiln_rocm_flash_online_finalize_f32_kernel<<<grid,
                                                  block,
                                                  0,
                                                  static_cast<hipStream_t>(stream)>>>(
        static_cast<const float*>(acc),
        static_cast<const float*>(row_m),
        static_cast<const float*>(row_l),
        static_cast<float*>(out),
        static_cast<float*>(lse),
        n,
        head_dim);
    hipError_t err = hipGetLastError();
    return err == hipSuccess ? 0 : static_cast<int>(err);
}
