// Phase 0.8 — cublasLt vs cublas-default MLP gate||up probe.
//
// Compares the BF16-input BF16-output FP32-compute matmul at the Qwen3.5-4B
// MLP gate||up shape `[B*T, 2560] @ [2560, 18432]` between:
//
//   (1) `cublasGemmEx` with `CUBLAS_GEMM_DEFAULT_TENSOR_OP`
//       — the locked-in candle path (mirrors
//         vendor/candle-core/src/cuda_backend/mod.rs:2625's call shape).
//   (2) `cublasLtMatmul` with `cublasLtMatmulAlgoGetHeuristic` algorithm
//       selection + an explicit workspace.
//
// The probe times each path with CUDA events, medians over `iters`
// iterations, and reports per-shape ms + the chosen algo ID + the
// workspace bytes selected by the heuristic.
//
// Builds via crates/kiln-blas/build.rs and links cublasLt + cublas + cudart.

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

// Public result struct mirrored on the Rust side in `probe_ffi::ProbeResult`.
extern "C" {

struct ProbeResult {
    int bt;
    int k;
    int n;
    float ms_cublas_default;
    float ms_cublaslt_heuristic;
    int chosen_algo_id;
    uint64_t chosen_workspace_bytes;
    int iters;
    int ok;
    int err_code;
};

}  // extern "C"

// Error-code conventions for `err_code` on the Rust side. Keep these
// in sync with the probe markdown.
constexpr int ERR_CUDA_ALLOC = 1;
constexpr int ERR_CUBLAS_INIT = 2;
constexpr int ERR_CUBLAS_GEMMEX = 3;
constexpr int ERR_CUBLASLT_INIT = 4;
constexpr int ERR_CUBLASLT_DESC = 5;
constexpr int ERR_CUBLASLT_HEURISTIC = 6;
constexpr int ERR_CUBLASLT_MATMUL = 7;

namespace {

// Helper: random BF16 fill on-device using a deterministic pattern.
// We don't need real Gaussian inputs for timing — repeated bit-patterns
// match the BLAS path's compute cost.
__global__ void fill_bf16_pattern_kernel(__nv_bfloat16* x, int n, uint32_t seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    // Linear congruential then quantize to BF16 around [-0.1, 0.1].
    uint32_t v = seed * (uint32_t)idx + 0x9E3779B9u;
    v = (v ^ (v >> 16)) * 0x85EBCA6Bu;
    float f = ((float)(v & 0xFFFF) - 32768.0f) / 327680.0f;
    x[idx] = __float2bfloat16(f);
}

cudaError_t fill_bf16_pattern(__nv_bfloat16* x, int n, uint32_t seed,
                              cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    fill_bf16_pattern_kernel<<<grid, block, 0, stream>>>(x, n, seed);
    return cudaGetLastError();
}

// Median of an unsorted vector of floats.
float median(std::vector<float>& v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

}  // anonymous namespace

extern "C" int kiln_blas_cublaslt_mlp_probe(int bt, int k, int n, int iters,
                                            ProbeResult* out) {
    *out = ProbeResult{};
    out->bt = bt;
    out->k = k;
    out->n = n;
    out->iters = iters;

    cudaStream_t stream;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        out->err_code = ERR_CUDA_ALLOC;
        return out->err_code;
    }

    // ------------------------------------------------------------------
    // Allocate BF16 A [bt, k], B [k, n], C [bt, n] on the device.
    // ------------------------------------------------------------------
    __nv_bfloat16 *dA = nullptr, *dB = nullptr, *dC = nullptr;
    size_t bytes_a = (size_t)bt * (size_t)k * sizeof(__nv_bfloat16);
    size_t bytes_b = (size_t)k * (size_t)n * sizeof(__nv_bfloat16);
    size_t bytes_c = (size_t)bt * (size_t)n * sizeof(__nv_bfloat16);
    if (cudaMalloc(&dA, bytes_a) != cudaSuccess ||
        cudaMalloc(&dB, bytes_b) != cudaSuccess ||
        cudaMalloc(&dC, bytes_c) != cudaSuccess) {
        out->err_code = ERR_CUDA_ALLOC;
        goto cleanup;
    }
    fill_bf16_pattern(dA, bt * k, 1u, stream);
    fill_bf16_pattern(dB, k * n, 2u, stream);
    cudaStreamSynchronize(stream);

    // ==================================================================
    // (1) `cublasGemmEx` baseline (CUBLAS_GEMM_DEFAULT_TENSOR_OP).
    // ==================================================================
    {
        cublasHandle_t blas;
        if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS) {
            out->err_code = ERR_CUBLAS_INIT;
            goto cleanup;
        }
        cublasSetStream(blas, stream);

        const float alpha_f = 1.0f, beta_f = 0.0f;

        // Warm-up.
        cublasStatus_t st = cublasGemmEx(
            blas, CUBLAS_OP_N, CUBLAS_OP_N,
            n, bt, k,
            &alpha_f,
            dB, CUDA_R_16BF, n,
            dA, CUDA_R_16BF, k,
            &beta_f,
            dC, CUDA_R_16BF, n,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        if (st != CUBLAS_STATUS_SUCCESS) {
            out->err_code = ERR_CUBLAS_GEMMEX;
            cublasDestroy(blas);
            goto cleanup;
        }
        cudaStreamSynchronize(stream);

        std::vector<float> times(iters);
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        for (int i = 0; i < iters; ++i) {
            cudaEventRecord(e0, stream);
            cublasGemmEx(
                blas, CUBLAS_OP_N, CUBLAS_OP_N,
                n, bt, k,
                &alpha_f,
                dB, CUDA_R_16BF, n,
                dA, CUDA_R_16BF, k,
                &beta_f,
                dC, CUDA_R_16BF, n,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            cudaEventRecord(e1, stream);
            cudaEventSynchronize(e1);
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, e0, e1);
            times[i] = ms;
        }
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        cublasDestroy(blas);
        out->ms_cublas_default = median(times);
    }

    // ==================================================================
    // (2) `cublasLtMatmul` with explicit heuristic + workspace.
    // ==================================================================
    {
        cublasLtHandle_t lt;
        if (cublasLtCreate(&lt) != CUBLAS_STATUS_SUCCESS) {
            out->err_code = ERR_CUBLASLT_INIT;
            goto cleanup;
        }

        // Build matmul descriptor: FP32 compute, BF16 scale type for
        // the alpha/beta is unsupported on Ampere; use FP32 alpha/beta
        // with FP32 compute. Operands are BF16.
        cublasLtMatmulDesc_t matmulDesc = nullptr;
        if (cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F)
            != CUBLAS_STATUS_SUCCESS) {
            out->err_code = ERR_CUBLASLT_DESC;
            cublasLtDestroy(lt);
            goto cleanup;
        }

        cublasOperation_t op_n = CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA,
                                       &op_n, sizeof(op_n));
        cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB,
                                       &op_n, sizeof(op_n));

        // Layouts. cublasLt's convention is column-major; the equivalent
        // of candle's row-major `[bt, k] @ [k, n] = [bt, n]` is the
        // column-major `[n, k] @ [k, bt] = [n, bt]` with operands swapped.
        cublasLtMatrixLayout_t aDesc = nullptr, bDesc = nullptr, cDesc = nullptr;
        cublasLtMatrixLayoutCreate(&aDesc, CUDA_R_16BF, n, k, n);
        cublasLtMatrixLayoutCreate(&bDesc, CUDA_R_16BF, k, bt, k);
        cublasLtMatrixLayoutCreate(&cDesc, CUDA_R_16BF, n, bt, n);

        // Preference: 32 MiB workspace cap (cublasLt will report what
        // it actually wants; we record that).
        cublasLtMatmulPreference_t pref = nullptr;
        cublasLtMatmulPreferenceCreate(&pref);
        uint64_t ws_limit = 32ull * 1024ull * 1024ull;
        cublasLtMatmulPreferenceSetAttribute(
            pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &ws_limit, sizeof(ws_limit));

        cublasLtMatmulHeuristicResult_t heuristic_result = {};
        int returned_results = 0;
        cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(
            lt, matmulDesc, aDesc, bDesc, cDesc, cDesc,
            pref, 1, &heuristic_result, &returned_results);
        if (st != CUBLAS_STATUS_SUCCESS || returned_results == 0) {
            out->err_code = ERR_CUBLASLT_HEURISTIC;
            cublasLtMatmulPreferenceDestroy(pref);
            cublasLtMatrixLayoutDestroy(aDesc);
            cublasLtMatrixLayoutDestroy(bDesc);
            cublasLtMatrixLayoutDestroy(cDesc);
            cublasLtMatmulDescDestroy(matmulDesc);
            cublasLtDestroy(lt);
            goto cleanup;
        }
        // Extract algo ID (best-effort; the API exposes a int via the
        // CUBLASLT_ALGO_CONFIG_ID attribute).
        int algo_id = -1;
        cublasLtMatmulAlgoConfigGetAttribute(
            &heuristic_result.algo, CUBLASLT_ALGO_CONFIG_ID,
            &algo_id, sizeof(algo_id), nullptr);

        size_t ws_bytes = heuristic_result.workspaceSize;
        out->chosen_algo_id = algo_id;
        out->chosen_workspace_bytes = (uint64_t)ws_bytes;

        void* d_ws = nullptr;
        if (ws_bytes > 0 && cudaMalloc(&d_ws, ws_bytes) != cudaSuccess) {
            out->err_code = ERR_CUDA_ALLOC;
            cublasLtMatmulPreferenceDestroy(pref);
            cublasLtMatrixLayoutDestroy(aDesc);
            cublasLtMatrixLayoutDestroy(bDesc);
            cublasLtMatrixLayoutDestroy(cDesc);
            cublasLtMatmulDescDestroy(matmulDesc);
            cublasLtDestroy(lt);
            goto cleanup;
        }

        const float alpha_f = 1.0f, beta_f = 0.0f;

        // Warm-up.
        st = cublasLtMatmul(
            lt, matmulDesc,
            &alpha_f,
            dB, aDesc,
            dA, bDesc,
            &beta_f,
            dC, cDesc,
            dC, cDesc,
            &heuristic_result.algo,
            d_ws, ws_bytes,
            stream);
        if (st != CUBLAS_STATUS_SUCCESS) {
            out->err_code = ERR_CUBLASLT_MATMUL;
            if (d_ws) cudaFree(d_ws);
            cublasLtMatmulPreferenceDestroy(pref);
            cublasLtMatrixLayoutDestroy(aDesc);
            cublasLtMatrixLayoutDestroy(bDesc);
            cublasLtMatrixLayoutDestroy(cDesc);
            cublasLtMatmulDescDestroy(matmulDesc);
            cublasLtDestroy(lt);
            goto cleanup;
        }
        cudaStreamSynchronize(stream);

        std::vector<float> times(iters);
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        for (int i = 0; i < iters; ++i) {
            cudaEventRecord(e0, stream);
            cublasLtMatmul(
                lt, matmulDesc,
                &alpha_f,
                dB, aDesc,
                dA, bDesc,
                &beta_f,
                dC, cDesc,
                dC, cDesc,
                &heuristic_result.algo,
                d_ws, ws_bytes,
                stream);
            cudaEventRecord(e1, stream);
            cudaEventSynchronize(e1);
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, e0, e1);
            times[i] = ms;
        }
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        if (d_ws) cudaFree(d_ws);
        cublasLtMatmulPreferenceDestroy(pref);
        cublasLtMatrixLayoutDestroy(aDesc);
        cublasLtMatrixLayoutDestroy(bDesc);
        cublasLtMatrixLayoutDestroy(cDesc);
        cublasLtMatmulDescDestroy(matmulDesc);
        cublasLtDestroy(lt);

        out->ms_cublaslt_heuristic = median(times);
    }

    out->ok = 1;

cleanup:
    if (dA) cudaFree(dA);
    if (dB) cudaFree(dB);
    if (dC) cudaFree(dC);
    cudaStreamDestroy(stream);
    return out->err_code;
}
