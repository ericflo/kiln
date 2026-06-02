// Phase R.6 — hipBLASLt vs hipblas-default MLP gate||up probe.
//
// ROCm analog of `crates/kiln-blas/csrc/cublaslt_probe.cu`. Compares
// the BF16-input BF16-output FP32-compute matmul at the Qwen3.5-4B
// MLP gate||up shape `[B*T, 2560] @ [2560, 18432]` between:
//
//   (1) `hipblasLtMatmul` with the implicit heuristic (algo == nullptr,
//       default search preferences) — the ROCm analog of the cuBLAS
//       CUBLAS_GEMM_DEFAULT_TENSOR_OP "default" baseline.
//   (2) `hipblasLtMatmul` with explicit `hipblasLtMatmulAlgoGetHeuristic`
//       algo selection + an explicit workspace.
//
// The probe times each path with HIP events, medians over `iters`
// iterations, and reports per-shape ms + the chosen algo ID + the
// workspace bytes selected by the heuristic.
//
// Builds via crates/kiln-rocblas/build.rs and links hipBLASLt + the
// HIP runtime only (no legacy hipblas — see the baseline note below).
//
// # hipBLASLt vs cuBLASLt deviations
//
// - Baseline: cuBLAS's *legacy* `cublasGemmEx` +
//   `CUBLAS_GEMM_DEFAULT_TENSOR_OP` lives in libcublas. Its hipBLAS
//   analog (`hipblasGemmEx`) lives in libhipblas, which the
//   kiln-rocblas build.rs does NOT link (only hipblaslt + amdhip64 +
//   stdc++) and which the R.6 link smoke test does not provide.
//   To stay on exactly the `-lhipblaslt -lamdhip64` link surface we
//   use the hipBLASLt implicit-heuristic path (algo == nullptr) as the
//   "default" baseline instead.
// - Compute type: `CUBLAS_COMPUTE_32F` -> `HIPBLAS_COMPUTE_32F`.
// - Algo-config-id: hipBLASLt has no `CUBLASLT_ALGO_CONFIG_ID` getter;
//   we hash the opaque 16-byte `hipblasLtMatmulAlgo_t::data` field
//   (FNV-1a) into a stable non-negative id, same as hipblaslt_matmul.cu.

#include <hipblaslt/hipblaslt.h>
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

// Public result struct mirrored on the Rust side in
// `probe_ffi::ProbeResult`. Field names / order / types match
// cublaslt_probe.cu exactly.
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

// Error-code conventions for `err_code` on the Rust side. Numeric
// values match cublaslt_probe.cu.
constexpr int ERR_CUDA_ALLOC = 1;
constexpr int ERR_CUBLAS_INIT = 2;
constexpr int ERR_CUBLAS_GEMMEX = 3;
constexpr int ERR_CUBLASLT_INIT = 4;
constexpr int ERR_CUBLASLT_DESC = 5;
constexpr int ERR_CUBLASLT_HEURISTIC = 6;
constexpr int ERR_CUBLASLT_MATMUL = 7;

namespace {

// Helper: deterministic BF16 fill on-device. We don't need real
// Gaussian inputs for timing — repeated bit-patterns match the BLAS
// path's compute cost.
__global__ void fill_bf16_pattern_kernel(__hip_bfloat16* x, int n, uint32_t seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    // Linear congruential then quantize to BF16 around [-0.1, 0.1].
    uint32_t v = seed * (uint32_t)idx + 0x9E3779B9u;
    v = (v ^ (v >> 16)) * 0x85EBCA6Bu;
    float f = ((float)(v & 0xFFFF) - 32768.0f) / 327680.0f;
    x[idx] = __float2bfloat16(f);
}

hipError_t fill_bf16_pattern(__hip_bfloat16* x, int n, uint32_t seed,
                             hipStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    fill_bf16_pattern_kernel<<<grid, block, 0, stream>>>(x, n, seed);
    return hipGetLastError();
}

// Median of an unsorted vector of floats.
float median(std::vector<float>& v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// hipBLASLt has no algo-config-id getter. Hash the opaque 16-byte algo
// `data` field with FNV-1a for a stable, reproducible non-negative id.
int algo_id_from_blob(const hipblasLtMatmulAlgo_t& algo) {
    uint32_t h = 2166136261u;
    const unsigned char* p =
        reinterpret_cast<const unsigned char*>(&algo.data[0]);
    for (size_t i = 0; i < sizeof(algo.data); ++i) {
        h ^= static_cast<uint32_t>(p[i]);
        h *= 16777619u;
    }
    return static_cast<int>(h & 0x7FFFFFFFu);
}

}  // anonymous namespace

extern "C" int kiln_blas_hipblaslt_mlp_probe(int bt, int k, int n, int iters,
                                             ProbeResult* out) {
    *out = ProbeResult{};
    out->bt = bt;
    out->k = k;
    out->n = n;
    out->iters = iters;

    hipStream_t stream;
    if (hipStreamCreate(&stream) != hipSuccess) {
        out->err_code = ERR_CUDA_ALLOC;
        return out->err_code;
    }

    // ------------------------------------------------------------------
    // Allocate BF16 A [bt, k], B [k, n], C [bt, n] on the device.
    // ------------------------------------------------------------------
    __hip_bfloat16 *dA = nullptr, *dB = nullptr, *dC = nullptr;
    size_t bytes_a = (size_t)bt * (size_t)k * sizeof(__hip_bfloat16);
    size_t bytes_b = (size_t)k * (size_t)n * sizeof(__hip_bfloat16);
    size_t bytes_c = (size_t)bt * (size_t)n * sizeof(__hip_bfloat16);
    if (hipMalloc(&dA, bytes_a) != hipSuccess ||
        hipMalloc(&dB, bytes_b) != hipSuccess ||
        hipMalloc(&dC, bytes_c) != hipSuccess) {
        out->err_code = ERR_CUDA_ALLOC;
        goto cleanup;
    }
    fill_bf16_pattern(dA, bt * k, 1u, stream);
    fill_bf16_pattern(dB, k * n, 2u, stream);
    hipStreamSynchronize(stream);

    // ==================================================================
    // (1) hipBLASLt "default" baseline — `hipblasLtMatmul` with the
    //     implicit heuristic (`algo == nullptr`, default search
    //     preferences), the ROCm analog of the cuBLAS
    //     CUBLAS_GEMM_DEFAULT_TENSOR_OP baseline.
    //
    //     NOTE (deviation): the cuBLASLt exemplar timed `cublasGemmEx`
    //     from the *legacy* cublas library. The hipBLAS analog
    //     (`hipblasGemmEx`) lives in `libhipblas`, which the
    //     kiln-rocblas build.rs does NOT link (it links only
    //     hipblaslt + amdhip64 + stdc++, and the R.6 link smoke test
    //     uses exactly `-lhipblaslt -lamdhip64`). To keep the probe
    //     self-contained against that link surface we use the
    //     hipBLASLt implicit-heuristic path as the "default" baseline
    //     instead. `ms_cublas_default` is still populated as the
    //     default-path reference number.
    // ==================================================================
    {
        hipblasLtHandle_t lt_d;
        if (hipblasLtCreate(&lt_d) != HIPBLAS_STATUS_SUCCESS) {
            out->err_code = ERR_CUBLAS_INIT;
            goto cleanup;
        }

        hipblasLtMatmulDesc_t descD = nullptr;
        if (hipblasLtMatmulDescCreate(&descD, HIPBLAS_COMPUTE_32F, HIP_R_32F)
            != HIPBLAS_STATUS_SUCCESS) {
            out->err_code = ERR_CUBLAS_GEMMEX;
            hipblasLtDestroy(lt_d);
            goto cleanup;
        }
        hipblasOperation_t op_nd = HIPBLAS_OP_N;
        hipblasLtMatmulDescSetAttribute(descD, HIPBLASLT_MATMUL_DESC_TRANSA,
                                        &op_nd, sizeof(op_nd));
        hipblasLtMatmulDescSetAttribute(descD, HIPBLASLT_MATMUL_DESC_TRANSB,
                                        &op_nd, sizeof(op_nd));

        hipblasLtMatrixLayout_t aD = nullptr, bD = nullptr, cD = nullptr;
        hipblasLtMatrixLayoutCreate(&aD, HIP_R_16BF, n, k, n);
        hipblasLtMatrixLayoutCreate(&bD, HIP_R_16BF, k, bt, k);
        hipblasLtMatrixLayoutCreate(&cD, HIP_R_16BF, n, bt, n);

        const float alpha_f = 1.0f, beta_f = 0.0f;

        // Warm-up with the implicit heuristic (algo == nullptr). Give
        // a 32 MiB workspace so the default path can pick a tiled algo.
        uint64_t ws_def = 32ull * 1024ull * 1024ull;
        void* d_ws_def = nullptr;
        if (hipMalloc(&d_ws_def, ws_def) != hipSuccess) {
            d_ws_def = nullptr;
            ws_def = 0;
        }
        hipblasStatus_t st = hipblasLtMatmul(
            lt_d, descD,
            &alpha_f,
            dB, aD,
            dA, bD,
            &beta_f,
            dC, cD,
            dC, cD,
            nullptr,  // implicit heuristic — "default" algo selection
            d_ws_def, ws_def,
            stream);
        if (st != HIPBLAS_STATUS_SUCCESS) {
            out->err_code = ERR_CUBLAS_GEMMEX;
            if (d_ws_def) hipFree(d_ws_def);
            hipblasLtMatrixLayoutDestroy(aD);
            hipblasLtMatrixLayoutDestroy(bD);
            hipblasLtMatrixLayoutDestroy(cD);
            hipblasLtMatmulDescDestroy(descD);
            hipblasLtDestroy(lt_d);
            goto cleanup;
        }
        hipStreamSynchronize(stream);

        std::vector<float> times(iters);
        hipEvent_t e0, e1;
        hipEventCreate(&e0);
        hipEventCreate(&e1);
        for (int i = 0; i < iters; ++i) {
            hipEventRecord(e0, stream);
            hipblasLtMatmul(
                lt_d, descD,
                &alpha_f,
                dB, aD,
                dA, bD,
                &beta_f,
                dC, cD,
                dC, cD,
                nullptr,
                d_ws_def, ws_def,
                stream);
            hipEventRecord(e1, stream);
            hipEventSynchronize(e1);
            float ms = 0.0f;
            hipEventElapsedTime(&ms, e0, e1);
            times[i] = ms;
        }
        hipEventDestroy(e0);
        hipEventDestroy(e1);
        if (d_ws_def) hipFree(d_ws_def);
        hipblasLtMatrixLayoutDestroy(aD);
        hipblasLtMatrixLayoutDestroy(bD);
        hipblasLtMatrixLayoutDestroy(cD);
        hipblasLtMatmulDescDestroy(descD);
        hipblasLtDestroy(lt_d);
        out->ms_cublas_default = median(times);
    }

    // ==================================================================
    // (2) `hipblasLtMatmul` with explicit heuristic + workspace.
    // ==================================================================
    {
        hipblasLtHandle_t lt;
        if (hipblasLtCreate(&lt) != HIPBLAS_STATUS_SUCCESS) {
            out->err_code = ERR_CUBLASLT_INIT;
            goto cleanup;
        }

        // Build matmul descriptor: FP32 compute, FP32 scale type for
        // alpha/beta. Operands are BF16.
        hipblasLtMatmulDesc_t matmulDesc = nullptr;
        if (hipblasLtMatmulDescCreate(&matmulDesc, HIPBLAS_COMPUTE_32F, HIP_R_32F)
            != HIPBLAS_STATUS_SUCCESS) {
            out->err_code = ERR_CUBLASLT_DESC;
            hipblasLtDestroy(lt);
            goto cleanup;
        }

        hipblasOperation_t op_n = HIPBLAS_OP_N;
        hipblasLtMatmulDescSetAttribute(matmulDesc, HIPBLASLT_MATMUL_DESC_TRANSA,
                                        &op_n, sizeof(op_n));
        hipblasLtMatmulDescSetAttribute(matmulDesc, HIPBLASLT_MATMUL_DESC_TRANSB,
                                        &op_n, sizeof(op_n));

        // Layouts. hipBLASLt's convention is column-major; the
        // equivalent of the row-major `[bt, k] @ [k, n] = [bt, n]` is
        // the column-major `[n, k] @ [k, bt] = [n, bt]` with operands
        // swapped.
        hipblasLtMatrixLayout_t aDesc = nullptr, bDesc = nullptr, cDesc = nullptr;
        hipblasLtMatrixLayoutCreate(&aDesc, HIP_R_16BF, n, k, n);
        hipblasLtMatrixLayoutCreate(&bDesc, HIP_R_16BF, k, bt, k);
        hipblasLtMatrixLayoutCreate(&cDesc, HIP_R_16BF, n, bt, n);

        // Preference: 32 MiB workspace cap (hipBLASLt will report what
        // it actually wants; we record that).
        hipblasLtMatmulPreference_t pref = nullptr;
        hipblasLtMatmulPreferenceCreate(&pref);
        uint64_t ws_limit = 32ull * 1024ull * 1024ull;
        hipblasLtMatmulPreferenceSetAttribute(
            pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &ws_limit, sizeof(ws_limit));

        hipblasLtMatmulHeuristicResult_t heuristic_result = {};
        int returned_results = 0;
        hipblasStatus_t st = hipblasLtMatmulAlgoGetHeuristic(
            lt, matmulDesc, aDesc, bDesc, cDesc, cDesc,
            pref, 1, &heuristic_result, &returned_results);
        if (st != HIPBLAS_STATUS_SUCCESS || returned_results == 0) {
            out->err_code = ERR_CUBLASLT_HEURISTIC;
            hipblasLtMatmulPreferenceDestroy(pref);
            hipblasLtMatrixLayoutDestroy(aDesc);
            hipblasLtMatrixLayoutDestroy(bDesc);
            hipblasLtMatrixLayoutDestroy(cDesc);
            hipblasLtMatmulDescDestroy(matmulDesc);
            hipblasLtDestroy(lt);
            goto cleanup;
        }
        // Extract a stable algo ID by hashing the opaque algo blob
        // (hipBLASLt exposes no config-id getter).
        int algo_id = algo_id_from_blob(heuristic_result.algo);

        size_t ws_bytes = heuristic_result.workspaceSize;
        out->chosen_algo_id = algo_id;
        out->chosen_workspace_bytes = (uint64_t)ws_bytes;

        void* d_ws = nullptr;
        if (ws_bytes > 0 && hipMalloc(&d_ws, ws_bytes) != hipSuccess) {
            out->err_code = ERR_CUDA_ALLOC;
            hipblasLtMatmulPreferenceDestroy(pref);
            hipblasLtMatrixLayoutDestroy(aDesc);
            hipblasLtMatrixLayoutDestroy(bDesc);
            hipblasLtMatrixLayoutDestroy(cDesc);
            hipblasLtMatmulDescDestroy(matmulDesc);
            hipblasLtDestroy(lt);
            goto cleanup;
        }

        const float alpha_f = 1.0f, beta_f = 0.0f;

        // Warm-up.
        st = hipblasLtMatmul(
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
        if (st != HIPBLAS_STATUS_SUCCESS) {
            out->err_code = ERR_CUBLASLT_MATMUL;
            if (d_ws) hipFree(d_ws);
            hipblasLtMatmulPreferenceDestroy(pref);
            hipblasLtMatrixLayoutDestroy(aDesc);
            hipblasLtMatrixLayoutDestroy(bDesc);
            hipblasLtMatrixLayoutDestroy(cDesc);
            hipblasLtMatmulDescDestroy(matmulDesc);
            hipblasLtDestroy(lt);
            goto cleanup;
        }
        hipStreamSynchronize(stream);

        std::vector<float> times(iters);
        hipEvent_t e0, e1;
        hipEventCreate(&e0);
        hipEventCreate(&e1);
        for (int i = 0; i < iters; ++i) {
            hipEventRecord(e0, stream);
            hipblasLtMatmul(
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
            hipEventRecord(e1, stream);
            hipEventSynchronize(e1);
            float ms = 0.0f;
            hipEventElapsedTime(&ms, e0, e1);
            times[i] = ms;
        }
        hipEventDestroy(e0);
        hipEventDestroy(e1);
        if (d_ws) hipFree(d_ws);
        hipblasLtMatmulPreferenceDestroy(pref);
        hipblasLtMatrixLayoutDestroy(aDesc);
        hipblasLtMatrixLayoutDestroy(bDesc);
        hipblasLtMatrixLayoutDestroy(cDesc);
        hipblasLtMatmulDescDestroy(matmulDesc);
        hipblasLtDestroy(lt);

        out->ms_cublaslt_heuristic = median(times);
    }

    out->ok = 1;

cleanup:
    if (dA) hipFree(dA);
    if (dB) hipFree(dB);
    if (dC) hipFree(dC);
    hipStreamDestroy(stream);
    return out->err_code;
}
