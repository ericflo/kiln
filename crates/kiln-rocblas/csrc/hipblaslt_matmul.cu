// hipBLASLt matmul executor for `kiln_rocblas::HipblasLtMatmulHandle`.
//
// Phase R.6 of #1082 — the ROCm analog of
// `crates/kiln-blas/csrc/cublaslt_matmul.cu`. Token-for-token port of
// the cuBLASLt executor onto hipBLASLt, which is a near-drop-in API.
// Exposes the same small C surface that the Rust side
// (`crates/kiln-rocblas/src/...`) drives via FFI.
//
// # Design
//
// - One `KilnHipblasLtCtx` per (device, thread of access). Holds the
//   `hipblasLtHandle_t`. Cheap to create; expensive to *churn* — keep
//   it alive for the program's lifetime.
//
// - Each `kiln_blas_hipblaslt_matmul` call takes pure device pointers
//   plus a request descriptor. No allocation inside this function —
//   the caller pre-allocates workspace via `WorkspacePool`'s policy.
//
// - The first call to a given `(shape, dtype, layout)` runs
//   `hipblasLtMatmulAlgoGetHeuristic` to pick an algo; subsequent calls
//   reuse the cached algo blob via `algo_blob_in`. The cache itself
//   lives in `kiln_rocblas::AlgoCache` on the Rust side.
//
// - Row-major in/row-major out is the kiln-tensor convention.
//   hipBLASLt is column-major; we emit the equivalent column-major
//   matmul `C^T = B^T @ A^T` by swapping the operands as in the
//   probe.
//
// - Compute type is `HIPBLAS_COMPUTE_32F` for BF16/F16 IO (matches
//   the F32 promotion idiom). F32 IO uses `HIPBLAS_COMPUTE_32F` as
//   well.
//
// # Determinism stance
//
// The same algo + same workspace + same stream produces bit-identical
// outputs. The serialized `hipblasLtMatmulAlgo_t` blob pins the algo
// selection so it is reproducible across calls.
//
// # Error codes
//
// Mirrored on the Rust side in the rocblas handle. Keep in sync. The
// numeric values match `cublaslt_matmul.cu` exactly so the shared
// `FfiError` mapping is identical across CUDA / ROCm.
//
// # hipBLASLt vs cuBLASLt deviations
//
// - Compute type: cuBLASLt's `CUBLAS_COMPUTE_32F` -> hipBLASLt's
//   `HIPBLAS_COMPUTE_32F` (`hipblasComputeType_t`, from
//   hipblas-common). (The R.6 brief tentatively named it
//   `HIPBLASLT_COMPUTE_F32`; that symbol does not exist in this ROCm
//   — the real name is `HIPBLAS_COMPUTE_32F`.)
// - Algo-config-id: hipBLASLt exposes no
//   `cublasLtMatmulAlgoConfigGetAttribute` / `CUBLASLT_ALGO_CONFIG_ID`
//   getter. `hipblasLtMatmulAlgo_t` is `{ uint8_t data[16]; size_t
//   max_workspace_bytes; }`. We derive a stable, reproducible
//   `chosen_algo_id` by hashing the opaque 16-byte `data` field (the
//   algo's identity) with FNV-1a. This is purely an observability /
//   cache-key value; correctness rides on the serialized blob, not the
//   id.

#include <hipblaslt/hipblaslt.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>

// ----------------------------------------------------------------------
// Error codes (negative on failure; 0 on success).
// Numeric values MUST match cublaslt_matmul.cu.
// ----------------------------------------------------------------------

extern "C" {

constexpr int KILN_BLAS_OK = 0;
constexpr int KILN_BLAS_ERR_CTX_CREATE = -1;
constexpr int KILN_BLAS_ERR_CTX_NULL = -2;
constexpr int KILN_BLAS_ERR_DESC_CREATE = -3;
constexpr int KILN_BLAS_ERR_LAYOUT_CREATE = -4;
constexpr int KILN_BLAS_ERR_PREFERENCE = -5;
constexpr int KILN_BLAS_ERR_HEURISTIC = -6;
constexpr int KILN_BLAS_ERR_MATMUL = -7;
constexpr int KILN_BLAS_ERR_UNSUPPORTED_DTYPE = -8;
constexpr int KILN_BLAS_ERR_UNSUPPORTED_EPILOGUE = -9;
constexpr int KILN_BLAS_ERR_INVALID_SHAPE = -10;
constexpr int KILN_BLAS_ERR_ALGO_DESERIALIZE = -11;
constexpr int KILN_BLAS_ERR_ALGO_BLOB_TOO_SMALL = -12;

// ----------------------------------------------------------------------
// Opaque context. One per (device, owning Rust handle).
// ----------------------------------------------------------------------

struct KilnHipblasLtCtx {
    hipblasLtHandle_t lt;
    int device_index;
};

// ----------------------------------------------------------------------
// Public C surface.
// ----------------------------------------------------------------------

// dtype codes — kept in sync with the rocblas handle.
constexpr int KILN_DTYPE_BF16 = 0;
constexpr int KILN_DTYPE_F16 = 1;
constexpr int KILN_DTYPE_F32 = 2;

// epilogue codes — kept in sync with backend_matmul.rs's Epilogue enum.
constexpr int KILN_EPI_IDENTITY = 0;
constexpr int KILN_EPI_BIAS = 1;
constexpr int KILN_EPI_RELU = 2;
constexpr int KILN_EPI_GELU = 3;
// SiLU is not a native hipBLASLt epilogue we route through; route
// through the kt-side activation kernel instead. We keep the code so
// the Rust side can validate that it falls back rather than silently
// producing a wrong result.
constexpr int KILN_EPI_SILU = 4;
constexpr int KILN_EPI_BIAS_SILU = 5;
constexpr int KILN_EPI_BIAS_GELU = 6;

// Request descriptor — pure data, no host pointers. Field names,
// order, and C types match cublaslt_matmul.cu's
// KilnCublasLtMatmulSpec exactly (only the struct NAME changes).
struct KilnHipblasLtMatmulSpec {
    int32_t m;
    int32_t n;
    int32_t k;
    int32_t dtype_in;        // KILN_DTYPE_*
    int32_t dtype_out;       // KILN_DTYPE_*
    int32_t a_transposed;    // 0 = N, non-zero = T
    int32_t b_transposed;    // 0 = N, non-zero = T
    int32_t epilogue;        // KILN_EPI_*
};

// Maximum bytes we'll serialize a hipBLASLt algo into. The
// `hipblasLtMatmulAlgo_t` struct is documented as trivially
// serializable; current size is ~24 bytes ({ uint8_t data[16]; size_t
// max_workspace_bytes; }). We give 256 to be safe (matches cuBLASLt).
[[maybe_unused]] constexpr uint64_t KILN_BLAS_ALGO_BLOB_MAX = 256;

// Create a context for the current device. Pass a non-null out param.
int kiln_blas_hipblaslt_ctx_create(KilnHipblasLtCtx** out_ctx);

// Destroy a context. Safe to call with a null pointer (no-op).
int kiln_blas_hipblaslt_ctx_destroy(KilnHipblasLtCtx* ctx);

// Run a matmul. All pointer args are device pointers; the caller is
// responsible for workspace allocation (use the heuristic's
// `chosen_workspace_bytes` on the first call to size it).
//
// Algo selection:
// - If `algo_blob_in_len > 0`, deserialize the algo from
//   `algo_blob_in` and use it directly (skips the heuristic search).
// - Otherwise, run `hipblasLtMatmulAlgoGetHeuristic` and serialize the
//   chosen algo into `algo_blob_out` (caller-provided buffer, at
//   least `KILN_BLAS_ALGO_BLOB_MAX` bytes).
//
// On success returns `KILN_BLAS_OK` (0). On error returns a negative
// error code.
int kiln_blas_hipblaslt_matmul(
    KilnHipblasLtCtx* ctx,
    hipStream_t stream,
    const KilnHipblasLtMatmulSpec* spec,
    const void* a_ptr,
    const void* b_ptr,
    void* c_ptr,
    const void* bias_ptr,
    float alpha,
    float beta,
    void* workspace_ptr,
    uint64_t workspace_bytes,
    const uint8_t* algo_blob_in,
    uint64_t algo_blob_in_len,
    uint8_t* algo_blob_out,
    uint64_t* algo_blob_out_len,
    int32_t* chosen_algo_id,
    uint64_t* chosen_workspace_bytes);

}  // extern "C"

// ----------------------------------------------------------------------
// Internal helpers
// ----------------------------------------------------------------------

namespace {

// Map kiln dtype code → hipBLASLt hipDataType.
bool resolve_hip_dtype(int kind, hipDataType* out) {
    switch (kind) {
        case KILN_DTYPE_BF16: *out = HIP_R_16BF; return true;
        case KILN_DTYPE_F16: *out = HIP_R_16F; return true;
        case KILN_DTYPE_F32: *out = HIP_R_32F; return true;
        default: return false;
    }
}

// Map kiln epilogue code → hipblasLtEpilogue_t. Returns false if the
// epilogue is not natively supported by hipBLASLt (e.g., SiLU — needs
// a kt-side post-op kernel).
bool resolve_hipblaslt_epilogue(int kind, hipblasLtEpilogue_t* out) {
    switch (kind) {
        case KILN_EPI_IDENTITY: *out = HIPBLASLT_EPILOGUE_DEFAULT; return true;
        case KILN_EPI_BIAS: *out = HIPBLASLT_EPILOGUE_BIAS; return true;
        case KILN_EPI_RELU: *out = HIPBLASLT_EPILOGUE_RELU; return true;
        case KILN_EPI_GELU: *out = HIPBLASLT_EPILOGUE_GELU; return true;
        case KILN_EPI_BIAS_GELU: *out = HIPBLASLT_EPILOGUE_GELU_BIAS; return true;
        // SiLU / BIAS_SILU are not routed natively — caller must lower
        // to a separate activation kernel and bias-add pass.
        case KILN_EPI_SILU:
        case KILN_EPI_BIAS_SILU:
        default:
            return false;
    }
}

// hipBLASLt has no algo-config-id getter (cuBLASLt's
// CUBLASLT_ALGO_CONFIG_ID). The algo's identity lives in its opaque
// 16-byte `data` field. We hash it with FNV-1a to produce a stable,
// reproducible, non-negative id for observability / cache keying.
int32_t algo_id_from_blob(const hipblasLtMatmulAlgo_t& algo) {
    uint32_t h = 2166136261u;  // FNV-1a offset basis
    const unsigned char* p =
        reinterpret_cast<const unsigned char*>(&algo.data[0]);
    for (size_t i = 0; i < sizeof(algo.data); ++i) {
        h ^= static_cast<uint32_t>(p[i]);
        h *= 16777619u;  // FNV-1a prime
    }
    // Mask to a non-negative int32 so it round-trips through the
    // `int*` out param the way the cuBLASLt config-id did.
    return static_cast<int32_t>(h & 0x7FFFFFFFu);
}

}  // anonymous namespace

// ----------------------------------------------------------------------
// Implementation
// ----------------------------------------------------------------------

extern "C" int kiln_blas_hipblaslt_ctx_create(KilnHipblasLtCtx** out_ctx) {
    if (out_ctx == nullptr) return KILN_BLAS_ERR_CTX_NULL;
    *out_ctx = nullptr;
    KilnHipblasLtCtx* ctx = new (std::nothrow) KilnHipblasLtCtx{};
    if (ctx == nullptr) return KILN_BLAS_ERR_CTX_CREATE;
    if (hipblasLtCreate(&ctx->lt) != HIPBLAS_STATUS_SUCCESS) {
        delete ctx;
        return KILN_BLAS_ERR_CTX_CREATE;
    }
    if (hipGetDevice(&ctx->device_index) != hipSuccess) {
        ctx->device_index = 0;
    }
    *out_ctx = ctx;
    return KILN_BLAS_OK;
}

extern "C" int kiln_blas_hipblaslt_ctx_destroy(KilnHipblasLtCtx* ctx) {
    if (ctx == nullptr) return KILN_BLAS_OK;
    hipblasLtDestroy(ctx->lt);
    delete ctx;
    return KILN_BLAS_OK;
}

extern "C" int kiln_blas_hipblaslt_matmul(
    KilnHipblasLtCtx* ctx,
    hipStream_t stream,
    const KilnHipblasLtMatmulSpec* spec,
    const void* a_ptr,
    const void* b_ptr,
    void* c_ptr,
    const void* bias_ptr,
    float alpha,
    float beta,
    void* workspace_ptr,
    uint64_t workspace_bytes,
    const uint8_t* algo_blob_in,
    uint64_t algo_blob_in_len,
    uint8_t* algo_blob_out,
    uint64_t* algo_blob_out_len,
    int32_t* chosen_algo_id,
    uint64_t* chosen_workspace_bytes) {

    if (ctx == nullptr) return KILN_BLAS_ERR_CTX_NULL;
    if (spec == nullptr) return KILN_BLAS_ERR_DESC_CREATE;
    if (spec->m <= 0 || spec->n <= 0 || spec->k <= 0) {
        return KILN_BLAS_ERR_INVALID_SHAPE;
    }

    hipDataType hip_dt_in, hip_dt_out;
    if (!resolve_hip_dtype(spec->dtype_in, &hip_dt_in)) {
        return KILN_BLAS_ERR_UNSUPPORTED_DTYPE;
    }
    if (!resolve_hip_dtype(spec->dtype_out, &hip_dt_out)) {
        return KILN_BLAS_ERR_UNSUPPORTED_DTYPE;
    }

    hipblasLtEpilogue_t lt_epi;
    if (!resolve_hipblaslt_epilogue(spec->epilogue, &lt_epi)) {
        return KILN_BLAS_ERR_UNSUPPORTED_EPILOGUE;
    }

    // Build the matmul descriptor. HIPBLAS_COMPUTE_32F + HIP_R_32F
    // scale type works for all our dtypes (BF16/F16/F32 IO).
    hipblasLtMatmulDesc_t matmul_desc = nullptr;
    if (hipblasLtMatmulDescCreate(&matmul_desc, HIPBLAS_COMPUTE_32F, HIP_R_32F)
        != HIPBLAS_STATUS_SUCCESS) {
        return KILN_BLAS_ERR_DESC_CREATE;
    }

    int ret = KILN_BLAS_OK;
    hipblasLtMatrixLayout_t a_desc = nullptr, b_desc = nullptr, c_desc = nullptr;
    hipblasLtMatmulPreference_t pref = nullptr;

    // Row-major in/out — convert to the equivalent column-major form
    // by swapping operands: C[m, n] = A[m, k] @ B[k, n] (row-major)
    // is equivalent to C_col[n, m] = B_col[n, k] @ A_col[k, m] where
    // X_col is X viewed column-major. (Identical to the cuBLASLt
    // path — hipBLASLt shares the column-major convention.)
    //
    // Transposes apply *in row-major space*: a_transposed=true means
    // A was passed as [k, m] in row-major, but we view it as [m, k]
    // logically; in column-major that's [m, k] (so we use OP_T).
    hipblasOperation_t op_a = spec->a_transposed ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    hipblasOperation_t op_b = spec->b_transposed ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    // The "primary operand" in our column-major view is B (because
    // we swapped). hipBLASLt's TRANSA / TRANSB attributes refer to its
    // first / second arguments, which after the swap are B / A.
    hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSA,
                                    &op_b, sizeof(op_b));
    hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSB,
                                    &op_a, sizeof(op_a));

    // Set the epilogue.
    hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_EPILOGUE,
                                    &lt_epi, sizeof(lt_epi));

    // Bias is keyed off the epilogue. hipBLASLt expects a per-column
    // (== per-N) bias vector with the output dtype.
    if (bias_ptr != nullptr &&
        (spec->epilogue == KILN_EPI_BIAS ||
         spec->epilogue == KILN_EPI_BIAS_GELU)) {
        hipblasLtMatmulDescSetAttribute(
            matmul_desc, HIPBLASLT_MATMUL_DESC_BIAS_POINTER,
            &bias_ptr, sizeof(bias_ptr));
        // Bias dtype defaults to output dtype.
    }

    // Layouts. After the operand swap, the first hipBLASLt operand is
    // B and the second is A. Column-major leading dims mirror the
    // cuBLASLt path exactly:
    //   B_layout = [n, k, n]  (rows=n, cols=k, ld=n)  — used as "A" in hipBLASLt
    //   A_layout = [k, m, k]  (rows=k, cols=m, ld=k)  — used as "B" in hipBLASLt
    //   C_layout = [n, m, n]
    // with op_a = op_b = HIPBLAS_OP_N for the non-transposed case.
    // For the transposed cases, swap rows/cols on the layout AND set
    // op_a or op_b to T.
    int64_t b_rows = spec->b_transposed ? spec->k : spec->n;
    int64_t b_cols = spec->b_transposed ? spec->n : spec->k;
    int64_t b_ld = b_rows;
    int64_t a_rows = spec->a_transposed ? spec->m : spec->k;
    int64_t a_cols = spec->a_transposed ? spec->k : spec->m;
    int64_t a_ld = a_rows;
    int64_t c_rows = spec->n;
    int64_t c_cols = spec->m;
    int64_t c_ld = c_rows;

    if (hipblasLtMatrixLayoutCreate(&b_desc, hip_dt_in, b_rows, b_cols, b_ld)
        != HIPBLAS_STATUS_SUCCESS ||
        hipblasLtMatrixLayoutCreate(&a_desc, hip_dt_in, a_rows, a_cols, a_ld)
        != HIPBLAS_STATUS_SUCCESS ||
        hipblasLtMatrixLayoutCreate(&c_desc, hip_dt_out, c_rows, c_cols, c_ld)
        != HIPBLAS_STATUS_SUCCESS) {
        ret = KILN_BLAS_ERR_LAYOUT_CREATE;
        goto cleanup;
    }

    // Preference: limit workspace to caller's budget.
    {
        if (hipblasLtMatmulPreferenceCreate(&pref) != HIPBLAS_STATUS_SUCCESS) {
            ret = KILN_BLAS_ERR_PREFERENCE;
            goto cleanup;
        }
        hipblasLtMatmulPreferenceSetAttribute(
            pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &workspace_bytes, sizeof(workspace_bytes));
    }

    {
        hipblasLtMatmulAlgo_t algo;
        bool have_algo = false;
        uint64_t ws_actual = 0;
        int32_t algo_id_out = -1;

        if (algo_blob_in != nullptr && algo_blob_in_len > 0) {
            if (algo_blob_in_len < sizeof(hipblasLtMatmulAlgo_t)) {
                ret = KILN_BLAS_ERR_ALGO_DESERIALIZE;
                goto cleanup;
            }
            std::memcpy(&algo, algo_blob_in, sizeof(hipblasLtMatmulAlgo_t));
            have_algo = true;
            // We don't know the workspace bytes for the cached algo
            // without re-querying; trust the caller's workspace_bytes.
            ws_actual = workspace_bytes;
            algo_id_out = algo_id_from_blob(algo);
        }

        if (!have_algo) {
            hipblasLtMatmulHeuristicResult_t h_result = {};
            int returned = 0;
            hipblasStatus_t hst = hipblasLtMatmulAlgoGetHeuristic(
                ctx->lt, matmul_desc, b_desc, a_desc, c_desc, c_desc,
                pref, 1, &h_result, &returned);
            if (hst != HIPBLAS_STATUS_SUCCESS || returned == 0) {
                ret = KILN_BLAS_ERR_HEURISTIC;
                goto cleanup;
            }
            algo = h_result.algo;
            ws_actual = h_result.workspaceSize;
            algo_id_out = algo_id_from_blob(algo);

            if (algo_blob_out != nullptr && algo_blob_out_len != nullptr) {
                if (*algo_blob_out_len < sizeof(hipblasLtMatmulAlgo_t)) {
                    ret = KILN_BLAS_ERR_ALGO_BLOB_TOO_SMALL;
                    goto cleanup;
                }
                std::memcpy(algo_blob_out, &algo, sizeof(hipblasLtMatmulAlgo_t));
                *algo_blob_out_len = sizeof(hipblasLtMatmulAlgo_t);
            } else if (algo_blob_out_len != nullptr) {
                *algo_blob_out_len = 0;
            }
        }

        if (chosen_algo_id != nullptr) *chosen_algo_id = algo_id_out;
        if (chosen_workspace_bytes != nullptr) *chosen_workspace_bytes = ws_actual;

        // Reject if the algo's workspace exceeds the caller's budget.
        if (ws_actual > workspace_bytes && workspace_bytes > 0) {
            ret = KILN_BLAS_ERR_PREFERENCE;
            goto cleanup;
        }

        hipblasStatus_t st = hipblasLtMatmul(
            ctx->lt, matmul_desc,
            &alpha,
            b_ptr, b_desc,
            a_ptr, a_desc,
            &beta,
            c_ptr, c_desc,
            c_ptr, c_desc,
            &algo,
            workspace_ptr, workspace_bytes,
            stream);
        if (st != HIPBLAS_STATUS_SUCCESS) {
            ret = KILN_BLAS_ERR_MATMUL;
            goto cleanup;
        }
    }

cleanup:
    if (pref) hipblasLtMatmulPreferenceDestroy(pref);
    if (a_desc) hipblasLtMatrixLayoutDestroy(a_desc);
    if (b_desc) hipblasLtMatrixLayoutDestroy(b_desc);
    if (c_desc) hipblasLtMatrixLayoutDestroy(c_desc);
    if (matmul_desc) hipblasLtMatmulDescDestroy(matmul_desc);
    return ret;
}
