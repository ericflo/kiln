// cublasLt matmul executor for `kiln_blas::CublasLtMatmulHandle`.
//
// Phase 2.x of #1082 — lifts the Phase 0.8 probe into a reusable
// matmul executor. Exposes a small C surface that the Rust side
// (`crates/kiln-blas/src/cublaslt_handle.rs`) drives via FFI.
//
// # Design
//
// - One `KilnCublasLtCtx` per (device, thread of access). Holds the
//   `cublasLtHandle_t`. Cheap to create; expensive to *churn* — keep
//   it alive for the program's lifetime.
//
// - Each `kiln_blas_cublaslt_matmul` call takes pure device pointers
//   plus a request descriptor. No allocation inside this function —
//   the caller pre-allocates workspace via `WorkspacePool`'s policy.
//
// - The first call to a given `(shape, dtype, layout)` runs
//   `cublasLtMatmulAlgoGetHeuristic` to pick an algo; subsequent calls
//   reuse the cached algo blob via `algo_blob_in`. The cache itself
//   lives in `kiln_blas::AlgoCache` on the Rust side.
//
// - Row-major in/row-major out is the kiln-tensor convention.
//   cublasLt is column-major; we emit the equivalent column-major
//   matmul `C^T = B^T @ A^T` by swapping the operands as in the
//   probe.
//
// - Compute type is `CUBLAS_COMPUTE_32F` for BF16/F16 IO (matches
//   `forward.rs:3454,3517`'s F32 promotion idiom). F32 IO uses
//   `CUBLAS_COMPUTE_32F` as well.
//
// # Determinism stance
//
// The same algo + same workspace + same stream produces bit-identical
// outputs. The `CUBLAS_WORKSPACE_CONFIG=:4096:8` env var (Phase 0
// determinism stance) constrains cublasLt's workspace partitioning so
// the algo selection is reproducible.
//
// # Error codes
//
// Mirrored on the Rust side in `cublaslt_handle::FfiError`. Keep in
// sync.

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>

// ----------------------------------------------------------------------
// Error codes (negative on failure; 0 on success).
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

struct KilnCublasLtCtx {
    cublasLtHandle_t lt;
    int device_index;
};

// ----------------------------------------------------------------------
// Public C surface.
// ----------------------------------------------------------------------

// dtype codes — kept in sync with cublaslt_handle.rs.
constexpr int KILN_DTYPE_BF16 = 0;
constexpr int KILN_DTYPE_F16 = 1;
constexpr int KILN_DTYPE_F32 = 2;

// epilogue codes — kept in sync with backend_matmul.rs's Epilogue enum.
constexpr int KILN_EPI_IDENTITY = 0;
constexpr int KILN_EPI_BIAS = 1;
constexpr int KILN_EPI_RELU = 2;
constexpr int KILN_EPI_GELU = 3;
// SiLU is not a native cublasLt epilogue; route through the kt-side
// activation kernel instead. We keep the code so the Rust side can
// validate that it falls back rather than silently producing a wrong
// result.
constexpr int KILN_EPI_SILU = 4;
constexpr int KILN_EPI_BIAS_SILU = 5;
constexpr int KILN_EPI_BIAS_GELU = 6;

// Request descriptor — pure data, no host pointers.
struct KilnCublasLtMatmulSpec {
    int32_t m;
    int32_t n;
    int32_t k;
    int32_t dtype_in;        // KILN_DTYPE_*
    int32_t dtype_out;       // KILN_DTYPE_*
    int32_t a_transposed;    // 0 = N, non-zero = T
    int32_t b_transposed;    // 0 = N, non-zero = T
    int32_t epilogue;        // KILN_EPI_*
};

// Maximum bytes we'll serialize a cublasLt algo into. The
// `cublasLtMatmulAlgo_t` struct is documented as opaque + ABI-stable
// across CUDA minor versions; current size is ~80 bytes. We give 256
// to be safe.
[[maybe_unused]] constexpr uint64_t KILN_BLAS_ALGO_BLOB_MAX = 256;

// Create a context for the current device. Pass a non-null out param.
int kiln_blas_cublaslt_ctx_create(KilnCublasLtCtx** out_ctx);

// Destroy a context. Safe to call with a null pointer (no-op).
int kiln_blas_cublaslt_ctx_destroy(KilnCublasLtCtx* ctx);

// Run a matmul. All pointer args are device pointers; the caller is
// responsible for workspace allocation (use the heuristic's
// `chosen_workspace_bytes` on the first call to size it).
//
// Algo selection:
// - If `algo_blob_in_len > 0`, deserialize the algo from
//   `algo_blob_in` and use it directly (skips the heuristic search).
// - Otherwise, run `cublasLtMatmulAlgoGetHeuristic` and serialize the
//   chosen algo into `algo_blob_out` (caller-provided buffer, at
//   least `KILN_BLAS_ALGO_BLOB_MAX` bytes).
//
// On success returns `KILN_BLAS_OK` (0). On error returns a negative
// error code.
int kiln_blas_cublaslt_matmul(
    KilnCublasLtCtx* ctx,
    cudaStream_t stream,
    const KilnCublasLtMatmulSpec* spec,
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

// Map kiln dtype code → cublasLt cudaDataType_t.
bool resolve_cuda_dtype(int kind, cudaDataType_t* out) {
    switch (kind) {
        case KILN_DTYPE_BF16: *out = CUDA_R_16BF; return true;
        case KILN_DTYPE_F16: *out = CUDA_R_16F; return true;
        case KILN_DTYPE_F32: *out = CUDA_R_32F; return true;
        default: return false;
    }
}

// Map kiln epilogue code → cublasLtEpilogue_t. Returns false if the
// epilogue is not natively supported by cublasLt (e.g., SiLU — needs
// a kt-side post-op kernel).
bool resolve_cublaslt_epilogue(int kind, cublasLtEpilogue_t* out) {
    switch (kind) {
        case KILN_EPI_IDENTITY: *out = CUBLASLT_EPILOGUE_DEFAULT; return true;
        case KILN_EPI_BIAS: *out = CUBLASLT_EPILOGUE_BIAS; return true;
        case KILN_EPI_RELU: *out = CUBLASLT_EPILOGUE_RELU; return true;
        case KILN_EPI_GELU: *out = CUBLASLT_EPILOGUE_GELU; return true;
        case KILN_EPI_BIAS_GELU: *out = CUBLASLT_EPILOGUE_GELU_BIAS; return true;
        // SiLU / BIAS_SILU are not native — caller must lower to a
        // separate activation kernel and bias-add pass.
        case KILN_EPI_SILU:
        case KILN_EPI_BIAS_SILU:
        default:
            return false;
    }
}

}  // anonymous namespace

// ----------------------------------------------------------------------
// Implementation
// ----------------------------------------------------------------------

extern "C" int kiln_blas_cublaslt_ctx_create(KilnCublasLtCtx** out_ctx) {
    if (out_ctx == nullptr) return KILN_BLAS_ERR_CTX_NULL;
    *out_ctx = nullptr;
    KilnCublasLtCtx* ctx = new (std::nothrow) KilnCublasLtCtx{};
    if (ctx == nullptr) return KILN_BLAS_ERR_CTX_CREATE;
    if (cublasLtCreate(&ctx->lt) != CUBLAS_STATUS_SUCCESS) {
        delete ctx;
        return KILN_BLAS_ERR_CTX_CREATE;
    }
    if (cudaGetDevice(&ctx->device_index) != cudaSuccess) {
        ctx->device_index = 0;
    }
    *out_ctx = ctx;
    return KILN_BLAS_OK;
}

extern "C" int kiln_blas_cublaslt_ctx_destroy(KilnCublasLtCtx* ctx) {
    if (ctx == nullptr) return KILN_BLAS_OK;
    cublasLtDestroy(ctx->lt);
    delete ctx;
    return KILN_BLAS_OK;
}

extern "C" int kiln_blas_cublaslt_matmul(
    KilnCublasLtCtx* ctx,
    cudaStream_t stream,
    const KilnCublasLtMatmulSpec* spec,
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

    cudaDataType_t cuda_dt_in, cuda_dt_out;
    if (!resolve_cuda_dtype(spec->dtype_in, &cuda_dt_in)) {
        return KILN_BLAS_ERR_UNSUPPORTED_DTYPE;
    }
    if (!resolve_cuda_dtype(spec->dtype_out, &cuda_dt_out)) {
        return KILN_BLAS_ERR_UNSUPPORTED_DTYPE;
    }

    cublasLtEpilogue_t lt_epi;
    if (!resolve_cublaslt_epilogue(spec->epilogue, &lt_epi)) {
        return KILN_BLAS_ERR_UNSUPPORTED_EPILOGUE;
    }

    // Build the matmul descriptor. CUBLAS_COMPUTE_32F + CUDA_R_32F
    // scale type works for all our dtypes (BF16/F16/F32 IO).
    cublasLtMatmulDesc_t matmul_desc = nullptr;
    if (cublasLtMatmulDescCreate(&matmul_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F)
        != CUBLAS_STATUS_SUCCESS) {
        return KILN_BLAS_ERR_DESC_CREATE;
    }

    int ret = KILN_BLAS_OK;
    cublasLtMatrixLayout_t a_desc = nullptr, b_desc = nullptr, c_desc = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;

    // Row-major in/out — convert to the equivalent column-major form
    // by swapping operands: C[m, n] = A[m, k] @ B[k, n] (row-major)
    // is equivalent to C_col[n, m] = B_col[n, k] @ A_col[k, m] where
    // X_col is X viewed column-major.
    //
    // In column-major cublasLt notation, the leading dim is the
    // number of rows. Here "B as the first operand" has shape
    // [n, k] in column-major, so its leading dim is n. "A as the
    // second operand" has shape [k, m] in column-major, leading dim
    // k. Output C^col has shape [n, m], leading dim n.
    //
    // Transposes apply *in row-major space*: a_transposed=true means
    // A was passed as [k, m] in row-major, but we view it as [m, k]
    // logically; in column-major that's [m, k] (so we use OP_T).
    cublasOperation_t op_a = spec->a_transposed ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t op_b = spec->b_transposed ? CUBLAS_OP_T : CUBLAS_OP_N;
    // The "primary operand" in our column-major view is B (because
    // we swapped). cublasLt's TRANSA / TRANSB attributes refer to its
    // first / second arguments, which after the swap are B / A.
    cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                    &op_b, sizeof(op_b));
    cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB,
                                    &op_a, sizeof(op_a));

    // Set the epilogue.
    cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                    &lt_epi, sizeof(lt_epi));

    // Bias is keyed off the epilogue. cublasLt expects a per-column
    // (== per-N) bias vector with the output dtype.
    if (bias_ptr != nullptr &&
        (spec->epilogue == KILN_EPI_BIAS ||
         spec->epilogue == KILN_EPI_BIAS_GELU)) {
        cublasLtMatmulDescSetAttribute(
            matmul_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
            &bias_ptr, sizeof(bias_ptr));
        // Bias dtype defaults to output dtype on Hopper / Ampere.
    }

    // Layouts. After the operand swap, the first cublasLt operand is
    // B and the second is A. Column-major leading dims:
    //   B as cm[n, k]: leading dim n (unless transposed in row-major,
    //                  then row-major is [n, k], col-major is [k, n],
    //                  leading dim k — but we encode transpose via
    //                  CUBLAS_OP_T and present the *non-transposed*
    //                  storage shape).
    //
    // Simpler: the *storage layout* in column-major matches the
    // *row-major* dim ordering of the operand:
    //   - row-major M×K stored row-major == col-major K×M stored col-
    //     major. Leading dim = M (#rows in col-major).
    //   - so for cublasLt, layout(A_row_major[M, K]) has rows=K,
    //     cols=M, ld=K. With op=T (default for our row-major view),
    //     the math is A^T_col = A_row.
    //
    // For the standard "non-transposed" row-major case (a_transposed=
    // false, b_transposed=false):
    //   - Input A is row-major [m, k]; storage layout in col-major:
    //     rows=k, cols=m, ld=k. Combined with op_a=N (matches our
    //     row-major intent? no — see next sentence).
    //
    // The simpler restatement the probe uses: present
    //   B_layout = [n, k, n]  (rows=n, cols=k, ld=n)  — used as "A" in cublasLt
    //   A_layout = [k, m, k]  (rows=k, cols=m, ld=k)  — used as "B" in cublasLt
    //   C_layout = [n, m, n]
    // with op_a = op_b = CUBLAS_OP_N (no transpose in column-major
    // space). That works when the user's row-major matmul is
    // C[m,n] = A[m,k] @ B[k,n] (i.e., a_transposed=false, b_transposed=false).
    //
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

    if (cublasLtMatrixLayoutCreate(&b_desc, cuda_dt_in, b_rows, b_cols, b_ld)
        != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatrixLayoutCreate(&a_desc, cuda_dt_in, a_rows, a_cols, a_ld)
        != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatrixLayoutCreate(&c_desc, cuda_dt_out, c_rows, c_cols, c_ld)
        != CUBLAS_STATUS_SUCCESS) {
        ret = KILN_BLAS_ERR_LAYOUT_CREATE;
        goto cleanup;
    }

    // Preference: limit workspace to caller's budget.
    {
        if (cublasLtMatmulPreferenceCreate(&pref) != CUBLAS_STATUS_SUCCESS) {
            ret = KILN_BLAS_ERR_PREFERENCE;
            goto cleanup;
        }
        cublasLtMatmulPreferenceSetAttribute(
            pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &workspace_bytes, sizeof(workspace_bytes));
    }

    {
        cublasLtMatmulAlgo_t algo;
        bool have_algo = false;
        uint64_t ws_actual = 0;
        int algo_id_out = -1;

        if (algo_blob_in != nullptr && algo_blob_in_len > 0) {
            if (algo_blob_in_len < sizeof(cublasLtMatmulAlgo_t)) {
                ret = KILN_BLAS_ERR_ALGO_DESERIALIZE;
                goto cleanup;
            }
            std::memcpy(&algo, algo_blob_in, sizeof(cublasLtMatmulAlgo_t));
            have_algo = true;
            // We don't know the workspace bytes for the cached algo
            // without re-querying; trust the caller's workspace_bytes.
            ws_actual = workspace_bytes;
            cublasLtMatmulAlgoConfigGetAttribute(
                &algo, CUBLASLT_ALGO_CONFIG_ID,
                &algo_id_out, sizeof(algo_id_out), nullptr);
        }

        if (!have_algo) {
            cublasLtMatmulHeuristicResult_t h_result = {};
            int returned = 0;
            cublasStatus_t hst = cublasLtMatmulAlgoGetHeuristic(
                ctx->lt, matmul_desc, b_desc, a_desc, c_desc, c_desc,
                pref, 1, &h_result, &returned);
            if (hst != CUBLAS_STATUS_SUCCESS || returned == 0) {
                ret = KILN_BLAS_ERR_HEURISTIC;
                goto cleanup;
            }
            algo = h_result.algo;
            ws_actual = h_result.workspaceSize;
            cublasLtMatmulAlgoConfigGetAttribute(
                &algo, CUBLASLT_ALGO_CONFIG_ID,
                &algo_id_out, sizeof(algo_id_out), nullptr);

            if (algo_blob_out != nullptr && algo_blob_out_len != nullptr) {
                if (*algo_blob_out_len < sizeof(cublasLtMatmulAlgo_t)) {
                    ret = KILN_BLAS_ERR_ALGO_BLOB_TOO_SMALL;
                    goto cleanup;
                }
                std::memcpy(algo_blob_out, &algo, sizeof(cublasLtMatmulAlgo_t));
                *algo_blob_out_len = sizeof(cublasLtMatmulAlgo_t);
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

        cublasStatus_t st = cublasLtMatmul(
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
        if (st != CUBLAS_STATUS_SUCCESS) {
            ret = KILN_BLAS_ERR_MATMUL;
            goto cleanup;
        }
    }

cleanup:
    if (pref) cublasLtMatmulPreferenceDestroy(pref);
    if (a_desc) cublasLtMatrixLayoutDestroy(a_desc);
    if (b_desc) cublasLtMatrixLayoutDestroy(b_desc);
    if (c_desc) cublasLtMatrixLayoutDestroy(c_desc);
    if (matmul_desc) cublasLtMatmulDescDestroy(matmul_desc);
    return ret;
}
