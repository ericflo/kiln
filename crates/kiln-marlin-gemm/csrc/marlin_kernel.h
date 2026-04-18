// Kiln Marlin W4A16 GEMM C-ABI wrapper.
//
// Exposes a fixed-signature C entry point around the vendored
// IST-DASLab/marlin (commit 1f25790) `marlin_cuda` function. Marlin is a
// 4-bit weight, 16-bit activation matrix multiply that uses tensor-core
// `mma.m16n8k16.f32.f16.f16.f32` instructions; the kernel itself is FP16
// activation-only. The Rust crate (`kiln-marlin-gemm`) provides a BF16
// activation interface by casting bf16 -> fp16 on the way in and fp16 -> bf16
// on the way out.
//
// Layout of inputs (all CUDA device pointers, all row-major contiguous):
//
//   A:   fp16, shape [prob_m, prob_k]
//   B:   int32 packed weights, shape [prob_k / 16, prob_n * 16 / 8] in the
//        Marlin tile/permute layout (NOT raw GPTQ qweight). See the Python
//        reference packer in upstream `marlin/__init__.py::Layer.pack` and
//        the Rust port in `kiln-marlin-gemm`'s tests for the exact ordering.
//   C:   fp16, shape [prob_m, prob_n] (output, written by the kernel)
//   s:   fp16, shape [prob_k / groupsize, prob_n] (per-group scales, also
//        permuted into Marlin's expected scale layout)
//
// `workspace` must be an int32 device buffer of at least
// `(prob_n / 128) * max_par` zero-initialized entries.

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int kiln_marlin_status_t;

// Status codes mirror the upstream constants: 0 on success, 1 if the problem
// shape is unsupported under the chosen tile sizes (`prob_n % thread_n != 0`,
// etc.), 2 if no kernel template was instantiated for the chosen
// (thread_m_blocks, thread_n_blocks, thread_k_blocks, group_blocks) tuple.
kiln_marlin_status_t kiln_marlin_w4a16_gemm(
    const void *A,
    const void *B,
    void *C,
    void *s,
    int prob_m,
    int prob_n,
    int prob_k,
    void *workspace,
    int groupsize,
    int dev,
    void *stream,
    int thread_k,
    int thread_n,
    int sms,
    int max_par
);

#ifdef __cplusplus
}
#endif
