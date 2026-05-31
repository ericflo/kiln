export const meta = {
  name: 'metal-gemm-design',
  description: 'Judge-panel design of a kiln-owned matrix-core Metal GEMM + backend architecture (#1082)',
  phases: [
    { title: 'Design', detail: 'independent expert designs of the kiln Metal matmul backend' },
    { title: 'Critique', detail: 'adversarial review of each design' },
  ],
}

const CONTEXT = String.raw`
KILN / Metal backend, issue #1082. We are building a kiln-OWNED Metal
compute backend on Apple Silicon (M-series, UMA), targeting Qwen3.5-4B
inference, and we are REMOVING all candle dependencies — including
candle-metal-kernels (not just candle-core). The end state: kiln owns its
MSL kernels and dispatches them via objc2-metal directly.

THE IMMEDIATE PROBLEM (highest leverage):
- kiln-tensor's matmul op ('ops::matmul', a DeviceOp2 'MatmulOp') has cpu_fwd
  + cuda_fwd but NO metal_fwd. So on Metal, every compute-bound matmul
  (prefill QKV/O at seq>1, MLP gate/up/down at prefill, LM head, any bs>1)
  falls back to a HOST ROUND-TRIP and runs on the CPU. This is the opposite
  of maxing out the hardware.
- The bs=1 DECODE projections use a separate custom kiln MSL kernel
  ('metal_transposed_coop_gemv') — a simdgroup-SCALAR GEMV (no matrix
  cores). Memory-bound; works but not provably optimal.
- candle's own reference is 'candle_metal_kernels::call_mlx_gemm' — an
  MLX steel-gemm using simdgroup_matrix (Apple matrix cores). We must MATCH
  OR BEAT it with a KILN-OWNED MSL kernel (not call it).

GOAL: lowest latency + max hardware utilization in EVERY config:
  - bs=1 decode (M=1 GEMV, memory-bound: bandwidth + coalescing dominate)
  - bs>1 / prefill (M>1 GEMM, compute-bound: simdgroup_matrix matrix cores)
  - continuous batching, heterogeneous batch, short + long context.
Do NOT contort to candle's kernel patterns; design what is right for Metal.

QWEN3.5-4B matmul shapes (the fixed kernel zoo — specialize, don't
parameterize what's constant): hidden=2560, gate||up=[B*T,2560]@[2560,18432],
down=[B*T,9216]@[9216,2560], QKV=[B*T,2560]@[2560,~4096], O=[B*T,4096]@[4096,2560],
LM head=[B*T,2560]@[2560,152064]. Weights are pre-TRANSPOSED and BF16
(weight_t is [K, N]); activations BF16; decode B*T in {1,2,4,8,16,32,64}.

SUBSTRATE AVAILABLE:
- objc2 0.6 + objc2-metal 0.3. metal_types.rs already aliases RawDevice/
  RawBuffer/RawCommandQueue/RawComputePipelineState/RawLibrary (objc2-metal
  protocol objects). Apple Silicon UMA: MTLStorageModeShared buffers are
  CPU+GPU addressable (zero-copy).
- Existing kiln custom kernels compile MSL via new_library_with_source +
  dispatch via an objc2 MTLComputeCommandEncoder (set_buffer/set_bytes/
  dispatch_threads). simdgroup_matrix MSL API: simdgroup_float8x8 /
  simdgroup_load / simdgroup_multiply_accumulate / simdgroup_store (8x8 BF16
  matrix tiles on Apple GPUs).
- MatmulOp contract to satisfy: a [.., M, K] @ b [.., K, N] -> [.., M, N],
  batched over leading dims, BF16/F16/F32, validated contiguous.

VALIDATION: an M1 (16 GiB). No model checkpoint available — validate via
(a) numerical PARITY of the kiln GEMM vs the kt CPU matmul at Qwen shapes,
(b) MICROBENCH (wall-clock per matmul) of kiln-GEMM vs the CPU-fallback vs
candle's call_mlx_gemm at the Qwen shapes + the decode-bs sweep.
`

const SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    approach_name: { type: 'string' },
    summary: { type: 'string', description: '2-3 sentence thesis of this design' },
    gemm_kernel_design: { type: 'string', description: 'Concrete MSL design for the matrix-core GEMM: threadgroup tiling, simdgroup_matrix usage, threadgroup-memory staging, the bs=1 (M=1) vs bs>1 path split, BF16 accumulation-in-F32, handling K=2560/9216, N up to 152064, batched + strided/offset inputs. Include actual MSL kernel source or detailed pseudo-MSL for the core tiled loop.' },
    dispatch_and_wiring: { type: 'string', description: 'How metal_fwd resolves (batch,M,N,K), allocates the Shared output, selects the M=1 vs M>1 kernel variant, sets buffers/bytes, computes grid/threadgroup sizes, and returns the kt Tensor. Where the pipeline is cached.' },
    per_config_strategy: { type: 'string', description: 'How this design is optimal across bs=1 decode (memory-bound), bs>1/prefill (compute-bound), long context, continuous batching. What kernel variant each config uses and why.' },
    candle_removal_sequencing: { type: 'string', description: 'How this drops candle-metal-kernels: substrate (objc2-metal device/queue/encoder/pipeline) + the order to migrate the remaining call_* op shaders (softmax/sdpa/rmsnorm/layernorm/elementwise/cast/index_select) to kiln MSL.' },
    risks: { type: 'string', description: 'Top correctness + perf risks and how to mitigate / validate them on the M1.' },
  },
  required: ['approach_name', 'summary', 'gemm_kernel_design', 'dispatch_and_wiring', 'per_config_strategy', 'candle_removal_sequencing', 'risks'],
}

const LENSES = [
  'MATRIX-CORE-MAXIMALIST: one heavily-tuned simdgroup_matrix tiled GEMM (MLX/steel-gemm class) for M>1, with double-buffered threadgroup-memory K-tiling; a separate bandwidth-optimal M=1 GEMV. Push for peak matrix-core utilization on prefill/bs>1.',
  'UNIFIED-PERSISTENT: a single persistent/megakernel-leaning design that handles the bs sweep {1..64} with one captured dispatch shape per bs bucket, minimizing launch overhead for the decode loop; matrix cores when M>=8, vector path below.',
  'SPECIALIZED-ZOO: lean hard into Qwen3.5-4B const-shape specialization — distinct hand-tuned kernels per matmul role (gate||up wide-N, down, QKV, LM-head vocab=152064), each monomorphized to its exact (M-bucket, K, N). Reject any hidden_dim runtime arg.',
  'PRAGMATIC-CORRECT-FIRST: simplest correct simdgroup_matrix GEMM that beats the CPU fallback and matches call_mlx_gemm within ~10%, with a crisp parity+microbench plan; explicitly note where MPSMatrixMultiplication (via objc2) would be the better build-vs-buy call and why custom MSL still wins for the fused/specialized shapes.',
]

phase('Design')
const designs = await pipeline(
  LENSES,
  (lens, _orig, i) => agent(
    `${CONTEXT}

Design lens for THIS proposal: ${lens}

Produce a complete, concrete, implementable design for the kiln-owned
matrix-core Metal GEMM (and how it slots into the kiln Metal backend)
under this lens. Be specific and technical — real MSL for the core loop,
real dispatch arithmetic, real per-config kernel selection. Assume an
expert Metal/MSL implementer will build exactly what you specify on an M1.
Return via StructuredOutput.`,
    { label: `design:${i}`, phase: 'Design', schema: SCHEMA }
  ),
  (design, _lens, i) => design ? agent(
    `Adversarially review this kiln Metal GEMM design for CORRECTNESS and
PERFORMANCE on Apple Silicon (M-series, UMA). Find: MSL/simdgroup_matrix
API mistakes, tiling/threadgroup-memory bugs, races, wrong stride/offset
or batch handling, BF16 accumulation precision issues, cases where it
would be SLOWER than candle's call_mlx_gemm or the CPU fallback, and any
config (bs=1 / bs>1 / long-context) it handles poorly. Be specific and
skeptical. Then give a verdict: is this design sound + competitive, and
what are the must-fix items before implementation?

DESIGN:
${JSON.stringify(design, null, 2)}`,
    { label: `critique:${i}`, phase: 'Critique' }
  ).then((critique) => ({ design, critique })) : null
)

const out = designs.filter(Boolean)
log(`${out.length} designs + critiques produced`)
return { designs: out }
