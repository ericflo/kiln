# Vulkan Backend Implementation Plan for Kiln

## Executive Summary

This plan details the implementation of a Vulkan backend for Kiln that delivers performance parity with the existing CUDA and Metal backends on AMD GPUs. The approach mirrors the existing Metal backend's design: a thin `VulkanBackend` implementing the `BackendRuntime` trait, with custom Vulkan compute shaders (`.comp` GLSL) for the hot-path kernels, compiled at runtime via `glslc` and dispatched via the `ash` Rust bindings.

**Key insight**: Kiln's backend abstraction is already designed for this. The `BackendRuntime` trait in `crates/kiln-model/src/backend/mod.rs` is the seam. Adding Vulkan requires:
1. A new crate `kiln-vulkan-kernel` with Vulkan compute shaders + Rust FFI
2. A new `crates/kiln-model/src/backend/vulkan.rs` implementing `BackendRuntime`
3. Build system wiring (Cargo features, device selection)

**What Vulkan brings**: Full support for AMD GPUs (RDNA2/3/4, CDNA), Intel Arc, and Qualcomm Adreno — platforms where CUDA doesn't work and Metal is unavailable. llama.cpp's Vulkan implementation (~17K lines in `ggml-vulkan.cpp`, 100+ `.comp` shaders) proves this is a mature, production-ready approach.

## Status: Phase 1 Complete — Foundation Working ✅

The Vulkan kernel dispatch pipeline is fully operational on AMD Radeon 8060S (RADV STRIX_HALO). The `dispatch_kernel` function correctly:
- Compiles GLSL shaders to SPIR-V at runtime via `glslc`
- Creates device-local and host-visible buffers
- Uploads input tensors, dispatches compute shaders, reads back output
- Produces correct results for element-wise kernels (add, multiply, etc.)

**Critical fix**: `STORAGE_TEXEL_BUFFER` + `BufferView` silently fails on RADV when `local_size_x > 1`. Switched to `STORAGE_BUFFER` + `DescriptorBufferInfo` which works correctly.

### Implemented Kernels (Phase 3-4)
- ✅ `dispatch_gdn_gates` — fused sigmoid(b) + -exp(A_log)*softplus(a + dt_bias)
- ✅ `dispatch_gdn_gated_rms_norm` — fused rms_norm(x, weight) * silu(z)
- ✅ `dispatch_causal_conv1d_update` — single-token decode depthwise conv1d + silu
- ✅ `dispatch_causal_conv1d_prefill` — multi-token prefill depthwise conv1d + silu
- ✅ `dispatch_gdn_forward_substitution` — triangular solve (I + A_strict)^{-1} (beta * V_prime)
- ✅ `dispatch_gdn_recurrent_step` — recurrent state update for GDN decode
- ✅ `dispatch_gdn_chunk_prep` — chunk prep (a_strict, b_mask, v_prime, q_s_scaled, decay_last_col, p_last)
- ✅ `dispatch_gdn_chunk_scan` — scan operation for chunkwise recurrence
- ✅ `dispatch_gdn_full_chunk_forward` — fused chunk prep + scan with state update
- ✅ `build_and_dispatch_pipeline` — common pipeline build + dispatch helper (reduces code duplication)

### Backend Wiring
- ✅ `VulkanBackend::gdn_gates()` wired to `dispatch_gdn_gates`
- ✅ `VulkanBackend::gdn_gated_rms_norm()` wired to `dispatch_gdn_gated_rms_norm`
- ✅ `VulkanBackend::causal_conv1d_update()` wired to `dispatch_causal_conv1d_update`
- ✅ `VulkanBackend::causal_conv1d_prefill()` wired to `dispatch_causal_conv1d_prefill`
- ✅ `VulkanBackend::gdn_forward_substitution()` wired to `dispatch_gdn_forward_substitution`
- ✅ `VulkanBackend::gdn_recurrent_step()` wired to `dispatch_gdn_recurrent_step`
- ✅ `VulkanBackend::gdn_chunk_prep()` wired to `dispatch_gdn_chunk_prep`
- ✅ `VulkanBackend::gdn_chunk_scan()` wired to `dispatch_gdn_chunk_scan`
- ✅ `VulkanBackend::gdn_full_chunk_forward()` wired to `dispatch_gdn_full_chunk_forward`
- ✅ `for_device()` Vulkan detection path active
- ✅ `select_device()` Vulkan path active
- ✅ Server prewarm hooks in place
- ✅ Kill switches (`KILN_DISABLE_*`) wired
- ✅ GLSLC defines: `-DFLOAT_TYPE=float -DUSE_BFLOAT16=1 -DUSE_SUBGROUP_ADD=1 -DUSE_SUBGROUP_CLUSTERED=1`

---

## Architecture Overview

### Current Backend Pattern

```
                        BackendRuntime trait
                        (kiln-model/src/backend/mod.rs)
                           │        │        │
              ┌────────────┤        │        ├────────────┐
              │            │        │        │            │
        CudaBackend   MetalBackend   CpuBackend   VulkanBackend (NEW)
              │            │        │            │
    kiln-flash-attn  candle-metal   all portable  kiln-vulkan-kernel
    kiln-gdn-kernel  sdpa+custom   candle ops    (new crate)
    kiln-marlin-gemm kernels
    kiln-conv1d-kernel
    kiln-rmsnorm-kernel
```

### Vulkan Kernel Strategy: Borrow from llama.cpp

llama.cpp's Vulkan implementation is the gold standard reference. We will:

1. **Reuse llama.cpp's Vulkan compute shaders** (`.comp` files in `ggml/src/ggml-vulkan/vulkan-shaders/`) — these are standard GLSL compute shaders compiled to SPIR-V at build time
2. **Adapt the shader interface** to match Kiln's tensor shapes and data layouts
3. **Use `vulkan-rs`** (the Rust Vulkan bindings) for device management, command buffers, and shader dispatch

### Why This Works for Kiln

Kiln's compute needs map directly to llama.cpp's Vulkan shader set:

| Kiln Need | llama.cpp Vulkan Equivalent |
|---|---|
| FlashAttention-2 prefill | `flash_attn.comp` + `flash_attn_base.glsl` |
| FlashAttention-2 paged decode | `flash_attn.comp` + gather |
| Gated DeltaNet forward-substitution | `gated_delta_net.comp` |
| Gated DeltaNet recurrent step | `gated_delta_net.comp` |
| GDN chunk prep/scan | Custom GLSL (small kernels) |
| GDN full chunk forward | `gated_delta_net.comp` variant |
| RMSNorm | `rms_norm.comp` |
| RMSNorm gated (GDN) | `rms_norm.comp` + `silu.comp` |
| Causal Conv1d | `conv2d_dw.comp` (depthwise) |
| GDN gates (sigmoid/softplus) | `sigmoid.comp` + `silu.comp` |
| Rotary embeddings | `rope_neox.comp` |
| LM head matmul | `mul_mm.comp` |
| GDN input projection | `mul_mm.comp` / `mul_mat_vec.comp` |
| Transposed GEMV (decode) | `mul_mat_vec_nc.comp` |
| Paged KV read | Custom gather shader |
| L2 Q/K norm | Custom shader (small) |

### Build-Time Shader Compilation

llama.cpp uses `vulkan-shaders-gen.cpp` to compile `.comp` GLSL files to SPIR-V `.spv` binaries, then embeds them as C++ byte arrays. We'll do the same in Rust:

1. A `build.rs` that invokes `glslc` to compile `.comp` files to SPIR-V
2. A Rust codegen step that converts `.spv` files into `include_bytes!()` byte arrays
3. At runtime, create `vk::ShaderModule` from the embedded SPIR-V
4. Pipeline caching to avoid recompilation across runs

This mirrors the existing CUDA pattern where `build.rs` invokes `nvcc` to compile `.cu` files to `.o` objects linked into the final binary.

---

## Phase Breakdown

### Phase 1: Foundation (2-3 days)
- `kiln-vulkan-kernel` crate skeleton
- Vulkan device detection + feature flag wiring
- SPIR-V shader compilation pipeline
- Minimal working `VulkanBackend` with `flash_attn_prefill`

### Phase 2: FlashAttention-2 (2-3 days)
- Vulkan FlashAttention-2 prefill kernel
- Vulkan FlashAttention-2 paged decode kernel
- Parity tests vs CUDA reference

### Phase 3: Gated DeltaNet Kernels (3-4 days)
- Vulkan GDN recurrent step kernel
- Vulkan GDN forward-substitution kernel
- Vulkan GDN chunk prep + scan kernels
- Vulkan GDN full chunk forward kernel
- Parity tests

### Phase 4: Supporting Kernels (2-3 days)
- RMSNorm, gated RMSNorm, RMSNorm-back
- Causal Conv1d update + prefill
- GDN gates (sigmoid/softplus)
- Rotary embeddings (RoPE)
- LM head matmul
- Transposed GEMV for decode
- Paged KV head-major read

### Phase 5: Integration & Optimization (2-3 days)
- `for_device()` Vulkan path
- `select_device()` Vulkan path
- Precompile at startup (like Metal)
- Pipeline caching
- AMD GPU profiling + tuning
- Environment kill switches

### Phase 6: Testing & Documentation (1-2 days)
- Parity tests against CUDA reference
- Performance benchmarks on AMD GPU
- README updates, build docs
- Desktop app Vulkan support

---

## Detailed Implementation

### New Crate: `kiln-vulkan-kernel`

```
crates/kiln-vulkan-kernel/
├── Cargo.toml
├── build.rs                    # glslc → SPIR-V → Rust byte arrays
├── csrc/
│   ├── shaders/
│   │   ├── flash_attn.comp          # FlashAttention-2 (from llama.cpp)
│   │   ├── flash_attn_base.glsl      # Shared FlashAttention GLSL
│   │   ├── flash_attn_mmq_funcs.glsl # Quant-aware (future)
│   │   ├── gated_delta_net.comp      # GDN recurrent (from llama.cpp)
│   │   ├── rms_norm.comp             # RMSNorm (from llama.cpp)
│   │   ├── rms_norm_partials.comp    # RMSNorm partials
│   │   ├── causal_conv1d.comp        # Depthwise conv (from llama.cpp)
│   │   ├── silu.comp                 # SiLU activation
│   │   ├── sigmoid.comp              # Sigmoid activation
│   │   ├── rope_neox.comp            # RoPE (from llama.cpp)
│   │   ├── mul_mm.comp               # General matmul (from llama.cpp)
│   │   ├── mul_mat_vec.comp          # MatVec (from llama.cpp)
│   │   ├── mul_mat_vec_nc.comp       # Non-contiguous (from llama.cpp)
│   │   ├── acc.comp                  # Accumulation (from llama.cpp)
│   │   ├── copy.comp                 # Buffer copy (from llama.cpp)
│   │   ├── soft_max.comp             # Softmax (from llama.cpp)
│   │   ├── add.comp                  # Element-wise add
│   │   ├── mul.comp                  # Element-wise mul
│   │   ├── exp.comp                  # Element-wise exp
│   │   ├── types.glsl                # Shared types
│   │   └── utils.glsl                # Shared utilities
│   ├── kiln_flash_attn.h             # C ABI header
│   ├── kiln_flash_attn.cu.h          # (not needed — pure Vulkan)
│   └── kiln_gdn_kernels.h            # C ABI header for GDN
├── src/
│   ├── lib.rs                       # Public FFI + Rust API
│   ├── shader_compilation.rs        # build.rs output: embedded SPIR-V
│   ├── vulkan_device.rs             # Vulkan device abstraction
│   ├── flash_attn.rs                # FlashAttention dispatch
│   ├── gdn_kernels.rs               # GDN kernel dispatch
│   ├── math_kernels.rs              # RMSNorm, RoPE, etc.
│   ├── pipeline_cache.rs            # Pipeline caching
│   └── memory.rs                    # Buffer management
└── tests/
    ├── flash_attn_parity.rs
    ├── gdn_parity.rs
    └── math_kernels_parity.rs
```

### Build System Changes

#### 1. `Cargo.toml` workspace changes

Add `crates/kiln-vulkan-kernel` to workspace members and `default-members`.

#### 2. `kiln-model/Cargo.toml` feature additions

Add `vulkan` feature alongside `cuda` and `metal`:

```toml
[features]
cuda = ["candle-core/cuda", ...]
metal = ["candle-core/metal", ...]
vulkan = ["dep:kiln-vulkan-kernel", "dep:vulkan-rs", "dep:glslc"]
```

#### 3. `kiln-vulkan-kernel/Cargo.toml`

```toml
[package]
name = "kiln-vulkan-kernel"
description = "Vulkan compute kernels for Kiln (GLSL compute shaders)"
build = "build.rs"

[dependencies]
vulkan-rs = { version = "1.11", features = ["default", "linked"] }
vulkan-hal = "0.11"
vulkan-loader = "0.11"
half = "2"
anyhow = "1"

[build-dependencies]
anyhow = "1"
```

### Shader Compilation Pipeline (`build.rs`)

This mirrors llama.cpp's `vulkan-shaders-gen.cpp` but in Rust:

```rust
// kiln-vulkan-kernel/build.rs
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=csrc/shaders/");
    
    // Find glslc (Vulkan GLSL compiler)
    let glslc = find_glslc();
    
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let shader_dir = manifest.join("csrc/shaders");
    let out_dir = env::var("OUT_DIR").unwrap();
    
    // Step 1: Compile each .comp file to SPIR-V
    for entry in walkdir::WalkDir::new(&shader_dir) {
        let path = entry.unwrap().into_path();
        if path.extension() == Some("comp".as_ref()) {
            let spv_path = path.with_extension("spv");
            Command::new(&glslc)
                .arg(&path)
                .arg("-o")
                .arg(&spv_path)
                .args(&["-D", "FLOAT_TYPE=float", "-D", "USE_BFLOAT16=1"])
                .status().expect("glslc failed");
        }
    }
    
    // Step 2: Generate Rust code that embeds all .spv files
    let rust_code = generate_embedded_code(&shader_dir, &out_dir);
    let output_path = PathBuf::from(out_dir).join("vulkan_shaders.rs");
    std::fs::write(&output_path, rust_code).unwrap();
    println!("cargo:rustc-link-lib=vulkan");
}
```

### Vulkan Device Abstraction

Kiln's CUDA backend uses `cudarc` for raw CUDA. For Vulkan we need:

```rust
// kiln-vulkan-kernel/src/vulkan_device.rs
use vulkan_rs::{
    device::{Device, Queue},
    instance::{Instance, PhysicalDevice},
    memory::MemoryType,
    pipeline::*,
    shader::ShaderModule,
    buffer::{Buffer, BufferCreateInfo},
    command::CommandBuffer,
    instance::Version,
};

pub struct VulkanDevice {
    instance: Instance,
    physical_device: PhysicalDevice,
    device: Device,
    compute_queue: Queue,
    vendor_id: u32,
    device_name: String,
    max_work_group_size: [u32; 3],
    max_push_constants: u32,
    pipeline_cache: std::sync::Mutex<HashMap<String, PipelineCache>>,
}

impl VulkanDevice {
    pub fn new(device_index: u32) -> Result<Self> {
        // Match llama.cpp's device selection:
        // 1. Try GGML_VK_VISIBLE_DEVICES env var
        // 2. Fall back to first Vulkan device
        // 3. Prefer AMD for gaming, Qualcomm for mobile
        
        let instance = Instance::new(Version::v1_2, false, &["VK_KHR_shader_float16_int8"])?;
        let physical_devices = instance.enumerate_physical_devices()?;
        
        let pd = find_best_physical_device(physical_devices, device_index)?;
        
        // Check bfloat16 support (critical for Kiln)
        let features = pd.properties2().device_features();
        if !features.shader_bfloat16_type {
            // Fallback: use float16 math path
        }
        
        let device = Device::new(pd, &["VK_KHR_shader_float16_int8", 
                                       "VK_KHR_shader_subgroup_extended_types"], 
                                 &[(Queue::COMPUTE, 0.0)])?;
        
        Ok(VulkanDevice {
            instance,
            physical_device: pd,
            device,
            compute_queue: device.queue(0, 0).unwrap(),
            vendor_id: pd.properties().vendor_id,
            device_name: pd.properties().device_name,
            max_work_group_size: pd.properties().limits().max_work_group_size(),
            max_push_constants: pd.properties().limits().max_push_constants(),
            pipeline_cache: std::sync::Mutex::new(HashMap::new()),
        })
    }
}
```

### Buffer Management

Vulkan requires explicit buffer allocation:

```rust
// kiln-vulkan-kernel/src/memory.rs
pub struct VulkanBuffer {
    buffer: Buffer,
    memory_type: MemoryType,
    size: u64,
    device: Device,
}

impl VulkanBuffer {
    pub fn create(device: &Device, size: u64) -> Result<Self> {
        let buffer = Buffer::new(device, BufferCreateInfo {
            size, usage: BufferUsage::STORAGE_BUFFER, ..Default::default()
        })?;
        let memory = buffer.allocate_memory(
            device, MemoryAllocateFlags::DEFAULT,
            MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        Ok(VulkanBuffer { buffer, memory_type: memory.memory_type(), size, device })
    }
}
```

### Pipeline Caching

Vulkan pipeline compilation is expensive (can take 100s of ms). We cache:

```rust
// kiln-vulkan-kernel/src/pipeline_cache.rs
pub struct PipelineCache {
    cache: vk::PipelineCache,
    pipelines: HashMap<String, vk::Pipeline>,
}
// Persist to disk: ~/.cache/kiln/vulkan_pipelines.bin
```

### Shader Dispatch Pattern

Each kernel follows this pattern (mirroring llama.cpp's approach):

```rust
// 1. Create pipeline (cached)
let pipeline = cache.get_or_create_pipeline("flash_attn", shader_module, ...)?;

// 2. Bind buffers
command_buffer.bind_pipeline(pipeline);
command_buffer.bind_descriptor_sets(...);

// 3. Set push constants (kernel parameters)
command_buffer.push_constants(pipeline.layout(), 0, &push_constant_bytes);

// 4. Dispatch compute shader
command_buffer.dispatch(workgroup_count_x, workgroup_count_y, workgroup_count_z);

// 5. Submit to compute queue
compute_queue.submit([command_buffer]).wait();
```

### `VulkanBackend` Implementation

This is `crates/kiln-model/src/backend/vulkan.rs`. It mirrors the existing `CudaBackend` and `MetalBackend` exactly:

```rust
#[derive(Debug)]
pub struct VulkanBackend {
    device: Device,           // candle-core Vulkan device
    vulkan_device: VulkanDevice,  // kiln-vulkan-kernel abstraction
    gdn_enabled: bool,
    gdn_gates_enabled: bool,
    fused_conv1d_enabled: bool,
}

impl VulkanBackend {
    pub fn new(device: Device) -> Self {
        let gdn_enabled = std::env::var("KILN_DISABLE_GDN_KERNEL").is_err();
        let gdn_gates_enabled = gdn_enabled 
            && std::env::var("KILN_DISABLE_FUSED_GDN_GATES").is_err();
        let fused_conv1d_enabled = std::env::var("KILN_DISABLE_FUSED_CONV1D").is_err();
        Self {
            device,
            vulkan_device: VulkanDevice::new(0).expect("Vulkan init failed"),
            gdn_enabled,
            gdn_gates_enabled,
            fused_conv1d_enabled,
        }
    }
}

impl BackendRuntime for VulkanBackend {
    fn name(&self) -> &'static str { "vulkan" }
    fn device(&self) -> &Device { &self.device }
    
    fn supports_flash_attn_prefill(&self) -> bool { true }
    fn supports_flash_attn_prefill_head_major(&self) -> bool { true }
    fn supports_flash_attn_paged_decode(&self) -> bool { true }
    fn supports_gdn_forward_substitution(&self) -> bool { self.gdn_enabled }
    fn supports_gdn_recurrent_step(&self) -> bool { self.gdn_enabled }
    fn supports_gdn_chunk_prep(&self) -> bool { self.gdn_enabled }
    fn supports_gdn_chunk_scan(&self) -> bool { self.gdn_enabled }
    fn supports_gdn_full_chunk_forward(&self) -> bool { self.gdn_enabled }
    fn supports_gdn_gates(&self) -> bool { self.gdn_gates_enabled }
    fn supports_gdn_gated_rms_norm(&self) -> bool { self.gdn_enabled }
    fn supports_causal_conv1d_update(&self) -> bool { self.fused_conv1d_enabled }
    fn supports_causal_conv1d_prefill(&self) -> bool { self.fused_conv1d_enabled }
    
    // All kernel methods: Ok(Some(out)) on success, Ok(None) to decline
    fn flash_attn_prefill(&self, q, k, v, scale, causal) -> Result<Option<Tensor>> {
        if q.dtype() != DType::BF16 { return Ok(None); }
        kiln_vulkan_kernel::flash_attn(q, k, v, scale, causal)
    }
    
    fn flash_attn_paged_decode(&self, q, k_pool, v_pool, block_table, ...) 
        -> Result<Option<Tensor>> {
        if q.dtype() != DType::BF16 { return Ok(None); }
        kiln_vulkan_kernel::flash_attn_paged_decode(q, k_pool, v_pool, block_table, ...)
    }
    
    fn gdn_recurrent_step(&self, q, k, v, beta, g, state) -> Result<Option<Tensor>> {
        if q.dtype() != DType::BF16 { return Ok(None); }
        kiln_vulkan_kernel::gdn_recurrent(q, k, v, beta, g, state)
    }
    
    fn gdn_forward_substitution(&self, a_strict, v_prime, beta) -> Result<Option<Tensor>> {
        if a_strict.dtype() != DType::BF16 { return Ok(None); }
        kiln_vulkan_kernel::gdn_forward_sub(a_strict, v_prime, beta)
    }
    
    // ... all other trait methods
}
```

### Kernel-by-Kernel Implementation Plan

#### 4.1 FlashAttention-2 Prefill

**Source**: llama.cpp's `flash_attn.comp` + `flash_attn_base.glsl` + `flash_attn_mmq_funcs.glsl`

**Kiln adaptation**: Kiln uses `[batch, seq_len, num_heads, head_dim]` (token-major). llama.cpp uses `[N, K, H]` (row-major). We need to:

1. Copy the base shader from llama.cpp's `ggml/src/ggml-vulkan/vulkan-shaders/flash_attn.comp`
2. Adapt the GLSL `main()` to match Kiln's tensor layout
3. Set up push constants for `softmax_scale`, `causal`, `head_dim`
4. Use descriptor sets for Q/K/V/O buffers

**Key spec constants** (from llama.cpp's approach):
```glsl
layout (constant_id = 0) const uint Br = 16;    // Q row block size
layout (constant_id = 1) const uint Bc = 128;   // KV col block size
layout (constant_id = 2) const uint D_split = 1; // D dimension split
layout (constant_id = 3) const uint SubGroupSize = 32;
layout (constant_id = 4) const uint WorkGroupSize = 256;
```

**Workgroup layout**: `gl_WorkGroupID.z` = head index, `gl_WorkGroupID.y` = batch, `gl_WorkGroupID.x` = seq position within block.

#### 4.2 FlashAttention-2 Paged Decode

**Approach**: Vulkan doesn't have direct indexable-atomics for gather. Two options:

1. **Block-table gather shader**: A pre-pass that gathers K/V from the paged pool into a contiguous buffer, then calls the standard FlashAttention kernel
2. **Direct paged shader**: Rewrite `flash_attn.comp` to read from the block table directly using a per-thread block lookup

**Recommended**: Option 1 (gather + FA) for correctness and simpler shader code. The gather is O(batch * max_blocks * block_size) which is small for decode.

```glsl
// gather_paged.comp
layout(local_size_x = 256) in;
layout(binding = 0) readonly buffer k_pool { float data[]; };
layout(binding = 1) readonly buffer v_pool { float data[]; };
layout(binding = 2) readonly buffer block_table { uint data[]; };
layout(binding = 3) writeonly buffer k_out { float data[]; };
layout(binding = 4) writeonly buffer v_out { float data[]; };

void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint seq = idx / max_blocks;
    uint block = idx % max_blocks;
    uint block_id = block_table[seq * max_blocks + block];
    uint offset = block_id * page_block_size;
    k_out[idx * page_block_size + : ] = k_pool[offset + : ];
    v_out[idx * page_block_size + : ] = v_pool[offset + : ];
}
```

#### 4.3 Gated DeltaNet Kernels

**Source**: llama.cpp's `gated_delta_net.comp` — this is the most complex shader.

**Kiln shapes**:
- `q`: `[B, H, dk]` bf16
- `k`: `[B, H, dk]` bf16  
- `v`: `[B, H, dv]` bf16
- `beta`: `[B, H]` bf16
- `g`: `[B, H]` bf16 (decay, pre-exp)
- `state`: `[B, H, dk, dv]` bf16 (mutated in-place)
- `out`: `[B, H, dv]` bf16

**Shader mapping** (from llama.cpp's `gated_delta_net.comp`):

```glsl
// Kiln adaptation of llama.cpp's gated_delta_net.comp
#version 450

layout(local_size_x_id = 2, local_size_y = 1, local_size_z = 1) in;

layout(constant_id = 0) const uint S_V = 128;       // state_size (dk)
layout(constant_id = 1) const uint KDA = 0;         // key_decay_axis
layout(constant_id = 2) const uint SUBGROUP_SIZE = 32;
layout(constant_id = 3) const uint LANES_PER_COLUMN = 32;

layout(binding = 0) readonly buffer QBuf     { float data_q[]; };
layout(binding = 1) readonly buffer KBuf     { float data_k[]; };
layout(binding = 2) readonly buffer VBuf     { float data_v[]; };
layout(binding = 3) readonly buffer GBuf     { float data_g[]; };
layout(binding = 4) readonly buffer BetaBuf  { float data_beta[]; };
layout(binding = 5) readonly buffer StateBuf { float data_state[]; };
layout(binding = 6) writeonly buffer DstBuf  { float data_dst[]; };

// Shared memory for state sharding
shared float s_shard[ROWS_PER_LANE];

void main() {
    uint head_id = gl_WorkGroupID.x;
    uint seq_id = gl_WorkGroupID.y;
    uint lane = gl_SubgroupInvocationID % LANES_PER_COLUMN;
    uint col = gl_WorkGroupID.z * COLS_PER_WG + (gl_SubgroupInvocationID / LANES_PER_COLUMN);
    
    // Read state shard
    uint state_base = (seq_id * H + head_id) * S_V * S_V;
    for (uint r = 0; r < ROWS_PER_LANE; r++) {
        s_shard[r] = data_state[state_base + col * S_V + r * LANES_PER_COLUMN + lane];
    }
    
    // Recurrence: S *= exp(g) + k * delta
    // output = S @ q
    // (full loop from llama.cpp, adapted for Kiln's shapes)
    
    // Write updated state back
    for (uint r = 0; r < ROWS_PER_LANE; r++) {
        data_state[state_base + col * S_V + r * LANES_PER_COLUMN + lane] = s_shard[r];
    }
}
```

#### 4.4 GDN Chunk Prep + Scan

These are smaller kernels that compute intermediate values for the chunkwise recurrence:

```glsl
// gdn_chunk_prep.comp
// Inputs: g[B,H,C], v[B,H,C,dv], kkt[B,H,C,C], qkt[B,H,C,C], 
//          ks_entry[B,H,C,dv], q_s[B,H,C,dv]
// Outputs: a_strict[B,H,C,C], b_mask[B,H,C,C], v_prime[B,H,C,dv],
//          q_s_scaled[B,H,C,dv], decay_last_col[B,H,C], p_last[B,H]

// Each workgroup = one (batch, head) slot
// Tid walks across C*C + C*dv elements
```

#### 4.5 GDN Full Chunk Forward

Combines chunk prep + scan into one pass. Reuses the same shader infrastructure as chunk prep but with the accumulation loop fused in.

#### 4.6 Supporting Kernels

| Kernel | Source | Notes |
|--------|--------|-------|
| RMSNorm | `rms_norm.comp` from llama.cpp | Already exists, minor shape adaptation |
| Gated RMSNorm | `rms_norm.comp` + `silu.comp` | Fused: RMSNorm × SiLU(z) |
| Causal Conv1d update | `conv2d_dw.comp` from llama.cpp | Depthwise conv, kernel_size=4 |
| Causal Conv1d prefill | `conv2d_dw.comp` variant | Same kernel, seq_len > 1 |
| GDN gates | `sigmoid.comp` + `silu.comp` | beta=sigmoid(b), g=-exp(A_log)*softplus(a+dt_bias) |
| RoPE | `rope_neox.comp` from llama.cpp | Already exists |
| LM head | `mul_mm.comp` from llama.cpp | Small matmul: [vocab] × [hidden] |
| Transposed GEMV | `mul_mat_vec_nc.comp` from llama.cpp | For LoRA delta decode |
| L2 Q/K norm | Custom small shader | Per-row L2 normalization |
| Paged KV read | Custom gather shader | Block table → contiguous K/V |
| Softmax | `soft_max.comp` from llama.cpp | Per-row softmax on last dim |
| Copy | `copy.comp` from llama.cpp | Buffer-to-buffer copy |
| Add/mul/exp | Element-wise unary/binary | Standard GLSL per-element |

### Environment Kill Switches

Mirror the existing pattern:

```rust
const DISABLE_VULKAN_GDN_KERNEL: &str = "KILN_DISABLE_GDN_KERNEL";
const DISABLE_VULKAN_FUSED_GDN_GATES: &str = "KILN_DISABLE_FUSED_GDN_GATES";
const DISABLE_VULKAN_FUSED_CONV1D: &str = "KILN_DISABLE_FUSED_CONV1D";
const DISABLE_VULKAN_GDN_GATED_RMSNORM: &str = "KILN_DISABLE_FUSED_GDN_GATED_RMS_NORM";
const DISABLE_VULKAN_GDN_QK_NORM: &str = "KILN_DISABLE_VULKAN_GDN_QK_NORM";
const DISABLE_VULKAN_RMSNORM: &str = "KILN_DISABLE_VULKAN_RMSNORM";
const DISABLE_VULKAN_MLP_GATE_UP: &str = "KILN_DISABLE_VULKAN_MLP_GATE_UP";
const DISABLE_VULKAN_LM_HEAD: &str = "KILN_DISABLE_VULKAN_LM_HEAD";
const DISABLE_VULKAN_TRANSPOSED_GEMV: &str = "KILN_DISABLE_VULKAN_TRANSPOSED_GEMV";
const DISABLE_VULKAN_GDN_IN_PROJ: &str = "KILN_DISABLE_VULKAN_GDN_IN_PROJ";
const DISABLE_VULKAN_PAGED_KV_READ: &str = "KILN_DISABLE_VULKAN_PAGED_KV_READ";
const GGML_VK_VISIBLE_DEVICES: &str = "GGML_VK_VISIBLE_DEVICES";
```

### Vulkan Shader Build Configuration

```glsl
// Preprocessor defines passed to glslc:
-DFLOAT_TYPE=float           // Use float32 for accuracy
-DUSE_BFLOAT16=1             // Enable bfloat16 via VK_KHR_shader_bfloat16
-DUSE_SUBGROUP_ADD=1         // Enable subgroup arithmetic for reductions
-DUSE_SUBGROUP_CLUSTERED=1   // Enable clustered reductions (AMD optimization)
-DGGML_VULKAN_BFLOAT16_GLSLC_SUPPORT  // bfloat16 extension support
```

### AMD GPU Specific Optimizations

llama.cpp already has AMD-specific paths. We'll port them:

1. **RDNA2/3 cooperative matrix**: Use `VK_AMDX_shader_invocation_reordering` for better occupancy
2. **Subgroup clustered reductions**: `subgroupClusteredAdd` for faster cross-warp reductions
3. **Float16 dot product**: `VK_KHR_shader_float16_int8` for accelerated F16 math
4. **Work group size tuning**: `gl_MaxComputeWorkGroupSize` per GPU model
5. **Push constant limits**: `gl_MaxPushConstants` — use descriptor sets for large params

### Integration with Existing Codebase

#### 5.1 `for_device()` in `backend/mod.rs`

Add a `vulkan` feature-gated branch:

```rust
// crates/kiln-model/src/backend/mod.rs
pub fn for_device(device: &Device) -> Arc<dyn BackendRuntime> {
    match device {
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => Arc::new(cuda::CudaBackend::new(device.clone())),
        #[cfg(feature = "vulkan")]
        Device::Vulkan(_) => Arc::new(vulkan::VulkanBackend::new(device.clone())),
        #[cfg(feature = "metal")]
        Device::Metal(_) => Arc::new(metal::MetalBackend::new(device.clone())),
        _ => Arc::new(cpu::CpuBackend::new(device.clone())),
    }
}
pub mod vulkan;  // new module
```

#### 5.2 `select_device()` in `device.rs`

Add Vulkan detection before Metal:

```rust
// crates/kiln-server/src/device.rs
pub fn select_device() -> Result<Device> {
    #[cfg(feature = "cuda")]
    if candle_core::utils::cuda_is_available() {
        tracing::info!("CUDA available — using GPU device 0");
        return Device::new_cuda(0).context("failed to initialize CUDA device");
    }

    #[cfg(feature = "vulkan")]
    if candle_core::utils::vulkan_is_available() {
        tracing::info!("Vulkan available — using GPU device 0 (AMD)");
        return Device::new_vulkan(0).context("failed to initialize Vulkan device");
    }

    #[cfg(feature = "metal")]
    if candle_core::utils::metal_is_available() {
        tracing::info!("Metal available — using Apple Silicon GPU");
        return Device::new_metal(0).context("failed to initialize Metal device");
    }

    tracing::info!("no GPU feature active — using CPU");
    Ok(Device::Cpu)
}
```

#### 5.3 Candle-core Vulkan Device

**Critical dependency**: candle-core 0.10.2 must support `Device::Vulkan`. Check if it does:

- candle-core 0.10.2 has `Device::Cuda` and `Device::Metal` variants
- Vulkan device support may need to be added to candle-core, OR
- We can use a **standalone Vulkan device** approach where `VulkanBackend` owns its own `vk::Device` and manages its own buffers, with tensor data copied to/from candle-core CPU tensors via the FFI boundary

**Recommended approach**: Follow llama.cpp's pattern. llama.cpp's Vulkan backend is a **standalone backend** — it doesn't depend on candle-core's Vulkan device. We'll do the same:

```rust
// vulkan.rs - VulkanBackend owns its own vk::Device
// Tensors are passed as raw pointers to the kernel crate
// Output is a fresh Vulkan buffer, then copied to CPU for candle tensor wrapping

pub struct VulkanBackend {
    device: Device,           // The candle device (for type identity)
    vk_device: VulkanDevice,  // Our own Vulkan device
    // ... kernel state
}

fn flash_attn_prefill(&self, q: &Tensor, k: &Tensor, v: &Tensor, ...) 
    -> Result<Option<Tensor>> {
    // 1. Extract raw pointers from candle tensors (if on Vulkan)
    // 2. Call kiln_vulkan_kernel::flash_attn()
    // 3. Wrap result in a candle Tensor on the Vulkan device
}
```

#### 5.4 Startup Prewarm

Mirror the Metal prewarm pattern:

```rust
// main.rs
#[cfg(feature = "vulkan")]
fn precompile_vulkan_kernels(device: &Device) {
    if !matches!(device, candle_core::Device::Vulkan(_)) {
        return;
    }
    let start = std::time::Instant::now();
    match kiln_model::backend::vulkan::precompile_custom_kernels(device) {
        Ok(()) => tracing::info!(
            elapsed_ms = start.elapsed().as_millis() as u64,
            "Vulkan custom kernels precompiled during background prewarm"
        ),
        Err(err) => tracing::warn!(
            error = %err,
            "Vulkan kernel precompile failed; falling back to lazy compilation"
        ),
    }
}

#[cfg(not(feature = "vulkan"))]
fn precompile_vulkan_kernels(_device: &Device) {}
```

#### 5.5 Health Endpoint

The `/health` endpoint already surfaces `backend_name`. Vulkan will automatically appear as `"vulkan"` from `VulkanBackend::name()`.

#### 5.6 Desktop App Integration

- The desktop app's Settings window needs a Vulkan checkbox alongside CUDA/Metal
- The auto-download binary matrix needs a `vulkan` build target
- GPU detection in the desktop app needs to query Vulkan physical devices

### Build System Details

#### `cargo build --features vulkan`

```bash
# Requires:
# - glslc (Vulkan GLSL compiler, part of Vulkan SDK)
# - vulkan-loader (Vulkan runtime library)
# - vulkan-headers (Vulkan SDK headers)

# Linux:
sudo apt install vulkan-tools vulkan-validationlayers glslc
# or: pacman -S vulkan-headers vulkan-loader glslc

# Build:
cargo build --release --features vulkan
```

#### `KILN_VULKAN_DEVICE` env var

```rust
// Allow user to select which Vulkan device
fn find_best_vulkan_device() -> u32 {
    let env_device = std::env::var("KILN_VULKAN_DEVICE")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(0);
    
    // llama.cpp uses GGML_VK_VISIBLE_DEVICES for device selection
    // We'll support both
    if let Ok(ggml_devs) = std::env::var("GGML_VK_VISIBLE_DEVICES") {
        // Parse comma-separated list, pick first
        return ggml_devs.split(',').next().unwrap_or("0").parse().unwrap_or(0);
    }
    env_device
}
```

### Shader Source Management

Two approaches for managing the GLSL shader source:

**Approach A**: Vendor llama.cpp's shaders directly (copy `.comp` files into `csrc/shaders/`)
- Pros: Full feature parity with llama.cpp, well-tested
- Cons: License compliance (llama.cpp is MIT), version drift

**Approach B**: Use a git submodule for llama.cpp's vulkan-shaders directory
- Pros: Automatic updates, clear provenance
- Cons: Build complexity, submodule management

**Approach C**: Write original GLSL shaders from scratch
- Pros: Clean IP, no license concerns
- Cons: More work, less battle-tested

**Recommended**: **Approach A** with clear attribution. llama.cpp is MIT-licensed. Copy the relevant `.comp` files into `kiln-vulkan-kernel/csrc/shaders/` with a LICENSE-LLAMA.cpp file referencing the source. This is the fastest path to a working Vulkan backend.

### Dependency on candle-core Vulkan Support

Check candle-core 0.10.2 for Vulkan device support:
- candle-core has `Device::Cuda` and `Device::Metal` variants
- Vulkan support may need to be added to candle-core via a PR, OR
- We use a **host-memory intermediate** approach where Vulkan buffers are managed independently

**Fallback strategy**: If candle-core doesn't have `Device::Vulkan`, we:
1. Run the Vulkan backend on a standalone `vk::Device`
2. Host tensors are kept in CPU memory
3. The `VulkanBackend` manages its own buffer lifecycle
4. Tensor data is copied host→GPU→host at kernel boundaries
5. This is slower than native Vulkan tensors but works with candle-core 0.10.2

The long-term fix is to add `Device::Vulkan` to candle-core (similar to how `Device::Metal` was added). This would require candle-core to depend on `vulkan-rs`.

### Testing Strategy

#### Phase 1: Parity Tests

Mirror the existing CUDA test pattern:

```rust
// kiln-vulkan-kernel/tests/flash_attn_parity.rs
#[test]
fn test_flash_attn_vs_cuda_reference() {
    // 1. Run on CUDA device (reference)
    let cuda_device = Device::new_cuda(0).unwrap();
    let q_ref = Tensor::randn(0.0, 1.0, (1, 64, 4, 128), &cuda_device)
        .unwrap().to_dtype(DType::BF16).unwrap();
    let k_ref = q_ref.clone();
    let v_ref = q_ref.clone();
    let (out_ref, lse_ref) = flash_attn(&q_ref, &k_ref, &v_ref, 0.125, true).unwrap();
    
    // 2. Run on Vulkan device
    let vulkan_device = Device::new_vulkan(0).unwrap();
    let q_v = Tensor::randn(0.0, 1.0, (1, 64, 4, 128), &vulkan_device)
        .unwrap().to_dtype(DType::BF16).unwrap();
    let k_v = q_v.clone();
    let v_v = q_v.clone();
    let (out_v, lse_v) = flash_attn(&q_v, &k_v, &v_v, 0.125, true).unwrap();
    
    // 3. Compare (BF16 has ~3 decimal places of precision)
    let out_f32 = out_ref.to_dtype(DType::F32).unwrap().flatten_all().unwrap();
    let out_v_f32 = out_v.to_dtype(DType::F32).unwrap().flatten_all().unwrap();
    let max_diff: f32 = out_f32
        .to_vec1::<f32>().unwrap()
        .iter()
        .zip(out_v_f32.to_vec1::<f32>().unwrap().iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    
    assert!(max_diff < 0.01, "max diff {} exceeds tolerance", max_diff);
}
```

#### Phase 2: Performance Benchmarks

```bash
# Benchmark on AMD GPU:
KILN_VULKAN_DEVICE=0 cargo bench --features vulkan
KILN_VULKAN_DEVICE=0 cargo bench --features cuda  # for comparison

# Metrics:
# - Tokens/sec (prefill, decode)
# - TTFT (time to first token)
# - VRAM usage
# - Kernel compilation time (first run)
# - Pipeline cache hit rate (subsequent runs)
```

#### Phase 3: AMD GPU Profiling

Use `rocprof` or `radeontop` to profile Vulkan workloads:

```bash
# AMD GPU profiling
rocprof --stats --output-dir /tmp/rocprof ./target/release/kiln serve

# Real-time monitoring
radeontop
```

### Performance Expectations

Based on llama.cpp's Vulkan performance on AMD hardware:

| Operation | CUDA (RTX 4090) | Vulkan (RX 7900 XTX) | Vulkan (RX 6700 XT) |
|-----------|-----------------|------------------------|----------------------|
| FA2 prefill (64 tokens) | ~1.0x baseline | ~0.85-0.95x | ~0.60-0.75x |
| FA2 paged decode | ~1.0x | ~0.80-0.90x | ~0.55-0.70x |
| GDN recurrent step | ~1.0x | ~0.85-0.95x | ~0.60-0.75x |
| GDN forward-sub | ~1.0x | ~0.80-0.90x | ~0.55-0.70x |
| RMSNorm | ~1.0x | ~0.90-0.95x | ~0.70-0.80x |
| Causal Conv1d | ~1.0x | ~0.85-0.95x | ~0.60-0.75x |

**Key insight**: Vulkan on AMD hardware is typically 70-95% of CUDA performance on comparable-tier hardware. The gap is wider on entry-level GPUs and narrower on high-end GPUs.

### Risks and Mitigations

#### Risk 1: Vulkan SDK Availability

**Problem**: `glslc` and Vulkan SDK aren't installed by default on most Linux distros.

**Mitigation**:
- Pre-compiled SPIR-V binaries shipped with the crate (like CUDA `.o` files)
- Build script falls back to pre-compiled shaders if `glslc` not found
- Docker image includes Vulkan SDK
- Documentation with one-command Vulkan SDK install

#### Risk 2: AMD GPU Fragmentation

**Problem**: Different AMD GPUs (RDNA2 vs RDNA3 vs CDNA) have different capabilities.

**Mitigation**:
- Runtime GPU detection (like llama.cpp's `ggml_vk_get_device_description`)
- Capability queries for bfloat16, subgroup features, work group limits
- Graceful degradation: if bfloat16 isn't supported, use float16 math path
- Per-GPU tuning parameters (work group sizes, block sizes)

#### Risk 3: Shader Compilation Latency

**Problem**: Vulkan pipeline compilation can take 100ms+ on first use.

**Mitigation**:
- Precompile all pipelines at startup (like Metal's `precompile_custom_kernels`)
- Pipeline cache persisted to `~/.cache/kiln/vulkan_pipelines.bin`
- Async precompile in background thread (already exists for Metal)
- Warmup request that triggers all shader compilation

#### Risk 4: Shader Bug Detection

**Problem**: GLSL shader bugs are harder to debug than CUDA kernel bugs.

**Mitigation**:
- Start with llama.cpp's battle-tested shaders (not writing from scratch)
- Vulkan validation layers in debug builds (`VK_LAYER_KHRONOS_validation`)
- `KILN_VULKAN_DEBUG=1` enables validation layers
- Parity tests against CUDA reference for every kernel
- SPIR-V disassembly for debugging (`spirv-dis`)

#### Risk 5: candle-core Vulkan Device Support

**Problem**: candle-core 0.10.2 may not have `Device::Vulkan`.

**Mitigation**:
- Short-term: Host-memory intermediate (tensors stay on CPU, Vulkan manages its own buffers)
- Long-term: PR to add `Device::Vulkan` to candle-core
- Alternative: Fork candle-core with Vulkan support
- This is the **biggest blocker** — need to verify candle-core 0.10.2 Vulkan support first

#### Risk 6: License Compliance

**Problem**: Using llama.cpp's shaders requires MIT license attribution.

**Mitigation**:
- MIT is permissive — no compatibility issues
- Add LICENSE-LLAMA.cpp file in `kiln-vulkan-kernel/`
- Document shader provenance in crate README
- Consider writing original shaders for long-term IP cleanliness

### Phased Rollout Plan

#### Week 1-2: Foundation
- [ ] Create `kiln-vulkan-kernel` crate
- [ ] Implement SPIR-V shader compilation pipeline
- [ ] Implement `VulkanDevice` abstraction
- [ ] Implement buffer management
- [ ] Add `vulkan` feature flag to workspace
- [ ] Add `Device::Vulkan` to candle-core (or verify it exists)

#### Week 3-4: FlashAttention-2
- [ ] Port `flash_attn.comp` from llama.cpp
- [ ] Implement `flash_attn_prefill` in `VulkanBackend`
- [ ] Implement paged gather + `flash_attn_paged_decode`
- [ ] Parity tests vs CUDA reference
- [ ] Performance benchmark on target AMD GPU

#### Week 5-6: Gated DeltaNet
- [ ] Port `gated_delta_net.comp` from llama.cpp
- [ ] Implement `gdn_recurrent_step`
- [ ] Implement `gdn_forward_substitution`
- [ ] Implement `gdn_chunk_prep` + `gdn_chunk_scan`
- [ ] Implement `gdn_full_chunk_forward`
- [ ] Parity tests
- [ ] Performance benchmark

#### Week 7-8: Supporting Kernels
- [ ] RMSNorm, gated RMSNorm
- [ ] Causal Conv1d (update + prefill)
- [ ] GDN gates
- [ ] RoPE
- [ ] LM head
- [ ] Transposed GEMV
- [ ] Paged KV read
- [ ] All parity tests pass

#### Week 9: Integration
- [ ] `for_device()` Vulkan path
- [ ] `select_device()` Vulkan path
- [ ] Startup prewarm
- [ ] Pipeline caching
- [ ] Environment kill switches
- [ ] Health endpoint verification
- [ ] Desktop app Vulkan checkbox

#### Week 10: Polish
- [ ] AMD GPU profiling + tuning
- [ ] Performance regression tests
- [ ] Documentation
- [ ] Docker image with Vulkan SDK
- [ ] CI/CD Vulkan test job
- [ ] Release notes

### Summary of Key Decisions

| Decision | Choice | Rationale |
|----------|--------|----------|
| Shader source | Copy from llama.cpp (MIT) | Fastest path to production, battle-tested |
| Vulkan bindings | `vulkan-rs` crate | Rust-native, well-maintained |
| Shader compilation | `glslc` at build time | Mirrors nvcc pattern, produces portable SPIR-V |
| Kernel dispatch | Push constants + descriptor sets | Standard Vulkan pattern, minimal overhead |
| Buffer management | Own `VulkanBuffer` type | Full control over memory allocation |
| Pipeline caching | In-memory + disk persistence | Avoids 100ms+ recompilation |
| Device selection | `KILN_VULKAN_DEVICE` / `GGML_VK_VISIBLE_DEVICES` | llama.cpp compatibility |
| Kill switches | `KILN_DISABLE_*` env vars | Consistent with existing pattern |
| Tensor interface | Raw pointers via `storage_and_layout()` | Mirrors CUDA backend pattern |
| License | MIT (llama.cpp) + attribution | Permissive, no compatibility issues |
| candle-core dep | Standalone Vulkan device | Avoids candle-core Vulkan dependency |
| Shader formats | `.comp` GLSL → `.spv` → embedded bytes | Standard Vulkan pipeline |

### Next Steps (Immediate)

1. **Verify candle-core 0.10.2 Vulkan support** — does it have `Device::Vulkan`? If not, this is the #1 blocker. Options:
   - PR to add it to candle-core
   - Use host-memory intermediate (tensors on CPU, Vulkan manages own buffers)
   - Fork candle-core

2. **Set up Vulkan dev environment** on the target AMD GPU system:
   ```bash
   # Install Vulkan SDK
   sudo apt install vulkan-tools vulkan-validationlayers glslc vulkan-headers
   
   # Verify Vulkan works
   vulkaninfo --summary
   glslc --version
   ```

3. **Create `kiln-vulkan-kernel` crate** skeleton with just the build system and one trivial shader (identity copy) to validate the pipeline works

4. **Run llama.cpp with Vulkan** on the target GPU to verify the hardware works:
   ```bash
   cd /home/ericflo/Development/llama.cpp
   ./build_vulkan.sh
   ./build/bin/main -m <model> -p "hello" -ngl 99
   ```

5. **Start with FlashAttention-2** — it's the highest-value kernel and has the best-tested source in llama.cpp
