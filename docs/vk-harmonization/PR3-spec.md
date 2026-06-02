# PR3 — `MatmulOp::vulkan_fwd` + `VulkanStorage`↔`VkTensor` zero-copy bridge

**Branch:** `feat/vk-tape-harmonization`
**Issue:** #1082 (Vulkan training kt-tape harmonization, PR3 of PR1–PR7)
**Status:** SPEC ONLY — not implemented. PR1 (AdamW dispatch) and PR2
(`Device::Vulkan` first-class kt storage + host↔device copy pair +
`from_arc_buffer`/`buffer_arc`) are already merged on this branch.

This is the **perf-critical** PR of the series. It makes Vulkan matmul run
on-device through the existing GLSL GEMM instead of the dispatch2 host
round-trip, and it kills the documented per-call D2H/H2D bounce that the
~10 `vulkan_*_last_axis` wrappers in `vulkan_storage.rs` currently pay.

> ⚠ **ADVERSARIAL-REVIEW CORRECTION — scope expansion.** Matmul is **not** the
> only forward op the Vulkan training graph needs. `sum_axis` (reduce.rs),
> `{mul,add,sub}_scalar` (scalar.rs), and `log_softmax_last_dim` have **no
> `vulkan_fwd`**, and Vulkan (unlike Metal) has **no host-fallback** in
> `dispatch1/dispatch2`, so the PR5 backward composites (e.g.
> `RmsNormKtBackward`) will **hard-error** at `tape.backward()`. Either expand
> THIS PR to also add the Metal-style Vulkan host-fallback + the hot op ports, or
> insert a dedicated **PR3.5 — Vulkan backward op-coverage** before PR4. See
> `REVIEW-NOTES.md` §"[major] op-coverage" and the PR5 spec correction banner.

---

## 0. TL;DR for the implementer

Two independent, separately-shippable pieces. Do them in this order:

1. **Bridge first** (`vulkan_storage.rs`): add two thin, *pure-metadata*
   helpers — `vk_tensor_from_kt_storage(&Tensor) -> VkTensor` (kt → kernel,
   no copy) and `kt_tensor_from_vk(VkTensor, DType, dims) -> Tensor`
   (kernel → kt, no copy) — built on the PR2 primitives that *already
   exist*: `VulkanStorage::buffer_arc()`, `VulkanStorage::from_arc_buffer()`,
   and `VkTensor::from_buffer(Arc<VulkanBuffer>, ...)`. No new upstream
   kernel API is required; option (2) in the `vulkan_storage.rs:130`
   TODO is the chosen design and PR2 already landed its kt half.

2. **`MatmulOp::vulkan_fwd`** (`ops/matmul.rs`): add the missing trait
   method (only `cuda_fwd` and `metal_fwd` exist today), gated F32 +
   contiguous, sharing `validate()`. It reshapes `[..,M,K]@[..,K,N]` to the
   rank the kernel wants, bridges both inputs in, calls
   `vk_ops::matmul::vk_matmul_no_grad` (2-D) or
   `vk_ops::matmul_batched::vk_matmul_batched_no_grad` (3-D), and bridges
   the result back. Then rewrite the ~10 `vulkan_*_last_axis` wrappers to
   use the new bridge (mechanical; can be a follow-up commit in the same PR).

Then the kt-tensor-level parity test (`vulkan_matmul_f32_parity`) compares
`ops::matmul` on `Device::Vulkan(0)` against the CPU reference at real LoRA
shapes, gated behind device probe, max_abs_err threshold mirrored from the
Metal OPD gate.

---

## 1. Current state (grep-confirmed anchors — line numbers drift, the
##    function/string anchors do not)

### 1.1 `crates/kiln-tensor/src/ops/matmul.rs`
- `impl DeviceOp2 for MatmulOp` at **line 49**.
- `cpu_fwd` at **line 61** (the canonical reference; F32-accumulate triple
  loop; output is fresh `Tensor::from_parts(..., TensorId::next())`).
- `cuda_fwd` at **line 113** — F32/BF16/F16 + contiguous gate, then
  `crate::cuda_matmul(a, b)`.
- `metal_fwd` at **line 138** — BF16-only + contiguous gate, calls
  `validate(a,b)?` then `crate::metal_matmul(a, b)`. **This is the template
  to mirror.**
- **`vulkan_fwd` does NOT exist on `MatmulOp`.** The `DeviceOp2` trait
  supplies a default `vulkan_fwd` returning `Ok(None)`
  (`device_op.rs:96`), so today a Vulkan matmul hits the dispatch2
  no-Metal-no-CPU-fallback path. Per `device_op.rs:234`, for a non-CPU,
  non-Metal device that returns `None`, dispatch2 calls `op.cpu_fwd(a, b)`
  **directly on the Vulkan tensors** — which fails the `downcast_cpu`
  in `cpu_fwd` (storage is `VulkanStorage`, not `CpuStorage`) and errors
  with `"a storage must be CpuStorage"`. So matmul on Vulkan is currently
  **broken/erroring**, not silently host-falling-back. (Confirm by reading
  `device_op.rs:222-245`.) PR3 fixes this.
- `validate(a, b)` at **line 174** — rank≥2, equal ranks, leading-axis
  match, contraction-dim match, equal dtype, dtype∈{F32,BF16,F16}, both
  contiguous. **`vulkan_fwd` must call this** (mirror `metal_fwd:150`).

### 1.2 Dispatch (`crates/kiln-tensor/src/device_op.rs`)
- `dispatch2` at **line 204**; the `Device::Vulkan(_) => op.vulkan_fwd(a, b)?`
  arm is at **line 217** — already wired, no change needed.

### 1.3 `crates/kiln-tensor/src/vulkan_storage.rs` (PR2-landed primitives)
- `VulkanStorage` struct at **line 51**: holds
  `buffer: Arc<VulkanBuffer>`, `byte_len: usize`, `dtype`,
  `device: Device::Vulkan(idx)`, `vulkan_device: Arc<VulkanDevice>`.
- `from_buffer` at **line 104**, `from_arc_buffer` at **line 130** (the
  zero-copy wrap; validates `len % size_in_bytes == 0` for non-packed),
  `buffer()` at **line 158**, `buffer_arc()` at **line 164** (Arc clone,
  no device copy), `vulkan_device()` at **line 170**.
- The 3 zero-copy options are documented at **lines 491–505** on
  `vulkan_softmax_last_axis`. **PR3 chooses option (2)** (kt-side
  `from_arc_buffer`, which PR2 already added) and the existing
  `VkTensor::from_buffer(Arc<VulkanBuffer>, ...)` for the kt→kernel
  direction. No upstream `kiln-vulkan-kernel` change required.
- `host_to_vulkan_copy` (line 277) / `vulkan_to_host_copy` (line 372) /
  `primary_vulkan_device` (line 245) are the H2D/D2H + device-cache
  primitives. They stay; the bridge is the *zero-copy* path that replaces
  the per-call D2H+H2D inside the wrappers.

### 1.4 The real GEMM — `crates/kiln-vulkan-kernel/src/vk_ops/`
- **`matmul.rs`**: `vk_matmul_no_grad(a: &VkTensor, b: &VkTensor)`
  (**line 77**) — rank-2 only, **F32-only** (`check_matmul_shapes:56`
  hard-errors on non-F32), `a:[M,K] @ b:[K,N] -> [M,N]`. Output buffer
  comes from `buffer_pool::pool_alloc_f32` (**bucket-rounded** — see
  §5 GOTCHA). Dispatches `vk_matmul_f32.comp`.
- **`matmul_batched.rs`**: `vk_matmul_batched_no_grad`
  (**line 101**) — rank-3 only, F32-only, `[B,M,K]@[B,K,N]->[B,M,N]`,
  `vk_matmul_batched_f32.comp`.
- **`matmul_bf16w.rs`**: `vk_matmul_bf16w_no_grad(x, weight)` (**line 298**)
  — this is **NOT** a general A@B. It is `out = x_f32 @ W_bf16.T` with `W`
  a frozen **transposed** `[out_dim, hidden]` weight, x F32, out F32. It is
  the LoRA frozen-base projection, not `MatmulOp`. **Do not** route
  `MatmulOp::vulkan_fwd` through it (semantics + transpose mismatch). It is
  documented here only so the implementer does not reach for it.
- **There is no general BF16×BF16 → BF16 Vulkan GEMM today.** The only
  BF16 GEMM is the frozen-weight transposed path above. See §3 dtype scope.

### 1.5 `crates/kiln-vulkan-blas/src/backend_matmul.rs`
- `VulkanBackendMatmul::plan` (**line 80**) is a **pure planner / no-op**:
  it resolves a `VkWorkgroupConfig`, reports `bytes_written`, and **never
  dispatches a shader**. The doc comment at the top says the dispatch
  "lands behind the Vulkan feature when the kiln-vulkan-kernel matmul
  wrapper extension ships." **PR3 does NOT use kiln-vulkan-blas.** The real
  GEMM is `vk_ops::matmul` in kiln-vulkan-kernel (§1.4). State this loudly
  in the `vulkan_fwd` doc-comment so a future reader does not wire the
  planner.

### 1.6 `VkTensor` (`crates/kiln-vulkan-kernel/src/vk_tensor.rs`)
- `VkDType` enum (**line 29**): `F32 | Bf16` only.
- `from_buffer(storage: Arc<VulkanBuffer>, shape, dtype, device)`
  (**line 179**) — wraps an existing `Arc<VulkanBuffer>` as a leaf,
  **no copy, no autograd**. This is the kt→kernel zero-copy entry point.
- `buffer() -> &Arc<VulkanBuffer>` (**line 114**), `device()` (**line 110**),
  `shape()` (**line 102**), `num_elements()` (**line 134**).
- BF16 device buffers are padded up to whole u32 words
  (`device_buffer_bytes:43`); kt's `packed_buffer_bytes(n)` for BF16 is
  `n*2` with **no** word-rounding. They agree for even element counts; for
  odd BF16 element counts the kernel buffer is 2 bytes larger. F32 (the
  only PR3 dtype) is unaffected. Note this in the bridge for a future BF16
  enablement, but PR3 is F32-only so it is not a live concern.

---

## 2. Data / ownership model (the load-bearing part)

`VulkanStorage` and `VkTensor` are **two wrappers around the same
`Arc<VulkanBuffer>`**. `VulkanBuffer` is not `Clone`; its `Drop` calls
`vkFreeMemory` exactly once. The Arc refcount is what makes the bridge
sound: cloning the Arc bumps the count, and the device memory is freed
only when the last wrapper (kt or kernel) drops.

```
                 Arc<VulkanBuffer>  (refcount-driven; Drop => vkFreeMemory once)
                  ^             ^
   VulkanStorage  |             |  VkTensor (leaf, grad_fn=None)
   { buffer, ───┘             └─── { storage, shape, dtype=F32, device,
     byte_len,                          grad_fn:None, requires_grad:false,
     dtype, device,                     op_id, param_id:None }
     vulkan_device }
```

### 2.1 kt → kernel: `vk_tensor_from_kt_storage`
- Downcast `t.storage()` to `&VulkanStorage`; `Arc::clone` via
  `buffer_arc()` (refcount bump, **no device copy**).
- Map dtype: `DType::F32 -> VkDType::F32`; everything else is rejected
  before reaching here (the op gate is F32-only). Keep the match
  exhaustive with an explicit error arm so a future BF16 enablement is a
  one-line change.
- `VkTensor::from_buffer(arc, shape, VkDType::F32, Arc::clone(vk_device))`.
- **Contiguity / offset:** the kernel assumes a tightly-packed,
  `start_offset == 0` row-major buffer. `MatmulOp::validate()` already
  requires `is_contiguous()`. But "contiguous" can still carry a nonzero
  `start_offset` (a narrowed-but-contiguous view sharing a parent buffer).
  The kernel cannot express an offset. **The bridge MUST assert
  `t.layout().start_offset() == 0`** and return an error otherwise (the op
  caller can `.contiguous()` to materialize a zero-offset buffer). Do not
  silently produce wrong results. (`vulkan_to_host_copy` handles offsets via
  host gather; the zero-copy bridge cannot — this is the one real
  behavioral narrowing vs the D2H path, and it is correct to reject.)

### 2.2 kernel → kt: `kt_tensor_from_vk`
- Take the result `VkTensor`, `Arc::clone(vk.buffer())`.
- `VulkanStorage::from_arc_buffer(vk_device, device_index, DType::F32,
  arc, logical_byte_len)` where **`logical_byte_len = element_count *
  4`**, NOT `buffer.size()` (the pool over-allocates — see §5 GOTCHA).
  `from_arc_buffer` validates `len % size_in_bytes == 0`, so passing the
  logical length is both correct and checked.
- Wrap: `Tensor::from_parts(Arc::new(storage),
  Layout::contiguous(out_shape), TensorId::next())`.
- **TensorId:** the output is a *fresh* tensor — allocate
  `TensorId::next()`, exactly as `cpu_fwd`/`metal_matmul` do. The kernel's
  `op_id`/`param_id` are autograd-tape concerns internal to
  `kiln-vulkan-kernel` and must NOT leak into kt's `TensorId`. Forward-only
  here: PR3 attaches no backward (`MatmulOp::bwd()` is `None`; matmul
  backward lands under kiln-autograd / the kt-tape work, out of scope).

### 2.3 device_index / vulkan_device threading
- Read `device_index` from the input storage's `Device::Vulkan(i)`.
- Reuse the input's `Arc<VulkanDevice>` (`vulkan_device()`); both inputs
  share a device (dispatch2 already errors on cross-device at
  `device_op.rs:205`). Pass that same Arc into the kernel `VkTensor`s and
  into the output `VulkanStorage` so no second `VulkanDevice::new()` or
  `primary_vulkan_device` lookup happens on the hot path.

---

## 3. dtype / shape / contiguity contract for `MatmulOp::vulkan_fwd`

Mirror `metal_fwd` precisely, but the Vulkan support matrix differs:

| dtype | rank 2 | rank 3 (batched) | rank ≥ 4 |
|-------|--------|------------------|----------|
| F32   | `vk_matmul_no_grad` | `vk_matmul_batched_no_grad` | flatten leading batch dims to one (§4.2), then batched |
| BF16  | **Ok(None)** → CPU fallback | Ok(None) | Ok(None) |
| F16   | **Ok(None)** → CPU fallback | Ok(None) | Ok(None) |

- **F32-only on Vulkan today.** `check_matmul_shapes` (matmul.rs:64)
  hard-rejects non-F32, and there is no general BF16 A@B GEMM (§1.4). So
  `vulkan_fwd` gates `a.dtype()==F32 && b.dtype()==F32`; otherwise
  `return Ok(None)` and the dispatcher falls back to CPU. **Caveat:** the
  current dispatch2 CPU fallback for a non-Metal device runs `cpu_fwd` on
  the *device* tensors (device_op.rs:234), which errors for Vulkan storage.
  Two acceptable resolutions, pick one and state it in the PR:
  - **(A, preferred, minimal):** in `vulkan_fwd`, for the unsupported-dtype
    case, do the host bounce explicitly — `to_device(Cpu)` both inputs,
    `MatmulOp.cpu_fwd`, `to_device(Vulkan)` the result — i.e. extend the
    Metal-style host fallback to Vulkan *inside the op*. This keeps the
    dispatch2 change out of PR3.
  - **(B):** widen the `device_op.rs:226` Metal-scoped host-fallback arm to
    also cover `Device::Vulkan`. Slightly broader blast radius; defer
    unless the team wants the symmetry now.
  PR3 should ship **(A)** for BF16/F16 so no matmul *errors* on Vulkan,
  while F32 takes the fast on-device path. Document this in the
  doc-comment.
- **Contiguous-only:** `validate()` already enforces it. Additionally the
  bridge asserts `start_offset == 0` (§2.1).
- **Zero-size:** if `element_count == 0` for the output, return an empty
  contiguous tensor without dispatching (mirror `metal_matmul:303`); the
  GEMM ensures M,N,K > 0 (`dispatch_matmul_f32:30`) and would otherwise
  error.

---

## 4. Exact changes, site by site

### 4.1 `crates/kiln-tensor/src/vulkan_storage.rs` — add the bridge (new code)

Add, near the `from_arc_buffer` / `buffer_arc` accessors (after line ~173),
two free functions (or `impl VulkanStorage` assoc fns; free fns match the
file's `host_to_vulkan_copy` style):

```rust
/// kt → kernel zero-copy bridge. Wrap a kt Vulkan tensor's device buffer
/// as a kernel `VkTensor` leaf WITHOUT copying device memory (Arc refcount
/// bump only). The caller guarantees F32 + contiguous + start_offset==0
/// (MatmulOp::validate + the assert below). Chosen design = option (2)
/// from the zero-copy TODO at the top of vulkan_softmax_last_axis.
#[cfg(feature = "vulkan")]
pub(crate) fn vk_tensor_from_kt_storage(
    t: &crate::Tensor,
) -> Result<kiln_vulkan_kernel::vk_tensor::VkTensor> {
    use kiln_vulkan_kernel::vk_tensor::{VkDType, VkTensor};
    let vk = t.storage().as_any().downcast_ref::<VulkanStorage>()
        .ok_or_else(|| Error::Msg("vk bridge: tensor must be Vulkan-backed".into()))?;
    if t.layout().start_offset() != 0 {
        return Err(Error::Msg(
            "vk bridge: zero-copy requires start_offset==0; call .contiguous()".into()));
    }
    let vk_dtype = match t.dtype() {
        DType::F32 => VkDType::F32,
        other => return Err(Error::Msg(format!(
            "vk bridge: only F32 supported on the zero-copy path (got {other})"))),
    };
    Ok(VkTensor::from_buffer(
        vk.buffer_arc(),
        t.shape().to_vec(),
        vk_dtype,
        Arc::clone(vk.vulkan_device()),
    ))
}

/// kernel → kt zero-copy bridge. Wrap a kernel `VkTensor`'s result buffer
/// as kt `VulkanStorage` WITHOUT copying. `logical_byte_len` is the packed
/// element-count * dtype size — NOT `buffer.size()`, which the pool
/// over-allocates (see PR3 spec §5).
#[cfg(feature = "vulkan")]
pub(crate) fn kt_tensor_from_vk(
    vk: &kiln_vulkan_kernel::vk_tensor::VkTensor,
    device_index: usize,
    dtype: DType,
    out_shape: Vec<usize>,
) -> Result<crate::Tensor> {
    let n: usize = out_shape.iter().product();
    let logical_byte_len = (n * dtype.size_in_bytes()) as u64;
    let storage = VulkanStorage::from_arc_buffer(
        Arc::clone(vk.device()),
        device_index,
        dtype,
        Arc::clone(vk.buffer()),
        logical_byte_len,
    )?;
    crate::Tensor::from_parts(
        Arc::new(storage),
        crate::Layout::contiguous(out_shape),
        crate::TensorId::next(),
    )
}
```

Add the `vulkan_matmul` orchestration function in the same file (mirrors
`metal_matmul`'s role; lives in `vulkan_storage.rs` because that is where
the `kiln_vulkan_kernel` imports already are and where the other
`vulkan_*` wrappers live):

```rust
/// F32 GEMM `a[..,M,K] @ b[..,K,N] -> [..,M,N]` on Vulkan, on-device.
/// Caller (MatmulOp::vulkan_fwd) has gated F32 + contiguous and run
/// MatmulOp::validate(). Routes through kiln-vulkan-kernel's vk_ops::matmul
/// (NOT kiln-vulkan-blas, which is a no-op planner). Zero-copy in and out.
#[cfg(feature = "vulkan")]
pub fn vulkan_matmul(a: &crate::Tensor, b: &crate::Tensor) -> Result<crate::Tensor> {
    use kiln_vulkan_kernel::vk_ops::matmul::vk_matmul_no_grad;
    use kiln_vulkan_kernel::vk_ops::matmul_batched::vk_matmul_batched_no_grad;

    let ar = a.rank();
    let a_shape = a.shape();
    let b_shape = b.shape();
    let m = a_shape[ar - 2];
    let k = a_shape[ar - 1];
    let n = b_shape[b.rank() - 1];
    let batch_dims: Vec<usize> = a_shape[..ar - 2].to_vec();
    let batch: usize = batch_dims.iter().product::<usize>().max(1);

    let device_index = match a.device() {
        Device::Vulkan(i) => i,
        _ => return Err(Error::Msg("vulkan_matmul: input not on Vulkan".into())),
    };

    // Output shape = batch_dims .. M, N.
    let mut out_shape = batch_dims.clone();
    out_shape.push(m);
    out_shape.push(n);

    // Empty output: no dispatch (kernel requires M,N,K > 0).
    if m * n * k == 0 || batch == 0 {
        let storage = VulkanStorage::zeros(
            Arc::clone(
                a.storage().as_any().downcast_ref::<VulkanStorage>().unwrap().vulkan_device()),
            device_index, DType::F32, out_shape.iter().product())?;
        return crate::Tensor::from_parts(
            Arc::new(storage), crate::Layout::contiguous(out_shape), crate::TensorId::next());
    }

    let vk_out = if ar == 2 {
        // 2-D fast path.
        let va = vk_tensor_from_kt_storage(a)?;
        let vb = vk_tensor_from_kt_storage(b)?;
        vk_matmul_no_grad(&va, &vb)
            .map_err(|e| Error::Msg(format!("vulkan_matmul: vk_matmul_no_grad: {e}")))?
    } else {
        // Higher-rank: flatten ALL leading dims to one batch axis. The
        // kernel's batched matmul is strictly rank-3; reshape is
        // metadata-only on a contiguous, zero-offset buffer, so we build
        // 3-D VkTensors directly from the same Arc buffers.
        let va = {
            let v = vk_tensor_from_kt_storage(a)?;
            kiln_vulkan_kernel::vk_tensor::VkTensor::from_buffer(
                Arc::clone(v.buffer()), vec![batch, m, k], v.dtype(), Arc::clone(v.device()))
        };
        let vb = {
            let v = vk_tensor_from_kt_storage(b)?;
            kiln_vulkan_kernel::vk_tensor::VkTensor::from_buffer(
                Arc::clone(v.buffer()), vec![batch, k, n], v.dtype(), Arc::clone(v.device()))
        };
        vk_matmul_batched_no_grad(&va, &vb)
            .map_err(|e| Error::Msg(format!("vulkan_matmul: vk_matmul_batched_no_grad: {e}")))?
    };

    kt_tensor_from_vk(&vk_out, device_index, DType::F32, out_shape)
}
```

Re-export `vulkan_matmul` from `lib.rs` (the `#[cfg(feature = "vulkan")]
pub use vulkan_storage::{ ... }` block at lib.rs:158): add `vulkan_matmul`
to the list. The two `pub(crate)` bridge fns need not be re-exported but
SHOULD be used to rewrite the existing wrappers (§4.4).

### 4.2 `crates/kiln-tensor/src/ops/matmul.rs` — add `vulkan_fwd`

Insert immediately after `metal_fwd` (after line 152), before `bwd()`:

```rust
    #[cfg(feature = "vulkan")]
    fn vulkan_fwd(&self, a: &Tensor, b: &Tensor) -> Result<Option<Tensor>> {
        // #1082 PR3: route compute-bound Vulkan matmul through the
        // kiln-owned vk_ops GEMM (vk_matmul_f32 / vk_matmul_batched_f32)
        // via the zero-copy VulkanStorage<->VkTensor bridge — NOT through
        // kiln-vulkan-blas (a no-op planner). F32-only today (the GLSL
        // GEMM is F32; there is no general BF16 A@B kernel yet); BF16/F16
        // take the explicit host fallback below so matmul never errors on
        // Vulkan. Contiguous-only (same contract as cuda_fwd / metal_fwd).
        if a.dtype() != DType::F32 || b.dtype() != DType::F32 {
            // Unsupported dtype on Vulkan: explicit host bounce (the
            // dispatch2 generic fallback would run cpu_fwd on the device
            // tensors and error on the VulkanStorage downcast).
            let dev = a.device();
            let a_cpu = a.to_device(crate::Device::Cpu)?;
            let b_cpu = b.to_device(crate::Device::Cpu)?;
            return match self.cpu_fwd(&a_cpu, &b_cpu)? {
                Some(t) => Ok(Some(t.to_device(dev)?)),
                None => Ok(None),
            };
        }
        if !a.is_contiguous() || !b.is_contiguous() {
            return Ok(None);
        }
        // Only dispatch when storage is actually Vulkan (under the vulkan
        // feature a tensor can still be CPU-backed); otherwise CPU path.
        if a.storage().as_any().downcast_ref::<crate::VulkanStorage>().is_none() {
            return Ok(None);
        }
        validate(a, b)?;
        Ok(Some(crate::vulkan_matmul(a, b)?))
    }
```

`use` note: `matmul.rs` already imports `DType`, `Tensor`, `Result`,
`Error`. `crate::Device` / `crate::VulkanStorage` / `crate::vulkan_matmul`
are referenced by path so no `use` edits are required beyond confirming
they resolve under `--features vulkan`.

### 4.3 No change to `device_op.rs`
The `Device::Vulkan(_) => op.vulkan_fwd(a, b)?` arm (line 217) already
routes. (Resolution (A) in §3 keeps the BF16/F16 host fallback inside the
op, so the dispatch2 generic fallback at line 234 is never reached for
matmul. If the team later prefers resolution (B), that is a separate
one-line dispatch2 edit, out of PR3's required scope.)

### 4.4 Rewrite the `vulkan_*_last_axis` wrappers to use the bridge
**(can be a second commit in the same PR; mechanical; kills the D2H/H2D
bounce the wrappers document).** Each of these currently does
D2H read_back → H2D upload → kernel → D2H → H2D (4 host transfers/call):

- `vulkan_softmax_last_axis` (vulkan_storage.rs:521)
- `vulkan_rmsnorm_last_axis` (line 724)
- `vulkan_l2norm_last_axis` (line 997)
- `vulkan_activation_unary` (line 1183)
- `vulkan_index_select_dim0`, `vulkan_argmax_last_axis`, `vulkan_cast`,
  `vulkan_elementwise_binary`, `vulkan_masked_fill` (later in the file;
  grep `read_back` / `upload_data` to find all ~10).

For each, replace the leading "D2H input bytes → create device-local →
H2D into VkTensor" block with a single `vk_tensor_from_kt_storage(x)?`
call, and replace the trailing "D2H kernel result → H2D into kt
VulkanStorage" block with a single `kt_tensor_from_vk(&vk_out,
device_index, dtype, shape)?`. **Watch the BF16 `device_buffer_bytes`
word-padding (§1.6) for any BF16-carrying wrapper** — but all current
wrappers are F32-only, so the logical-length contract holds. The
`vulkan_rmsnorm_last_axis` `w - 1.0` host transform (line 820) must stay
on the *weight* path (it genuinely mutates host bytes); only the
non-transformed buffers move to the bridge. Keep this wrapper's weight
path as-is, bridge only `x` and the output.

---

## 5. GOTCHAS (do not skip)

1. **Pool over-allocation ≠ logical length.** `vk_matmul_no_grad`'s
   output comes from `buffer_pool::pool_alloc_f32`, which rounds to a
   `bucket_for()` size (min 64 KB; e.g. a `[3,4]` F32 = 48 B buffer is
   physically ≥ 64 KB). `VulkanBuffer::size()` returns the *bucket* size.
   `kt_tensor_from_vk` MUST compute `logical_byte_len = n * 4` itself and
   pass that to `from_arc_buffer` (which validates `% size_in_bytes`).
   Passing `buffer.size()` would set a wrong `byte_len` and corrupt every
   downstream `vulkan_to_host_copy` (it reads `byte_len` bytes). This is
   the single highest-risk detail in the PR.

2. **`start_offset != 0` on a "contiguous" view.** Contiguity does not
   imply zero offset. The kernel takes a bare `vk::Buffer` with no offset
   field. `vk_tensor_from_kt_storage` MUST reject nonzero offset (§2.1).
   Without this, a narrowed parameter (shared parent buffer) silently
   reads from the wrong base.

3. **Batched reshape is metadata-only, but only on zero-offset
   contiguous buffers.** The rank≥3 path rebuilds `[batch,M,K]` VkTensors
   from the same Arc. Safe because the buffer is row-major packed and
   `start_offset==0` (asserted). Do NOT attempt this for strided inputs.

4. **kiln-vulkan-blas is a trap.** It compiles, has tests, names itself
   `"vulkan"`, and has a `BackendMatmul` impl — but `plan()` dispatches
   nothing. Routing `vulkan_fwd` there yields *uninitialized output*. The
   doc-comment on `vulkan_fwd` must say "uses vk_ops::matmul, not
   kiln-vulkan-blas."

5. **No backward.** `MatmulOp::bwd()` returns `None`. The kernel's
   `vk_matmul` (with-grad) builds a `MatmulBackward` autograd node on the
   *kernel* tape — irrelevant to kt's forward-only dispatch. Use
   `vk_matmul_no_grad` / `vk_matmul_batched_no_grad` (no tape node, no
   wasted allocs). kt-tape matmul backward is a later PR.

6. **F32 accumulation parity.** The CPU reference (`cpu_fwd`) accumulates
   in F32; `vk_matmul_f32.comp` accumulates in F32. So F32-in/F32-out
   parity is tight (the existing kernel test asserts `< 1e-5` at K≤33,
   `< 1e-4` at K=33). Threshold scales with K (more summands → more ULP).
   See §6.

---

## 6. Parity acceptance thresholds (mirror the Metal OPD gate)

The Metal matmul gate (`metal_ops_parity.rs:556 matmul_matrix_core_f32_parity`)
uses a **relative** bound `d < 0.02 * max(|ref|, 1.0)` because its inputs
are BF16. **PR3 is F32-in/F32-out**, which is much tighter — mirror the
kernel-level F32 test (`vk_matmul_parity.rs`) and the Metal F32 op gates
(`metal_ops_parity.rs:137` softmax `< 1e-5`):

- **F32, K ≤ 64:** `max_abs_err < 1e-5`.
- **F32, 64 < K ≤ 4096:** `max_abs_err < 1e-4` (more summands; matches the
  kernel test's `1e-4` at K=33 with headroom).
- **F32, K > 4096 (real Qwen K=2560/4096 contractions):** `max_abs_err <
  1e-3` AND `max_abs_err < 1e-4 * max(|ref|,1)` (whichever is met; large-K
  F32 GEMM accumulates ~K ULPs, so an absolute-only bound is fragile at
  large result magnitudes — keep both, assert the OR).
- **Output device:** assert `got.device() == Device::Vulkan(0)` (the
  result must stay on Vulkan — proves zero-copy, no accidental host
  fallback). Mirror `metal_ops_parity.rs:583`.
- **Output shape:** assert exact `[..,M,N]`.

These thresholds are the F32 analogue of the Metal OPD `max_abs_err ~1e-5`
gate the task references; F32 GEMM cannot hold a flat `1e-5` past small K,
so the bands above are the honest, K-scaled version.

---

## 7. BOUNDED test plan (runs without any training loop)

All device tests gate behind a probe and **return early (pass) when no
Vulkan device is present** — exactly like `vk_matmul_parity.rs:26` and the
Metal `metal()` helper. None of these run a training step, a model, or any
unbounded loop.

### 7.1 Pure-CPU unit tests (no GPU; always run) — in `ops/matmul.rs#[cfg(test)]`
- **`vulkan_fwd_bf16_falls_back_without_panicking`** — construct two CPU
  BF16 tensors (no Vulkan device needed because the dtype gate returns
  before any device touch when storage is CPU). Assert
  `MatmulOp.vulkan_fwd` returns `Ok(Some(_))` via the host-fallback path or
  `Ok(None)` cleanly (depending on chosen wiring) — i.e. **never errors**.
  *(If `vulkan_fwd` is `#[cfg(feature="vulkan")]`, this test is also
  feature-gated.)*

### 7.2 Bridge unit tests (GPU, bounded, single dispatch) — new test file
`crates/kiln-tensor/tests/vulkan_matmul_parity.rs`
- **`vulkan_matmul_f32_parity`** — the core FD/parity test. For each LoRA
  shape (§7.4), build identical F32 data on CPU and on `Device::Vulkan(0)`
  (`Tensor::from_vec_on(Device::Vulkan(0), ...)` → `host_to_vulkan_copy`),
  run `ops::matmul` on both, read the Vulkan result back via
  `to_device(Cpu)` (`vulkan_to_host_copy`), compare with the §6 threshold.
  Assert device stays Vulkan and shape is exact. **One dispatch per shape;
  ~8 small shapes; completes in well under a second on GPU.**
- **`vulkan_matmul_batched_f32_parity`** — same, rank-3 `[B,M,K]@[B,K,N]`
  and one rank-4 case to exercise the leading-dim flatten (§4.2).
- **`vulkan_matmul_rejects_nonzero_offset`** — narrow a contiguous Vulkan
  tensor to a nonzero `start_offset`, assert `ops::matmul` either errors
  with the offset message OR (if the op `.contiguous()`-materializes first)
  produces a correct result; pin whichever behavior is implemented. This
  guards GOTCHA #2.
- **`vulkan_matmul_zero_size`** — `[0,K]@[K,N]` returns an empty
  `[0,N]` tensor on Vulkan without dispatching.

### 7.3 Bridge round-trip unit test (GPU, no kernel) — same file
- **`bridge_roundtrip_preserves_bytes`** — upload F32 data to a Vulkan
  tensor, `vk_tensor_from_kt_storage` → `kt_tensor_from_vk` (no kernel
  call), read back, assert bit-identical. Proves the Arc bridge does not
  copy or corrupt and that `logical_byte_len` is correct (GOTCHA #1).
  *(Requires the two bridge fns be reachable from the test — either make
  them `pub` under `#[cfg(feature="vulkan")]` or add a `#[doc(hidden)]`
  test shim. Prefer a tiny `pub` re-export `vulkan_matmul` and exercise the
  bridge transitively via `vulkan_matmul` if you do not want to widen the
  bridge fns' visibility.)*

### 7.4 LoRA parity shapes (real, small)
LoRA delta is `(x @ A.T) @ B.T`; the two matmuls have shapes:
- `x@A.T`: `[batch, in] @ [in, rank]` → `[batch, rank]`
- `h@B.T`: `[batch, rank] @ [rank, out]` → `[batch, out]`

Use these `(M,K,N)` cases (all F32, all small, all single-dispatch):
```
(16, 64, 8)      # batch=16, in=64, rank=8  (x@A.T)
(16, 8, 64)      # rank=8, out=64           (h@B.T)
(17, 33, 19)     # non-tile-aligned (mirrors kernel test tile-boundary)
(1, 2560, 16)    # M=1 decode-ish, Qwen K
(8, 2560, 16)    # small-batch prefill-ish, Qwen K (K>4096-band? no: 2560 -> 1e-4 band)
(4, 128, 4096)   # wide-N out projection, K=128
```
Batched: `([2], 8, 64, 16)`, `([3], 17, 33, 19)`, `([2,2], 4, 32, 8)`.

### 7.5 Cargo invocations (bounded; each has a timeout in CI/local)
```
# Compile gate (must be 0 — base already is):
CARGO_TARGET_DIR=/home/ericflo/Development/kiln/target \
  cargo check -p kiln-tensor --features vulkan

# Named unit + parity tests ONLY (never the full suite):
CARGO_TARGET_DIR=/home/ericflo/Development/kiln/target \
  cargo test -p kiln-tensor --features vulkan --test vulkan_matmul_parity
CARGO_TARGET_DIR=/home/ericflo/Development/kiln/target \
  cargo test -p kiln-vulkan-kernel --test vk_matmul_parity   # regression: bridge must not change kernel parity
```

---

## 8. Human-gated GPU-soak steps (EXPLICITLY SEPARATED — do NOT run
##    autonomously; the host has hard-crashed twice on long runs)

These are for a human at the machine, after the bounded tests pass:

1. **Single-step reachability smoke** (bounded, but device-touching): run
   `vulkan_matmul_parity::vulkan_matmul_f32_parity` alone with
   `--nocapture` and confirm "result stays on Vulkan" + thresholds.
2. **Wrapper-rewrite perf A/B** (manual, time-boxed): before/after the §4.4
   rewrite, run the existing `bench-results/vulkan-strix-halo-baseline.md`
   decode/prefill micro-bench (the one in MEMORY: 0.5→14.3 tok/s baseline)
   and confirm decode tok/s does not regress and ideally improves (fewer
   host bounces). **Time-box to a few iterations; do not leave running.**
3. **One short real-model forward** (human-supervised, bounded steps): a
   single prefill+1-decode-token Qwen forward on Vulkan to confirm the
   matmul fast path produces sane logits. **NOT a training loop. NOT
   multi-step. Kill after one token.**
4. **Memory-leak spot check:** run the bridge round-trip test under a
   handful of iterations and confirm `buffer_pool::pool_stats()` total
   bytes do not grow unbounded across calls (Arc drop returns buffers to
   the pool). Bounded; human reads the printout.

None of the above is part of CI; all are operator-driven and bounded.

---

## 9. Prerequisites and what this unblocks

**Prerequisite PRs (already merged on this branch):**
- **PR1** — Vulkan AdamW through `backend.dispatch_adamw_step`
  (retired `VkAdamWBook`). Independent; not strictly required for matmul,
  but part of the substrate.
- **PR2** — `Device::Vulkan` first-class kt storage: `host_to_vulkan_copy`
  / `vulkan_to_host_copy`, `VulkanStorage::from_arc_buffer` + `buffer_arc`,
  un-NYI constructors. **Hard dependency** — the bridge is built entirely
  on PR2's `from_arc_buffer`/`buffer_arc` and the H2D/D2H pair the tests
  use to construct/read Vulkan tensors.

**This PR (PR3) unblocks:**
- The remaining `vulkan_*_last_axis` wrappers' zero-copy rewrite (§4.4) —
  the documented D2H/H2D bounce removal across ~10 ops.
- **PR4+** kt-tape Vulkan forward graph: with on-device matmul + the
  bridge, the elementwise/norm/softmax ops can be chained device-resident
  without per-op host round-trips, which is the precondition for a
  kt-tape-authoritative Vulkan forward (the Metal-equivalent already
  landed — see commit b9aa7219 "kt-tape-authoritative GRPO training on
  Apple Metal").
- Vulkan matmul backward (a later PR under kiln-autograd / kt-tape):
  `MatmulOp::bwd` + `vk_matmul`'s grad path can reuse this same bridge.

**Mirror of how Metal already did it (the harmonized template):**
- Metal: `MatmulOp::metal_fwd` (matmul.rs:138) → `crate::metal_matmul`
  (metal_matmul.rs:265) → kiln-owned MSL GEMM, output is a fresh
  `MetalStorage` tensor with `TensorId::next()`, gated dtype + contiguous,
  shares `validate()`, parity-tested in `metal_ops_parity.rs:556`.
- Vulkan (PR3): `MatmulOp::vulkan_fwd` → `crate::vulkan_matmul`
  (vulkan_storage.rs) → kiln-owned GLSL GEMM via the zero-copy bridge,
  fresh `VulkanStorage` tensor with `TensorId::next()`, gated F32 +
  contiguous, shares `validate()`, parity-tested in
  `vulkan_matmul_parity.rs`. **Same shape, F32-only (vs Metal BF16) until a
  general BF16 Vulkan GEMM lands.**

---

## 10. Open risks + de-risking

| Risk | Likelihood | Impact | De-risk |
|------|-----------|--------|---------|
| `kt_tensor_from_vk` uses `buffer.size()` (bucket size) instead of logical len → corrupt readback | Med (easy mistake) | High | GOTCHA #1 + the `bridge_roundtrip_preserves_bytes` test asserts exact bytes; `from_arc_buffer` `% size_in_bytes` check catches gross errors |
| nonzero `start_offset` produces silent wrong result | Med | High | `vk_tensor_from_kt_storage` rejects offset; `vulkan_matmul_rejects_nonzero_offset` test |
| BF16/F16 matmul errors on Vulkan (dispatch2 cpu_fwd on device tensor) | High if not handled | Med | Resolution (A): explicit host fallback inside `vulkan_fwd` (§3); unit test `vulkan_fwd_bf16_falls_back_without_panicking` |
| Implementer routes through kiln-vulkan-blas (no-op planner) → uninitialized output | Low-Med | Critical | GOTCHA #4; doc-comment; the planner has no dispatch entry point so it won't even type-check into this path |
| Batched reshape on a non-contiguous/offset buffer | Low | High | gated by `validate()` contiguity + offset assert; reshape only metadata on zero-offset row-major |
| Pool buffer aliasing: output Arc shares a pooled buffer that gets recycled while kt still holds it | Low | High | `pool_alloc_*` only recycles a slot when `Arc::strong_count == 1` (buffer_pool.rs:68); kt holds a clone, so strong_count ≥ 2 → never recycled under us. Verified by reading `pool_alloc_device_local`. Spot-checked by soak step #4 |
| `vulkan_device()` Arc mismatch between two inputs on "same" device index | Low | Med | dispatch2 errors on cross-device (device_op.rs:205); both inputs share the cached `primary_vulkan_device` Arc (single physical device today) |
| Large-K F32 parity flakiness | Low | Low | K-scaled thresholds (§6), OR-of-abs-and-relative at large K |
| BF16 `device_buffer_bytes` word-padding vs kt `packed_buffer_bytes` mismatch when BF16 enabled later | Future | Med | Documented (§1.6); not live in PR3 (F32-only); leave a `// TODO(bf16): odd-count word padding` at the dtype match |

**Highest-leverage de-risk:** land the bridge + `bridge_roundtrip_preserves_bytes`
test FIRST and prove byte-exact round-trip before wiring the GEMM. Every
other risk is downstream of the bridge being correct.
