# PR4 — Backward into the Tape (`VkBwdAdapter`, op-family by op-family)

Issue: #1082
Branch: `feat/vk-tape-harmonization`
Status: **SPEC ONLY — not implemented.** PR1+PR2 are code on this branch; PR3–PR7 are specs.
Parent plan: [`docs/vulkan-train-harmonization-plan.md`](../../vulkan-train-harmonization-plan.md) §4 "PR4 — Backward into the Tape (the long pole)".

> All `file:line` anchors below were read out of the worktree at branch HEAD
> (`b94feeac`). Line numbers drift; every anchor is paired with a **grep
> string** so the implementer can re-confirm. **Grep first, trust the string,
> not the number.**

---

## 0. One-paragraph summary

The Vulkan leaf kernels already carry a complete eager autograd: every forward
op family records an `Arc<dyn VkBackwardOp>`
(`crates/kiln-vulkan-kernel/src/vk_tensor.rs:61`, grep `pub trait VkBackwardOp`)
that owns Arc-cloned input `VkTensor`s and computes input grads via on-device
kernels (`MatmulBackward`, `RmsNormBackward`, `RopeBackward`,
`SoftmaxLastDimBackward`, `FlceBackward`, `GrpoBackward`,
`GdnChunkwiseBackward`, `OpdLossBackward`, …; 33 impls, see §3 inventory). The
shared substrate walks a *different* graph: `kiln_autograd::Tape`
(`crates/kiln-autograd/src/tape.rs`) of `Box<dyn BackwardOp>`
(`crates/kiln-autograd/src/backward_op.rs:30`, grep `pub trait BackwardOp`),
whose `apply(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>>` works
over `kiln_tensor::Tensor`. PR4 introduces one thin adapter, **`VkBwdAdapter`**,
that wraps a `VkBackwardOp` and presents it as a `BackwardOp`: on `apply`, it
downcasts the incoming `grad_output` `Tensor`'s storage to `VulkanStorage`,
zero-copy-wraps it as a `VkTensor`, runs the existing vk_ops backward kernel,
and rewraps each returned `VkTensor` back into a `Tensor(VulkanStorage)`
preserving the input `TensorId`. It is landed **op-family by op-family behind a
fallback** so there is no flag day. This mirrors exactly what
`kiln-opd-loss-kernel/src/kt_tape.rs` (the CUDA/Metal OPD reverse-KL composite)
already did to bridge a fused kernel backward into the shared tape.

---

## 1. Where the adapter lives (crate / module) — and the cycle that forces it

The adapter needs to name three things at once:

1. `kiln_autograd::{BackwardOp, Tape}` — the shared-tape trait it implements.
2. `kiln_vulkan_kernel::vk_tensor::{VkTensor, VkBackwardOp, VkDType}` — the leaf
   backward it wraps.
3. `kiln_tensor::{Tensor, VulkanStorage, DType, ...}` — the storage it
   downcasts / rewraps.

**Cargo edge analysis (verified):**

- `kiln-tensor` depends on `kiln-vulkan-kernel` as a **normal, optional** dep
  under the `vulkan` feature
  (`crates/kiln-tensor/Cargo.toml`, grep `kiln-vulkan-kernel = { path`, and
  `vulkan = ["dep:kiln-vulkan-kernel"]`).
- `kiln-vulkan-kernel` depends on `kiln-tensor` only as a **dev-dependency**
  (`crates/kiln-vulkan-kernel/Cargo.toml:30`, under `[dev-dependencies]`). So
  the real build edge is `kiln-tensor → kiln-vulkan-kernel`. The adapter **cannot**
  live in `kiln-vulkan-kernel` (it would need `kiln-tensor::Tensor` in non-dev code,
  reversing the edge into a cycle).
- `kiln-autograd` is only a **dev-dependency** of `kiln-tensor`
  (`crates/kiln-tensor/Cargo.toml:111` `[dev-dependencies]`, grep
  `kiln-autograd = { workspace = true }`). So `kiln-tensor` production code
  **cannot** implement `kiln_autograd::BackwardOp` either.

The crate that already names **all three in non-dev code under one feature** is
`kiln-model`:

- `crates/kiln-model/Cargo.toml:28` `kiln-vulkan-kernel = { workspace = true, optional = true }`
- `crates/kiln-model/Cargo.toml:34` `kiln-autograd = { workspace = true, optional = true }`
- `crates/kiln-model/Cargo.toml:49` `kiln-tensor = { workspace = true }`

And it is the precedent home: the CUDA/Metal kt-tape `BackwardOp` impls live in
`crates/kiln-model/src/tape_forward.rs` (grep `impl BackwardOp for RmsNormKtBackward`,
`impl BackwardOp for FlashAttnBackward`, …), and the entire `VkTensor` forward
lives in `crates/kiln-model/src/vk_forward.rs` (grep
`use kiln_vulkan_kernel::vk_tensor` / `use kiln_vulkan_kernel::{VkDType, VkTensor`).

**Decision: the adapter lives in `kiln-model`, in a new module
`crates/kiln-model/src/vk_bwd_adapter.rs`, gated `#[cfg(feature = "vulkan")]`.**
It is `mod`-declared from `crates/kiln-model/src/lib.rs` next to the existing
`vk_forward` declaration (grep `mod vk_forward` / `pub mod vk_forward` in
`crates/kiln-model/src/lib.rs` to match the exact visibility convention).

### 1.1 Required Cargo changes (mandatory — without these it will not compile)

`kiln-model`'s `vulkan` feature today is
(`crates/kiln-model/Cargo.toml:94`, grep `^vulkan = `):

```toml
vulkan = ["dep:kiln-vulkan-kernel", "dep:kiln-tensor-id", "kiln-core/vulkan", "dep:half", "dep:bytemuck"]
```

It is **missing two activations** that `cuda` and `metal` already carry
(`crates/kiln-model/Cargo.toml:76` and `:85`):

- `"dep:kiln-autograd"` — so `kiln_autograd::BackwardOp`/`Tape` are in scope.
- `"kiln-tensor/vulkan"` — so `kiln_tensor::VulkanStorage` and the
  `host_to_vulkan_copy` / `vulkan_to_host_copy` / bridge fns are reachable
  (they are exported only behind `kiln-tensor`'s own `vulkan` feature:
  `crates/kiln-tensor/src/lib.rs:158`, grep `pub use vulkan_storage::{`).

The change:

```toml
vulkan = ["dep:kiln-vulkan-kernel", "dep:kiln-tensor-id", "kiln-core/vulkan", "dep:half", "dep:bytemuck", "dep:kiln-autograd", "kiln-tensor/vulkan"]
```

Confirm `dep:half`/`dep:bytemuck` are still listed once (do not duplicate).
After this edit, run the gate from §7.1 before touching any `.rs`.

---

## 2. The two traits, side by side (the exact contract to bridge)

`VkBackwardOp` — `crates/kiln-vulkan-kernel/src/vk_tensor.rs:61` (grep `pub trait VkBackwardOp`):

```rust
pub trait VkBackwardOp: Send + Sync + std::fmt::Debug {
    fn op_name(&self) -> &'static str;
    fn input_refs(&self) -> &[VkTensor];                       // saved inputs, in fwd order
    fn backward(&self, grad_out: &VkTensor) -> Result<Vec<Option<VkTensor>>>;
}
```

`BackwardOp` — `crates/kiln-autograd/src/backward_op.rs:30` (grep `pub trait BackwardOp`):

```rust
pub trait BackwardOp: Send + Sync + std::fmt::Debug {
    fn name(&self) -> &'static str;
    fn input_count(&self) -> usize;
    fn apply(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>>;
    fn requires_input(&self, _idx: usize) -> bool { true }     // default
}
```

The mapping is one-to-one:

| `BackwardOp`                     | source on the wrapped `VkBackwardOp`                                    |
| ------------------------------- | ---------------------------------------------------------------------- |
| `name()`                        | `inner.op_name()` (same stable label; reused as the parity-tolerance key) |
| `input_count()`                 | `inner.input_refs().len()`                                              |
| `apply(grad_output: &Tensor)`   | downcast → `VkTensor` → `inner.backward(&vk_grad)` → rewrap each result |
| `requires_input(idx)`           | default `true` (the vk kernels read all saved inputs; see §3 notes)     |

**Result-vector contract (load-bearing — both walkers enforce it):**
`apply` MUST return a `Vec` of length exactly `input_count()`, in the **same
order** the forward consumed inputs. `None` entries pass through unchanged
(non-differentiable inputs, e.g. `RmsNormBackward` returns `[Some(dx), None]`
for the frozen weight — `crates/kiln-vulkan-kernel/src/vk_ops/rmsnorm.rs:132`,
grep `Ok(vec![Some(grad_x), None])`). The kt tape walker checks this length at
`crates/kiln-autograd/src/tape.rs:236` (grep `returned {} grads for {} inputs`);
the vk walker checks the analogous invariant at
`crates/kiln-vulkan-kernel/src/vk_autograd.rs:104` (grep
`returned {} grads for {} inputs`). The adapter must preserve element count and
order **exactly**; never drop a `None`.

---

## 3. Op-family inventory + landing order

33 `VkBackwardOp` impls exist today (grep `impl VkBackwardOp` across
`crates/kiln-vulkan-kernel/src/vk_ops/`). PR4 ports them in two waves, each
family independently behind the fallback (§4.3).

### Wave 1 — the core training path (matmul / rmsnorm / rope / softmax)

| Family   | Struct (`grep`)            | File:line                                            | arity | grad slots |
| -------- | -------------------------- | ---------------------------------------------------- | ----- | ---------- |
| matmul   | `MatmulBackward`           | `vk_ops/matmul.rs:94`  (`impl VkBackwardOp for Matmul`) | 2     | `[Some,Some]` |
| matmul-bf16w | `MatmulBf16wBackward`   | `vk_ops/matmul_bf16w.rs:208`                         | 1     | `[Some]` (frozen W) |
| matmul-batched | `MatmulBatchedBackward` | `vk_ops/matmul_batched.rs:141`                     | 2     | `[Some,Some]` |
| rmsnorm  | `RmsNormBackward`          | `vk_ops/rmsnorm.rs:103`                              | 2     | `[Some,None]` (frozen W) |
| rope     | `RopeBackward`             | `vk_ops/rope.rs:120`                                 | 1     | `[Some]` |
| softmax  | `SoftmaxLastDimBackward`   | `vk_ops/softmax.rs:95`                               | 1     | `[Some]` |
| (support) elementwise `Add/Sub/Mul/Div` | `vk_ops/elementwise.rs:95/113/132/152` | 2 each | mixed |
| (support) `ReshapeBackward`/`Transpose2dBackward` | `vk_ops/shape.rs:24/180` | 1 | `[Some]` |
| (support) `SumAllBackward`/`MeanAllBackward` | `vk_ops/reduce.rs:239/260` | 1 | `[Some]` |

Wave-1 families are pure tensor ops with no host-side metadata (see §3.1) and
are FD-checkable with no kernel-specific oracle. Land matmul first (the perf
spine; it has an existing CPU oracle `naive_matmul` in
`crates/kiln-vulkan-kernel/tests/vk_matmul_parity.rs`).

### Wave 2 — fused / metadata-carrying families (flce / gdn / opd)

| Family   | Struct                     | File:line                  | arity | notes |
| -------- | -------------------------- | -------------------------- | ----- | ----- |
| flce     | `FlceBackward`             | `vk_ops/flce.rs:200` (struct `:187`) | 1 (`hidden`) | host-side label/teacher metadata closed over in the struct |
| grpo     | `GrpoBackward`             | `vk_ops/flce.rs:348` (struct `:335`) | 1 (`hidden`) | host-side advantage/logprob metadata in struct |
| gdn      | `GdnChunkwiseBackward`     | `vk_ops/gdn_chunkwise.rs:1396` (inputs `[q,k,v,beta,g]` `:1203`) | 5 | the heaviest backward; chunkwise state |
| gdn-gates | `GdnGatesBackward`        | `vk_ops/gdn_gates.rs:198`  | (grep) | |
| gdn-rms  | `GdnGatedRmsNormBackward`  | `vk_ops/gdn_gated_rms_norm.rs:171` | (grep) | |
| opd      | `OpdLossBackward`          | `vk_ops/opd.rs:392` (`[Some(dh)]` `:401`) | 1 (`hidden`) | top-K reverse-KL; the **direct analogue** of the CUDA/Metal kt_tape OPD backward already shipped |

**§3.1 — host-side metadata is already inside the `VkBackwardOp` struct.**
This is the key simplification: families like `FlceBackward`, `GrpoBackward`,
`OpdLossBackward` close over their teacher/label/advantage arrays *inside the
struct* at forward-record time (e.g. `OpdLossBackward` holds `self.state` with
`topk_idx_buf` etc.; `FlceBackward`/`GrpoBackward` hold `inputs: [hidden]` plus
host metadata fields). The adapter therefore **does not** need to thread any
side metadata through the `BackwardOp::apply` signature — it wraps the existing
`Arc<dyn VkBackwardOp>` whole and only translates the grad tensor at the
boundary. This is exactly the pattern
`CudaOpdTopKReverseKlPhaseBBackward` uses
(`crates/kiln-opd-loss-kernel/src/kt_tape.rs:143`, grep
`pub struct CudaOpdTopKReverseKlPhaseBBackward`): saved tensors + host arrays
live in the struct; `apply` only consumes `grad_output`.

---

## 4. The adapter — concrete implementation

### 4.1 Data / ownership model (read this before writing code)

- **`VulkanStorage` owns `Arc<VulkanBuffer>`.**
  `crates/kiln-tensor/src/vulkan_storage.rs:51` (struct), field `buffer: Arc<VulkanBuffer>`
  (`:54`). Accessors: `buffer_arc(&self) -> Arc<VulkanBuffer>` (`:164`, refcount
  bump, **no device copy**), `vulkan_device(&self) -> &Arc<VulkanDevice>` (`:170`),
  `byte_len()`, `device() -> Device::Vulkan(i)`.
- **`VkTensor` also owns `Arc<VulkanBuffer>`.**
  `crates/kiln-vulkan-kernel/src/vk_tensor.rs:72` (`storage: Arc<VulkanBuffer>`),
  constructed zero-copy from an existing Arc via
  `VkTensor::from_buffer(Arc<VulkanBuffer>, shape, VkDType, Arc<VulkanDevice>)`
  (`:179`, grep `pub fn from_buffer`).
- **Therefore the bridge is a refcount bump in both directions** — the same
  device memory is shared; no D2H/H2D. This is the PR3 zero-copy guarantee
  (`VulkanStorage::from_arc_buffer`, `crates/kiln-tensor/src/vulkan_storage.rs:130`,
  grep `pub fn from_arc_buffer`). The currently-shipping
  `vulkan_softmax_last_axis` (`:521`) is the **worked round-trip template**, but
  it still bounces through the host (D2H read_back → H2D upload, `:562`–`:614`)
  because PR3's zero-copy wiring of *that op* is a separate task; PR4 must use
  the **zero-copy** `from_buffer`/`from_arc_buffer` path, not the host bounce.
  See §8 risk R3.

**Dtype / contiguity contract:**

- vk kernels are F32 on the hot path (`vk_matmul` asserts F32-only at
  `crates/kiln-vulkan-kernel/src/vk_ops/matmul.rs:64`, grep
  `Phase B is F32-only`). The adapter maps `DType::F32 ↔ VkDType::F32`
  and `DType::BF16 ↔ VkDType::Bf16` (mapping precedent:
  `crates/kiln-tensor/src/vulkan_storage.rs:577` `DType::F32 => VkDType::F32`
  and `:1658` the BF16 pair). Any other dtype → hard `Err` (do not silently
  cast).
- `VkTensor` storage is **always C-contiguous** (module doc
  `crates/kiln-vulkan-kernel/src/vk_tensor.rs:7`). The adapter must reject a
  non-contiguous `grad_output` (`grad_output.is_contiguous()` is false) with an
  `Err`, OR materialize via `grad_output.contiguous()?` first. **Prefer
  `contiguous()?`** — it is a no-op when already contiguous and matches how
  `host_to_vulkan_copy` handles it (`crates/kiln-tensor/src/vulkan_storage.rs:281`,
  grep `let contig = cpu.contiguous()`). Grad tensors flowing from the tape
  walker are contiguous in practice, so this is defensive.
- **Shape:** carry `grad_output.shape().to_vec()` onto the wrapped `VkTensor`.
  The vk kernel validates shapes internally and the kt walker re-checks each
  returned grad's shape against the recorded input at
  `crates/kiln-vulkan-kernel/src/vk_autograd.rs:116` (vk side) and implicitly on
  the kt side via the consumer op; the adapter need not re-assert but should not
  reshape.

### 4.2 `TensorId` preservation (the subtle, load-bearing part)

The kt tape keys gradients by **output/input `TensorId`**
(`Tape::record(output, inputs, op)` captures `output.id()` and each
`input.id()` — `crates/kiln-autograd/src/tape.rs:86`, grep `pub fn record`). On
backward, the walker accumulates the i-th returned grad against the i-th
recorded **input id** (`crates/kiln-autograd/src/tape.rs`, the per-input
accumulation loop after `node.op.apply`). **The adapter does NOT assign ids** —
the tape walker does, using the recorded `node.input_ids[i]`. The adapter's only
job is to return grads **in input order** with `None` in the right slots; the
id-binding is the walker's. (Contrast the *vk-native* walker, which keys by
`op_id` and maps back to `param_id` at the end —
`crates/kiln-vulkan-kernel/src/vk_autograd.rs:139`. On the kt tape this is the
walker's `TensorId` map, so the adapter is simpler: it never sees ids.)

This is why the §0 phrase "preserving input `TensorId`" resolves to: **emit the
grad for input *i* in slot *i*; the walker binds it to `node.input_ids[i]`.**
The `Tensor` the adapter mints for a grad gets a *fresh* `TensorId::next()` (it
is a new tensor) — that fresh id is irrelevant; only the slot position matters.

### 4.3 The adapter struct + fallback gate

```rust
// crates/kiln-model/src/vk_bwd_adapter.rs   (NEW, #[cfg(feature = "vulkan")])
use std::sync::Arc;
use kiln_autograd::BackwardOp;
use kiln_tensor::{DType, Device, Error, Result, Tensor, VulkanStorage};
use kiln_vulkan_kernel::vk_tensor::{VkBackwardOp, VkDType, VkTensor};

/// Presents a leaf `VkBackwardOp` to the shared `kiln_autograd::Tape`.
/// Zero-copy at the storage boundary: `grad_output`'s `Arc<VulkanBuffer>`
/// is refcount-bumped into a `VkTensor`, the leaf kernel runs, and each
/// returned `VkTensor`'s `Arc<VulkanBuffer>` is refcount-bumped back into
/// a fresh `Tensor(VulkanStorage)`.
#[derive(Debug)]
pub struct VkBwdAdapter {
    inner: Arc<dyn VkBackwardOp>,
    /// Vulkan physical-device index, captured at record time so the
    /// rewrapped output `Tensor`s report the right `Device::Vulkan(i)`.
    device_index: usize,
}

impl VkBwdAdapter {
    pub fn new(inner: Arc<dyn VkBackwardOp>, device_index: usize) -> Self {
        Self { inner, device_index }
    }
}

fn dtype_to_vk(d: DType) -> Result<VkDType> {
    match d {
        DType::F32 => Ok(VkDType::F32),
        DType::BF16 => Ok(VkDType::Bf16),
        other => Err(Error::Msg(format!(
            "VkBwdAdapter: unsupported grad dtype {other} (F32/BF16 only)"
        ))),
    }
}
fn vk_to_dtype(d: VkDType) -> DType {
    match d { VkDType::F32 => DType::F32, VkDType::Bf16 => DType::BF16 }
}

/// kt Tensor(VulkanStorage) -> VkTensor, zero-copy (Arc bump).
fn tensor_to_vk(t: &Tensor) -> Result<VkTensor> {
    let t = t.contiguous()?; // no-op when already contiguous
    let vs = t.storage().as_any().downcast_ref::<VulkanStorage>()
        .ok_or_else(|| Error::Msg("VkBwdAdapter: grad_output is not Vulkan-backed".into()))?;
    let vk_dtype = dtype_to_vk(vs.dtype())?;
    Ok(VkTensor::from_buffer(
        vs.buffer_arc(),                 // Arc<VulkanBuffer> refcount bump
        t.shape().to_vec(),
        vk_dtype,
        Arc::clone(vs.vulkan_device()),
    ))
}

/// VkTensor -> kt Tensor(VulkanStorage), zero-copy (Arc bump).
fn vk_to_tensor(v: &VkTensor, device_index: usize) -> Result<Tensor> {
    let dtype = vk_to_dtype(v.dtype());
    let storage = VulkanStorage::from_arc_buffer(
        Arc::clone(v.device()),
        device_index,
        dtype,
        Arc::clone(v.buffer()),          // Arc<VulkanBuffer> refcount bump
        v.byte_size() as u64,
    )?;
    Tensor::from_parts(
        Arc::new(storage),
        kiln_tensor::Layout::contiguous(v.shape().to_vec()),
        kiln_tensor::TensorId::next(),   // fresh id; slot position is what binds the grad
    )
}

impl BackwardOp for VkBwdAdapter {
    fn name(&self) -> &'static str { self.inner.op_name() }
    fn input_count(&self) -> usize { self.inner.input_refs().len() }

    fn apply(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let vk_grad = tensor_to_vk(grad_output)?;
        let vk_grads = self.inner.backward(&vk_grad)
            .map_err(|e| Error::Msg(format!("VkBwdAdapter[{}] backward: {e}", self.inner.op_name())))?;
        // Preserve length + slot order + None positions EXACTLY.
        vk_grads.into_iter().map(|opt| match opt {
            None => Ok(None),
            Some(v) => Ok(Some(vk_to_tensor(&v, self.device_index)?)),
        }).collect()
    }
}
```

Notes on exactness:

- `Tensor::from_parts(storage, layout, id)` — `crates/kiln-tensor/src/tensor.rs:381`
  (grep `pub fn from_parts`). It validates `layout.addressable_byte_size <=
  storage.byte_len()`; the contiguous layout over `v.shape()` always satisfies
  this for a kernel result.
- `VulkanStorage::from_arc_buffer(vulkan_device, device_index, dtype, Arc<VulkanBuffer>, size_bytes)`
  — `crates/kiln-tensor/src/vulkan_storage.rs:130`. `size_bytes = v.byte_size()`
  (`VkTensor::byte_size` = `num_elements() * dtype.byte_size()`,
  `crates/kiln-vulkan-kernel/src/vk_tensor.rs:138`). For BF16 the device buffer
  may carry a 4-byte-word pad; pass the **logical** byte size (`byte_size()`),
  matching how `to_bytes()` truncates to the logical size (`vk_tensor.rs:436`).
- `Error`/`Result`/`Layout`/`TensorId` are re-exported from `kiln_tensor`
  (grep `pub use` in `crates/kiln-tensor/src/lib.rs`).

### 4.4 The fallback — how a family lands behind it

The fallback is the **forward recorder's** choice, not the adapter's. PR5 owns
the recorders, but PR4 must give the recorder a clean switch so each family can
flip independently. The pattern mirrors the OPD `envelope_ok` gate
(`crates/kiln-opd-loss-kernel/src/kt_tape.rs:91`, grep `fn envelope_ok`):

- For a ported family, the Vulkan forward records onto the shared `Tape` via
  `tape.record(&out, &[inputs...], Box::new(VkBwdAdapter::new(grad_fn, dev_idx)))`
  — exactly like the OPD kt_tape recorder
  (`crates/kiln-opd-loss-kernel/src/kt_tape.rs:326`, grep `tape.record(`).
- For an **un-ported** family, the forward keeps recording the *vk-native* graph
  (the `Arc<dyn VkBackwardOp>` on the `VkTensor`) and the vk-native
  `vk_backward` walk (`crates/kiln-vulkan-kernel/src/vk_autograd.rs:72`) is used
  for it. No flag day: a build with only matmul/rmsnorm/rope/softmax ported runs
  those four on the kt tape and everything else on the vk-native walker.

**PR4's deliverable for the fallback is a single boolean policy fn** that the
PR5 recorders call, e.g. in `vk_bwd_adapter.rs`:

```rust
/// Families whose VkBwdAdapter is validated and may record onto the
/// shared kt Tape. Everything else falls through to the vk-native
/// `vk_backward` walk. Extended op-family-by-op-family as FD parity lands.
pub fn family_ported(op_name: &str) -> bool {
    matches!(op_name,
        "matmul" | "matmul_bf16w" | "rms_norm" | "rope" | "softmax_lastdim"
        // wave 2 appended here as each family's FD test goes green:
        // | "flce" | "grpo" | "gdn_chunkwise" | "opd_topk_kl"
    )
}
```

(Op names are the `op_name()` returns: `"matmul"` `matmul.rs:96`, `"rms_norm"`
`rmsnorm.rs:105`, `"opd_topk_kl"` `opd.rs:394`. **Grep each `fn op_name`** to
copy the exact literal — do not guess, e.g. softmax's literal must be read from
`vk_ops/softmax.rs` around `:96`.)

This keeps PR4 self-contained: the adapter + the gate fn ship together; PR5
wires the gate into the recorders.

---

## 5. The eager-DFS vs push-recorded-Tape reconciliation

Two distinct autograd engines exist; PR4 deliberately bridges them. The task
called out three reconciliation points — here is the precise status of each,
re-verified against the tree.

### 5.1 Topo order

- vk-native: **eager DFS** from loss, post-order push, reverse-walk
  (`crates/kiln-vulkan-kernel/src/vk_autograd.rs:152` `fn collect_topo`,
  consumers-before-producers via `order.iter().rev()` at `:93`).
- kt tape: **push-recorded**, already topo because each forward records before
  its consumers; walked in reverse insertion order
  (`crates/kiln-autograd/src/tape.rs:200`, grep `Insertion order is already
  topo-sorted producer-before-consumer`).

**Requirement:** when a family is ported, the **forward must `tape.record` in
the same order it executes** so reverse-insertion order is valid. This is free
for a straight-line forward (the common case) and is exactly how the OPD
recorder does it (record immediately after computing the output,
`kt_tape.rs:326`). No DFS is needed on the kt side. **Action item for PR5, noted
here so PR4's adapter contract is unambiguous:** do not record a node before its
inputs' producing nodes are recorded.

### 5.2 In-place version counter (anti-pattern 16)

- kt tape: every node captures `input_versions` + live `Arc<AtomicU64>` handles
  at record time (`crates/kiln-autograd/src/tape.rs:88`–`95`) and **panics/errs
  on drift** at backward (`:213`–`:226`, grep `version drifted`).
- vk side: `VkTensor` has **no** version counter; the buffer is shared by Arc and
  in-place ops would mutate it silently.

**Requirement:** any Vulkan in-place op whose result feeds a kt-recorded node
must bump the kt `Tensor`'s version via `Tensor::bump_version()`
(`crates/kiln-tensor/src/tensor.rs:876`, grep `pub fn bump_version`). The
adversarial finding in the parent plan (§5.1) holds: **this is NOT a live
corruption risk today** because the two engines are split by device and no
shared persistent tape crosses an in-place mutation. It becomes *managed* work
the moment a family is ported. Concretely, audit the in-place vk ops before
porting any family that consumes their output:
`vk_scatter_to_lastdim_slice_inplace` (grep, used in `vk_forward.rs:44`) and any
`*_inplace` kernel. For Wave 1 (matmul/rmsnorm/rope/softmax) **none of the saved
inputs are mutated in place** — they are pure functional ops — so Wave 1 lands
with zero version-counter work. Document this per-family in the PR description.

### 5.3 Grad accumulation

- vk-native: `vk_add_no_grad` accumulates multi-use grads
  (`crates/kiln-vulkan-kernel/src/vk_autograd.rs:126`).
- kt tape: the caller supplies the accumulator closure to `Tape::backward`
  (`crates/kiln-autograd/src/tape.rs:124`, the `F: FnMut(&Tensor, &Tensor)`
  param; typically `|a, b| kiln_tensor::ops::add(a, b)`).

**Requirement:** the accumulator passed to `Tape::backward` must be able to add
two `Tensor(VulkanStorage)` grads. That add must dispatch to a Vulkan
elementwise-add `vulkan_fwd` (the `AddOp` path through `dispatch2`,
`crates/kiln-tensor/src/device_op.rs:217`). **Prerequisite check:** confirm
`vulkan_elementwise_binary` is wired as `AddOp::vulkan_fwd` (it is exported,
`crates/kiln-tensor/src/lib.rs:160`, grep `vulkan_elementwise_binary`); if the
add path is not yet a `vulkan_fwd`, accumulation of a multi-use Vulkan grad will
error. For Wave 1 single-use parameter grads (LoRA A/B each used once) this does
not trigger; flag it for any fan-out family in Wave 2.

---

## 6. Mirror of how Metal already did the equivalent (the template)

Metal harmonized its fused OPD backward into the shared tape in commit
`948bbe0f`. The template is `crates/kiln-opd-loss-kernel/src/kt_tape.rs` (read in
full — it is 680 lines and is the canonical pattern). Direct correspondences:

| Metal/CUDA OPD kt_tape (template)                                   | Vulkan PR4 equivalent                                       |
| ------------------------------------------------------------------- | ---------------------------------------------------------- |
| `struct CudaOpdTopKReverseKlPhaseBBackward` holds saved tensors + host metadata (`kt_tape.rs:143`) | the existing `VkBackwardOp` struct already holds these (§3.1) — adapter wraps it whole |
| `impl BackwardOp` with `name`/`input_count`/`apply` (`kt_tape.rs:166`) | `impl BackwardOp for VkBwdAdapter` (§4.3)                   |
| `apply` dispatches by device, returns `vec![Some(d_hidden), None]` (`kt_tape.rs:219`) | `apply` translates grad, returns `inner.backward(...)` slot-for-slot |
| `envelope_ok` gate routes around the kt path off-envelope (`kt_tape.rs:91`) | `family_ported(op_name)` gate (§4.4) — fallback to vk-native |
| recorder `..._via_kt_tape` calls `tape.record(&y, &[h, w], Box::new(bwd))` (`kt_tape.rs:326`) | PR5 recorders call `tape.record(&out, &inputs, Box::new(VkBwdAdapter::new(gf, idx)))` |
| `requires_input(idx)` declares which saved inputs are read (`kt_tape.rs:240`) | default `true`; tighten per-family if a frozen input is unread |

The **one structural difference**: Metal's OPD backward is a *single composite*
that re-derives `d_hidden` analytically (`opd_top_k_reverse_kl_phase_b_bwd_composite_kt`,
`kt_tape.rs:223`). Vulkan's are *per-op-family fused kernels*. So Vulkan ports
**many small adapters** (one `VkBwdAdapter` instance per recorded node) rather
than one composite. The adapter is generic over `Arc<dyn VkBackwardOp>`, so this
is a single struct reused across all families — strictly simpler than Metal's
per-loss composite.

---

## 7. Bounded test plan (runs without long training)

> Host-safety: the dev host has hard-crashed on long runs. **Everything in §7.1
> and §7.2 is bounded** (single dispatch, tiny shapes, deterministic). The GPU
> soak in §7.3 is **human-gated** and must NOT be run autonomously.

### 7.1 Compile gates (always)

```
# from /home/ericflo/Development/kiln-vk-harmonize, target dir shared:
CARGO_TARGET_DIR=/home/ericflo/Development/kiln/target \
  cargo check -p kiln-model --features vulkan            # adapter compiles, Cargo edges resolve
CARGO_TARGET_DIR=/home/ericflo/Development/kiln/target \
  cargo check -p kiln-train --features vulkan            # confirmed 0 at base; must stay 0
```

Run the first one **immediately after the Cargo.toml edit in §1.1**, before any
`.rs`, to confirm the `dep:kiln-autograd` + `kiln-tensor/vulkan` activations
resolve without a cycle.

### 7.2 Bounded named tests (per family — FD parity)

These live in `crates/kiln-model/tests/vk_bwd_adapter_parity.rs` (NEW) and reuse
the **exact** FD harness already proven in
`crates/kiln-vulkan-kernel/tests/vk_matmul_parity.rs` (`fn fd_grad` central
difference at `:109`, `fn naive_matmul` at `:57`, `fn max_abs_diff` at `:49`).
Each test:

1. probes the device (`VulkanDevice::probe()`, return `Ok(())` / skip when
   absent — `vk_matmul_parity.rs:26` `fn vk_dev`);
2. builds a tiny F32 input on Vulkan (`Tensor::from_vec_on(... Device::Vulkan(0))`,
   un-NYI'd by PR2 — `crates/kiln-tensor/src/tensor.rs:241`);
3. runs the forward through the (PR5) recorder OR, for a PR4-only test, builds a
   `VkBwdAdapter` directly over a hand-constructed `VkBackwardOp` and calls
   `apply(grad)` — **this is the PR4-checkable unit** (does not need PR5);
4. compares the adapter's returned grad against a **central finite-difference**
   gradient of the same scalar loss (`fd_grad`, eps `1e-3`).

Named bounded tests (all `#[cfg(feature = "vulkan")]`, all skip cleanly w/o GPU):

| Test name                                       | Family   | Oracle                          | Tolerance |
| ----------------------------------------------- | -------- | ------------------------------- | --------- |
| `vk_bwd_adapter_matmul_fd_parity`               | matmul   | analytic + FD (`naive_matmul`)  | `max_abs_err < 1e-5` (F32) |
| `vk_bwd_adapter_rmsnorm_fd_parity`              | rmsnorm  | FD of `mean(rmsnorm(x,w))`      | `< 1e-4` (eps-sensitive) |
| `vk_bwd_adapter_rope_fd_parity`                 | rope     | FD of `mean(rope(x))`           | `< 1e-4` |
| `vk_bwd_adapter_softmax_fd_parity`              | softmax  | FD of `mean(softmax(x))`        | `< 1e-4` |
| `vk_bwd_adapter_preserves_none_slots`           | rmsnorm  | assert `apply` returns `[Some,None]`, frozen-weight slot is `None` | exact |
| `vk_bwd_adapter_input_count_matches`            | all wave1 | `input_count() == inner.input_refs().len()` | exact |
| `vk_bwd_adapter_rejects_non_vulkan_grad`        | n/a      | feed a CPU `Tensor` → expect `Err` | exact |
| `vk_bwd_adapter_rejects_unsupported_dtype`      | n/a      | feed an I64 grad → expect `Err` | exact |
| `vk_bwd_adapter_zero_copy_shares_buffer`        | matmul   | assert output storage's `Arc<VulkanBuffer>` strong-count > 1 after wrap (no copy) | exact |

**Acceptance threshold (mirrors the Metal OPD gate / §9.2 grand-plan gate):**
`max_abs_err ~1e-5` for F32 on the **analytic-comparable** ops (matmul); the
eps-perturbed FD oracle on the smooth fused ops uses `< 2e-3` to absorb central
-difference truncation (exactly as `vk_matmul_parity.rs:394` loosened to `2e-3`
for the LoRA composition). The **cross-engine** OPD parity gate stays at the
literal `1e-5`/`1e-4` band already enforced in
`crates/kiln-train/tests/vk_cuda_opd_parity.rs` (grep `1e-5` / `1e-4`).

Wave-2 families add, when ported:

| Test name                                       | Family | Oracle |
| ----------------------------------------------- | ------ | ------ |
| `vk_bwd_adapter_opd_fd_parity`                  | opd    | FD of the reverse-KL composite (mirror the OPD reverse-KL composite gate already in `kiln-opd-loss-kernel/src/kt_api.rs`, grep `2.0 * h`) |
| `vk_bwd_adapter_flce_fd_parity`                 | flce   | FD of `mean(flce_loss(hidden))` |
| `vk_bwd_adapter_gdn_chunkwise_fd_parity`        | gdn    | reuse `crates/kiln-vulkan-kernel/tests/vk_gdn_backward_parity.rs` FD harness (`:259` `(loss_p - loss_m)/(2.0*h)`) |

Run a **single** named test at a time, bounded:

```
CARGO_TARGET_DIR=/home/ericflo/Development/kiln/target \
  cargo test -p kiln-model --features vulkan vk_bwd_adapter_matmul_fd_parity -- --exact --nocapture
```

This is one GPU dispatch over a `[4,5]@[5,3]` matmul + its two backward
matmuls — well under any soak threshold. **Do NOT** `cargo test -p kiln-model
--features vulkan` (whole-suite) on the dev host.

### 7.3 Human-gated GPU soak (NOT run autonomously — describe only)

These are explicitly out of the autonomous validation ceiling and must be run by
a human on a stable GPU host:

1. **One-step real-model reachability:** a single forward+backward step of a
   small real model on `Device::Vulkan(0)` through the (PR5) recorders, asserting
   non-NaN grads land in the `GradStore` for the LoRA params. Gate at one step;
   reuse the existing single-step smoke pattern (`vk_train_smoke`, grep in
   `crates/kiln-train/src/vk_train.rs`). **Human-gated** — the host crashed twice
   on multi-step loops.
2. **Cross-engine OPD parity on real GPUs:**
   `cargo test -p kiln-train --features cuda,vulkan vk_cuda_opd` (needs both a
   CUDA and a Vulkan GPU present; the test skips otherwise —
   `crates/kiln-train/tests/vk_cuda_opd_parity.rs:16`). Enforces the literal
   `1e-5`/`1e-4` §9.2 gate.
3. **End-to-end short GRPO/SFT step parity** vk-native-walker vs kt-tape-walker
   on identical inputs: assert the per-param grads agree to `1e-5` (F32). Bounded
   to a handful of steps; **human-gated**.

---

## 8. Prerequisites + what PR4 unblocks

### Prerequisite PRs (must be merged first)

- **PR2 (storage keystone) — MERGED on this branch** (`b94feeac`). Provides
  `Tensor`-on-Vulkan constructors (`tensor.rs:241` un-NYI'd) and the
  `host_to_vulkan_copy`/`vulkan_to_host_copy` pair the FD tests use to build
  inputs and read grads back. Without it, no `Tensor(VulkanStorage)` exists for
  the adapter to consume.
- **PR3 (zero-copy bridge + `MatmulOp::vulkan_fwd`)** — provides
  `VulkanStorage::from_arc_buffer` (already present, `vulkan_storage.rs:130`) as
  the zero-copy rewrap the adapter's `vk_to_tensor` uses, and the
  `MatmulOp::vulkan_fwd` (NOT yet present — `crates/kiln-tensor/src/ops/matmul.rs`
  has only `cuda_fwd:113` / `metal_fwd:138`) needed so a *forward* matmul can run
  on the shared substrate at all. **PR4's adapter can be written and unit-tested
  against hand-built `VkBackwardOp`s without PR3's `vulkan_fwd`,** but the
  end-to-end recorder path (PR5) needs it. Order: PR3 → PR4 → PR5.

### What PR4 unblocks

- **PR5 (forward harmonization)** — once each family's backward can record onto
  the shared tape via `VkBwdAdapter`, the `try_tape_*_kt` recorders can add a
  Vulkan-resident branch (the parent plan §4 PR5).
- **PR6/PR7** — the orchestration flip and fork deletion depend on the shared
  tape being authoritative for Vulkan backward, which is precisely what PR4
  delivers per-family.

---

## 9. Open risks + de-risking

- **R1 — Cargo cycle / feature-unification.** Adding `kiln-tensor/vulkan` +
  `dep:kiln-autograd` to `kiln-model`'s `vulkan` feature could, in theory,
  surface a feature-unification surprise in a workspace build. **De-risk:** the
  *first* action is the Cargo edit + `cargo check -p kiln-model --features
  vulkan` (§7.1) — a clean 0 proves no cycle. CUDA/Metal already activate both,
  so the pattern is proven (`Cargo.toml:76`,`:85`).
- **R2 — `None`-slot / ordering drift.** If the adapter ever drops a `None` or
  reorders, the kt walker binds grads to the wrong `TensorId` *silently* (no
  shape error if shapes coincide). **De-risk:** `vk_bwd_adapter_preserves_none_slots`
  and `vk_bwd_adapter_input_count_matches` (§7.2) assert exact length + slot
  identity. The `.map(...).collect()` in `apply` (§4.3) preserves order by
  construction; never `.flatten()` or `.filter_map()`.
- **R3 — Accidental host bounce.** The shipping `vulkan_softmax_last_axis`
  round-trips through the host (`vulkan_storage.rs:562`–`:660`). If the
  implementer copies *that* as the bridge template instead of the zero-copy
  `from_buffer`/`from_arc_buffer` path, every backward op pays D2H+H2D and the
  harmonized path regresses badly (parent plan §6: zero-copy is the only real
  perf lever). **De-risk:** `vk_bwd_adapter_zero_copy_shares_buffer` (§7.2)
  asserts `Arc::strong_count` on the shared buffer is >1 after wrap (a copy would
  make a fresh Arc). Code-review checklist item: the adapter must contain **no**
  `read_back` / `upload_data` call.
- **R4 — In-place version drift when Wave 2 lands.** §5.2: a ported family that
  consumes the output of an in-place vk kernel will trip the anti-pattern-16
  version check (or worse, read stale state if the bump is missing). **De-risk:**
  Wave 1 is pure-functional (no in-place saved inputs) — ship it first to
  validate the adapter mechanics with zero version-counter work. Before each Wave
  2 family, grep its forward for `*_inplace` writes to a saved input and add the
  `bump_version()` call + an anomaly-detector run (`KILN_DETECT_ANOMALY`,
  `crates/kiln-autograd/src/tape.rs:180`) in the bounded test.
- **R5 — BF16 byte-size / padding mismatch.** BF16 device buffers pad to a
  4-byte word (`vk_tensor.rs:43` `device_buffer_bytes`); passing the padded
  device size to `from_arc_buffer` instead of the logical `byte_size()` would
  trip its multiple-of-`size_in_bytes` validation (`vulkan_storage.rs:138`).
  **De-risk:** §4.3 passes `v.byte_size()` (logical) explicitly; a BF16 adapter
  test asserts the rewrapped `Tensor`'s `byte_len()` equals the logical size.
  (Wave 1 is F32, so this only matters for the bf16w family.)
- **R6 — Grad accumulator can't add Vulkan tensors (§5.3).** A fan-out param
  whose grad is accumulated needs Vulkan elementwise-add as a `vulkan_fwd`.
  **De-risk:** confirm the add path before porting any fan-out family; Wave 1
  LoRA params are single-use so this is deferred, not blocking.

---

## 10. Reference index (re-verify with the grep strings; numbers drift)

| Claim | Location | grep |
| ----- | -------- | ---- |
| `VkBackwardOp` trait | `crates/kiln-vulkan-kernel/src/vk_tensor.rs:61` | `pub trait VkBackwardOp` |
| `BackwardOp` trait | `crates/kiln-autograd/src/backward_op.rs:30` | `pub trait BackwardOp` |
| `VkTensor::from_buffer` (zero-copy in) | `crates/kiln-vulkan-kernel/src/vk_tensor.rs:179` | `pub fn from_buffer` |
| `VkTensor::byte_size` | `crates/kiln-vulkan-kernel/src/vk_tensor.rs:138` | `pub fn byte_size` |
| `VulkanStorage` struct (`Arc<VulkanBuffer>`) | `crates/kiln-tensor/src/vulkan_storage.rs:51` | `pub struct VulkanStorage` |
| `VulkanStorage::from_arc_buffer` (zero-copy out) | `crates/kiln-tensor/src/vulkan_storage.rs:130` | `pub fn from_arc_buffer` |
| `VulkanStorage::buffer_arc` | `crates/kiln-tensor/src/vulkan_storage.rs:164` | `pub fn buffer_arc` |
| `VulkanStorage::vulkan_device` | `crates/kiln-tensor/src/vulkan_storage.rs:170` | `pub fn vulkan_device` |
| `host_to_vulkan_copy` / `vulkan_to_host_copy` | `crates/kiln-tensor/src/vulkan_storage.rs:277,372` | `pub fn host_to_vulkan_copy` |
| `vulkan_softmax_last_axis` (bridge worked example, host-bounce) | `crates/kiln-tensor/src/vulkan_storage.rs:521` | `pub fn vulkan_softmax_last_axis` |
| `VulkanStorage` exports (need `kiln-tensor/vulkan`) | `crates/kiln-tensor/src/lib.rs:158` | `pub use vulkan_storage::{` |
| `Tensor::from_parts` | `crates/kiln-tensor/src/tensor.rs:381` | `pub fn from_parts` |
| `Tensor::storage().as_any().downcast_ref` pattern | `crates/kiln-tensor/src/vulkan_storage.rs:285` | `downcast_ref::<VulkanStorage>` |
| `Tensor::bump_version` (anti-pattern 16) | `crates/kiln-tensor/src/tensor.rs:876` | `pub fn bump_version` |
| `Tape::record` | `crates/kiln-autograd/src/tape.rs:86` | `pub fn record` |
| `Tape::backward` (accumulator closure) | `crates/kiln-autograd/src/tape.rs:124` | `pub fn backward<F>` |
| tape walker grad-len check | `crates/kiln-autograd/src/tape.rs:236` | `returned {} grads for {} inputs` |
| tape walker version-drift err | `crates/kiln-autograd/src/tape.rs:213` | `version drifted` |
| vk-native walker (`vk_backward`) | `crates/kiln-vulkan-kernel/src/vk_autograd.rs:72` | `pub fn vk_backward` |
| vk-native topo DFS | `crates/kiln-vulkan-kernel/src/vk_autograd.rs:152` | `fn collect_topo` |
| Metal/CUDA OPD kt_tape (TEMPLATE) | `crates/kiln-opd-loss-kernel/src/kt_tape.rs:143,166,326` | `impl BackwardOp for CudaOpdTopKReverseKlPhaseBBackward` |
| OPD `envelope_ok` gate (fallback precedent) | `crates/kiln-opd-loss-kernel/src/kt_tape.rs:91` | `fn envelope_ok` |
| `MatmulBackward` | `crates/kiln-vulkan-kernel/src/vk_ops/matmul.rs:94` | `impl VkBackwardOp for MatmulBackward` |
| `RmsNormBackward` (`[Some,None]`) | `crates/kiln-vulkan-kernel/src/vk_ops/rmsnorm.rs:103,132` | `Ok(vec![Some(grad_x), None])` |
| `OpdLossBackward` | `crates/kiln-vulkan-kernel/src/vk_ops/opd.rs:392` | `impl VkBackwardOp for OpdLossBackward` |
| `FlceBackward`/`GrpoBackward` (host metadata in struct) | `crates/kiln-vulkan-kernel/src/vk_ops/flce.rs:200,348` | `impl VkBackwardOp for FlceBackward` |
| `GdnChunkwiseBackward` (5 inputs) | `crates/kiln-vulkan-kernel/src/vk_ops/gdn_chunkwise.rs:1396` | `inputs: [VkTensor; 5]` |
| FD harness (central diff) to reuse | `crates/kiln-vulkan-kernel/tests/vk_matmul_parity.rs:109,57,49` | `fn fd_grad` |
| cross-engine OPD parity gate (1e-5/1e-4) | `crates/kiln-train/tests/vk_cuda_opd_parity.rs:16` | `#![cfg(all(feature = "cuda"` |
| `kiln-model` vulkan feature (needs +autograd +tensor/vulkan) | `crates/kiln-model/Cargo.toml:94` | `^vulkan = ` |
| `kiln-model` cuda/metal feature (proven +autograd pattern) | `crates/kiln-model/Cargo.toml:76,85` | `dep:kiln-autograd` |
| `dispatch2` Vulkan arm (add accumulator path) | `crates/kiln-tensor/src/device_op.rs:217` | `Device::Vulkan(_) => op.vulkan_fwd(a, b)` |
| anti-pattern-16 anomaly detector | `crates/kiln-autograd/src/tape.rs:180` | `anomaly_detection_enabled` |
