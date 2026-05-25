# Metal Kernel Integration Pattern (Phase 4, #1082)

This doc describes the canonical way to wire a real Metal implementation
for a kiln-tensor `DeviceOp::metal_fwd` scaffold, using
`metal_softmax_last_axis` (in `crates/kiln-tensor/src/metal_storage.rs`)
as the reference implementation.

## Quick reference

The integration pattern leverages candle's Metal kernels through
`candle_nn::ops::*` (or `candle_metal_kernels::*` for lower-level access)
**without** going through host memory. On Apple Silicon UMA, the same
`metal::Buffer` (i.e. the same MTLBuffer) is shared between kt-Tensor and
candle Tensor — both hold their own `Arc<metal::Buffer>` referencing the
same Objective-C object via retain/release.

## The five-step pattern

For an op that takes one input tensor and returns one output tensor:

```rust
#[cfg(feature = "metal")]
pub fn metal_<op>(x: &crate::Tensor) -> Result<crate::Tensor> {
    use candle_core::{op::BackpropOp, DType as CandleDType,
                      MetalStorage as CandleMetalStorage,
                      Storage as CandleStorage, Tensor as CandleTensor};

    // 1. Validate kt-side preconditions (dtype, rank, contiguity).
    //    Downcast `x.storage()` to `MetalStorage` via `as_any()`.
    let kt_metal = x.storage()
        .as_any()
        .downcast_ref::<MetalStorage>()
        .ok_or_else(|| Error::Msg("input must be Metal-backed".to_string()))?;
    let candle_device_arc = kt_metal.candle_device().clone();
    let shape: Vec<usize> = x.shape().to_vec();
    let element_count: usize = x.element_count();
    let candle_dtype = match x.dtype() {
        DType::F32  => CandleDType::F32,
        DType::BF16 => CandleDType::BF16,
        DType::F16  => CandleDType::F16,
        other       => return Err(Error::Msg(format!("unsupported dtype {other}"))),
    };

    // 2. Wrap kt buffer in a candle MetalStorage. The kt-tensor's
    //    Arc<metal::Buffer> is the same type candle uses; we clone the
    //    Arc (cheap refcount bump on the Rust side) and pass it in.
    //    The candle MetalStorage and the kt MetalStorage now both
    //    reference the same underlying MTLBuffer.
    let candle_in_storage = CandleMetalStorage::new(
        Arc::clone(kt_metal.buffer()),
        (*candle_device_arc).clone(),
        element_count,
        candle_dtype,
    );
    let candle_in: CandleTensor = CandleTensor::from_storage(
        CandleStorage::Metal(candle_in_storage),
        shape.as_slice(),
        BackpropOp::none(),
        /*is_variable=*/ false,
    );

    // 3. Dispatch through candle_nn::ops or candle_metal_kernels.
    //    For ops candle already exposes:
    let candle_out: CandleTensor = candle_nn::ops::<op>(&candle_in)
        .map_err(|e| Error::Msg(format!("candle <op> failed: {e}")))?;

    // 4. Force contiguity if needed (some candle paths may produce
    //    non-contiguous outputs).
    let candle_out = candle_out.contiguous()
        .map_err(|e| Error::Msg(format!("contiguous: {e}")))?;

    // 5. Extract the result buffer back into kt space. Use
    //    storage_and_layout(), match on Storage::Metal, then
    //    `buffer().to_owned()` — `metal::Buffer: ToOwned` performs an
    //    NSObject `retain`, so kt's new Arc<Buffer> points to the same
    //    MTLBuffer the candle result owns. The candle Tensor drops at
    //    function exit (decrementing its Arc), but the MTLBuffer
    //    survives via the kt-side retain.
    let (out_storage_guard, _) = candle_out.storage_and_layout();
    let candle_out_metal = match &*out_storage_guard {
        CandleStorage::Metal(m) => m,
        _ => return Err(Error::Msg("candle returned non-Metal storage".into())),
    };
    let out_buffer_arc: Arc<candle_metal_kernels::metal::Buffer> =
        Arc::new(candle_out_metal.buffer().to_owned());
    let out_storage = MetalStorage::from_buffer(
        candle_device_arc, 0, x.dtype(), out_buffer_arc,
    )?;
    drop(out_storage_guard);
    crate::Tensor::from_parts(
        Arc::new(out_storage),
        crate::Layout::contiguous(shape),
        crate::TensorId::next(),
    )
}
```

## Wiring into a `metal_fwd` scaffold

The op file (e.g. `crates/kiln-tensor/src/ops/<op>.rs`) becomes:

```rust
#[cfg(feature = "metal")]
fn metal_fwd(&self, x: &Tensor) -> Result<Option<Tensor>> {
    // Same precondition gates as cuda_fwd so the dispatch surface
    // matches across backends:
    if !matches!(x.dtype(), DType::F32 | DType::BF16 | DType::F16) {
        return Ok(None);
    }
    if x.rank() == 0 || !x.is_contiguous() {
        return Ok(None);
    }
    Ok(Some(crate::metal_<op>(x)?))
}
```

Then add `pub use metal_storage::metal_<op>;` to `lib.rs` under
`#[cfg(feature = "metal")]`.

## Why this is zero-copy on Apple Silicon

Apple Silicon GPUs share physical memory with the CPU (Unified Memory
Architecture, UMA). An `MTLBuffer` in `MTLStorageModeShared` is
addressable from both the CPU and GPU at the same virtual address. When
candle allocates an output buffer in `softmax_last_dim`, it produces a
fresh `Arc<metal::Buffer>` pointing to a fresh MTLBuffer. We don't copy
that data — we clone the Objective-C handle (retain) and wrap a new
`Arc<Buffer>` around it. The MTLBuffer is now referenced by both
candle's Arc (about to drop) and kt's Arc (kept alive). When candle's
Arc drops, the inner `Drop` for `metal::Buffer` decrements the NSObject
retain count, but kt's Arc keeps it positive.

This is the same model candle uses internally — every candle `MetalStorage`
holds an `Arc<Buffer>`, and `MetalStorage: Clone` clones the Arc
(refcount bump) rather than copying bytes.

## Validation matrix

| Platform | Build | Runtime test | What it catches |
|---|---|---|---|
| Linux + `cargo check -p kiln-tensor` (no metal) | passes | n/a | Rust syntax for non-metal code paths |
| Linux + `cargo check --features metal -p kiln-tensor` | fails at upstream `objc2` (compile_error: "objc2 only works on Apple platforms") | n/a | Crate graph wiring (deps, features) up to objc2 |
| Mac + `cargo check --features metal -p kiln-tensor` | should pass | n/a | Full Rust type-check of kiln-tensor under metal |
| Mac + `cargo test --features metal -p kiln-tensor -- metal_softmax` with `KILN_TENSOR_METAL_TEST=1` | should pass | yes | Runtime correctness (parity with CPU softmax) |

Mac targets are not testable from the kiln RunPod fleet (A6000 Linux only).
This pattern was validated structurally on Linux through `cargo check`
up to the objc2 limit.

## Phase 7 (#1082) follow-up

Once candle is fully removed, the inner `candle_nn::ops::<op>` call
becomes a direct `candle_metal_kernels::call_<op>` invocation, or a
vendored MSL kernel under `crates/kiln-graph-metal/`. The public
`metal_<op>(&Tensor) -> Result<Tensor>` signature does not change —
call sites in the op scaffolds (`metal_fwd`) are stable through Phase 7.

## Ops with `metal_fwd` scaffolds awaiting wiring

As of this writing, `softmax_last_dim` is the first op to receive a
real Metal implementation. The following ops have `metal_fwd`
scaffolds currently returning `Ok(None)` (CPU fallback) and are
candidates for the same pattern:

- `ops/argmax.rs` — `candle_nn::ops::argmax` (or candle's `Tensor::argmax`)
- `ops/embedding.rs` — `Tensor::index_select` is already on candle's Metal path
- `ops/layernorm.rs` — `candle_nn::ops::layer_norm`
- `ops/rmsnorm.rs` — `candle_nn::ops::rms_norm`
- `ops/broadcast.rs` — `Tensor::broadcast_as`
- `ops/concat.rs` — `Tensor::cat`
- `ops/index_select.rs` — `Tensor::index_select`
- (See `git grep 'TODO(#1082, phase 4 Metal)'` for the full list.)

When wiring any of these, follow the same five-step pattern above.
