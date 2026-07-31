# Tensor substrate quickstart

Use this guide when changing `kiln-tensor`, adding an operation, or debugging a
device, dtype, shape, layout, or dispatch failure. `kiln-tensor` is an internal
workspace API: callers and implementations may change together, but every
change still needs an explicit behavioral contract.

## Start here

| Question | Source of truth |
| --- | --- |
| How do I create and inspect a tensor? | `crates/kiln-tensor/src/tensor.rs` |
| Which scalar types and packed formats exist? | `crates/kiln-tensor/src/dtype.rs` and `element.rs` |
| How is an operation dispatched? | `crates/kiln-tensor/src/device_op.rs` |
| Which operation modules are public? | `crates/kiln-tensor/src/ops/mod.rs` |
| Which backend capabilities are reported? | [Backend capability report](backend-capability-report.md) |
| Where does the tensor layer sit in the runtime? | [Architecture package boundaries](../ARCHITECTURE.md#package-boundaries) |

Start with a CPU reference test. Add accelerator evidence only after the
operation’s numerical, shape, dtype, and error behavior is unambiguous.

## Create, run, and verify one operation

This complete test constructs two CPU tensors, multiplies them, and checks the
four observable contracts:

```rust
use kiln_tensor::{DType, Device, Result, Tensor, ops};

#[test]
fn documented_cpu_matmul_contract() -> Result<()> {
    let lhs = Tensor::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], (2, 2))?;
    let rhs = Tensor::from_slice(&[5.0_f32, 6.0, 7.0, 8.0], (2, 2))?;

    let output = ops::matmul(&lhs, &rhs)?;

    assert_eq!(output.device(), Device::Cpu);
    assert_eq!(output.dtype(), DType::F32);
    assert_eq!(output.shape(), &[2, 2]);
    assert_eq!(output.to_vec::<f32>()?, vec![19.0, 22.0, 43.0, 50.0]);
    Ok(())
}
```

The checked-in version lives in
`crates/kiln-tensor/tests/docs_quickstart.rs`. Run it with:

```bash
scripts/cargo-bounded.sh test --locked \
  -p kiln-tensor --test docs_quickstart
```

`Tensor::from_slice` derives the dtype from the Rust element type, verifies
that the shape’s element count equals the slice length, and creates contiguous
CPU storage. `ops::matmul` dispatches from the tensors’ device. `to_vec`
returns logical row-major values and requires an exact element-type match.

## The tensor contract

A `Tensor` combines four kinds of state:

| Part | Meaning | Common mistake |
| --- | --- | --- |
| Storage | Owned backend memory and its dtype | Assuming a `Device` label alone makes CPU bytes valid GPU storage |
| Layout | Shape, strides, and start offset over that storage | Passing a view to a kernel that only accepts contiguous input |
| `TensorId` | Autograd and optimizer identity for this tensor node | Assuming a general view keeps its parent’s identity |
| Version | Mutation counter shared by aliases | Mutating storage without bumping the version and invalidating tape assumptions |

Treat device, dtype, shape, strides, and identity as part of the operation’s
public behavior. Do not validate only the numerical values.

## Device selection and movement

`Device` represents `Cpu`, `Cuda(index)`, `Metal(index)`, `Vulkan(index)`, or
`Rocm(index)`. The index is a logical device selection, not a marketing model,
vendor allowlist, or memory-size shortcut.

Production code receives the device chosen by the runtime’s capability and
configuration owners. Tensor operations must not branch on a GPU name, PCI ID,
host label, or one qualification machine.

Use the device-parametric constructors when the caller already owns a valid
selection:

```rust
let input = Tensor::from_vec_on(
    selected_device,
    vec![1.0_f32, 2.0, 3.0, 4.0],
    vec![2, 2],
)?;
let zeros = Tensor::zeros_on(selected_device, vec![2, 2], DType::F32)?;
```

The corresponding backend feature must be compiled. A request for
`Device::Vulkan(0)` without the `vulkan` feature, for example, returns an
explicit feature-disabled error.

`Tensor::to_device` supports same-device clones and the host/device transfers
implemented by the selected feature. Cross-backend and cross-GPU movement is
not implicit. If two inputs to `dispatch2` or three inputs to `dispatch3` live
on different devices, dispatch fails before the operation runs.

## Dtypes and readback

Typed constructors accept these Rust element types:

| Rust type | Tensor dtype |
| --- | --- |
| `f32` | `DType::F32` |
| `half::bf16` | `DType::BF16` |
| `half::f16` | `DType::F16` |
| `u32` | `DType::U32` |
| `u8` | `DType::U8` |
| `i64` | `DType::I64` |

FP8 and packed four-bit formats are storage formats, not ordinary Rust scalar
elements. Load their validated raw bytes through the owning weight or raw-byte
path; do not pretend a byte is one logical packed element.

There is no implicit cast during readback:

```rust
let tensor = Tensor::from_slice(&[1.0_f32, 2.0], (2,))?;
let values = tensor.to_vec::<f32>()?; // valid
let wrong = tensor.to_vec::<u32>();   // error: no implicit cast
```

Use `ops::cast` deliberately when a conversion is part of the algorithm.
Document the output dtype and tolerance for any lossy conversion.

## Shapes, layouts, and contiguity

The product of the shape dimensions must match the logical element count.
Shape errors include both values:

```text
Tensor::from_slice: shape [2, 2] has 4 elements but slice has 3
```

`narrow`, `reshape`, and `transpose` produce zero-copy views when their layout
rules permit it. A zero-copy view shares storage and the mutation version with
its parent, but receives a fresh `TensorId`. A plain clone preserves the ID.
`Parameter` separately owns one stable logical ID across its physical storage
variants.

`contiguous()` is a materialization boundary, not a harmless formatting call.
When it copies, it increments
`kiln_tensor::profile::contiguous_copy_count()`. A kernel should declare and
test its stride support. Do not insert an unmeasured `.contiguous()` merely to
make a backend path accept a view.

## Dispatch and fallback

`DeviceOp1`, `DeviceOp2`, and `DeviceOp3` define operations by input arity.
Each backend method returns:

| Result | Meaning |
| --- | --- |
| `Ok(Some(output))` | This backend produced the output |
| `Ok(None)` | No native implementation accepted this input; apply the dispatcher’s fallback policy |
| `Err(error)` | The operation or backend failed; stop dispatch |

CPU is the mandatory numerical reference. Missing native behavior is handled
differently by backend:

- Metal, Vulkan, and ROCm may stage inputs to CPU, run the reference
  implementation, and copy the result back. Each such event increments the
  backend-and-arity counters returned by
  `profile::device_op_host_fallback_counts()`.
- CUDA does not perform that host round trip. The dispatcher tries the CPU
  method against the original storage, so operations that require CPU storage
  fail and expose the missing native path.

A host fallback is a correctness bridge. It is not proof of acceptable
latency, throughput, or memory traffic. Decode and training hot paths must
assert that unexpected fallback counters remain zero and must be qualified on
the real device relevant to the claim.

## Add or change an operation

1. Define the operation’s accepted ranks, shapes, dtypes, devices, output, and
   failure cases before choosing a kernel.
2. Add or update its module under `crates/kiln-tensor/src/ops/`.
3. Use `DeviceOp1`, `DeviceOp2`, or `DeviceOp3` when the standard dispatcher
   owns the operation. Some multi-output or specialized operations have an
   explicit owning API instead; follow that existing boundary.
4. Implement `cpu_fwd` as the canonical numerical reference.
5. Validate shared shape and dtype rules consistently across every native
   backend method.
6. Return `Ok(None)` only when the documented fallback is safe. Return an
   actionable error for an invalid input or failed backend operation.
7. Export the convenience function from `ops/mod.rs`.
8. Add CPU arithmetic, shape, dtype, layout, and error tests.
9. Add same-input backend parity tests for every native implementation.
10. For a hot path, add evidence that no unexpected host fallback or
    contiguous copy occurred, then run the relevant local qualification
    workload.

Do not use a hardcoded device model to choose a kernel. Query the capabilities
that the kernel actually requires, validate the input geometry, and keep
machine identity in the resulting evidence.

## Verification ladder

Run the smallest relevant test first:

```bash
scripts/cargo-bounded.sh test --locked \
  -p kiln-tensor --test docs_quickstart

scripts/cargo-bounded.sh test --locked -p kiln-tensor
```

Then build and test the feature you changed on a suitable host:

```bash
cargo test --locked -p kiln-tensor --features cuda
cargo test --locked -p kiln-tensor --features rocm
cargo test --locked -p kiln-tensor --features vulkan
cargo test --locked -p kiln-tensor --features metal -- --test-threads=1
```

Feature compilation and tests that skip unavailable hardware are not
on-device proof. For an accelerator correctness or performance claim, run the
checked-in workload through `scripts/qualification/run.py`, require every case,
and retain the source-bound receipt.

## Failure triage

| Symptom | Check first |
| --- | --- |
| Constructor reports the wrong element count | Compare the value length with the product of the declared shape |
| `to_vec` reports a dtype mismatch | Read back with the exact element type, or cast explicitly before readback |
| Dispatch reports inputs on different devices | Move data once at the owning boundary; do not hide movement inside the operation |
| “feature is not enabled” | Build the owning crate and caller with the intended backend feature |
| “no backend produced output” | Confirm the native method or permitted host fallback exists for that device and input |
| Correct values but poor accelerator performance | Inspect host-fallback and contiguous-copy counters before changing the math |
| A view fails in a kernel | Check strides and the kernel’s support predicate; materialize only at an explicit, measured boundary |
| Backward reports a version mismatch | Find the in-place mutation that failed to bump or respect the shared version |

## Invariants worth preserving

- CPU remains a tested numerical reference.
- Device selection is capability-driven and explicit.
- Mixed-device inputs fail rather than moving silently.
- Dtype changes are explicit.
- Views stay zero-copy until an owned materialization boundary.
- In-place mutation updates the shared version.
- Clones preserve `TensorId`; ordinary view operations receive a fresh ID;
  `Parameter` preserves its own logical ID across storage variants.
- Fallback and copy counters remain visible in performance-sensitive paths.
- Errors name the violated device, dtype, shape, layout, or capability
  boundary and give the contributor a next check.
