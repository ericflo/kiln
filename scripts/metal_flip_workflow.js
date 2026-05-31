export const meta = {
  name: 'metal-candle-free-flip',
  description: 'Draft candle-free kt-native rewrites of metal.rs boundary kernels (#1082)',
  phases: [{ title: 'Draft', detail: 'one agent per kernel subsystem' }],
}

// Disjoint subsystem groups — each agent owns a set of functions and
// writes ONLY its own JSON file (no shared-file write conflict).
const GROUPS = [
  { tag: 'casc_gdn', fns: ['metal_gdn_gates_supports', 'metal_gdn_gates_bf16', 'metal_gated_rms_norm_supports', 'metal_gated_rms_norm_bf16', 'metal_gdn_recurrent_prefill_native_head_last_supports', 'metal_gdn_recurrent_prefill_native_head_last_bf16'] },
  { tag: 'casc_coop', fns: ['metal_transposed_coop_gemv_batch_bf16', 'metal_transposed_coop_gemv_bf16_with_tile', 'metal_rotary_table_batch_stride'] },
]

const PATTERN = String.raw`
You are migrating GPU kernel helper functions in
\`crates/kiln-model/src/backend/metal.rs\` OFF candle-core to be
candle-CORE-free and kt-native (#1082). The substrate stays
\`candle_metal_kernels\` + \`objc2_metal\` (those are KEPT). Only the
\`candle_core::\` Tensor/Device/DType/Storage surface is removed.

CRITICAL RULE: This is a PURE SUBSTRATE SWAP. You MUST preserve ALL
kernel logic BYTE-FOR-BYTE: every \`encoder.set_buffer(IDX, ...)\` index,
every \`encoder.set_bytes(IDX, &val)\`, every \`dispatch_threads\` /
\`dispatch_thread_groups\` size computation, every mode flag / row_group /
threadgroup-size calculation, every MSL kernel/pipeline name, every
shape/stride/dim computation, and the order of operations. Do NOT
"improve", reorder, or simplify anything. If you are unsure whether a
line changes, LEAVE IT UNCHANGED.

The ONLY allowed changes are (a)-(g):

(a) Signature types: \`&candle_core::Tensor\` -> \`&kiln_tensor::Tensor\`;
    return \`Result<candle_core::Tensor>\` -> \`Result<kiln_tensor::Tensor>\`;
    tuple returns like \`Result<(candle_core::Tensor, candle_core::Tensor)>\`
    -> the kt tuple. \`Result<u32>\` / \`Result<Vec<u32>>\` / \`-> bool\` are
    UNCHANGED.

(b) dtype/device literals: \`candle_core::DType::BF16\` ->
    \`kiln_tensor::DType::BF16\` (same for F32/F16/U32/U8/I64);
    \`candle_core::Device::Metal(_)\` -> \`kiln_tensor::Device::Metal(_)\`.

(c) Output allocation: replace
      \`let out = unsafe { candle_core::Tensor::empty(DIMS, candle_core::DType::X, SRC.device())? };\`
    with
      \`let out = kt_metal_alloc(SRC_METAL, kiln_tensor::DType::X, DIMS_AS_SLICE)?;\`
    where SRC_METAL is \`let SRC_metal = kt_metal(&SRC)?;\` computed
    earlier (SRC is the input tensor whose device/companion the output
    sits on, usually the first input \`x\`/\`q\`/\`gate\`). DIMS_AS_SLICE must
    be a \`&[usize]\` (e.g. \`x_dims.as_slice()\` or \`&[a, b, c]\`). If the
    original alloc'd MULTIPLE outputs, call kt_metal_alloc once per
    output.

(d) device / pipeline / encoder: replace the block
      \`let candle_core::Device::Metal(device) = SRC.device() else { anyhow::bail!(...) };
       let pipeline = metal_X_pipeline(device)?;   // (may be a different getter / multiple)
       let encoder = device.command_encoder()?;\`
    with
      \`let companion = SRC_metal.companion()?;
       let pipeline = metal_X_pipeline(&*companion)?;
       let encoder = companion.command_encoder()?;\`
    Keep the SAME pipeline-getter call(s) and the SAME order. The
    pipeline getters already accept \`&*companion\` (they take
    \`&dyn MetalPipelineHost\`). If a getter takes extra args (e.g. a
    tile), keep them: \`metal_X_pipeline(&*companion, tile)?\`.

(e) Storage extraction: replace each
      \`let (a_storage, a_layout) = a.storage_and_layout();\`
      \`... let a_metal = match &*a_storage { Storage::Metal(s) => s, _ => anyhow::bail!(...) };\`
    with a single
      \`let a_metal = kt_metal(&a)?;\`
    (compute these once near the top, after \`out\` is allocated, since
    \`kt_metal\` borrows the tensor). For the output buffer use
    \`let out_metal = kt_metal(&out)?;\`.

(f) buffer_o_kt: replace
      \`buffer_o_kt(a_metal.buffer(), &kt_layout_from_candle(a_layout), kt_dtype_from_candle(a.dtype()))\`
    with
      \`buffer_o_kt(a_metal.buffer().as_ref(), a.layout(), a.dtype())\`
    NOTE the \`.as_ref()\`: kt \`MetalStorage::buffer()\` returns
    \`&Arc<Buffer>\`, and buffer_o_kt wants \`&Buffer\`.

(g) \`.contiguous()?\`, \`.dims()\`, \`.dims2()\`, \`.dims3()\`, \`.dims4()\`,
    \`.dims1()\`, \`.rank()\`, \`.element_count()\`, \`.shape()\` all exist on
    kt Tensor unchanged — keep them as-is.

For pure \`_supports\` functions (return \`bool\`): only (a) + (b) apply —
swap the signature Tensor type and the candle_core::DType /
candle_core::Device literals. The \`.dims()\`/\`.dims2()\`/\`.rank()\` calls
stay. \`weight.dims() == &[hidden]\` becomes \`weight.dims() == [hidden]\`
(kt \`.dims()\` returns \`&[usize]\`; compare against an array literal).

If a function calls ANOTHER metal_* helper passing a tensor (e.g.
\`metal_transposed_coop_gemv_bf16\` calls
\`metal_transposed_coop_gemv_bf16_with_tile(x, weight_t)\`), pass the kt
tensors through unchanged — that callee will be flipped separately; do
NOT add any candle bridge.

WORKED EXAMPLE (the canonical pattern), BEFORE:
\`\`\`rust
pub(crate) fn metal_rms_norm_bf16(x: &candle_core::Tensor, weight: &candle_core::Tensor, eps: f32) -> Result<candle_core::Tensor> {
    let x_dims = x.dims().to_vec();
    let hidden = *x_dims.last().context("...")?;
    let rows: usize = x_dims[..x_dims.len() - 1].iter().product();
    let x = x.contiguous()?;
    let weight = weight.contiguous()?;
    let out = unsafe { candle_core::Tensor::empty(x_dims.as_slice(), candle_core::DType::BF16, x.device())? };
    if rows == 0 { return Ok(out); }
    let candle_core::Device::Metal(device) = x.device() else { anyhow::bail!("..."); };
    let pipeline = metal_rms_norm_pipeline(device)?;
    let encoder = device.command_encoder()?;
    encoder.set_label("kiln_rmsnorm_bf16");
    encoder.set_compute_pipeline_state(&pipeline);
    {
        let (x_storage, x_layout) = x.storage_and_layout();
        let (w_storage, w_layout) = weight.storage_and_layout();
        let (o_storage, o_layout) = out.storage_and_layout();
        let x_metal = match &*x_storage { Storage::Metal(s) => s, _ => anyhow::bail!("...") };
        let w_metal = match &*w_storage { Storage::Metal(s) => s, _ => anyhow::bail!("...") };
        let out_metal = match &*o_storage { Storage::Metal(s) => s, _ => anyhow::bail!("...") };
        let x_buf = buffer_o_kt(x_metal.buffer(), &kt_layout_from_candle(x_layout), kt_dtype_from_candle(x.dtype()));
        let w_buf = buffer_o_kt(w_metal.buffer(), &kt_layout_from_candle(w_layout), kt_dtype_from_candle(weight.dtype()));
        let out_buf = buffer_o_kt(out_metal.buffer(), &kt_layout_from_candle(o_layout), kt_dtype_from_candle(out.dtype()));
        encoder.set_buffer(0, Some(x_buf.buffer), x_buf.offset_in_bytes);
        // ... set_bytes, dispatch_threads UNCHANGED ...
    }
    Ok(out)
}
\`\`\`
AFTER:
\`\`\`rust
pub(crate) fn metal_rms_norm_bf16(x: &kiln_tensor::Tensor, weight: &kiln_tensor::Tensor, eps: f32) -> Result<kiln_tensor::Tensor> {
    let x_dims = x.dims().to_vec();
    let hidden = *x_dims.last().context("...")?;
    let rows: usize = x_dims[..x_dims.len() - 1].iter().product();
    let x = x.contiguous()?;
    let weight = weight.contiguous()?;
    let x_metal = kt_metal(&x)?;
    let out = kt_metal_alloc(x_metal, kiln_tensor::DType::BF16, &x_dims)?;
    if rows == 0 { return Ok(out); }
    let companion = x_metal.companion()?;
    let pipeline = metal_rms_norm_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_rmsnorm_bf16");
    encoder.set_compute_pipeline_state(&pipeline);
    {
        let w_metal = kt_metal(&weight)?;
        let out_metal = kt_metal(&out)?;
        let x_buf = buffer_o_kt(x_metal.buffer().as_ref(), x.layout(), x.dtype());
        let w_buf = buffer_o_kt(w_metal.buffer().as_ref(), weight.layout(), weight.dtype());
        let out_buf = buffer_o_kt(out_metal.buffer().as_ref(), out.layout(), out.dtype());
        encoder.set_buffer(0, Some(x_buf.buffer), x_buf.offset_in_bytes);
        // ... set_bytes, dispatch_threads UNCHANGED ...
    }
    Ok(out)
}
\`\`\`
Note: \`x_metal\` is computed BEFORE \`out\` (so kt_metal_alloc can use it),
and \`w_metal\`/\`out_metal\` are computed inside the block. If the input is
re-bound by \`.contiguous()\` (e.g. \`let x = x.contiguous()?;\`), call
\`kt_metal(&x)?\` on the contiguous binding.
`

phase('Draft')
const results = await parallel(GROUPS.map((g) => () =>
  agent(
    `${PATTERN}

Your assigned functions (in crates/kiln-model/src/backend/metal.rs):
${g.fns.map((f) => '  - ' + f).join('\n')}

Steps:
1. For EACH assigned function, find + read its full current source in
   crates/kiln-model/src/backend/metal.rs (grep "fn <name>(" to locate;
   read from the "pub(crate) fn <name>(" line through its closing "}" at
   column 0, inclusive).
2. Produce the candle-free kt-native rewrite per the (a)-(g) rules,
   preserving ALL kernel logic byte-for-byte.
3. Write a single output file at /tmp/metal_flip_${g.tag}.txt using the
   Write tool. Format: for EACH function, one block:
       ===FN:<exact_name>===
       <the COMPLETE rewritten function source, raw Rust, from the
        "pub(crate) fn <name>(" line through its closing "}" inclusive>
   Put the "===FN:<name>===" marker on its own line immediately before
   each function's source. Raw Rust only between markers — do NOT JSON-
   escape, do NOT wrap in code fences, do NOT add commentary. One block
   per assigned function, all in the one file.

After writing the file, reply with just: "<tag>: wrote N functions to
/tmp/metal_flip_${g.tag}.txt" and nothing else.`,
    { label: `flip:${g.tag}`, phase: 'Draft' }
  ).then((r) => ({ tag: g.tag, msg: r }))
))

for (const r of results.filter(Boolean)) {
  log(`${r.tag}: ${String(r.msg).slice(0, 200)}`)
}
return { groups: results.filter(Boolean).map((r) => r.tag) }
