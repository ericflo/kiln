export const meta = {
  name: 'metal-trait-bridge-removal',
  description: 'Drop the kt->candle->kt host-copy bridges in metal.rs BackendRuntime trait methods (#1082)',
  phases: [{ title: 'Flip', detail: 'one agent per trait-method group' }],
}

const GROUPS = [
  { tag: 'br_paged', fns: ['paged_kv_head_major_read', 'paged_kv_head_major_read_append_token_major', 'flash_attn_paged_decode_contiguous', 'flash_attn_paged_decode_contiguous_batch', 'flash_attn_paged_decode_contiguous_batch_dyn_seqlen'] },
  { tag: 'br_flash', fns: ['flash_attn_prefill', 'flash_attn_prefill_head_major', 'flash_attn_paged_decode'] },
  { tag: 'br_gdn', fns: ['gdn_chunk_prep', 'gdn_forward_substitution', 'gdn_full_chunk_forward', 'gdn_full_chunk_forward_head_last_into', 'gdn_in_proj_decode', 'gdn_recurrent_prefill_head_last', 'gdn_recurrent_step'] },
]

const PATTERN = String.raw`
You are removing the kt->candle->kt host-copy BRIDGES from
'BackendRuntime' trait methods in
'crates/kiln-model/src/backend/metal.rs' (#1082). The kernel HELPER
functions these methods call have ALREADY been flipped to take/return
'kiln_tensor::Tensor' (kt). The trait methods still wastefully bridge
their kt arguments to 'candle_core::Tensor' (via
'crate::forward::kt_logits_to_candle'), call the helper, then bridge the
result back (via 'crate::forward::candle_to_kt_activation'). On Metal
those bridges are a full host round-trip per call. Your job: delete the
bridges and pass kt directly.

The trait methods are kt-typed already (their params are
'&kiln_tensor::Tensor', '&mut kiln_tensor::Tensor', return
'kiln_tensor::Tensor'). Transformations:

(1) DELETE every input-bridge line of the form
      'let X = crate::forward::kt_logits_to_candle(X)?;'
    (and the 'let mut Xc = crate::forward::kt_logits_to_candle(X)?;'
    variant for '&mut' state). The kt param 'X' is used directly below.

(2) The deleted bridge shadowed the kt param with a candle local, so the
    body referenced it as '&X'. Now 'X' is already a '&kiln_tensor::Tensor'
    param, so change those call-site uses from '&X' to 'X'. Concretely,
    in the '*_supports(...)' and '*_bf16(...)' / helper calls, an arg that
    was '&q' / '&k_pool' / etc. becomes 'q' / 'k_pool'. (Args that were NOT
    bridged — e.g. plain 'usize' like 'start_slot', 'seq_len', 'kernel_size'
    — stay exactly as they are.)

(3) For a '&mut' state arg (e.g. 'conv_state', 'state', 'out' written in
    place): the old code did 'let mut state_c = kt_logits_to_candle(state)?;'
    ... 'helper(&mut state_c)' ... '*state = candle_to_kt_activation(&state_c)?;'.
    Replace with: pass the kt 'state' straight to the helper (which mutates
    it in place through its shared UMA buffer) and DELETE the '*state = ...'
    write-back line.

(4) For the candle 'sdpa(...)' call (the 'sdpa' symbol imported at the top
    of the module = candle metal SDPA), swap to the kt-native metal SDPA:
      'sdpa(&q_t, &k_t, &v_t, None, causal, scale, 1.0)?'
    becomes
      'kiln_tensor::metal_sdpa_last_axis(&q_t, &k_t, &v_t, scale, causal)?'
    (the kt op bakes in mask=None and softcapping=1.0, matching the only
    call shape used here). 'q_t'/'k_t'/'v_t' are kt tensors produced by
    kt '.transpose(...)?.contiguous()?' on the (now-undeleted) kt q/k/v —
    those transpose/contiguous calls work unchanged on kt tensors.

(5) DELETE the output bridges: 'Ok(Some(crate::forward::candle_to_kt_activation(&Y)?))'
    becomes 'Ok(Some(Y))'; for tuples,
    'Ok(Some((candle_to_kt_activation(&a)?, candle_to_kt_activation(&b)?)))'
    becomes 'Ok(Some((a, b)))'.

(6) Preserve EVERYTHING ELSE byte-for-byte: the guard/'return Ok(None)'
    early-exits, the '.context(...)' chains, dim computations, scale math,
    transposes, the 'self.disable.*' checks, NVTX ranges, comments. Do NOT
    reorder or restructure. Keep the method signature identical.

WORKED EXAMPLE — BEFORE:
    fn paged_kv_head_major_read(
        &self, k_pool: &kiln_tensor::Tensor, v_pool: &kiln_tensor::Tensor,
        start_slot: usize, seq_len: usize,
    ) -> Result<Option<(kiln_tensor::Tensor, kiln_tensor::Tensor)>> {
        let k_pool = crate::forward::kt_logits_to_candle(k_pool)?;
        let v_pool = crate::forward::kt_logits_to_candle(v_pool)?;
        if !metal_paged_kv_head_major_read_supports(&k_pool, &v_pool, start_slot, seq_len) {
            return Ok(None);
        }
        let (k_out, v_out) =
            metal_paged_kv_head_major_read_bf16(&k_pool, &v_pool, start_slot, seq_len)
                .context("metal paged_kv_head_major_read failed")?;
        Ok(Some((
            crate::forward::candle_to_kt_activation(&k_out)?,
            crate::forward::candle_to_kt_activation(&v_out)?,
        )))
    }
AFTER:
    fn paged_kv_head_major_read(
        &self, k_pool: &kiln_tensor::Tensor, v_pool: &kiln_tensor::Tensor,
        start_slot: usize, seq_len: usize,
    ) -> Result<Option<(kiln_tensor::Tensor, kiln_tensor::Tensor)>> {
        // #1082: kt-native — helpers take kt directly, no candle bridge.
        if !metal_paged_kv_head_major_read_supports(k_pool, v_pool, start_slot, seq_len) {
            return Ok(None);
        }
        let (k_out, v_out) =
            metal_paged_kv_head_major_read_bf16(k_pool, v_pool, start_slot, seq_len)
                .context("metal paged_kv_head_major_read failed")?;
        Ok(Some((k_out, v_out)))
    }
(Note: the method header may span multiple lines in the file — reproduce
it exactly as it appears.)
`

phase('Flip')
const results = await parallel(GROUPS.map((g) => () =>
  agent(
    `${PATTERN}

Your assigned BackendRuntime trait methods (in
crates/kiln-model/src/backend/metal.rs, inside the
"impl BackendRuntime for MetalBackend" block):
${g.fns.map((f) => '  - ' + f).join('\n')}

Steps:
1. For EACH method, find + read its full current source (grep
   "fn <name>(" inside the impl block; read from the "    fn <name>("
   line through its closing "    }" at 4-space indent, inclusive).
2. Apply transformations (1)-(6) to drop the candle bridges.
3. Write /tmp/metal_flip_${g.tag}.txt with one block per method:
       ===FN:<exact_name>===
       <COMPLETE rewritten method source, raw Rust, from the
        "    fn <name>(" line through its closing "    }" inclusive>
   Raw Rust only, no JSON, no code fences, no commentary between markers.
After writing, reply with just: "<tag>: wrote N methods".`,
    { label: `bridge:${g.tag}`, phase: 'Flip' }
  ).then((r) => ({ tag: g.tag, msg: r }))
))
for (const r of results.filter(Boolean)) log(`${r.tag}: ${String(r.msg).slice(0, 120)}`)
return { groups: results.filter(Boolean).map((r) => r.tag) }
