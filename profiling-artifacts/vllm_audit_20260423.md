# vLLM fused-recurrent GDN audit — 2026-04-23

## Scope

Task: audit vLLM's current fused recurrent GDN kernel for one bounded win that
could be ported into kiln's fused full-chunk prefill kernel.

Kiln target path:

- `crates/kiln-gdn-kernel/csrc/gdn_full_chunk_forward.cu`

Audited upstream source:

- repo: `vllm-project/vllm`
- file: `vllm/model_executor/layers/fla/ops/fused_recurrent.py`
- current commit inspected: `ccaf5ffaa3e1fb2a081b2c9e403ac0e4dfc142c8`

## Fresh-main preflight

- kiln `main` at `4ecd7ce` (`phase6: document failed post-403 front-half full-chunk retry (#405)`)
- PR `#405` is merged and doc-only
- `git diff --stat 57f67ae..4ecd7ce -- crates/kiln-gdn-kernel` is empty
- `gh pr list -R ericflo/kiln --state all --limit 20` shows no newer open or
  merged PR changing `crates/kiln-gdn-kernel/csrc/gdn_full_chunk_forward.cu`
  after `#405`

## Upstream commit audit

| Commit | Upstream change | Why it is not the requested one-diff full-chunk port |
| --- | --- | --- |
| `22dffca9822987f0e912bfd9635e94bbdd05def3` | Raise decode `BV` cap from 8 to 32 | Decode-only tile size knob. kiln full-chunk prefill already runs one thread per `dv` lane and has no equivalent `BV` cap. |
| `824058076c56164a3772a5f5829bd9662507e5a3` | Change recurrent state layout `[N, HV, K, V] -> [N, HV, V, K]` | Cross-cutting state ABI/layout refactor. Would touch decode + prefill kernels, Rust tensor layout assumptions, and tests. Not bounded to one kernel diff. |
| `9e19f8338b4098047175ca3119d5ae0368bcf24a` | Add packed recurrent decode fast path | Decode-only path with mixed-QKV and scalar gate inputs. kiln full-chunk prefill consumes chunk matrices (`kkt`, `qkt`, `ks_entry`, `q_s`), so this does not map to a thin port. |
| `d4cb783c10ffc091af7f09a3b052dceadc06d075` | Guard NULL-block decode padding | Continuous-batching decode bugfix. No equivalent state-index input exists in kiln's full-chunk prefill kernel. |

## Why the direct reuse frontier is exhausted

The last three bounded attempts already covered the only obvious "reuse the
same loaded data more aggressively" ideas from the fused recurrent mental model:

1. `#401` weighted-`W` / `decay_last` hoist
   - control: `3277.7 ms`
   - branch: `3566.0 ms`, `3686.3 ms`
   - verdict: slower, reverted
2. `#403` shared-`k_t` row staging
   - same-pod control: `3272.4 ms`
   - same-pod branch: `4704.0 ms`
   - verdict: slower, reverted
3. `#405` front-half triangular packing
   - same-pod cold control: `4785.1 ms`
   - same-pod branch: `3348.3 ms`
   - same-pod warm-`main` rerun: `3347.1 ms`
   - verdict: warm-arm artifact, reverted

## Conclusion

No single bounded win remains from the current vLLM
`fused_recurrent_gated_delta_rule` frontier that cleanly ports into kiln's
full-chunk prefill kernel.

The remaining upstream differences are either:

- decode-specific,
- bugfix-only, or
- wide state-layout refactors that exceed this task's allowed scope.

Correct outcome: doc-only PR, no edit to
`crates/kiln-gdn-kernel/csrc/gdn_full_chunk_forward.cu`.
