//! On-Policy Distillation top-K reverse-KL loss kernel.
//!
//! Computes the per-token reverse KL between a student LLM's distribution and
//! a teacher's distribution, restricted to the **teacher's top-K support**
//! (Fu et al. 2026, "Revisiting On-Policy Distillation", §3.1, Eq. 6–8).
//!
//! Given a sequence of student-sampled tokens, for each active position `t`
//! the teacher provides its top-K vocab indices `S_t = TopK_q(c_t)` and the
//! corresponding teacher logprobs over the full vocab. Both distributions
//! are renormalised over the K-element support and we compute
//!
//! ```text
//! KL_t = sum_{v in S_t} p_hat(v) * (log p_hat(v) - log q_hat(v))
//! ```
//!
//! where
//!
//! ```text
//! p_hat(v) = exp(s_v - logsumexp_{u in S_t}(s_u))                      (student renorm)
//! q_hat(v) = exp(t_v - logsumexp_{u in S_t}(t_u))                      (teacher renorm)
//! ```
//!
//! and `s_v = (hidden[t] @ head_t)[v]` is the student logit at the v-th
//! vocab position. The final loss is the mean of `KL_t` over **active**
//! positions (positions where `label_mask[t]` is true — typically only the
//! assistant tokens contribute to the loss).
//!
//! # Why a custom op and not just candle
//!
//! Naive candle: materialize the full `[T, V]` student logits tensor, gather
//! the K columns specified by the teacher's top-K indices, compute KL, and
//! backprop through the gather and the projection. For Qwen3.5-4B with
//! V = 151936 and T_active = 4096 this is **~9.7 GB of F32 logits**
//! before doing anything useful — the same memory pressure FLCE avoids
//! for standard cross-entropy.
//!
//! The OPD path is structurally cheaper than CE: we only need the K
//! per-token student logits the teacher cares about, not the full vocab.
//! We project `hidden[t]` against the `K` columns of `head_t` named by
//! `teacher_topk_indices[t, :]` — a per-token gather-then-matmul whose
//! peak intermediate is `[T_active, K]` (~5000× smaller than `[T_active, V]`).
//!
//! # API contract
//!
//! - `hidden`: `[1, T, H]` student hidden states (output of
//!   `model_forward_final_norm`), bf16 or f32 (kiln trainer uses bf16).
//! - `head_t`: `[H, V]` transposed LM head (matches kiln's `embed_tokens_t`
//!   layout), bf16 or f32.
//! - `teacher_topk_indices`: `[T_active, K]` flattened in row-major
//!   order. Holds the teacher's top-K vocab indices at each **active**
//!   position; positions in the same order as `active_positions` (see
//!   below). Dtype u32.
//! - `teacher_topk_logprobs`: `[T_active, K]` flattened in row-major
//!   order. Holds teacher log-probabilities at the K vocab positions
//!   (`log_softmax(teacher_logits)`). Dtype f32 (this is what every
//!   hosted-logprobs API returns and matches §3.2's `LogitSource` trait).
//! - `label_mask`: `[T]` booleans; the position-`t` logit contributes when
//!   `label_mask[t]` is true. The number of active positions must equal
//!   `T_active` and the order of active positions left-to-right in
//!   `hidden` must match the row order of `teacher_topk_indices`.
//!
//! Returns the **mean reverse KL** over active positions as a scalar f32
//! tensor. The trainer scales this by 1.0 (it is the loss directly) — note
//! that the per-token advantage used in the GRPO importance-sampling code
//! path is `-reverse_kl_per_token`, so a separate helper
//! [`opd_per_position_reverse_kl`] returns the per-position vector for
//! direct advantage construction.
//!
//! # Phase A vs Phase B
//!
//! Mirrors `kiln-flce-kernel`'s historical split (both legs now
//! collapsed to the kt-typed Phase B path):
//!
//! - **Phase A** — the pure-candle reference implementation (built
//!   `[T_active, K]` student logits via per-token gather + batched
//!   matmul, ran the renormalised reverse-KL in candle ops, and let
//!   candle autograd handle the backward). Deleted in the #1082 candle
//!   drop (2026-05-28); it is no longer part of this crate.
//! - **Phase B** — the live kt-typed path
//!   ([`opd_top_k_reverse_kl_phase_b_per_position_via_kt_tape`]): it
//!   runs the kt composite forward and dispatches the fused CUDA
//!   backward via the surviving FFI symbols `kiln_opd_topk_kl_bwd_{bf16,f32}`
//!   declared in [`phase_b`]. The candle `CustomOp1` wrapper that
//!   previously hosted this (`OpdLossCustomOp` / `opd_top_k_reverse_kl_phase_b`)
//!   was deleted in (#1082, 2026-05-28); see
//!   `docs/archive/candle-removal/opd-loss-kernel-candle-removal-stop-2026-05-28.md`.
//!
//! # Numerical contract
//!
//! Per [§9.2 of `docs/plans/grand-plan-for-extraordinarily-great-on-policy-distillation-for-everyone.md`]:
//! the same `(hidden, head_t, teacher_topk_indices, teacher_topk_logprobs,
//! label_mask)` tuple must produce KL values within 1e-5 across CPU / CUDA /
//! Metal. The parity tests in this crate enforce that at f32, and 1e-2
//! relative at bf16.
//!
//! [§9.2 of `docs/plans/grand-plan-for-extraordinarily-great-on-policy-distillation-for-everyone.md`]: ../../docs/plans/grand-plan-for-extraordinarily-great-on-policy-distillation-for-everyone.md

// `phase_b` retains only the fused-CUDA backward FFI declarations
// (`kiln_opd_topk_kl_bwd_{bf16,f32}`) and the top-k/dtype envelope gate
// (both pure kt — the candle `DType` reference was ported
// to `kiln_tensor::DType` in the candle-drop). The candle `CustomOp1`
// wrapper `OpdLossCustomOp`, the candle entry points
// `opd_top_k_reverse_kl_phase_b` and `_per_position`, the fused-FWD FFI
// symbols, and the `PerPositionMetrics` path were all removed in
// (#1082, 2026-05-28).
//
// (#1082) candle-drop: this crate is now 100% candle-free. The
// candle-typed glue that used to live here — the pure-candle Phase A
// reference path (`opd_top_k_reverse_kl_phase_a_per_position` & helpers),
// the candle `CustomOp1` kt-forward-op shim
// (`opd_top_k_reverse_kl_per_position_via_kt_forward_op`,
// `kt_forward_op.rs`), and the kt-tape production-caller adapters
// (`try_tape_opd_per_position_cuda` / `try_tape_opd_scalar_mean_cuda`,
// `tape_forward.rs`) — moved UP into `kiln-train::opd_tape_shim`
// (itself since candle-free). They call the kt-typed building
// blocks below (`kt_api`, `kt_tape`) across the crate boundary.
mod phase_b;

pub mod kt_api;
pub use kt_api::{
    OpdActiveMetadata, OpdLossError, OpdLossOutputKt, PerPositionMetricsKt,
    compute_per_position_metrics_kt, opd_top_k_reverse_kl_kt, opd_top_k_reverse_kl_per_position_kt,
    opd_top_k_reverse_kl_per_position_with_metadata_kt, opd_top_k_reverse_kl_with_metadata_kt,
};

#[cfg(any(feature = "cuda", feature = "rocm"))]
pub use kt_api::{
    opd_top_k_reverse_kl_phase_b_bwd_kt, opd_top_k_reverse_kl_phase_b_bwd_scalar_mean_unit_grad_kt,
    opd_top_k_reverse_kl_phase_b_bwd_scalar_mean_unit_grad_with_metadata_kt,
    opd_top_k_reverse_kl_phase_b_bwd_with_metadata_kt,
};

/// Phase 6a/CP-4 (#1082): kt-tape entry that drops the candle
/// CustomOp1 wrapper in favour of recording onto a `kiln_autograd::Tape`
/// directly. Same FFI symbols on the backward, same envelope. See
/// `kt_tape.rs` for the port rationale (mirroring the rmsnorm
/// sibling in commit `895162ca`).
mod kt_tape;
pub use kt_tape::{
    CudaOpdTopKReverseKlPhaseBBackward, opd_top_k_reverse_kl_phase_b_per_position_via_kt_tape,
    opd_top_k_reverse_kl_phase_b_unit_grad_via_kt_tape, opd_top_k_reverse_kl_phase_b_via_kt_tape,
};

/// Default chunk size when iterating along the active-token dimension. Used
/// by Phase B to bound the temporary `[chunk_T, K]` intermediate. For
/// typical OPD configs (T_active ≤ 8192, K = 32) the whole batch fits in
/// one chunk, but very-long-context training keeps the option open.
pub const DEFAULT_CHUNK_SIZE: usize = 4096;

// (#1082) candle-drop: the pure-candle Phase A reference path
// (`validate_inputs`, `opd_top_k_reverse_kl_phase_a_per_position`,
// `per_position_phase_a`, `gather_head_columns`, `log_softmax_last`)
// moved UP into `kiln-train::opd_tape_shim` so this crate could drop
// its `candle-core` dependency. The kt-typed forward + backward
// (`kt_api`, `kt_tape`) — which the relocated shim now calls across the
// crate boundary — stay here.

// (#1082, 2026-05-28) The candle `OpdLossCustomOp` re-export and the
// `compute_per_position_metrics` / `PerPositionMetrics` re-exports were
// removed alongside their `phase_b.rs` definitions. The kt-typed
// metrics surface (`compute_per_position_metrics_kt` /
// `PerPositionMetricsKt`) remains re-exported above for any future
// caller; no live caller exists today.
