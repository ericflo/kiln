# ECHO plan archive

Two plan documents, both archived 2026-08-26 because the work they planned is
landed in the live tree.

- [`echo-integration-plan.md`](echo-integration-plan.md) — the engineering
  contract: canonical trajectory schema (`kiln-train::trajectory`), the ECHO
  env-CE loss term (`kiln-train::echo`), the masking primitive
  (`kiln-train::trajectory_mask`), the capability/CLI/HTTP surface, and the
  Phase 0–4 rollout with acceptance gates.
- [`grand-plan-for-extraordinarily-great-echo-for-everyone.md`](grand-plan-for-extraordinarily-great-echo-for-everyone.md) —
  the companion user-facing arc (tiers, killer workflows, the "pit of
  success") written after Phases 0–3 shipped.

**Status: landed.** Verified against the live tree at archival time
(2026-08-26):

- `crates/kiln-train/src/echo.rs` (the loss term) and
  `crates/kiln-train/src/trajectory.rs` / `trajectory_mask.rs` (the masking
  layer) are live modules, still cited by the code's own doc comments.
- ECHO is **on by default** in `LossConfig::default()` (λ=0.05) — documented
  product behavior in README ("ECHO-by-default") and `docs/ECHO_GUIDE.md`;
  CHANGELOG carries the "Unreleased — ECHO" section.
- Landing commits: `8a9181a70` (#1502 ECHO fail-fast + honest receipts),
  `7c746208d` (#1512 ECHO-by-default on the fused GRPO tape root),
  `a8de9dd85` (#1518 eligibility lift restore), `0e0606f73` (#1531 OPD env-CE
  composition — the Phase 4 item), plus the docs truth passes `7bd4a2e56`
  (#1511) and `13528aa25` (#1536).

Kept as the historical design record. Their present-tense status lines
("Draft", "what ships after Phase 3") describe the 2026-05-18/19 point in
time, not current state. The technique's reference corpus stays live at
`docs/papers/echo/` (paper + blog), and the unlanded OPD roadmap stays live in
`docs/plans/` (grand OPD plan + `opd-onpolicy-roadmap.md`).
