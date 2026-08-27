# MTP training plan archive

- [`mtp-training-plan.md`](mtp-training-plan.md) — the design +
  implementation record for keeping the MTP draft head aligned with the
  tuned model (PR-A serving/adapter format, PR-B post-SFT draft-head
  training, validation strategy, and the "Follow-ups after PR-B" list).

**Status: landed (plan scope).** Verified against the live tree at archival
time (2026-08-26):

- PR-A shipped: `dc6b8df44` "MTP speculative decoding for tuned adapters:
  LoRA-threaded verify + adapter-carried MTP draft LoRA (#1508)".
- PR-B implemented: `ca3e8794e` "MTP PR-B: train the draft head whenever an
  adapter trains (post-SFT alignment phase) (#1515)"; `run_mtp_alignment_phase`
  is live in `crates/kiln-train/src/trainer/reporting.rs`, called from
  `sft.rs`; ROCm Muon-path training covered by `abf665759`.
- Follow-up #3 (per-adapter draft acceptance data) landed: `195e52122`
  (#1530) — always-on counters + `/v1/stats/mtp-acceptance`; operator
  validation follow-ups at `406161719` (#1516); debug machinery retired at
  `db4383326`.

Still open (owner workstream, not plan debt): the GRPO/OPD phase hookup,
`kiln self-improve` auto-inclusion, and the dashboard view — all three are
listed under "Follow-ups after PR-B" in the archived document. Archived
2026-08-26 with zero in-tree references to update.
