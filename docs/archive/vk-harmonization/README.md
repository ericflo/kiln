# Vulkan kt-tape harmonization records (archived)

Dated coordination record for #1082's Vulkan training kt-tape harmonization
(PR1–PR7): specs, bounded test scaffolds, adversarial review notes, and the
human-soak handoff.

**Status: fully landed.** The whole series was implemented and merged to
`main` via PR #1441 (`feat/vk-tape-harmonization`), including:

- PR1–PR5: AdamW seam, first-class `Device::Vulkan` kt storage, host-fallback +
  zero-copy bridge + hot-op ports, `VkBwdAdapter` validation, gate cascade and
  recorder proofs (GPU-validated on RADV Strix Halo).
- PR6 (`3b226d620`): orchestration flip — Vulkan SFT/GRPO/OPD routed through
  the shared `kiln_train::trainer` / `opd` path.
- PR7 (`a909d46ff`): deletion of the legacy fork (`vk_train.rs`,
  `vk_forward.rs`, server opt-out env family).

The specs are kept as historical design records; they no longer describe work
to be done. The authoritative plan itself is archived alongside these records:
[`vulkan-train-harmonization-plan.md`](vulkan-train-harmonization-plan.md)
(archived 2026-09-02 after the full series landed). The two pre-harmonization
design docs for the legacy fork — [`vk_native_training.md`](vk_native_training.md)
(GPU-resident VkTensor/autograd stack) and [`vk_native_gdn.md`](vk_native_gdn.md)
(GDN math + kernel phasing; the `vk_ops/gdn_*` kernels themselves remain live
in `crates/kiln-vulkan-kernel/`) — joined them here on 2026-09-03.
