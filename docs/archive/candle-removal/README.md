# Candle-removal records (#1082) — archived

Issue #1082 removed the candle / candle-core / candle-nn dependency from the
workspace and replaced it with kiln's own tensor + autograd substrate
(kiln-tensor, kiln-autograd, the kt-native kernel crates). That work is fully
landed: `Cargo.lock` contains zero candle packages, and the Metal, Vulkan,
CUDA, and CPU paths all run on the native substrate.

The dated plan/status/STOP documents here were the working coordination
records for that effort (May 2026). They are retained as historical evidence
of how the migration was staged and verified; their present-tense statements
about candle APIs no longer describe the codebase. For current behavior see
[`../../CONFIGURATION.md`](../../CONFIGURATION.md),
[`../metal/`](../metal/) for the Metal migration records, and the
kt-tape docs referenced from the kernel crate doc comments.
