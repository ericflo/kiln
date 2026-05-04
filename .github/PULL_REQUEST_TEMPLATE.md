## Summary

What changed and why, in 2-4 lines.

## Type of change

- [ ] Bug fix
- [ ] Performance improvement
- [ ] Kernel change (CUDA / Metal / Vulkan)
- [ ] Feature
- [ ] Docs / DX
- [ ] Infra / CI

## Related issue / PR

Link any related issues or prior PRs.

## Perf change?

If this PR changes performance, paste before/after `kiln-bench` median-of-3 numbers (per CONTRIBUTING.md "For performance changes"). Link the kernel crate or `forward.rs` region touched and the relevant `PROFILING.md` NVTX hot region.

## Checklist

- [ ] `cargo test` passes (skip for docs-only)
- [ ] `cargo build` passes (skip for docs-only)
- [ ] Read CONTRIBUTING.md
- [ ] No new dependency without a prior issue
- [ ] Scoped to one logical change
