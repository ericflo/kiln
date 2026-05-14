# Contributing to Kiln

## Welcome

Kiln is a single-GPU inference server with live LoRA training, written in pure Rust and tuned for one model — [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B). Contributions of all sizes are welcome: bug reports, performance improvements, kernel work (CUDA, Metal, MLX), documentation, examples, and developer-experience polish.

A note on scope: Kiln is deliberately a scalpel, not a framework. The scheduler, memory manager, and kernels are all tuned for Qwen3.5-4B's hybrid architecture (24 Gated DeltaNet + 8 GQA layers). PRs that add support for a second model family will be closed unless the design has been agreed in an issue first. The same goes for adding a Python sidecar process — the single-binary, single-process constraint is a core feature, not an accident.

## Before you start

- **File an issue first** for anything non-trivial — roughly 50+ lines of changes, any new dependency, any new kernel, or any new public API surface. A 5-minute conversation up front saves a 5-day rewrite later.
- **Read [`ARCHITECTURE.md`](ARCHITECTURE.md)** so your change fits the existing seams: `BackendRuntime` for device dispatch, the scheduler for batching, the block manager for paged KV, the `kiln-eval` crate + `crates/kiln-server/src/eval/` for the eval queue and scorer trait, and the kernel crates (`kiln-flash-attn`, `kiln-gdn-kernel`, `kiln-marlin-gemm`, `kiln-rmsnorm-kernel`, `kiln-conv1d-kernel`, `kiln-flce-kernel`) for fused ops.
- **Adding an eval scorer?** `kiln_eval::scorers::Scorer` is a serde-tagged enum, not a trait — append a variant with the right `#[serde(rename_all = "snake_case")]` `kind`, add the matching `kind_label()` arm, implement the scoring logic in `crates/kiln-eval/src/scorers/<name>.rs`, and dispatch it in `score_completion`. Add it to `auto_detect_scorer` in `synthesis.rs` if the target shape is recognizable, then document the request shape in `docs/EVAL_GUIDE.md` and `docs/site/evals.html`. Synthesis, the UI, and the CLI pick scorers up via `kind_label` once those four edits land — that's the seam to preserve.
- **For performance changes**, attach a before/after benchmark from `kiln-bench` (median of 3 back-to-back runs; A6000 if you have access). Cite the kernel crate or the specific region in `crates/kiln-model/src/forward.rs` you touched, and include the NVTX hot-region percentages from `PROFILING.md` if relevant.
- **Search closed PRs** before vendoring a kernel or proposing a fusion. Several speculative wins have already been measured and rejected — the closed-PR history is the cheapest way to avoid burning a weekend on a null result.

## Development setup

Install the stable Rust toolchain:

```bash
rustup toolchain install stable
rustup default stable
```

Build the default (no-GPU) configuration. This works on any host and is the fastest way to iterate on non-kernel code:

```bash
cargo build --locked
```

Build with Apple Silicon GPU support (M-series Macs, Xcode Command Line Tools sufficient):

```bash
cargo build --locked --features metal
```

Build with NVIDIA CUDA support (Linux + CUDA 12.0+):

```bash
cargo build --locked --features cuda
```

CUDA builds compile a fair amount of `nvcc` per architecture. To target only an A6000-class GPU and cut nvcc time by 3-4×, set `KILN_CUDA_ARCHS=86`:

```bash
KILN_CUDA_ARCHS=86 cargo build --release --features cuda
```

Run the test suite. The skipped `test_health_with_real_backend` depends on a live network backend and is intentionally excluded from CI. Env-mutating tests are now serialized via an internal `ENV_LOCK` mutex and run safely in parallel. CI on macOS additionally pins the Metal feature tests to `--test-threads=1` because of a known race in `candle-metal`'s `MetalDevice::new`:

```bash
cargo test --locked -- --skip test_health_with_real_backend
```

If you have `cargo-nextest` installed, it runs the same tests in parallel and is noticeably faster:

```bash
cargo nextest run --locked
```

Run the license / source / bans policy check that CI runs:

```bash
cargo install --locked cargo-deny  # if not already installed
cargo deny check --all-features
```

This validates that any new dependency satisfies the workspace's MIT/Apache-compatible license allowlist and other policy rules in `deny.toml`. CI pins cargo-deny to a specific action SHA (see `.github/workflows/ci.yml`); local runs with the latest version are fine for catching gross violations before pushing.

## Running the server locally

See [`QUICKSTART.md`](QUICKSTART.md) for the full zero-to-running walkthrough — model download, config, first chat completion, first SFT POST. The default port is `8420` and the embedded web dashboard lives at `/ui`.

## Submitting changes

- Branch from `main`. Forks and direct branches are both fine.
- One logical change per PR. Small PRs land faster and are easier to bisect when something regresses.
- Run `cargo build --locked` and `cargo test --locked` (with the documented skips above) before pushing. CI will run them again on Linux default-features and macOS Metal — see `.github/workflows/ci.yml`.
- Open the PR with a **plain title** — no project prefix. Describe what changed and why in the body.
- For performance PRs, include the bench numbers in the body. Median of 3 runs, hardware noted, and ideally the relevant `PROFILING.md` region cited so reviewers can sanity-check the math ceiling.

## Code style

- Match the surrounding style. There is no enforced `cargo fmt` policy yet, but keep new code reasonably tidy.
- **Avoid adding dependencies casually.** Kiln deliberately keeps the dep tree small; every new crate is a build-time cost, an attack-surface increase, and a maintenance burden. Justify new deps in the PR body.
- **No new `unwrap()` in the request path.** Prefer `?` with helpful errors. The error-message style added in PR #545 is a good reference: say what failed, why, and what to try next, instead of bubbling up a bare `io::Error`.
- Keep comments short and load-bearing. Explain *why*, not *what* — the code already shows what.

## What we will probably reject

Setting expectations honestly so you don't waste your time:

- PRs that add support for a second model family. Kiln is scoped to Qwen3.5-4B on purpose; the entire perf story depends on that focus.
- PRs that introduce a Python sidecar process or a second copy of the model in VRAM. Pure Rust + single process is a core constraint.
- PRs that bypass safety checks: `--no-verify` git hooks, deleting tests rather than fixing them, removing assertions because they fire, etc.
- Speculative performance "optimizations" without a profile and a bench. If the change is meant to make Kiln faster, prove it.
- Large refactors with no behavior change. Kiln favors incremental cleanup that ships alongside real work.

## License

Kiln is MIT-licensed (see [`LICENSE`](LICENSE)). By submitting a pull request you agree to release your contribution under the MIT License.

## Questions / discussion

Open an issue or a discussion on [`ericflo/kiln`](https://github.com/ericflo/kiln). There is no Discord or Slack — GitHub is the canonical place for design conversations and bug reports.
