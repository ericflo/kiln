//! `kiln-replay` — verify and inspect deterministic LoRA replay artifacts.
//!
//! Mode 1 (no GPU): walk the parent chain from a LoRA adapter directory,
//! recompute every `replay_hash` from `replay.jsonl` + parent, and confirm
//! the chain matches what `lineage.json` records. This is the fast-path
//! integrity check — it does not retrain anything and never touches the GPU.
//!
//! Usage:
//!   kiln-replay verify <adapter-dir>          # verify a single chain
//!   kiln-replay show   <adapter-dir>          # print the chain (root→leaf)
//!
//! The adapter root is inferred as the parent directory of <adapter-dir>;
//! parent names are resolved as `<adapter-root>/<name>`.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use kiln_train::replay;

fn print_usage() {
    eprintln!("usage:");
    eprintln!("  kiln-replay verify <adapter-dir>");
    eprintln!("  kiln-replay show   <adapter-dir>");
}

fn resolve_adapter_root(adapter_dir: &Path) -> PathBuf {
    adapter_dir
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."))
}

fn cmd_verify(adapter_dir: &Path) -> anyhow::Result<()> {
    let adapter_root = resolve_adapter_root(adapter_dir);
    let resolver = |name: &str| adapter_root.join(name);
    replay::verify_chain_integrity(adapter_dir, resolver)?;
    println!("OK: replay chain at {} verifies", adapter_dir.display());
    Ok(())
}

fn cmd_show(adapter_dir: &Path) -> anyhow::Result<()> {
    let adapter_root = resolve_adapter_root(adapter_dir);
    let resolver = |name: &str| adapter_root.join(name);
    let chain = replay::walk_parent_chain(adapter_dir, resolver)?;
    println!("chain length: {}", chain.len());
    for (i, (dir, lineage)) in chain.iter().enumerate() {
        let parent = lineage
            .parent_lora
            .as_ref()
            .map(|p| p.name.as_str())
            .unwrap_or("<root>");
        println!(
            "{idx:>3} {hash}  parent={parent}  base={base}  dir={dir}",
            idx = i,
            hash = lineage.replay_hash,
            parent = parent,
            base = lineage.base_model.id,
            dir = dir.display(),
        );
    }
    Ok(())
}

fn run() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let cmd = match args.next() {
        Some(c) => c,
        None => {
            print_usage();
            anyhow::bail!("missing command");
        }
    };
    let adapter_dir = match args.next() {
        Some(p) => PathBuf::from(p),
        None => {
            print_usage();
            anyhow::bail!("missing adapter dir");
        }
    };
    if args.next().is_some() {
        print_usage();
        anyhow::bail!("unexpected extra arguments");
    }
    match cmd.as_str() {
        "verify" => cmd_verify(&adapter_dir),
        "show" => cmd_show(&adapter_dir),
        other => {
            print_usage();
            anyhow::bail!("unknown command: {other}");
        }
    }
}

fn main() -> ExitCode {
    if let Err(e) = run() {
        eprintln!("kiln-replay: {e:#}");
        return ExitCode::from(1);
    }
    ExitCode::SUCCESS
}
