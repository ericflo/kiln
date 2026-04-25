#!/usr/bin/env python3
"""H17b — apply the pre-registered decision rule (vLLM v0.20.0 retest).

Direct lineage: scripts/h15c_compare.py (PR #530, vLLM 0.19.1).

Inputs:
    docs/phase-c29-v3-vllm-020/vllm_020_alpha_per_seed.json
    docs/phase-c29-v3-vllm-020/kiln_alpha_per_seed.json

Decision rule (set BEFORE the run, do NOT adjust after):

    delta = vllm_020_median_alpha - kiln_median_alpha

    delta >= +0.05          → external_ceiling_exists
    -0.02 <= delta < +0.05  → mtp_head_quality_ceiling
    delta <  -0.02          → kiln_above_vllm
    mtp_supported == false  → vllm_020_mtp_unsupported_dense_4b

Emits:
    docs/phase-c29-v3-vllm-020/verdict.json
    docs/phase-c29-v3-vllm-020/compare.json
    docs/phase-c29-v3-vllm-020/compare.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


DECISION_RULE = {
    "external_ceiling_exists": {
        "bound": "delta >= +0.05",
        "next_action": (
            "External system hits materially higher α. Queue a serving-"
            "difference bisect (sampler / KV layout / quantization path / "
            "RoPE / MTP-head wiring) to find what kiln diverges on."
        ),
    },
    "mtp_head_quality_ceiling": {
        "bound": "-0.02 <= delta < +0.05",
        "next_action": (
            "α is upper-bounded by checkpoint quality. Deprioritize MTP-"
            "side α work and redirect Phase 7 to decode-path optimizations "
            "that do not depend on raising α (e.g. post-#521 prefix-cache + "
            "CUDA graphs wins, SGLang RadixAttention port from #526)."
        ),
    },
    "kiln_above_vllm": {
        "bound": "delta < -0.02",
        "next_action": (
            "Unexpected — kiln α exceeds vLLM at same workload. Sanity-check "
            "vLLM args (spec method, num_speculative_tokens, sampler). If "
            "confirmed, document and queue next H."
        ),
    },
    "vllm_020_mtp_unsupported_dense_4b": {
        "bound": "vllm_020_alpha_per_seed.json.mtp_supported == false",
        "next_action": (
            "vLLM v0.20.0 also fails to serve Qwen3.5-4B dense + native "
            "MTP on A6000 (mirrors PR #530's PR-#530 0.19.1 segfault + "
            "PR #532's SGLang segfault). Both external-OSS-server "
            "candidates from PR #531 are now blocked. Queue the hand-"
            "rolled HF transformers H18 reference (PR #531 candidate 8) "
            "as the next external-α reference path; kiln-native-ceiling "
            "verdict from H15b stands until H18 lands."
        ),
    },
}


def apply_rule(delta: float) -> str:
    if delta >= 0.05:
        return "external_ceiling_exists"
    if delta >= -0.02:
        return "mtp_head_quality_ceiling"
    return "kiln_above_vllm"


def load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"missing input: {path}")
    return json.loads(path.read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--vllm",
        default="docs/phase-c29-v3-vllm-020/vllm_020_alpha_per_seed.json",
    )
    ap.add_argument(
        "--kiln",
        default="docs/phase-c29-v3-vllm-020/kiln_alpha_per_seed.json",
    )
    ap.add_argument(
        "--out-dir",
        default="docs/phase-c29-v3-vllm-020",
    )
    args = ap.parse_args()

    vllm = load(Path(args.vllm))
    kiln = load(Path(args.kiln))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mtp_supported = vllm.get("mtp_supported", False)
    if not mtp_supported:
        verdict = {
            "verdict": "vllm_020_mtp_unsupported_dense_4b",
            "decision_rule": DECISION_RULE,
            "kiln_median_alpha": kiln.get("median_alpha"),
            "vllm_020_median_alpha": None,
            "delta": None,
            "next_action": DECISION_RULE["vllm_020_mtp_unsupported_dense_4b"][
                "next_action"
            ],
            "reason": vllm.get("reason", "unknown"),
            "vllm_version": vllm.get("vllm_version", "unknown"),
        }
    else:
        kiln_med = float(kiln["median_alpha"])
        vllm_med = float(vllm["median_alpha"])
        delta = vllm_med - kiln_med
        v = apply_rule(delta)
        verdict = {
            "verdict": v,
            "decision_rule": DECISION_RULE,
            "kiln_median_alpha": kiln_med,
            "vllm_020_median_alpha": vllm_med,
            "delta": delta,
            "next_action": DECISION_RULE[v]["next_action"],
            "vllm_version": vllm.get("vllm_version", "unknown"),
        }

    compare = {
        "kiln": kiln,
        "vllm_020": vllm,
        "verdict": verdict,
    }

    (out_dir / "verdict.json").write_text(json.dumps(verdict, indent=2))
    (out_dir / "compare.json").write_text(json.dumps(compare, indent=2))

    md = []
    md.append(
        "# H17b — vLLM v0.20.0 α microbench vs kiln (free pre-step retest)\n\n"
    )
    md.append(f"**Verdict:** `{verdict['verdict']}`\n\n")
    md.append(f"**vLLM version:** `{verdict.get('vllm_version','unknown')}`\n\n")
    if mtp_supported:
        md.append(f"- kiln median α: **{verdict['kiln_median_alpha']:.4f}**\n")
        md.append(
            f"- vLLM v0.20.0 median α: **{verdict['vllm_020_median_alpha']:.4f}**\n"
        )
        md.append(f"- Δ (vllm_020 − kiln): **{verdict['delta']:+.4f}**\n\n")
    else:
        md.append(f"- kiln median α: **{verdict['kiln_median_alpha']}**\n")
        md.append(
            f"- vLLM v0.20.0 median α: **UNAVAILABLE** ({verdict['reason'][:300]})\n\n"
        )

    md.append("## Decision rule (pre-registered, unchanged from PR #530/#531/#532)\n\n")
    md.append("| verdict | bound | next action |\n")
    md.append("|---|---|---|\n")
    for name, spec in DECISION_RULE.items():
        md.append(f"| `{name}` | `{spec['bound']}` | {spec['next_action']} |\n")
    md.append("\n")

    md.append("## Per-seed table\n\n")
    if mtp_supported:
        md.append(
            "| seed | kiln α | vLLM v0.20 α | kiln accept/steps | "
            "vLLM accept/proposed |\n"
        )
        md.append("|---|---|---|---|---|\n")
        kiln_per = {r["seed"]: r for r in kiln["per_seed"]}
        vllm_per = {r["seed"]: r for r in vllm["per_seed"]}
        for seed in sorted(set(kiln_per) | set(vllm_per)):
            kr = kiln_per.get(seed, {})
            vr = vllm_per.get(seed, {})
            md.append(
                f"| {seed} | "
                f"{kr.get('alpha', 0):.4f} | "
                f"{vr.get('alpha', 0):.4f} | "
                f"{kr.get('n_accept', 0)}/{kr.get('n_steps', 0)} | "
                f"{vr.get('n_accepted', 0)}/{vr.get('n_proposed', 0)} |\n"
            )
        md.append("\n")
    else:
        md.append("| seed | kiln α | notes |\n|---|---|---|\n")
        for r in kiln["per_seed"]:
            md.append(f"| {r['seed']} | {r['alpha']:.4f} | vLLM v0.20.0 unavailable |\n")
        md.append("\n")

    md.append(f"## Next action\n\n{verdict['next_action']}\n")

    (out_dir / "compare.md").write_text("".join(md))

    print(json.dumps(verdict, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
