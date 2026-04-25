#!/usr/bin/env python3
"""H15c — apply the pre-registered decision rule.

Inputs:
    docs/phase-c29-v3-vllm/vllm_alpha_per_seed.json  (from h15c_vllm_alpha_dump.py)
    docs/phase-c29-v3-vllm/kiln_alpha_per_seed.json  (from h15c_kiln_alpha_from_csv.py)

Decision rule (set BEFORE the run, do NOT adjust after):

    delta = vllm_median_alpha - kiln_median_alpha

    delta >= +0.05          → external_ceiling_exists
    -0.02 <= delta < +0.05  → mtp_head_quality_ceiling
    delta <  -0.02          → kiln_above_vllm

Emits:
    docs/phase-c29-v3-vllm/verdict.json
    docs/phase-c29-v3-vllm/compare.json
    docs/phase-c29-v3-vllm/compare.md
"""
from __future__ import annotations

import argparse
import json
import sys
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
    "vllm_mtp_unsupported": {
        "bound": "vllm_alpha_per_seed.json.mtp_supported == false",
        "next_action": (
            "vLLM does not yet support Qwen3.5-4B native MTP at the installed "
            "version. Ship this as a doc-only redirect PR documenting the gap, "
            "queue the next H from PR #527 §'Queued next action'."
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
        default="docs/phase-c29-v3-vllm/vllm_alpha_per_seed.json",
    )
    ap.add_argument(
        "--kiln",
        default="docs/phase-c29-v3-vllm/kiln_alpha_per_seed.json",
    )
    ap.add_argument(
        "--out-dir",
        default="docs/phase-c29-v3-vllm",
    )
    args = ap.parse_args()

    vllm = load(Path(args.vllm))
    kiln = load(Path(args.kiln))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mtp_supported = vllm.get("mtp_supported", False)
    if not mtp_supported:
        verdict = {
            "verdict": "vllm_mtp_unsupported",
            "decision_rule": DECISION_RULE,
            "kiln_median_alpha": kiln.get("median_alpha"),
            "vllm_median_alpha": None,
            "delta": None,
            "next_action": DECISION_RULE["vllm_mtp_unsupported"]["next_action"],
            "reason": vllm.get("reason", "unknown"),
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
            "vllm_median_alpha": vllm_med,
            "delta": delta,
            "next_action": DECISION_RULE[v]["next_action"],
        }

    compare = {
        "kiln": kiln,
        "vllm": vllm,
        "verdict": verdict,
    }

    (out_dir / "verdict.json").write_text(json.dumps(verdict, indent=2))
    (out_dir / "compare.json").write_text(json.dumps(compare, indent=2))

    # Markdown summary
    md = []
    md.append("# H15c — vLLM α microbench vs kiln (external-reference upper bound)\n\n")
    md.append(f"**Verdict:** `{verdict['verdict']}`\n\n")
    if mtp_supported:
        md.append(f"- kiln median α: **{verdict['kiln_median_alpha']:.4f}**\n")
        md.append(f"- vLLM median α: **{verdict['vllm_median_alpha']:.4f}**\n")
        md.append(f"- Δ (vllm − kiln): **{verdict['delta']:+.4f}**\n\n")
    else:
        md.append(f"- kiln median α: **{verdict['kiln_median_alpha']}**\n")
        md.append(f"- vLLM median α: **UNAVAILABLE** ({verdict['reason']})\n\n")

    md.append("## Decision rule (pre-registered)\n\n")
    md.append("| verdict | bound | next action |\n")
    md.append("|---|---|---|\n")
    for name, spec in DECISION_RULE.items():
        md.append(f"| `{name}` | `{spec['bound']}` | {spec['next_action']} |\n")
    md.append("\n")

    md.append("## Per-seed table\n\n")
    if mtp_supported:
        md.append("| seed | kiln α | vLLM α | kiln accept/steps | vLLM accept/proposed |\n")
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
            md.append(f"| {r['seed']} | {r['alpha']:.4f} | vLLM unavailable |\n")
        md.append("\n")

    md.append(f"## Next action\n\n{verdict['next_action']}\n")

    (out_dir / "compare.md").write_text("".join(md))

    print(json.dumps(verdict, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
