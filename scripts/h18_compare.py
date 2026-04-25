#!/usr/bin/env python3
"""H18 — apply the pre-registered decision rule (hand-rolled HF transformers).

Direct lineage: scripts/h17b_compare.py (PR #533, vLLM v0.20.0).

Inputs:
    docs/phase-c29-v3-hf/hf_alpha_per_seed.json
    docs/phase-c29-v3-hf/kiln_alpha_per_seed.json

Decision rule (set BEFORE the run, do NOT adjust after):

    delta = hf_median_alpha - kiln_median_alpha

    delta >= +0.05          → external_ceiling_exists_hf
    -0.02 <= delta < +0.05  → mtp_head_quality_ceiling
    delta <  -0.02          → kiln_above_hf
    mtp_supported == false  → hf_mtp_load_failure

Emits:
    docs/phase-c29-v3-hf/verdict.json
    docs/phase-c29-v3-hf/compare.json
    docs/phase-c29-v3-hf/compare.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


DECISION_RULE = {
    "external_ceiling_exists_hf": {
        "bound": "delta >= +0.05",
        "next_action": (
            "HF transformers reference α exceeds kiln α by ≥ 0.05. Kiln has "
            "α-quality work to do — queue an α-improvement task to bisect the "
            "kiln-vs-HF MTP-head implementation difference (likely candidates "
            "from prior phases: Marlin W4A16 quantization drift, fc.weight "
            "transpose convention, or pre_fc_norm swap). The kiln-native-"
            "ceiling verdict from H15b is *displaced* by this finding."
        ),
    },
    "mtp_head_quality_ceiling": {
        "bound": "-0.02 <= delta < +0.05",
        "next_action": (
            "Both implementations land within ±0.05 of each other. The "
            "pretrained MTP head's checkpoint quality is the dominant ceiling. "
            "Accept and refocus Phase 7 on non-α decode-path wins (post-PR "
            "#521 prefix-cache + CUDA graph work, SGLang RadixAttention port "
            "from PR #526). The H15b `kiln_native_ceiling` verdict stands as "
            "the operative checkpoint-quality ceiling."
        ),
    },
    "kiln_above_hf": {
        "bound": "delta < -0.02",
        "next_action": (
            "Kiln α exceeds the HF transformers reference by > 0.02 — "
            "unexpected for an MTP head that should be implementation-"
            "deterministic at greedy decode. Sanity-check the HF reference "
            "(MTP head loader prefix, pre_fc_norm swap convention, RoPE "
            "position threading). If confirmed, the H15b kiln-native-ceiling "
            "verdict stands as a safe lower bound and the H17/H18 family "
            "closes."
        ),
    },
    "hf_mtp_load_failure": {
        "bound": "hf_alpha_per_seed.json.mtp_supported == false",
        "next_action": (
            "Hand-rolled HF transformers reference also failed to load or "
            "drive the pretrained MTP head end-to-end. With vLLM 0.19.1 + "
            "v0.20.0 (PR #530, #533), SGLang main HEAD (PR #532), AND HF "
            "transformers all blocked, no external α reference is "
            "available for Qwen3.5-4B + native MTP on A6000 sm_86. Close the "
            "external-α reference family with the `kiln_native_ceiling` "
            "verdict from H15b as the operative checkpoint-quality "
            "ceiling. Phase 7 next steps: refocus on non-α decode-path wins "
            "(prefix-cache + CUDA graphs, RadixAttention port)."
        ),
    },
}


def apply_rule(delta: float) -> str:
    if delta >= 0.05:
        return "external_ceiling_exists_hf"
    if delta >= -0.02:
        return "mtp_head_quality_ceiling"
    return "kiln_above_hf"


def load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"missing input: {path}")
    return json.loads(path.read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--hf",
        default="docs/phase-c29-v3-hf/hf_alpha_per_seed.json",
    )
    ap.add_argument(
        "--kiln",
        default="docs/phase-c29-v3-hf/kiln_alpha_per_seed.json",
    )
    ap.add_argument(
        "--out-dir",
        default="docs/phase-c29-v3-hf",
    )
    args = ap.parse_args()

    hf = load(Path(args.hf))
    kiln = load(Path(args.kiln))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mtp_supported = hf.get("mtp_supported", False)
    if not mtp_supported:
        verdict = {
            "verdict": "hf_mtp_load_failure",
            "decision_rule": DECISION_RULE,
            "kiln_median_alpha": kiln.get("median_alpha"),
            "hf_median_alpha": None,
            "delta": None,
            "next_action": DECISION_RULE["hf_mtp_load_failure"]["next_action"],
            "reason": hf.get("reason", "unknown"),
            "transformers_version": hf.get("transformers_version", "unknown"),
            "torch_version": hf.get("torch_version", "unknown"),
        }
    else:
        kiln_med = float(kiln["median_alpha"])
        hf_med = float(hf["median_alpha"])
        delta = hf_med - kiln_med
        v = apply_rule(delta)
        verdict = {
            "verdict": v,
            "decision_rule": DECISION_RULE,
            "kiln_median_alpha": kiln_med,
            "hf_median_alpha": hf_med,
            "delta": delta,
            "next_action": DECISION_RULE[v]["next_action"],
            "transformers_version": hf.get("transformers_version", "unknown"),
            "torch_version": hf.get("torch_version", "unknown"),
        }

    compare = {
        "kiln": kiln,
        "hf": hf,
        "verdict": verdict,
    }

    (out_dir / "verdict.json").write_text(json.dumps(verdict, indent=2))
    (out_dir / "compare.json").write_text(json.dumps(compare, indent=2))

    md = []
    md.append("# H18 — hand-rolled HF transformers MTP α reference vs kiln\n\n")
    md.append(f"**Verdict:** `{verdict['verdict']}`\n\n")
    md.append(
        f"**transformers:** `{verdict.get('transformers_version','unknown')}` "
        f"**torch:** `{verdict.get('torch_version','unknown')}`\n\n"
    )
    if mtp_supported:
        md.append(f"- kiln median α: **{verdict['kiln_median_alpha']:.4f}**\n")
        md.append(f"- HF transformers median α: **{verdict['hf_median_alpha']:.4f}**\n")
        md.append(f"- Δ (hf − kiln): **{verdict['delta']:+.4f}**\n\n")
    else:
        md.append(f"- kiln median α: **{verdict['kiln_median_alpha']}**\n")
        md.append(
            f"- HF median α: **UNAVAILABLE** ({(verdict.get('reason') or '')[:300]})\n\n"
        )

    md.append(
        "## Decision rule (pre-registered, derived from PR #531/#533)\n\n"
    )
    md.append("| verdict | bound | next action |\n")
    md.append("|---|---|---|\n")
    for name, spec in DECISION_RULE.items():
        md.append(f"| `{name}` | `{spec['bound']}` | {spec['next_action']} |\n")
    md.append("\n")

    md.append("## Per-seed table\n\n")
    if mtp_supported:
        md.append(
            "| seed | kiln α | HF α | kiln accept/steps | HF accept/steps |\n"
        )
        md.append("|---|---|---|---|---|\n")
        kiln_per = {r["seed"]: r for r in kiln["per_seed"]}
        hf_per = {r["seed"]: r for r in hf["per_seed"]}
        for seed in sorted(set(kiln_per) | set(hf_per)):
            kr = kiln_per.get(seed, {})
            hr = hf_per.get(seed, {})
            md.append(
                f"| {seed} | "
                f"{kr.get('alpha', 0):.4f} | "
                f"{hr.get('alpha', 0):.4f} | "
                f"{kr.get('n_accept', 0)}/{kr.get('n_steps', 0)} | "
                f"{hr.get('n_accept', 0)}/{hr.get('n_steps', 0)} |\n"
            )
        md.append("\n")
    else:
        md.append("| seed | kiln α | notes |\n|---|---|---|\n")
        for r in kiln["per_seed"]:
            md.append(f"| {r['seed']} | {r['alpha']:.4f} | HF unavailable |\n")
        md.append("\n")

    md.append(f"## Next action\n\n{verdict['next_action']}\n")

    (out_dir / "compare.md").write_text("".join(md))

    print(json.dumps(verdict, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
