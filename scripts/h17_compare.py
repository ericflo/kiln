#!/usr/bin/env python3
"""H17 — apply the pre-registered decision rule for SGLang α vs kiln α.

Inputs:
    docs/archive/phase-c/phase-c29-v3-sglang/sglang_alpha_per_seed.json  (from h17_sglang_alpha_dump.py)
    docs/archive/phase-c/phase-c29-v3-sglang/kiln_alpha_per_seed.json    (from h15c_kiln_alpha_from_csv.py)

Decision rule (set BEFORE the run in PR #531 audit, do NOT adjust after):

    delta = sglang_median_alpha - kiln_median_alpha

    delta >= +0.05                    → external_ceiling_exists
    -0.02 <= delta < +0.05            → mtp_head_quality_ceiling
    delta <  -0.02                    → kiln_above_sglang
    sglang.mtp_supported == false     → sglang_mtp_unsupported_dense_4b

Emits:
    docs/archive/phase-c/phase-c29-v3-sglang/verdict.json
    docs/archive/phase-c/phase-c29-v3-sglang/compare.json
    docs/archive/phase-c/phase-c29-v3-sglang/compare.md
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
            "SGLang MTP heads beat kiln MTP heads at same workload. Queue an "
            "H18 serving-difference bisect (sampler / KV layout / quantization "
            "path / RoPE / MTP-head wiring) to find what kiln diverges on. "
            "Since kiln is Marlin W4A16 and SGLang is BF16 this confounder must "
            "be isolated first (re-run kiln in BF16 or port SGLang's quant "
            "path)."
        ),
    },
    "mtp_head_quality_ceiling": {
        "bound": "-0.02 <= delta < +0.05",
        "next_action": (
            "Both implementations hit the pretrained MTP head's quality "
            "ceiling. No external α headroom available from SGLang either. "
            "Close the external-α reference family. Pivot Phase 7 to: "
            "(a) MTP head retraining (H18 candidate), or "
            "(b) non-MTP speculative decoding (Eagle-style draft model "
            "trained from scratch on kiln logs)."
        ),
    },
    "kiln_above_sglang": {
        "bound": "delta < -0.02",
        "next_action": (
            "Unexpected — kiln α exceeds SGLang at same workload. Sanity-"
            "check SGLang args (spec_algorithm, num_draft_tokens, sampler, "
            "dtype). If confirmed, no external reference path can help raise "
            "α. Close H17 family and document kiln-native ceiling."
        ),
    },
    "sglang_mtp_unsupported_dense_4b": {
        "bound": "sglang_alpha_per_seed.json.mtp_supported == false",
        "next_action": (
            "SGLang does not yet support Qwen3.5-4B dense + native MTP "
            "end-to-end at the installed version. Escalate to free pre-step "
            "(vLLM v0.20.0 retest of PR #530 segfault) — if also fails, queue "
            "hand-rolled HF transformers H18 reference (PR #531 candidate 8)."
        ),
    },
}


def apply_rule(delta: float) -> str:
    if delta >= 0.05:
        return "external_ceiling_exists"
    if delta >= -0.02:
        return "mtp_head_quality_ceiling"
    return "kiln_above_sglang"


def load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"missing input: {path}")
    return json.loads(path.read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sglang",
        default="docs/archive/phase-c/phase-c29-v3-sglang/sglang_alpha_per_seed.json",
    )
    ap.add_argument(
        "--kiln",
        default="docs/archive/phase-c/phase-c29-v3-sglang/kiln_alpha_per_seed.json",
    )
    ap.add_argument(
        "--vllm-v020",
        default="docs/archive/phase-c/phase-c29-v3-sglang/vllm_v020_alpha_per_seed.json",
        help="Optional vLLM v0.20.0 retest dump (free pre-step). Absent = skipped.",
    )
    ap.add_argument(
        "--out-dir",
        default="docs/archive/phase-c/phase-c29-v3-sglang",
    )
    args = ap.parse_args()

    sglang = load(Path(args.sglang))
    kiln = load(Path(args.kiln))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vllm_v020 = None
    vllm_v020_path = Path(args.vllm_v020)
    if vllm_v020_path.exists():
        vllm_v020 = json.loads(vllm_v020_path.read_text())

    mtp_supported = sglang.get("mtp_supported", False)
    if not mtp_supported:
        verdict = {
            "verdict": "sglang_mtp_unsupported_dense_4b",
            "decision_rule": DECISION_RULE,
            "kiln_median_alpha": kiln.get("median_alpha"),
            "sglang_median_alpha": None,
            "delta": None,
            "next_action": DECISION_RULE["sglang_mtp_unsupported_dense_4b"][
                "next_action"
            ],
            "reason": sglang.get("reason", "unknown"),
        }
    else:
        kiln_med = float(kiln["median_alpha"])
        sglang_med = float(sglang["median_alpha"])
        delta = sglang_med - kiln_med
        v = apply_rule(delta)
        verdict = {
            "verdict": v,
            "decision_rule": DECISION_RULE,
            "kiln_median_alpha": kiln_med,
            "sglang_median_alpha": sglang_med,
            "delta": delta,
            "next_action": DECISION_RULE[v]["next_action"],
        }

    # Layer vLLM v0.20.0 retest verdict under a separate key so it
    # doesn't contaminate the canonical H17 decision.
    if vllm_v020 is not None:
        if vllm_v020.get("mtp_supported", False):
            vllm_alpha = float(vllm_v020["median_alpha"])
            if mtp_supported:
                vllm_delta = vllm_alpha - float(kiln["median_alpha"])
                if vllm_delta >= 0.05:
                    vllm_verdict = "vllm_v020_external_ceiling_exists"
                elif vllm_delta >= -0.02:
                    vllm_verdict = "vllm_v020_mtp_head_quality_ceiling"
                else:
                    vllm_verdict = "vllm_v020_kiln_above_vllm"
            else:
                vllm_verdict = "vllm_v020_supported"
                vllm_delta = None
            verdict["vllm_v020_free_prestep"] = {
                "verdict": vllm_verdict,
                "vllm_median_alpha": vllm_alpha,
                "delta_vs_kiln": vllm_delta,
            }
        else:
            verdict["vllm_v020_free_prestep"] = {
                "verdict": "vllm_v020_still_unsupported",
                "reason": vllm_v020.get("reason", "unknown"),
            }

    compare = {
        "kiln": kiln,
        "sglang": sglang,
        "vllm_v020": vllm_v020,
        "verdict": verdict,
    }

    (out_dir / "verdict.json").write_text(json.dumps(verdict, indent=2))
    (out_dir / "compare.json").write_text(json.dumps(compare, indent=2))

    # Markdown summary
    md: list[str] = []
    md.append(
        "# H17 — SGLang α microbench vs kiln (external-reference upper bound)\n\n"
    )
    md.append(f"**Verdict:** `{verdict['verdict']}`\n\n")
    if mtp_supported:
        md.append(f"- kiln median α: **{verdict['kiln_median_alpha']:.4f}**\n")
        md.append(f"- SGLang median α: **{verdict['sglang_median_alpha']:.4f}**\n")
        md.append(f"- Δ (sglang − kiln): **{verdict['delta']:+.4f}**\n\n")
    else:
        md.append(f"- kiln median α: **{verdict['kiln_median_alpha']}**\n")
        md.append(
            f"- SGLang median α: **UNAVAILABLE** ({verdict.get('reason', '')})\n\n"
        )

    if "vllm_v020_free_prestep" in verdict:
        vp = verdict["vllm_v020_free_prestep"]
        md.append(f"### vLLM v0.20.0 free pre-step\n\n")
        md.append(f"- verdict: `{vp['verdict']}`\n")
        if "vllm_median_alpha" in vp:
            md.append(f"- vLLM v0.20.0 median α: `{vp['vllm_median_alpha']:.4f}`\n")
        if vp.get("delta_vs_kiln") is not None:
            md.append(f"- Δ (vllm_v020 − kiln): `{vp['delta_vs_kiln']:+.4f}`\n")
        md.append("\n")

    md.append("## Decision rule (pre-registered, from PR #531)\n\n")
    md.append("| verdict | bound | next action |\n")
    md.append("|---|---|---|\n")
    for name, spec in DECISION_RULE.items():
        md.append(
            f"| `{name}` | `{spec['bound']}` | {spec['next_action']} |\n"
        )
    md.append("\n")

    md.append("## Per-seed table\n\n")
    if mtp_supported:
        md.append(
            "| seed | kiln α | SGLang α | kiln accept/steps | SGLang accept/proposed |\n"
        )
        md.append("|---|---|---|---|---|\n")
        kiln_per = {r["seed"]: r for r in kiln["per_seed"]}
        sglang_per = {r["seed"]: r for r in sglang["per_seed"]}
        for seed in sorted(set(kiln_per) | set(sglang_per)):
            kr = kiln_per.get(seed, {})
            sr = sglang_per.get(seed, {})
            md.append(
                f"| {seed} | "
                f"{kr.get('alpha', 0):.4f} | "
                f"{sr.get('alpha', 0):.4f} | "
                f"{kr.get('n_accept', 0)}/{kr.get('n_steps', 0)} | "
                f"{sr.get('n_accepted', 0)}/{sr.get('n_proposed', 0)} |\n"
            )
        md.append("\n")
    else:
        md.append("| seed | kiln α | notes |\n|---|---|---|\n")
        for r in kiln["per_seed"]:
            md.append(
                f"| {r['seed']} | {r['alpha']:.4f} | SGLang unavailable |\n"
            )
        md.append("\n")

    md.append(f"## Next action\n\n{verdict['next_action']}\n")

    (out_dir / "compare.md").write_text("".join(md))

    print(json.dumps(verdict, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
