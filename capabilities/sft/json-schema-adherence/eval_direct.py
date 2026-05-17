"""Direct eval — load the kiln 4B model + a kiln-trained LoRA adapter in
Python (HF transformers + PEFT), run inference on the eval prompts, score
with the JSON-schema rubric. No kiln-server needed.

Usage:
    python3 eval_direct.py --adapter /tmp/opd-onpolicy-v1/opd-onpolicy-v1 \
        --out judgments/opd-onpolicy-v1.json
    python3 eval_direct.py --adapter base --out judgments/baseline-direct.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rubric import score_response

MODEL_PATH = "/workspace/kiln/Qwen3.5-4B"
EVAL_FILE = Path(__file__).parent / "datasets" / "eval.jsonl"

SYSTEM = (
    "You are a strict structured-output assistant. Given a user request and a "
    "JSON Schema, you reply with ONE JSON object that:\n"
    "  1. parses as valid JSON,\n"
    "  2. validates against the schema,\n"
    "  3. has substantive content (no placeholder strings),\n"
    "  4. is the ENTIRE response — no preamble, no postamble, no markdown "
    "fences, no commentary.\n"
)


def build_messages(query, schema):
    schema_str = json.dumps(schema, indent=2)
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": (
                f"Request: {query}\n\nJSON Schema:\n{schema_str}\n\n"
                "Return only the JSON object."
            ),
        },
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", default="base", help="adapter dir or 'base'")
    p.add_argument("--out", required=True)
    p.add_argument("--max-tokens", type=int, default=1200)
    p.add_argument("--max-prompts", type=int, default=None)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    print(f"loading base model from {MODEL_PATH}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    model.eval()

    if args.adapter not in ("base", "none", ""):
        from peft import PeftModel

        print(f"loading adapter from {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()
        model.eval()

    prompts = []
    with open(EVAL_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            prompts.append(json.loads(line))
    if args.max_prompts:
        prompts = prompts[: args.max_prompts]
    print(f"eval prompts: {len(prompts)}")

    judgments = []
    t0 = time.time()
    for i, prompt in enumerate(prompts):
        messages = build_messages(prompt["query"], prompt["schema"])
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        ids = tok(text, return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            out = model.generate(
                **ids,
                max_new_tokens=args.max_tokens,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        gen = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
        score = score_response(gen, prompt["schema"])
        score["id"] = prompt["id"]
        score["response_preview"] = gen[:200]
        judgments.append(score)
        if (i + 1) % 5 == 0 or i + 1 == len(prompts):
            avg = sum(j["composite"] for j in judgments) / len(judgments)
            print(f"  [{i + 1}/{len(prompts)}] composite_so_far={avg:.4f}")

    n = len(judgments)
    summary = {
        "n": n,
        "parses": sum(j["parses"] for j in judgments) / n,
        "validates": sum(j["validates"] for j in judgments) / n,
        "is_pure": sum(j["is_pure"] for j in judgments) / n,
        "is_substantive": sum(j["is_substantive"] for j in judgments) / n,
        "composite": sum(j["composite"] for j in judgments) / n,
        "elapsed_s": time.time() - t0,
    }
    out_dict = {"summary": summary, "judgments": judgments}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out_dict, f, indent=2)
    print(f"\nwrote {args.out}")
    print(f"  composite = {summary['composite']:.4f}")
    print(f"  parses    = {summary['parses']:.4f}")
    print(f"  validates = {summary['validates']:.4f}")
    print(f"  is_pure   = {summary['is_pure']:.4f}")
    print(f"  is_subs.  = {summary['is_substantive']:.4f}")


if __name__ == "__main__":
    main()
