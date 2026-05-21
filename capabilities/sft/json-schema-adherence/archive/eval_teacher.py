"""Eval Qwen3.6-27B (the OPD teacher) on the same rubric as eval_kiln.py.

Computes the "oracle ceiling" — the score we'd expect if distillation
fully captured the teacher's behavior. Caller is responsible for ensuring
no other process holds the GPU.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import minijinja
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from rubric import score_response

ROOT = Path(__file__).parent
SYSTEM = (
    "You are a strict structured-output assistant. Given a user request and a "
    "JSON Schema, you reply with ONE JSON object that:\n"
    "  1. parses as valid JSON,\n"
    "  2. validates against the schema,\n"
    "  3. has substantive content (no placeholder strings),\n"
    "  4. is the ENTIRE response — no preamble, no postamble, no markdown "
    "fences, no commentary.\n"
)


def build_user(query: str, schema: dict) -> str:
    return (
        f"Request: {query}\n\nJSON Schema:\n{json.dumps(schema, indent=2)}\n\n"
        "Return only the JSON object."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/workspace/Qwen3.6-27B")
    ap.add_argument("--eval-file", default=str(ROOT / "datasets" / "eval.jsonl"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--quant", choices=["none", "4bit"], default="4bit")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained("/workspace/kiln/Qwen3.5-4B")
    tpl_text = (ROOT.parent / "Qwen3.5-4B" / "chat_template.jinja").read_text()
    env = minijinja.Environment(templates={"chat": tpl_text})

    quant_cfg = None
    if args.quant == "4bit":
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    print("[eval_teacher] loading 27B...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": 0},
        quantization_config=quant_cfg,
    )
    model.eval()
    lm = getattr(model, "language_model", model)
    print(f"[eval_teacher] loaded in {time.time()-t0:.1f}s")

    eval_rows = [json.loads(l) for l in open(args.eval_file)]
    rows = []
    t_start = time.time()
    for i, p in enumerate(eval_rows):
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": build_user(p["query"], p["schema"])},
        ]
        prompt_text = env.render_template(
            "chat", messages=messages, add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_ids = tok(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
        with torch.inference_mode():
            out = model.generate(
                prompt_ids,
                max_new_tokens=args.max_tokens,
                do_sample=False,
                eos_token_id=tok.convert_tokens_to_ids("<|im_end|>"),
                pad_token_id=tok.eos_token_id,
            )
        assistant_ids = out[0][prompt_ids.shape[1]:].tolist()
        if assistant_ids and assistant_ids[-1] == tok.convert_tokens_to_ids("<|im_end|>"):
            assistant_ids = assistant_ids[:-1]
        text = tok.decode(assistant_ids, skip_special_tokens=False)
        s = score_response(text, p["schema"])
        s["id"] = p["id"]
        s["response_chars"] = len(text)
        rows.append(s)
        if (i+1) % 5 == 0 or (i+1) == len(eval_rows):
            print(f"  {i+1}/{len(eval_rows)} composite_so_far={sum(r['composite'] for r in rows)/len(rows):.4f}")

    def mean(k):
        return sum(r[k] for r in rows) / len(rows)
    agg = {
        "model": "qwen3.6-27b@4bit-offline",
        "n": len(rows),
        "wall_time_s": round(time.time()-t_start, 1),
        "parses": round(mean("parses"), 4),
        "validates": round(mean("validates"), 4),
        "is_pure": round(mean("is_pure"), 4),
        "is_substantive": round(mean("is_substantive"), 4),
        "composite": round(mean("composite"), 4),
        "per_prompt": rows,
    }
    Path(args.out).write_text(json.dumps(agg, indent=2))
    print(f"[eval_teacher] saved {args.out}: composite={agg['composite']:.4f}")


if __name__ == "__main__":
    main()
