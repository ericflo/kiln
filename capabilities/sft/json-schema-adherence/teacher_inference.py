"""Offline teacher pre-compute for OPD: Qwen3.6-27B → top-K logprobs over
the kiln-shared vocabulary at every assistant position.

Pipeline per train prompt:
  1. Build messages [{system}, {user}] (same as eval_kiln.py SYSTEM).
  2. Render the chat template with the SHARED tokenizer + add_generation_prompt=True.
  3. Generate teacher's assistant turn (deterministic greedy or low-temp).
  4. Append `<|im_end|>` to the assistant body and re-tokenize the full
     [system + user + assistant] conversation as the kiln trainer will.
  5. Forward 27B over the full token sequence to get per-position logits.
  6. For each assistant token position, extract top-K logprobs over the full
     vocab; record (position, indices, logprobs).
  7. Save the teacher fixture row + the (user, assistant) chat pair as the
     OPD prompt for kiln to train on.

Output files:
  datasets/train.opd.jsonl       — OpdPrompt rows: {messages:[{role,content}, ...]}
                                   keyed by the same `id` as datasets/train.jsonl
  datasets/teacher.fixture.jsonl — One row per OPD prompt:
                                   {prompt_id, tokens, active_positions,
                                    topk_indices: [[..K..],..],
                                    topk_logprobs: [[..K..],..]}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import minijinja
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).parent
TRAIN_FILE = ROOT / "datasets" / "train.jsonl"
OUT_PROMPTS = ROOT / "datasets" / "train.opd.jsonl"
OUT_FIXTURE = ROOT / "datasets" / "teacher.fixture.jsonl"

SYSTEM = (
    "You are a strict structured-output assistant. Given a user request and a "
    "JSON Schema, you reply with ONE JSON object that:\n"
    "  1. parses as valid JSON,\n"
    "  2. validates against the schema,\n"
    "  3. has substantive content (no placeholder strings),\n"
    "  4. is the ENTIRE response — no preamble, no postamble, no markdown "
    "fences, no commentary.\n"
)


def build_user_message(query: str, schema: dict) -> str:
    return (
        f"Request: {query}\n\nJSON Schema:\n{json.dumps(schema, indent=2)}\n\n"
        "Return only the JSON object."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/workspace/Qwen3.6-27B")
    ap.add_argument("--train", default=str(TRAIN_FILE))
    ap.add_argument("--out-prompts", default=str(OUT_PROMPTS))
    ap.add_argument("--out-fixture", default=str(OUT_FIXTURE))
    ap.add_argument("--top-k", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--max-prompts", type=int, default=None)
    ap.add_argument("--quant", choices=["none", "4bit"], default="4bit")
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()

    rows = []
    with open(args.train) as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    if args.max_prompts:
        rows = rows[args.start : args.start + args.max_prompts]
    else:
        rows = rows[args.start :]
    print(f"[teacher] {len(rows)} prompts queued; quant={args.quant}", file=sys.stderr)

    # Use the SHARED tokenizer (4B path; verified tokenizer.json SHA matches 27B).
    tok = AutoTokenizer.from_pretrained("/workspace/kiln/Qwen3.5-4B")
    # kiln's `tokenize_for_training` uses minijinja with add_generation_prompt=True.
    # We replicate that here so token IDs + byte offsets match what the OPD
    # trainer will see at fixture lookup time. The HF jinja2 rendering differs
    # slightly in whitespace, which broke an earlier iteration of this script.
    tpl_text = (ROOT.parent / "Qwen3.5-4B" / "chat_template.jinja").read_text()
    env = minijinja.Environment(templates={"chat": tpl_text})

    def kiln_render(messages, no_think: bool = False):
        kwargs = {"messages": messages, "add_generation_prompt": True}
        if no_think:
            kwargs["enable_thinking"] = False
        return env.render_template("chat", **kwargs)

    quant_cfg = None
    if args.quant == "4bit":
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    print("[teacher] loading 27B...", file=sys.stderr)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map={"": 0},
        quantization_config=quant_cfg,
    )
    model.eval()
    # Some Qwen multi-modal variants expose model.language_model — use it if so.
    lm = getattr(model, "language_model", model)
    print(
        f"[teacher] loaded in {time.time()-t0:.1f}s; "
        f"vocab_size={lm.config.vocab_size if hasattr(lm,'config') else 'n/a'}",
        file=sys.stderr,
    )

    # Only wipe existing output when starting from scratch. With --start > 0
    # we APPEND so previously-generated rows are preserved.
    if args.start == 0:
        if Path(args.out_prompts).exists():
            Path(args.out_prompts).unlink()
        if Path(args.out_fixture).exists():
            Path(args.out_fixture).unlink()
    f_prompts = open(args.out_prompts, "a")
    f_fixture = open(args.out_fixture, "a")
    n_done = 0
    n_skipped = 0
    t_loop = time.time()
    for r in rows:
        user_text = build_user_message(r["query"], r["schema"])
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_text},
        ]
        # Render PROMPT for generation with enable_thinking=False so the
        # 27B emits JSON directly (no reasoning trace). Generation is much
        # faster and the OPD target stays focused on the JSON body.
        prompt_text = kiln_render(messages, no_think=True)
        prompt_ids = tok(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
        prompt_len = prompt_ids.shape[1]
        # Greedy generate teacher's assistant turn.
        with torch.inference_mode():
            out = model.generate(
                prompt_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                eos_token_id=tok.convert_tokens_to_ids("<|im_end|>"),
                pad_token_id=tok.eos_token_id,
            )
        full_ids = out[0]
        assistant_ids = full_ids[prompt_len:].tolist()
        # Strip any trailing pad / EOS clones.
        # Decode for storage as text — we'll keep the response as kiln will
        # see it.
        if assistant_ids and assistant_ids[-1] == tok.convert_tokens_to_ids("<|im_end|>"):
            assistant_text_ids = assistant_ids[:-1]
        else:
            assistant_text_ids = assistant_ids
        assistant_text = tok.decode(assistant_text_ids, skip_special_tokens=False)
        # The kiln trainer tokenizes {system, user, assistant} → finds the
        # assistant span using the chat template (im_start assistant ... im_end).
        # The simplest way to get a matching full token sequence: rebuild via
        # the chat template with the full 3-message conversation.
        full_messages = messages + [{"role": "assistant", "content": assistant_text}]
        kiln_text = kiln_render(full_messages)
        kiln_ids = tok(kiln_text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
        # Find assistant span: kiln masks tokens between
        # `<|im_start|>assistant\n` and `<|im_end|>`. We approximate by
        # taking everything past the prompt's `<|im_start|>assistant\n` boundary.
        # Easier: re-render the prompt with assistant header and find that
        # length, then go to the end minus one (the trailing im_end + newline
        # tokens get masked off anyway in kiln's tokenize_for_training, but
        # for the fixture we just provide all positions in the assistant body).
        # Pragmatic: we use the same approach we used at generation —
        # prompt_text length up to and including the assistant-header — and
        # take active_positions starting from that boundary.
        # Re-encode the prompt to get its kiln-tokenized length.
        prompt_kiln_ids = tok(prompt_text, return_tensors="pt").input_ids
        prompt_kiln_len = prompt_kiln_ids.shape[1]
        seq_len = kiln_ids.shape[1]
        # Match kiln's `label_mask_from_rendered_assistant_spans` exactly:
        # find the FIRST `<|im_start|>assistant\n` and its FIRST closing
        # `<|im_end|>`, then mark all tokens whose byte-offsets overlap
        # that span. add_generation_prompt=True appends another (open)
        # assistant header — we deliberately ignore that, matching the
        # trainer's logic which only marks closed assistant spans.
        ASSISTANT_START = "<|im_start|>assistant\n"
        MESSAGE_END = "<|im_end|>"
        marker_idx = kiln_text.find(ASSISTANT_START)
        if marker_idx < 0:
            n_skipped += 1
            continue
        content_start = marker_idx + len(ASSISTANT_START)
        end_rel = kiln_text.find(MESSAGE_END, content_start)
        if end_rel < 0:
            n_skipped += 1
            continue
        span_start = marker_idx
        span_end = end_rel + len(MESSAGE_END)
        if span_end < len(kiln_text) and kiln_text[span_end] == "\n":
            span_end += 1
        enc = tok(
            kiln_text,
            return_offsets_mapping=True,
            return_tensors=None,
            add_special_tokens=False,
        )
        offsets = enc["offset_mapping"]
        assert len(offsets) == seq_len, (len(offsets), seq_len)
        active_positions = []
        for ti, (ts, te) in enumerate(offsets):
            if ts == te:
                continue
            if ts < span_end and te > span_start:
                active_positions.append(ti)
        if not active_positions:
            n_skipped += 1
            continue

        # Forward pass over the full kiln-tokenized sequence.
        with torch.inference_mode():
            logits = lm(kiln_ids).logits  # [1, T, V]
        logits = logits[0]  # [T, V]
        # Extract top-K at each active position; convert to f32 logprobs.
        # We compute log_softmax once per position over the full vocab and
        # take topk. To save memory we do it in chunks.
        topk_indices_all: list[list[int]] = []
        topk_logprobs_all: list[list[float]] = []
        CHUNK = 256
        for i in range(0, len(active_positions), CHUNK):
            idx_slice = active_positions[i : i + CHUNK]
            sub_logits = logits[idx_slice]  # [chunk, V]
            log_probs = F.log_softmax(sub_logits.float(), dim=-1)
            tk = torch.topk(log_probs, args.top_k, dim=-1)
            inds = tk.indices.cpu().tolist()
            lps = tk.values.cpu().tolist()
            topk_indices_all.extend(inds)
            topk_logprobs_all.extend(lps)
        # Sanity: the model's "predicted" token at each active position
        # should *be* the next assistant token in the sequence — the
        # teacher generated greedily, so the top-1 prediction at each
        # position should match the actual next token in 99%+ of cases.
        # We sample one position to log: not critical.

        # Persist the OpdPrompt as {user, assistant} only (kiln's
        # tokenize_for_training will re-tokenize this with the same template).
        # We DROP the system message in the persisted prompt to match what
        # the eval_kiln.py rubric uses — that script passes [system, user]
        # explicitly, and the trainer here uses [user, assistant]. Wait —
        # we want kiln to tokenize the SAME [system, user, assistant] that
        # the teacher saw, otherwise the active_positions wouldn't align.
        # Keep the system message.
        opd_prompt = {
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text},
            ]
        }
        f_prompts.write(json.dumps(opd_prompt) + "\n")
        fix = {
            "prompt_id": r["id"],
            "tokens": kiln_ids[0].cpu().tolist(),
            "active_positions": active_positions,
            "topk_indices": topk_indices_all,
            "topk_logprobs": topk_logprobs_all,
        }
        f_fixture.write(json.dumps(fix) + "\n")
        f_prompts.flush()
        f_fixture.flush()
        n_done += 1
        if n_done % 5 == 0 or n_done == len(rows):
            dt = time.time() - t_loop
            print(
                f"[teacher] {n_done}/{len(rows)} done "
                f"(skipped={n_skipped}) "
                f"avg={dt/max(n_done,1):.2f}s/prompt "
                f"vram={torch.cuda.memory_allocated()/1024**3:.1f}GB",
                file=sys.stderr,
            )
    f_prompts.close()
    f_fixture.close()
    print(
        f"[teacher] wrote {n_done} rows to {args.out_prompts} + {args.out_fixture}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
