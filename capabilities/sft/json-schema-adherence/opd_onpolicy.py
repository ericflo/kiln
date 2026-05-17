"""TRUE on-policy distillation (OPD).

For each training step:
  1. Sample a trajectory from the STUDENT model (with current LoRA applied).
  2. Forward the TEACHER over the (prompt + student_response) tokens.
  3. Forward the STUDENT over the same tokens.
  4. At each student-response position, compute per-token reverse-KL of the
     student's distribution against the teacher's (Lu 2025 §3.1):
         KL_t = sum_k p_θ(k|...) · (log p_θ(k|...) − log p_T(k|...))
     where p_θ is the student, p_T is the teacher, and k ranges over the
     teacher's top-K support (Fu et al. 2026 — full-vocab KL is too noisy).
  5. Backward + AdamW step on the LoRA parameters.

This is the literal Lu / Tinker-cookbook recipe, run on a single 48 GB GPU.
Both models live in memory simultaneously: Qwen3.5-4B student in bf16 (~8 GB)
and Qwen3.6-27B teacher in 4-bit NF4 via bitsandbytes (~18 GB).

Output: a PEFT-format LoRA adapter at `--output-dir`. A separate utility can
convert it to kiln's safetensors layout for serving; for the prototype we
also evaluate the adapter directly via PyTorch inference using `rubric.py`.

Usage:
    python3 opd_onpolicy.py \
        --student /workspace/kiln/Qwen3.5-4B \
        --teacher /workspace/Qwen3.6-27B \
        --train datasets/train.jsonl \
        --output-dir adapters/opd-onpolicy-v1 \
        --rank 16 --lr 1e-5 --epochs 1 --top-k 32 \
        --max-prompts 38
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
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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


def build_user_message(query: str, schema: dict) -> str:
    return (
        f"Request: {query}\n\nJSON Schema:\n{json.dumps(schema, indent=2)}\n\n"
        "Return only the JSON object."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default="/workspace/kiln/Qwen3.5-4B")
    ap.add_argument("--teacher", default="/workspace/Qwen3.6-27B")
    ap.add_argument("--train", default=str(ROOT / "datasets" / "train.jsonl"))
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--top-k", type=int, default=32)
    ap.add_argument("--max-prompts", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--samples-per-prompt", type=int, default=1)
    ap.add_argument("--seed", type=int, default=4218)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    # -------------------------------------------------------------------
    # Tokenizer + chat template (kiln-canonical via minijinja)
    # -------------------------------------------------------------------
    tok = AutoTokenizer.from_pretrained(args.student)
    tpl_text = (Path(args.student) / "chat_template.jinja").read_text()
    env = minijinja.Environment(templates={"chat": tpl_text})

    def kiln_render(messages, gen_prompt=False, no_think=False):
        kwargs = {
            "messages": messages,
            "add_generation_prompt": gen_prompt,
        }
        if no_think:
            kwargs["enable_thinking"] = False
        return env.render_template("chat", **kwargs)

    im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
    pad_id = tok.pad_token_id or tok.eos_token_id

    # -------------------------------------------------------------------
    # Load training prompts
    # -------------------------------------------------------------------
    prompts = []
    with open(args.train) as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    if args.max_prompts:
        prompts = prompts[: args.max_prompts]
    print(f"[opd] {len(prompts)} train prompts", file=sys.stderr, flush=True)

    # -------------------------------------------------------------------
    # Load STUDENT in bf16 with PEFT LoRA on all linear layers
    # -------------------------------------------------------------------
    print("[opd] loading student (bf16) ...", file=sys.stderr, flush=True)
    t0 = time.time()
    student = AutoModelForCausalLM.from_pretrained(
        args.student,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="eager",
    )
    student.config.use_cache = False
    print(f"[opd] student loaded in {time.time()-t0:.1f}s", file=sys.stderr, flush=True)

    # Apply LoRA to all linear modules (LoRA Without Regret §)
    lora_cfg = LoraConfig(
        r=args.rank,
        lora_alpha=32,
        # All linear layers — Schulman (2025): attention-only is worse than MLP.
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    student = get_peft_model(student, lora_cfg)
    student.print_trainable_parameters()

    # -------------------------------------------------------------------
    # Load TEACHER in 4-bit NF4
    # -------------------------------------------------------------------
    print("[opd] loading teacher (4-bit) ...", file=sys.stderr, flush=True)
    t0 = time.time()
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher,
        dtype=torch.bfloat16,
        device_map={"": 0},
        quantization_config=quant,
        attn_implementation="eager",
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher_lm = getattr(teacher, "language_model", teacher)
    print(
        f"[opd] teacher loaded in {time.time()-t0:.1f}s, vram={torch.cuda.memory_allocated()/1024**3:.1f}GB",
        file=sys.stderr, flush=True,
    )

    # -------------------------------------------------------------------
    # Optimizer over LoRA params only
    # -------------------------------------------------------------------
    lora_params = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(lora_params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    print(
        f"[opd] LoRA trainable params: {sum(p.numel() for p in lora_params)/1e6:.1f}M",
        file=sys.stderr, flush=True,
    )

    # -------------------------------------------------------------------
    # Training loop — TRUE on-policy
    # -------------------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    global_step = 0
    total_steps = args.epochs * len(prompts) * args.samples_per_prompt
    t_loop_start = time.time()

    for epoch in range(args.epochs):
        for prompt_idx, p in enumerate(prompts):
            for sample_idx in range(args.samples_per_prompt):
                # === 1. Build the prompt seen by the student ===
                user_text = build_user_message(p["query"], p["schema"])
                messages = [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user_text},
                ]
                # Render WITH add_generation_prompt=True and enable_thinking=False
                # so the student starts JSON directly.
                prompt_text = kiln_render(messages, gen_prompt=True, no_think=True)
                prompt_ids = tok(
                    prompt_text, return_tensors="pt", add_special_tokens=False
                ).input_ids.to(student.device)
                prompt_len = prompt_ids.shape[1]

                # === 2. SAMPLE FROM STUDENT (this is the on-policy bit) ===
                student.eval()
                with torch.inference_mode():
                    student.config.use_cache = True
                    gen_out = student.generate(
                        prompt_ids,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=(args.temperature > 0),
                        temperature=args.temperature,
                        top_p=args.top_p,
                        eos_token_id=im_end_id,
                        pad_token_id=pad_id,
                    )
                    student.config.use_cache = False
                student.train()

                full_ids = gen_out[0]  # [prompt_len + gen_len]
                gen_len = full_ids.shape[0] - prompt_len
                if gen_len < 2:
                    continue  # nothing to learn

                # === 3. STUDENT forward (with LoRA, autograd ON) over full sequence ===
                # We want student logits at student-sampled positions.
                input_ids_2d = full_ids.unsqueeze(0).to(student.device)
                # student logits[i] = prediction for token i+1
                student_out = student(input_ids_2d, use_cache=False)
                student_logits = student_out.logits[0]  # [T, V]

                # === 4. TEACHER forward (no grad) over same tokens ===
                with torch.inference_mode():
                    teacher_out = teacher_lm(input_ids_2d)
                    teacher_logits = teacher_out.logits[0]  # [T, V]

                # === 5. Active positions: where the student GENERATED (so the
                #     logit at position t-1 predicted token t). For a token at
                #     position t to be a student-generated token, t in [prompt_len, full_len].
                #     We compute loss at positions t-1 in [prompt_len-1, full_len-2],
                #     where the prediction TARGET is a student-generated token.
                active = list(range(prompt_len - 1, full_ids.shape[0] - 1))
                if not active:
                    continue
                active_t = torch.tensor(active, device=student.device)

                # === 6. Top-K reverse KL on teacher's top-K support ===
                # Gather student + teacher logits at active positions
                s_logits_active = student_logits.index_select(0, active_t)  # [T_act, V]
                t_logits_active = teacher_logits.index_select(0, active_t)  # [T_act, V]
                # Teacher's top-K indices
                t_log_probs = F.log_softmax(t_logits_active.float(), dim=-1)
                topk_vals, topk_idx = torch.topk(t_log_probs, args.top_k, dim=-1)
                # Renormalize teacher over its top-K support
                t_supp_log = F.log_softmax(
                    t_log_probs.gather(-1, topk_idx), dim=-1
                )  # [T_act, K]
                # Student logits at those same K indices, renormalized
                s_supp_logits = s_logits_active.float().gather(-1, topk_idx)  # [T_act, K]
                s_supp_log = F.log_softmax(s_supp_logits, dim=-1)
                s_supp = s_supp_log.exp()
                # Reverse KL: KL(student || teacher) = sum_k p_s * (log p_s - log p_t)
                per_pos_kl = (s_supp * (s_supp_log - t_supp_log)).sum(-1)  # [T_act]
                loss = per_pos_kl.mean()

                # === 7. Backward + step ===
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
                optimizer.step()

                global_step += 1
                loss_val = loss.item()
                if global_step % 1 == 0 or global_step == total_steps:
                    elapsed = time.time() - t_loop_start
                    print(
                        f"[opd] step={global_step}/{total_steps} epoch={epoch+1}/{args.epochs} "
                        f"prompt={prompt_idx} gen_len={gen_len} active={len(active)} "
                        f"loss={loss_val:.4f} elapsed={elapsed:.1f}s "
                        f"vram={torch.cuda.memory_allocated()/1024**3:.1f}GB",
                        file=sys.stderr,
                        flush=True,
                    )

    print("[opd] training complete, saving adapter ...", file=sys.stderr, flush=True)
    student.save_pretrained(str(output_dir))
    tok.save_pretrained(str(output_dir))
    print(f"[opd] saved PEFT adapter to {output_dir}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
