#!/usr/bin/env python3
"""Build a small constrained-RLM experiment from trajectory-trainer turns.

The builder samples real tool-call turns from trajectory-trainer, materializes
them with the production materializer, and emits:

- same-tool-eventually EvalSuite JSON;
- root-action SFT rows for the fixed-window RLM controller;
- ECHO-ready rollout groups that preserve action/observation alternation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sqlite3
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pi_rlm_harness import (  # noqa: E402
    DEFAULT_MODEL,
    INPUT_TOKENS,
    OUTPUT_TOKENS,
    RLM_SYSTEM_PROMPT,
    TURN_BREAK,
    RlmEnvironment,
    TokenBudget,
    action_text,
    json_dumps,
    normalize_tool_call,
    parse_arguments,
    pretty_json,
    sanitize_id,
    tool_name,
)

DEFAULT_DB = "/data/.clouderic-internal/repos/apps/trajectory-trainer/trajectories.db"
DEFAULT_MATERIALIZE = "/data/apps/trajectory-trainer/scripts/materialize_turn.py"
DEFAULT_OUT = ".kiln/pi-rlm-trace-experiment"


def load_materializer(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("trajectory_materialize_turn", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import materializer from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def candidate_rows(
    conn: sqlite3.Connection,
    *,
    min_input_tokens: int,
    max_input_tokens: int,
    seed: int,
    candidate_limit: int | None,
) -> list[sqlite3.Row]:
    rows = conn.execute(
        """
        SELECT id, session_id, model, split, num_input_messages, input_tokens, output_tokens
        FROM turns
        WHERE num_tools > 0
          AND response_json IS NOT NULL
          AND input_tokens BETWEEN ? AND ?
        """,
        (min_input_tokens, max_input_tokens),
    ).fetchall()
    rng = random.Random(seed)
    rng.shuffle(rows)
    if candidate_limit is not None:
        rows = rows[:candidate_limit]
    return rows


def prompt_from_env(env: RlmEnvironment, depth: int, step: int) -> str:
    body = {
        "depth": depth,
        "step": step,
        "environment_summary": env.summary(),
        "recent_observations": env.observations[-8:],
    }
    return env.budget.clip(pretty_json(body), INPUT_TOKENS)


def latest_user_index(messages: list[dict[str, Any]]) -> int:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "user":
            return index
    return max(0, len(messages) - 1)


def target_from_chosen(chosen: dict[str, Any]) -> dict[str, Any] | None:
    calls = chosen.get("tool_calls")
    if not isinstance(calls, list) or not calls:
        return None
    first = calls[0]
    if not isinstance(first, dict):
        return None
    fn = first.get("function")
    if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
        return None
    return {
        "name": fn["name"],
        "arguments": parse_arguments(fn.get("arguments")),
    }


def tool_names(tools: list[dict[str, Any]]) -> set[str]:
    return {name for name in (tool_name(t) for t in tools) if name}


def materialize_examples(
    conn: sqlite3.Connection,
    materializer: Any,
    candidates: list[sqlite3.Row],
    *,
    budget: TokenBudget,
    max_action_tokens: int,
    total: int,
    require_single_tool_call: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    examples: list[dict[str, Any]] = []
    skips: dict[str, int] = {
        "no_row": 0,
        "materialize_error": 0,
        "no_target": 0,
        "multiple_tool_calls": 0,
        "unknown_target_tool": 0,
        "target_action_too_long": 0,
        "empty_messages": 0,
    }

    for row in candidates:
        if len(examples) >= total:
            break
        turn_id = str(row["id"])
        raw = materializer.get_turn_by_id(conn, turn_id)
        if raw is None:
            skips["no_row"] += 1
            continue
        try:
            turn = materializer.normalize_turn(raw)
            record = json.loads(materializer.format_prompt_chosen_jsonl(turn))
        except Exception:
            skips["materialize_error"] += 1
            continue

        chosen = record.get("chosen") or {}
        calls = chosen.get("tool_calls")
        if not isinstance(calls, list) or not calls:
            skips["no_target"] += 1
            continue
        if require_single_tool_call and len(calls) != 1:
            skips["multiple_tool_calls"] += 1
            continue
        target = target_from_chosen(chosen)
        if target is None:
            skips["no_target"] += 1
            continue

        messages = record.get("prompt_messages") or []
        tools = record.get("tools") or []
        if not isinstance(messages, list) or not messages:
            skips["empty_messages"] += 1
            continue
        if target["name"] not in tool_names(tools):
            skips["unknown_target_tool"] += 1
            continue
        target_action = {
            "action": "finish",
            "tool_call": {
                "name": target["name"],
                "arguments": target["arguments"],
            },
        }
        if budget.count(action_text(target_action)) > max_action_tokens:
            skips["target_action_too_long"] += 1
            continue

        meta = dict(record.get("metadata") or {})
        meta.update(
            {
                "turn_id": record.get("id") or turn_id,
                "session_id": record.get("session_id"),
                "model": record.get("model"),
                "split": record.get("split"),
                "target_tool": target["name"],
            }
        )
        examples.append(
            {
                "id": str(record.get("id") or turn_id),
                "messages": messages,
                "tools": tools,
                "target": target,
                "prompt_chosen": record,
                "metadata": meta,
            }
        )
    return examples, skips


def suite_from_examples(name: str, examples: list[dict[str, Any]], model: str) -> dict[str, Any]:
    return {
        "name": name,
        "model": model,
        "description": (
            "Random production trajectory-trainer turns scored by whether the "
            "RLM harness eventually emits the same first Pi tool call."
        ),
        "default_scorer": {
            "kind": "tool_call",
            "name_match": "case_insensitive",
            "args": "structural",
        },
        "examples": [
            {
                "id": ex["id"],
                "messages": ex["messages"],
                "tools": ex["tools"],
                "target": json_dumps({"tool_calls": [ex["target"]]}),
                "metadata": ex["metadata"],
            }
            for ex in examples
        ],
    }


def add_sft_row(
    rows: list[dict[str, Any]],
    env: RlmEnvironment,
    *,
    depth: int,
    step: int,
    action: dict[str, Any],
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    prompt_messages = [
        {"role": "system", "content": RLM_SYSTEM_PROMPT},
        {"role": "user", "content": prompt_from_env(env, depth, step)},
    ]
    row = {
        "messages": prompt_messages + [{"role": "assistant", "content": action_text(action)}],
        "metadata": metadata
        | {
            "depth": depth,
            "step": step,
            "action": action.get("action"),
        },
    }
    rows.append(row)
    return prompt_messages


def build_trace_for_example(
    ex: dict[str, Any],
    *,
    budget: TokenBudget,
    state_dir: Path,
    use_spawn: bool,
    source_index: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    request_id = sanitize_id(f"trace-{ex['id']}")
    upstream_request = {
        "model": ex["metadata"].get("model") or DEFAULT_MODEL,
        "messages": ex["messages"],
        "tools": ex["tools"],
        "metadata": {"request_id": request_id},
    }
    env = RlmEnvironment(
        request_id=request_id,
        parent_id=None,
        depth=0,
        messages=ex["messages"],
        tools=ex["tools"],
        upstream_request=upstream_request,
        state_dir=state_dir,
        budget=budget,
    )
    rows: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    trajectory: list[dict[str, Any]] = []
    base_meta = {
        "source": "trajectory_trainer_materialized_turn",
        "turn_id": ex["id"],
        "source_index": source_index,
        "target_tool": ex["target"]["name"],
        "scaffold": "recursive_spawn" if use_spawn else "direct_inspect",
    }

    def train_action(step: int, action: dict[str, Any]) -> Any:
        prompt_messages = add_sft_row(rows, env, depth=0, step=step, action=action, metadata=base_meta)
        output = env.inspect(action) if action.get("action") == "inspect" else None
        if output is not None:
            obs = env.observe(action, output)
            trajectory.append({"role": "assistant", "content": action_text(action), "kind": "action"})
            trajectory.append({"role": "tool", "content": json_dumps(obs), "kind": "observation"})
        else:
            trajectory.append({"role": "assistant", "content": action_text(action), "kind": "action"})
        actions.append(action)
        return prompt_messages, output

    first_prompt_messages, _ = train_action(0, {"action": "inspect", "target": "summary"})
    train_action(1, {"action": "inspect", "target": "tools"})

    final_action = {
        "action": "finish",
        "tool_call": {
            "name": ex["target"]["name"],
            "arguments": ex["target"]["arguments"],
        },
    }

    if use_spawn:
        msg_index = latest_user_index(ex["messages"])
        spawn_action = {
            "action": "spawn_agent",
            "task": (
                "Identify the single next Pi-visible tool call required by this "
                "conversation. Return a final finish.tool_call with the tool name "
                "and JSON arguments only."
            ),
            "message_refs": [msg_index],
            "max_iters": 3,
        }
        add_sft_row(rows, env, depth=0, step=2, action=spawn_action, metadata=base_meta)
        actions.append(spawn_action)
        trajectory.append({"role": "assistant", "content": action_text(spawn_action), "kind": "action"})

        child_messages = env.materialize_child_messages(spawn_action)
        child_env = RlmEnvironment(
            request_id=sanitize_id(f"{request_id}.child0"),
            parent_id=request_id,
            depth=1,
            messages=child_messages,
            tools=ex["tools"],
            upstream_request=upstream_request,
            state_dir=state_dir,
            budget=budget,
        )
        child_meta = base_meta | {"scaffold": "recursive_child", "parent_request_id": request_id}
        child_inspect = {"action": "inspect", "target": "summary"}
        add_sft_row(rows, child_env, depth=1, step=0, action=child_inspect, metadata=child_meta)
        child_env.observe(child_inspect, child_env.inspect(child_inspect))
        add_sft_row(rows, child_env, depth=1, step=1, action=final_action, metadata=child_meta)

        child_result = {
            "child_request_id": child_env.request_id,
            "depth": 1,
            "content": "",
            "tool_call": normalize_tool_call(ex["target"]),
            "fallback_used": False,
            "trace_steps": 2,
        }
        obs = env.observe(spawn_action, child_result)
        trajectory.append({"role": "tool", "content": json_dumps(obs), "kind": "observation"})
        add_sft_row(rows, env, depth=0, step=3, action=final_action, metadata=base_meta)
        actions.append(final_action)
        trajectory.append({"role": "assistant", "content": action_text(final_action), "kind": "action"})
    else:
        train_action(2, {"action": "inspect", "target": "message", "index": latest_user_index(ex["messages"])})
        add_sft_row(rows, env, depth=0, step=3, action=final_action, metadata=base_meta)
        actions.append(final_action)
        trajectory.append({"role": "assistant", "content": action_text(final_action), "kind": "action"})

    echo_group = {
        "messages": first_prompt_messages,
        "rollouts": [
            {
                "text": TURN_BREAK.join(action_text(action) for action in actions),
                "reward": 1.0,
                "trajectory": trajectory,
            }
        ],
        "metadata": base_meta
        | {
            "source": "pi_rlm_trace_experiment",
            "trace_steps": len(actions),
            "final_tool": ex["target"]["name"],
            "echo_ready": any(seg.get("kind") == "observation" for seg in trajectory),
        },
    }
    return rows, echo_group


def build_training_rows(
    examples: list[dict[str, Any]],
    *,
    budget: TokenBudget,
    out_dir: Path,
    recursive_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    rng = random.Random(seed)
    sft_rows: list[dict[str, Any]] = []
    echo_groups: list[dict[str, Any]] = []
    scaffolds = {"direct_inspect": 0, "recursive_spawn": 0}
    for index, ex in enumerate(examples):
        use_spawn = rng.random() < recursive_fraction
        rows, echo_group = build_trace_for_example(
            ex,
            budget=budget,
            state_dir=out_dir / "state-shadow",
            use_spawn=use_spawn,
            source_index=index,
        )
        sft_rows.extend(rows)
        echo_groups.append(echo_group)
        scaffolds["recursive_spawn" if use_spawn else "direct_inspect"] += 1
    return sft_rows, echo_groups, scaffolds


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(pretty_json(value) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json_dumps(row) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=DEFAULT_DB, help="trajectory-trainer sqlite DB")
    parser.add_argument("--materialize", default=DEFAULT_MATERIALIZE, help="materialize_turn.py path")
    parser.add_argument("--out-dir", default=DEFAULT_OUT, help="directory for generated artifacts")
    parser.add_argument("--total", type=int, default=200, help="accepted materialized turns to export")
    parser.add_argument("--train-count", type=int, default=160, help="accepted turns used for SFT/ECHO")
    parser.add_argument("--seed", type=int, default=3141592653, help="sampling seed")
    parser.add_argument("--candidate-limit", type=int, help="limit shuffled candidate rows before filtering")
    parser.add_argument("--min-input-tokens", type=int, default=1)
    parser.add_argument("--max-input-tokens", type=int, default=60000)
    parser.add_argument("--max-action-tokens", type=int, default=OUTPUT_TOKENS)
    parser.add_argument("--allow-multiple-tool-calls", action="store_true")
    parser.add_argument("--recursive-fraction", type=float, default=0.35)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", help="Qwen3.5 tokenizer.json for exact SFT prompt clipping")
    parser.add_argument("--require-tokenizer", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.total <= 0:
        raise SystemExit("--total must be positive")
    if args.train_count <= 0 or args.train_count > args.total:
        raise SystemExit("--train-count must be in 1..--total")
    if not (0.0 <= args.recursive_fraction <= 1.0):
        raise SystemExit("--recursive-fraction must be in [0, 1]")

    db_path = Path(args.db)
    materialize_path = Path(args.materialize)
    out_dir = Path(args.out_dir)
    materializer = load_materializer(materialize_path)
    budget = TokenBudget(Path(args.tokenizer) if args.tokenizer else None, require_tokenizer=args.require_tokenizer)

    with connect(db_path) as conn:
        candidates = candidate_rows(
            conn,
            min_input_tokens=args.min_input_tokens,
            max_input_tokens=args.max_input_tokens,
            seed=args.seed,
            candidate_limit=args.candidate_limit,
        )
        examples, skips = materialize_examples(
            conn,
            materializer,
            candidates,
            budget=budget,
            max_action_tokens=args.max_action_tokens,
            total=args.total,
            require_single_tool_call=not args.allow_multiple_tool_calls,
        )

    if len(examples) < args.total:
        raise SystemExit(
            f"only accepted {len(examples)} examples from {len(candidates)} candidates; skips={skips}"
        )

    train = examples[: args.train_count]
    eval_examples = examples[args.train_count :]
    sft_rows, echo_groups, scaffolds = build_training_rows(
        train,
        budget=budget,
        out_dir=out_dir,
        recursive_fraction=args.recursive_fraction,
        seed=args.seed + 1,
    )

    write_json(out_dir / "all.suite.json", suite_from_examples("pi-rlm-trace-200-all", examples, args.model))
    write_json(out_dir / "train.suite.json", suite_from_examples("pi-rlm-trace-200-train", train, args.model))
    write_json(out_dir / "eval.suite.json", suite_from_examples("pi-rlm-trace-200-eval", eval_examples, args.model))
    write_jsonl(out_dir / "all.prompt_chosen.jsonl", [ex["prompt_chosen"] for ex in examples])
    write_jsonl(out_dir / "train.sft.jsonl", sft_rows)
    write_jsonl(out_dir / "train.echo.jsonl", echo_groups)

    tool_hist: dict[str, int] = {}
    for ex in examples:
        name = str(ex["target"]["name"])
        tool_hist[name] = tool_hist.get(name, 0) + 1
    manifest = {
        "db": str(db_path),
        "materialize": str(materialize_path),
        "out_dir": str(out_dir),
        "seed": args.seed,
        "total_examples": len(examples),
        "train_examples": len(train),
        "eval_examples": len(eval_examples),
        "sft_rows": len(sft_rows),
        "echo_groups": len(echo_groups),
        "tokenizer_exact": budget.exact,
        "tokenizer": str(budget.tokenizer_path) if budget.tokenizer_path else None,
        "filters": {
            "min_input_tokens": args.min_input_tokens,
            "max_input_tokens": args.max_input_tokens,
            "require_single_tool_call": not args.allow_multiple_tool_calls,
            "max_action_tokens": args.max_action_tokens,
        },
        "skips": skips,
        "scaffolds": scaffolds,
        "target_tool_histogram": dict(sorted(tool_hist.items(), key=lambda item: (-item[1], item[0]))),
        "artifacts": {
            "all_suite": str(out_dir / "all.suite.json"),
            "train_suite": str(out_dir / "train.suite.json"),
            "eval_suite": str(out_dir / "eval.suite.json"),
            "prompt_chosen": str(out_dir / "all.prompt_chosen.jsonl"),
            "sft": str(out_dir / "train.sft.jsonl"),
            "echo": str(out_dir / "train.echo.jsonl"),
        },
    }
    write_json(out_dir / "manifest.json", manifest)
    print(
        "built pi RLM trace experiment: "
        f"{len(train)} train examples, {len(eval_examples)} eval examples, "
        f"{len(sft_rows)} SFT rows, {len(echo_groups)} ECHO groups "
        f"at {out_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
