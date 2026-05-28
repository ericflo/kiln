#!/usr/bin/env python3
"""Build a small constrained-RLM experiment from trajectory-trainer turns.

The builder samples real tool-call turns from trajectory-trainer, materializes
them with the production materializer, and emits:

- same-tool-eventually EvalSuite JSON;
- root-action SFT rows for the fixed-window RLM controller;
- ECHO-ready rollout groups that preserve action/observation alternation.

OPD CE JSONL can still be emitted for archaeology with --emit-opd-ce, but the
default Pi RLM path is large-teacher SFT plus optional ECHO, not OPD.
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


def row_action(row: dict[str, Any]) -> dict[str, Any]:
    messages = row.get("messages") or []
    if not messages:
        return {}
    last = messages[-1]
    if not isinstance(last, dict):
        return {}
    try:
        value = json.loads(str(last.get("content") or "{}"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def row_token_count(row: dict[str, Any], budget: TokenBudget) -> int:
    total = 0
    for msg in row.get("messages") or []:
        if isinstance(msg, dict):
            total += budget.count(str(msg.get("content") or ""))
    return total


def copy_row_with_meta(row: dict[str, Any], **metadata: Any) -> dict[str, Any]:
    copied = dict(row)
    copied["metadata"] = dict(row.get("metadata") or {}) | metadata
    return copied


def build_balanced_sft_curriculum(
    rows: list[dict[str, Any]],
    *,
    budget: TokenBudget,
    seed: int,
    inspect_ratio_to_finish: float,
    finish_repeat: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Build SFT curricula that do not let inspect rows swamp final actions."""

    rng = random.Random(seed)
    by_kind: dict[str, list[dict[str, Any]]] = {"finish": [], "spawn_agent": [], "inspect": [], "other": []}
    for row in rows:
        kind = str((row.get("metadata") or {}).get("action") or row_action(row).get("action") or "other")
        by_kind.setdefault(kind, by_kind["other"]).append(row)

    finish = list(by_kind.get("finish", []))
    spawn = list(by_kind.get("spawn_agent", []))
    inspect = list(by_kind.get("inspect", []))
    other = list(by_kind.get("other", []))

    inspect_cap = min(len(inspect), int(max(0, round(len(finish) * inspect_ratio_to_finish))))
    rng.shuffle(inspect)
    rng.shuffle(spawn)
    rng.shuffle(other)

    curriculum: list[dict[str, Any]] = []
    for idx, row in enumerate(finish):
        curriculum.append(copy_row_with_meta(row, curriculum="balanced", curriculum_role="finish", repeat_index=0))
        for repeat_idx in range(1, max(1, finish_repeat)):
            curriculum.append(
                copy_row_with_meta(
                    row,
                    curriculum="balanced",
                    curriculum_role="finish_repeat",
                    repeat_index=repeat_idx,
                )
            )
    for row in spawn:
        curriculum.append(copy_row_with_meta(row, curriculum="balanced", curriculum_role="spawn"))
    for row in inspect[:inspect_cap]:
        curriculum.append(copy_row_with_meta(row, curriculum="balanced", curriculum_role="inspect"))
    for row in other:
        curriculum.append(copy_row_with_meta(row, curriculum="balanced", curriculum_role="other"))
    rng.shuffle(curriculum)

    final_weighted: list[dict[str, Any]] = []
    for row in finish:
        for repeat_idx in range(max(1, finish_repeat + 1)):
            final_weighted.append(
                copy_row_with_meta(
                    row,
                    curriculum="final_weighted",
                    curriculum_role="finish",
                    repeat_index=repeat_idx,
                )
            )
    rng.shuffle(final_weighted)

    stats = {
        "raw": {kind: len(values) for kind, values in sorted(by_kind.items())},
        "balanced_rows": len(curriculum),
        "final_weighted_rows": len(final_weighted),
        "inspect_cap": inspect_cap,
        "finish_repeat": finish_repeat,
        "token_rows": {
            "balanced_max": max((row_token_count(row, budget) for row in curriculum), default=0),
            "final_weighted_max": max((row_token_count(row, budget) for row in final_weighted), default=0),
        },
    }
    return curriculum, final_weighted, stats


def write_short_sft_variants(
    out_dir: Path,
    rows: list[dict[str, Any]],
    *,
    budget: TokenBudget,
    counts: list[int],
    prefix: str,
) -> dict[str, str]:
    paths: dict[str, str] = {}
    ranked = sorted(rows, key=lambda row: (row_token_count(row, budget), str((row.get("metadata") or {}).get("turn_id"))))
    for count in counts:
        if count <= 0:
            continue
        subset = ranked[: min(count, len(ranked))]
        path = out_dir / f"{prefix}.short{len(subset)}.jsonl"
        write_jsonl(path, subset)
        paths[f"{prefix}_short{len(subset)}"] = str(path)
    return paths


def write_stratified_sft_variants(
    out_dir: Path,
    rows: list[dict[str, Any]],
    *,
    budget: TokenBudget,
    counts: list[int],
    prefix: str,
) -> dict[str, str]:
    paths: dict[str, str] = {}
    indexed: list[tuple[int, dict[str, Any]]] = list(enumerate(rows))
    buckets: dict[str, list[tuple[int, dict[str, Any]]]] = {
        "finish": [],
        "spawn_agent": [],
        "inspect": [],
        "other": [],
    }
    for item in indexed:
        _, row = item
        kind = str((row.get("metadata") or {}).get("action") or row_action(row).get("action") or "other")
        if kind not in buckets:
            kind = "other"
        buckets[kind].append(item)
    for values in buckets.values():
        values.sort(key=lambda item: (row_token_count(item[1], budget), item[0]))

    ratios = {"finish": 0.55, "spawn_agent": 0.20, "inspect": 0.25}
    for count in counts:
        if count <= 0:
            continue
        target = min(count, len(rows))
        selected_ids: set[int] = set()
        selected: list[dict[str, Any]] = []
        for kind, ratio in ratios.items():
            want = min(len(buckets[kind]), int(round(target * ratio)))
            for idx, row in buckets[kind][:want]:
                if idx in selected_ids:
                    continue
                selected_ids.add(idx)
                selected.append(row)
        if len(selected) < target:
            remainder = sorted(
                (item for item in indexed if item[0] not in selected_ids),
                key=lambda item: (row_token_count(item[1], budget), item[0]),
            )
            for idx, row in remainder[: target - len(selected)]:
                selected_ids.add(idx)
                selected.append(row)
        selected.sort(key=lambda row: (str((row.get("metadata") or {}).get("turn_id")), str((row.get("metadata") or {}).get("action"))))
        path = out_dir / f"{prefix}.stratified{len(selected)}.jsonl"
        write_jsonl(path, selected)
        paths[f"{prefix}_stratified{len(selected)}"] = str(path)
    return paths


def echo_group_to_opd_example(group: dict[str, Any]) -> dict[str, Any]:
    rollouts = group.get("rollouts") or []
    rollout = rollouts[0] if rollouts and isinstance(rollouts[0], dict) else {}
    trajectory = rollout.get("trajectory") if isinstance(rollout, dict) else []
    teacher_response = str(rollout.get("text") or "")
    if not teacher_response and isinstance(trajectory, list):
        teacher_response = TURN_BREAK.join(
            str(seg.get("content") or "")
            for seg in trajectory
            if isinstance(seg, dict) and seg.get("kind") == "action"
        )
    return {
        "id": str((group.get("metadata") or {}).get("turn_id") or (group.get("metadata") or {}).get("source_index") or ""),
        "messages": group.get("messages") or [],
        "teacher_response": teacher_response,
        "trajectory": trajectory if isinstance(trajectory, list) else [],
        "metadata": dict(group.get("metadata") or {})
        | {
            "source": "pi_rlm_trace_experiment",
            "opd_objective": "cross_entropy",
            "teacher": "production_trace_chosen_action",
        },
    }


def build_opd_examples(echo_groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples = [echo_group_to_opd_example(group) for group in echo_groups]
    return [ex for ex in examples if ex.get("messages") and ex.get("teacher_response")]


def opd_example_token_count(example: dict[str, Any], budget: TokenBudget) -> int:
    total = sum(budget.count(str(msg.get("content") or "")) for msg in example.get("messages") or [] if isinstance(msg, dict))
    for seg in example.get("trajectory") or []:
        if isinstance(seg, dict):
            total += budget.count(str(seg.get("content") or ""))
    return total


def write_short_opd_variants(
    out_dir: Path,
    examples: list[dict[str, Any]],
    *,
    budget: TokenBudget,
    counts: list[int],
) -> dict[str, str]:
    paths: dict[str, str] = {}
    ranked = sorted(examples, key=lambda ex: (opd_example_token_count(ex, budget), str(ex.get("id") or "")))
    for count in counts:
        if count <= 0:
            continue
        subset = ranked[: min(count, len(ranked))]
        path = out_dir / f"train.opd.ce.short{len(subset)}.jsonl"
        write_jsonl(path, subset)
        paths[f"opd_ce_short{len(subset)}"] = str(path)
    return paths


def parse_short_counts(value: str) -> list[int]:
    counts = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        counts.append(int(part))
    return sorted(set(counts))


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
    parser.add_argument("--inspect-ratio-to-finish", type=float, default=1.0)
    parser.add_argument("--finish-repeat", type=int, default=1)
    parser.add_argument(
        "--short-counts",
        default="4,8,16,32,64,128,256,512,1024",
        help="comma-separated short curriculum sizes to emit",
    )
    parser.add_argument(
        "--emit-opd-ce",
        action="store_true",
        help="also emit legacy off-policy OPD cross-entropy JSONL; disabled by default",
    )
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
    if args.inspect_ratio_to_finish < 0.0:
        raise SystemExit("--inspect-ratio-to-finish must be >= 0")
    if args.finish_repeat < 1:
        raise SystemExit("--finish-repeat must be >= 1")

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
    curriculum_rows, final_weighted_rows, curriculum_stats = build_balanced_sft_curriculum(
        sft_rows,
        budget=budget,
        seed=args.seed + 2,
        inspect_ratio_to_finish=args.inspect_ratio_to_finish,
        finish_repeat=args.finish_repeat,
    )
    opd_examples = build_opd_examples(echo_groups) if args.emit_opd_ce else []
    short_counts = parse_short_counts(args.short_counts)

    suite_prefix = f"pi-rlm-trace-{len(examples)}"
    write_json(out_dir / "all.suite.json", suite_from_examples(f"{suite_prefix}-all", examples, args.model))
    write_json(out_dir / "train.suite.json", suite_from_examples(f"{suite_prefix}-train", train, args.model))
    write_json(out_dir / "eval.suite.json", suite_from_examples(f"{suite_prefix}-eval", eval_examples, args.model))
    write_jsonl(out_dir / "all.prompt_chosen.jsonl", [ex["prompt_chosen"] for ex in examples])
    write_jsonl(out_dir / "train.sft.jsonl", sft_rows)
    write_jsonl(out_dir / "train.sft.curriculum.jsonl", curriculum_rows)
    write_jsonl(out_dir / "train.sft.final_weighted.jsonl", final_weighted_rows)
    write_jsonl(out_dir / "train.echo.jsonl", echo_groups)
    if args.emit_opd_ce:
        write_jsonl(out_dir / "train.opd.ce.jsonl", opd_examples)
    short_artifacts: dict[str, str] = {}
    short_artifacts.update(
        write_short_sft_variants(
            out_dir,
            curriculum_rows,
            budget=budget,
            counts=short_counts,
            prefix="train.sft.curriculum",
        )
    )
    short_artifacts.update(
        write_stratified_sft_variants(
            out_dir,
            curriculum_rows,
            budget=budget,
            counts=short_counts,
            prefix="train.sft.curriculum",
        )
    )
    short_artifacts.update(
        write_short_sft_variants(
            out_dir,
            final_weighted_rows,
            budget=budget,
            counts=short_counts,
            prefix="train.sft.final_weighted",
        )
    )
    if args.emit_opd_ce:
        short_artifacts.update(write_short_opd_variants(out_dir, opd_examples, budget=budget, counts=short_counts))

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
        "sft_curriculum_rows": len(curriculum_rows),
        "sft_final_weighted_rows": len(final_weighted_rows),
        "echo_groups": len(echo_groups),
        "opd_ce_examples": len(opd_examples),
        "opd_ce_enabled": args.emit_opd_ce,
        "tokenizer_exact": budget.exact,
        "tokenizer": str(budget.tokenizer_path) if budget.tokenizer_path else None,
        "filters": {
            "min_input_tokens": args.min_input_tokens,
            "max_input_tokens": args.max_input_tokens,
            "require_single_tool_call": not args.allow_multiple_tool_calls,
            "max_action_tokens": args.max_action_tokens,
            "inspect_ratio_to_finish": args.inspect_ratio_to_finish,
            "finish_repeat": args.finish_repeat,
        },
        "skips": skips,
        "scaffolds": scaffolds,
        "curriculum": curriculum_stats,
        "target_tool_histogram": dict(sorted(tool_hist.items(), key=lambda item: (-item[1], item[0]))),
        "artifacts": ({
            "all_suite": str(out_dir / "all.suite.json"),
            "train_suite": str(out_dir / "train.suite.json"),
            "eval_suite": str(out_dir / "eval.suite.json"),
            "prompt_chosen": str(out_dir / "all.prompt_chosen.jsonl"),
            "sft": str(out_dir / "train.sft.jsonl"),
            "sft_curriculum": str(out_dir / "train.sft.curriculum.jsonl"),
            "sft_final_weighted": str(out_dir / "train.sft.final_weighted.jsonl"),
            "echo": str(out_dir / "train.echo.jsonl"),
        }
        | ({"opd_ce": str(out_dir / "train.opd.ce.jsonl")} if args.emit_opd_ce else {}))
        | short_artifacts,
    }
    write_json(out_dir / "manifest.json", manifest)
    print(
        "built pi RLM trace experiment: "
        f"{len(train)} train examples, {len(eval_examples)} eval examples, "
        f"{len(sft_rows)} raw SFT rows, {len(curriculum_rows)} curriculum SFT rows, "
        f"{len(echo_groups)} ECHO groups"
        + (f", {len(opd_examples)} OPD CE examples" if args.emit_opd_ce else "")
        + " "
        f"at {out_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
