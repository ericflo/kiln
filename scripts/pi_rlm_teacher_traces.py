#!/usr/bin/env python3
"""Generate large-teacher recursive Pi RLM traces.

This is the no-OPD distillation path:

1. Run a same-tool-eventually eval suite through the recursive harness using a
   larger OpenAI-compatible teacher model.
2. Annotate each saved harness state with pass/fail reward.
3. Export only passing non-fallback traces as fixed-window SFT rows and
   agentic GRPO/ECHO rollout groups.

The exported SFT rows teach controller actions. The exported ECHO groups are
for optional observation-modeling updates, usually with `loss.no_policy_loss`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pi_rlm_harness import (  # noqa: E402
    DEFAULT_UPSTREAM,
    INPUT_TOKENS,
    OUTPUT_TOKENS,
    RLM_SYSTEM_PROMPT,
    TURN_BREAK,
    RlmEnvironment,
    RlmController,
    RlmResult,
    TokenBudget,
    UpstreamClient,
    action_text,
    discover_tokenizer_path,
    eval_messages,
    export_echo_jsonl,
    export_sft_jsonl,
    first_json_object,
    json_dumps,
    pretty_json,
    run_eval_suite,
    sanitize_id,
    target_tool_call_from_text,
)


DEFAULT_OUT = ".kiln/pi-rlm-teacher-traces"


def api_key_default() -> str:
    return (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("DEEPINFRA_API_KEY")
        or "kiln"
    )


def clear_state_dir(state_dir: Path) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    for path in state_dir.glob("*.json"):
        path.unlink()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(pretty_json(value) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json_dumps(row) + "\n")


def completed_state_file(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    result = payload.get("result")
    return isinstance(result, dict) and bool(result.get("trace"))


def annotate_oracle_state(path: Path, *, teacher_model: str) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    result = payload.get("result")
    if not isinstance(result, dict) or not result.get("trace"):
        return False
    result["reward"] = 1.0
    result["eval"] = {
        "pass": True,
        "score": 1.0,
        "reason": "oracle target trace synthesis",
        "reward": 1.0,
        "teacher_model": teacher_model,
        "oracle_target_visible_to_teacher": True,
    }
    path.write_text(pretty_json(payload) + "\n", encoding="utf-8")
    return True


def first_json_array_or_object(text: str) -> Any:
    obj = first_json_object(text)
    if obj is not None:
        return obj
    stripped = text.strip()
    if stripped.startswith("```"):
        import re

        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(stripped):
        if ch != "[":
            continue
        try:
            value, _ = decoder.raw_decode(stripped[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, list):
            return value
    return None


def oracle_teacher_plan(
    upstream: UpstreamClient,
    *,
    model: str,
    env: RlmEnvironment,
    target: dict[str, Any],
    max_actions: int,
) -> tuple[list[dict[str, Any]], str]:
    body = {
        "task": (
            "Generate a training trace for a bounded Recursive Language Model controller. "
            "The target Pi-visible tool call is provided only to you as a teacher. "
            "Return the internal controller actions that would efficiently uncover enough "
            "state, then end with exactly that target finish.tool_call."
        ),
        "valid_internal_actions": [
            {"action": "inspect", "target": "summary"},
            {"action": "inspect", "target": "tools"},
            {"action": "inspect", "target": "message", "index": 0},
            {"action": "search", "query": "needle", "regex": False, "max_matches": 8},
            {"action": "slice", "index": 0, "start": 0, "length": 4096},
        ],
        "rules": [
            "Return JSON only: either an array of actions or {\"actions\": [...]}",
            f"Use at most {max_actions} internal inspect/search/slice actions before finish.",
            "Do not use subcall or spawn_agent in this oracle synthesis mode.",
            "The last action must be finish.tool_call and must equal target_tool_call exactly.",
            "No prose, comments, markdown, or explanations.",
        ],
        "target_tool_call": target,
        "environment_summary": env.summary(),
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You emit compact JSON controller-action traces for distillation."},
            {"role": "user", "content": env.budget.clip(pretty_json(body), INPUT_TOKENS)},
        ],
        "temperature": 0.0,
        "max_tokens": OUTPUT_TOKENS,
    }
    raw = upstream.first_text(payload)
    parsed = first_json_array_or_object(raw)
    actions_any = parsed.get("actions") if isinstance(parsed, dict) else parsed
    actions: list[dict[str, Any]] = []
    if isinstance(actions_any, list):
        for item in actions_any:
            if isinstance(item, dict):
                actions.append(item)
    return actions, raw


def finish_action_for(target: dict[str, Any]) -> dict[str, Any]:
    return {
        "action": "finish",
        "tool_call": {
            "name": target["name"],
            "arguments": target.get("arguments", {}),
        },
    }


def normalize_oracle_actions(
    actions: list[dict[str, Any]],
    *,
    target: dict[str, Any],
    max_actions: int,
) -> list[dict[str, Any]]:
    allowed_internal = {"inspect", "search", "slice"}
    normalized: list[dict[str, Any]] = []
    for action in actions:
        kind = str(action.get("action", "")).lower()
        if kind == "finish":
            break
        if kind in allowed_internal:
            normalized.append(action)
        if len(normalized) >= max_actions:
            break
    normalized.append(finish_action_for(target))
    return normalized


def oracle_prompt_messages(controller: RlmController, env: RlmEnvironment, depth: int, step: int) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": RLM_SYSTEM_PROMPT},
        {"role": "user", "content": controller.root_user_prompt(env, depth, step)},
    ]


def synthesize_oracle_trace(
    controller: RlmController,
    upstream: UpstreamClient,
    *,
    suite_name: str,
    example_id: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    target: dict[str, Any],
    max_actions: int,
) -> dict[str, Any]:
    request_id = sanitize_id(f"{suite_name}.{example_id}.oracle")
    env = RlmEnvironment(
        request_id=request_id,
        parent_id=None,
        depth=0,
        messages=messages,
        tools=tools,
        upstream_request={
            "model": upstream.model,
            "messages": messages,
            "tools": tools,
            "metadata": {"request_id": request_id, "suite": suite_name, "example_id": example_id},
        },
        state_dir=controller.state_dir,
        budget=controller.budget,
    )
    env.persist()
    planned, raw_plan = oracle_teacher_plan(
        upstream,
        model=upstream.model,
        env=env,
        target=target,
        max_actions=max_actions,
    )
    actions = normalize_oracle_actions(planned, target=target, max_actions=max_actions)
    trace: list[dict[str, Any]] = []
    sft_rows: list[dict[str, Any]] = []
    trajectory: list[dict[str, Any]] = []
    action_segments: list[str] = []
    final_result = RlmResult(request_id=request_id, depth=0)

    for step, action in enumerate(actions):
        prompt_messages = oracle_prompt_messages(controller, env, 0, step)
        action_json = action_text(action)
        row_metadata = {
            "source": "pi_rlm_large_teacher_oracle",
            "request_id": request_id,
            "suite": suite_name,
            "example_id": example_id,
            "depth": 0,
            "step": step,
            "teacher_model": upstream.model,
            "oracle_target_visible_to_teacher": True,
            "target_tool": target["name"],
        }
        sft_rows.append(
            {
                "messages": prompt_messages + [{"role": "assistant", "content": action_json}],
                "metadata": row_metadata,
            }
        )
        entry: dict[str, Any] = {
            "step": step,
            "prompt_messages": prompt_messages,
            "prompt_tokens": sum(env.budget.count(m["content"]) for m in prompt_messages),
            "raw": action_json,
            "action": action,
            "action_tokens": env.budget.count(action_json),
        }
        trace.append(entry)
        action_segments.append(action_json)
        trajectory.append({"role": "assistant", "content": action_json, "kind": "action"})
        validation_error = controller.validate_action(env, action)
        kind = str(action.get("action", "")).lower()
        if validation_error is not None:
            obs = env.observe(action, validation_error)
            entry["observation"] = obs
            trajectory.append({"role": "tool", "content": json_dumps(obs), "kind": "observation"})
            env.persist()
            continue
        if kind == "finish":
            final_result = controller.finish(action, trace)
            final_result.request_id = request_id
            final_result.depth = 0
            break
        output = controller.execute(env, action, 0)
        obs = env.observe(action, output)
        entry["observation"] = obs
        trajectory.append({"role": "tool", "content": json_dumps(obs), "kind": "observation"})
        env.persist()

    final_result.trace = trace
    final_result.fallback_used = False
    env.persist(final_result)
    echo_group = {
        "messages": trace[0]["prompt_messages"] if trace else oracle_prompt_messages(controller, env, 0, 0),
        "rollouts": [
            {
                "text": TURN_BREAK.join(action_segments),
                "reward": 1.0,
                "trajectory": trajectory,
            }
        ],
        "metadata": {
            "source": "pi_rlm_large_teacher_oracle",
            "request_id": request_id,
            "suite": suite_name,
            "example_id": example_id,
            "teacher_model": upstream.model,
            "trace_steps": len(trace),
            "final_tool": target["name"],
            "oracle_target_visible_to_teacher": True,
            "echo_ready": any(seg.get("kind") == "observation" for seg in trajectory),
        },
    }
    return {
        "request_id": request_id,
        "example_id": example_id,
        "target": target,
        "raw_plan": raw_plan,
        "planned_actions": planned,
        "actions": actions,
        "sft_rows": sft_rows,
        "echo_group": echo_group,
        "trace_steps": len(trace),
    }


def run_oracle_synthesis(
    controller: RlmController,
    upstream: UpstreamClient,
    suite_path: Path,
    *,
    output_report: Path,
    output_sft: Path,
    output_echo: Path,
    limit: int | None,
    max_actions: int,
    resume_existing: bool,
) -> dict[str, Any]:
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    suite_name = str(suite.get("name") or suite_path.stem)
    examples = suite.get("examples") or []
    if limit is not None:
        examples = examples[:limit]
    sft_rows: list[dict[str, Any]] = []
    echo_groups: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    existing = 0
    skipped = 0
    for index, example in enumerate(examples):
        if not isinstance(example, dict):
            skipped += 1
            continue
        target = target_tool_call_from_text(example.get("target"))
        if target is None:
            skipped += 1
            continue
        example_id = str(example.get("id") or f"example-{index}")
        request_id = sanitize_id(f"{suite_name}.{example_id}.oracle")
        state_path = controller.state_dir / f"{request_id}.json"
        if resume_existing and completed_state_file(state_path):
            existing += 1
            continue
        record = synthesize_oracle_trace(
            controller,
            upstream,
            suite_name=suite_name,
            example_id=example_id,
            messages=eval_messages(suite, example),
            tools=example.get("tools") or suite.get("tools") or [],
            target=target,
            max_actions=max_actions,
        )
        sft_rows.extend(record.pop("sft_rows"))
        echo_groups.append(record.pop("echo_group"))
        records.append(record)
    if existing:
        # Resumed runs may have most traces only on disk. Export from state so
        # prior successful examples are included without re-billing the teacher.
        for path in controller.state_dir.glob("*.json"):
            annotate_oracle_state(path, teacher_model=upstream.model)
        sft_count = export_sft_jsonl(controller.state_dir, output_sft)
        echo_count = export_echo_jsonl(controller.state_dir, output_echo)
    else:
        write_jsonl(output_sft, sft_rows)
        write_jsonl(output_echo, echo_groups)
        sft_count = len(sft_rows)
        echo_count = len(echo_groups)
    report = {
        "suite": suite_name,
        "suite_path": str(suite_path),
        "teacher_model": upstream.model,
        "mode": "oracle_target_trace_synthesis",
        "stats": {
            "examples": len(examples),
            "generated": len(records) + existing,
            "generated_new": len(records),
            "existing": existing,
            "skipped": skipped,
            "sft_rows": sft_count,
            "echo_groups": echo_count,
        },
        "examples": records,
    }
    write_json(output_report, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, help="EvalSuite JSON generated from production turns")
    parser.add_argument("--out-dir", default=DEFAULT_OUT, help="directory for report, state, and exported JSONL")
    parser.add_argument("--upstream", default=DEFAULT_UPSTREAM, help="large teacher OpenAI-compatible /v1 base URL")
    parser.add_argument("--model", required=True, help="large teacher model id")
    parser.add_argument("--api-key", default=api_key_default(), help="teacher API key")
    parser.add_argument("--limit", type=int, help="number of suite examples to run")
    parser.add_argument("--max-iters", type=int, default=8, help="max root harness actions per example")
    parser.add_argument("--max-depth", type=int, default=2, help="max recursive child-agent depth")
    parser.add_argument("--upstream-timeout", type=float, default=180.0, help="seconds per teacher request")
    parser.add_argument("--upstream-retries", type=int, default=5, help="retries for transient teacher API failures")
    parser.add_argument("--tokenizer", help="Qwen3.5 tokenizer.json for fixed-window accounting")
    parser.add_argument("--require-tokenizer", action="store_true")
    parser.add_argument("--adapter", help="optional adapter name when the teacher upstream is Kiln")
    parser.add_argument(
        "--allow-direct-fallback",
        action="store_true",
        help="allow full-context fallback; default is pure bounded RLM traces only",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="keep existing state JSON files instead of clearing the output state dir first",
    )
    parser.add_argument("--pass-reward", type=float, default=1.0)
    parser.add_argument("--fail-reward", type=float, default=0.0)
    parser.add_argument(
        "--oracle-target",
        action="store_true",
        help="show the recorded target to the teacher while synthesizing internal traces; exported prompts do not include it",
    )
    parser.add_argument("--oracle-max-actions", type=int, default=4, help="max internal inspect/search/slice actions before forced finish")
    parser.add_argument(
        "--export-all",
        action="store_true",
        help="export failed/fallback traces too; default exports passing non-fallback traces only",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    state_dir = out_dir / "state"
    report_path = out_dir / "teacher.report.json"
    sft_path = out_dir / "teacher.passed.sft.jsonl"
    echo_path = out_dir / "teacher.passed.echo.jsonl"
    manifest_path = out_dir / "manifest.json"

    if not args.resume:
        clear_state_dir(state_dir)

    tokenizer_path = discover_tokenizer_path(args.tokenizer)
    budget = TokenBudget(tokenizer_path, require_tokenizer=args.require_tokenizer)
    upstream = UpstreamClient(
        args.upstream,
        args.api_key,
        args.model,
        timeout=args.upstream_timeout,
        max_retries=args.upstream_retries,
    )
    controller = RlmController(
        upstream,
        state_dir,
        args.max_iters,
        args.max_depth,
        budget,
        args.adapter,
        allow_direct_fallback=args.allow_direct_fallback,
    )

    if args.oracle_target:
        report = run_oracle_synthesis(
            controller,
            upstream,
            Path(args.suite),
            output_report=report_path,
            output_sft=sft_path,
            output_echo=echo_path,
            limit=args.limit,
            max_actions=args.oracle_max_actions,
            resume_existing=args.resume,
        )
        sft_count = report["stats"]["sft_rows"]
        echo_count = report["stats"]["echo_groups"]
        manifest = {
            "suite": str(Path(args.suite)),
            "out_dir": str(out_dir),
            "state_dir": str(state_dir),
            "teacher": {
                "upstream": args.upstream,
                "model": args.model,
                "direct_fallback": args.allow_direct_fallback,
                "oracle_target": True,
            },
            "tokenizer": {
                "path": str(tokenizer_path) if tokenizer_path else None,
                "exact": budget.exact,
            },
            "eval": {
                "report": str(report_path),
                "limit": args.limit,
                "accuracy": None,
                "stats": report["stats"],
            },
            "exports": {
                "passed_only": True,
                "nonfallback_only": True,
                "sft": str(sft_path),
                "sft_rows": sft_count,
                "echo": str(echo_path),
                "echo_groups": echo_count,
            },
        }
        write_json(manifest_path, manifest)
        print(
            "teacher oracle traces: "
            f"generated {report['stats']['generated']} examples, "
            f"exported {sft_count} SFT rows and {echo_count} ECHO groups at {out_dir}",
            flush=True,
        )
        return 0

    report = run_eval_suite(
        controller,
        Path(args.suite),
        report_path,
        args.limit,
        model_override=args.model,
        pass_reward=args.pass_reward,
        fail_reward=args.fail_reward,
    )
    passed_only = not args.export_all
    nonfallback_only = not args.export_all
    sft_count = export_sft_jsonl(
        state_dir,
        sft_path,
        passed_only=passed_only,
        nonfallback_only=nonfallback_only,
    )
    echo_count = export_echo_jsonl(
        state_dir,
        echo_path,
        default_reward=args.pass_reward,
        passed_only=passed_only,
        nonfallback_only=nonfallback_only,
    )
    manifest = {
        "suite": str(Path(args.suite)),
        "out_dir": str(out_dir),
        "state_dir": str(state_dir),
        "teacher": {
            "upstream": args.upstream,
            "model": args.model,
            "direct_fallback": args.allow_direct_fallback,
        },
        "tokenizer": {
            "path": str(tokenizer_path) if tokenizer_path else None,
            "exact": budget.exact,
        },
        "eval": {
            "report": str(report_path),
            "limit": args.limit,
            "accuracy": report["accuracy"],
            "stats": report["stats"],
        },
        "exports": {
            "passed_only": passed_only,
            "nonfallback_only": nonfallback_only,
            "sft": str(sft_path),
            "sft_rows": sft_count,
            "echo": str(echo_path),
            "echo_groups": echo_count,
        },
    }
    write_json(manifest_path, manifest)
    print(
        "teacher traces: "
        f"{report['stats']['passed']}/{report['stats']['scored']} passed, "
        f"exported {sft_count} SFT rows and {echo_count} ECHO groups at {out_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
