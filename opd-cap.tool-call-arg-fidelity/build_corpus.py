"""Synthesize tool-call-arg-fidelity prompts.

Each prompt declares a tool schema in the system message, gives a user
request, and expects the assistant to emit a JSON tool call.

Output two JSONL files:
  prompts/h1-r16-6ep.jsonl  — training prompts (no eval examples)
  datasets/eval.jsonl       — held-out eval set with ground-truth schema

We do NOT include expected responses in the eval file because the rubric
scores purely on the JSON the model emits + the schema. Same prompt
shape; the schema is what matters.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

random.seed(20260517)

# 12 distinct tool shapes spanning the typical coding-agent surface
TOOLS = [
    {
        "name": "file_read",
        "schema": {
            "required": ["path"],
            "types": {"path": "string", "offset": "integer", "limit": "integer"},
            "allowed": ["path", "offset", "limit"],
        },
        "description": "Read a file from disk. `path` (str, required), `offset` (int, optional), `limit` (int, optional).",
        "asks": [
            "Read the first 50 lines of crates/kiln-train/src/opd.rs",
            "Show me the entirety of /workspace/config.toml",
            "Read lines 200-250 of src/main.py",
            "Open the README starting at line 30",
            "Read src/lib.rs",
        ],
    },
    {
        "name": "file_write",
        "schema": {
            "required": ["path", "content"],
            "types": {"path": "string", "content": "string", "mode": "string"},
            "allowed": ["path", "content", "mode"],
        },
        "description": "Write `content` (str) to `path` (str). `mode` (str, optional) = 'append' or 'overwrite'.",
        "asks": [
            "Save 'hello world' to /tmp/greet.txt",
            "Append the line 'export FOO=1' to ~/.bashrc",
            "Overwrite config.yaml with the YAML literal `debug: true`",
            "Create README.md containing '# My Project'",
        ],
    },
    {
        "name": "run_bash",
        "schema": {
            "required": ["command"],
            "types": {"command": "string", "timeout": "integer", "background": "boolean"},
            "allowed": ["command", "timeout", "background"],
        },
        "description": "Run a shell command. `command` (str, required), `timeout` (int seconds, optional), `background` (bool, optional).",
        "asks": [
            "Show git status",
            "Run the test suite with a 5 minute timeout",
            "Start the dev server in the background",
            "List files in current directory with sizes",
            "Run npm install",
        ],
    },
    {
        "name": "web_search",
        "schema": {
            "required": ["query"],
            "types": {"query": "string", "max_results": "integer"},
            "allowed": ["query", "max_results"],
        },
        "description": "Web search. `query` (str, required), `max_results` (int, optional, default 5).",
        "asks": [
            "Search for the latest Rust async runtime benchmark",
            "Find 3 results about React 19 release notes",
            "Look up how to install postgres on Ubuntu 22",
        ],
    },
    {
        "name": "list_files",
        "schema": {
            "required": ["directory"],
            "types": {"directory": "string", "recursive": "boolean", "pattern": "string"},
            "allowed": ["directory", "recursive", "pattern"],
        },
        "description": "List files. `directory` (str, required), `recursive` (bool, optional), `pattern` (str glob, optional).",
        "asks": [
            "List Python files under src/ recursively",
            "Show me what's in /workspace",
            "List all .toml files recursively under crates/",
        ],
    },
    {
        "name": "git_diff",
        "schema": {
            "required": [],
            "types": {"paths": "array", "staged": "boolean", "base": "string"},
            "allowed": ["paths", "staged", "base"],
        },
        "description": "Get git diff. `paths` (array of str, optional), `staged` (bool, optional), `base` (str ref, optional).",
        "asks": [
            "Show the staged diff",
            "Diff between main and HEAD for crates/kiln-train",
            "Show what's changed in src/main.rs",
            "Get the diff",
        ],
    },
    {
        "name": "http_request",
        "schema": {
            "required": ["url", "method"],
            "types": {"url": "string", "method": "string", "headers": "object",
                      "body": "string", "timeout": "integer"},
            "allowed": ["url", "method", "headers", "body", "timeout"],
        },
        "description": "Make an HTTP request. `url` (str, required), `method` (str, required), `headers` (obj, optional), `body` (str, optional), `timeout` (int s, optional).",
        "asks": [
            "POST {'foo': 1} to https://api.example.com/items with Content-Type application/json",
            "GET https://api.github.com/repos/foo/bar",
            "DELETE the resource at https://api.example.com/users/42",
        ],
    },
    {
        "name": "schedule_task",
        "schema": {
            "required": ["task", "when"],
            "types": {"task": "string", "when": "string", "retry_count": "integer", "priority": "string"},
            "allowed": ["task", "when", "retry_count", "priority"],
        },
        "description": "Schedule a task. `task` (str, required), `when` (str ISO8601 or cron, required), `retry_count` (int, optional), `priority` (str, optional).",
        "asks": [
            "Run the nightly backup at 2am every day with high priority",
            "Schedule a retry-3-times task to ping the health endpoint at 2026-06-01T14:00:00Z",
        ],
    },
    {
        "name": "code_search",
        "schema": {
            "required": ["query"],
            "types": {"query": "string", "case_sensitive": "boolean", "file_pattern": "string", "max_matches": "integer"},
            "allowed": ["query", "case_sensitive", "file_pattern", "max_matches"],
        },
        "description": "Grep the codebase. `query` (str, required), `case_sensitive` (bool, optional), `file_pattern` (str glob, optional), `max_matches` (int, optional).",
        "asks": [
            "Find case-insensitive uses of `apply_chat_template` in *.rs files, max 20 matches",
            "Search for TODO comments",
            "Grep for the string ERROR in log files",
        ],
    },
    {
        "name": "delete_file",
        "schema": {
            "required": ["path"],
            "types": {"path": "string", "force": "boolean", "recursive": "boolean"},
            "allowed": ["path", "force", "recursive"],
        },
        "description": "Delete file or directory. `path` (str, required), `force` (bool, optional), `recursive` (bool, optional).",
        "asks": [
            "Delete /tmp/old.log",
            "Force-delete the build/ directory recursively",
        ],
    },
    {
        "name": "set_env",
        "schema": {
            "required": ["name", "value"],
            "types": {"name": "string", "value": "string", "scope": "string"},
            "allowed": ["name", "value", "scope"],
        },
        "description": "Set environment variable. `name` (str, required), `value` (str, required), `scope` (str, optional) = 'session' or 'persistent'.",
        "asks": [
            "Set DATABASE_URL to postgres://localhost/dev for the session",
            "Persistently set RUST_LOG to debug",
        ],
    },
    {
        "name": "create_branch",
        "schema": {
            "required": ["name"],
            "types": {"name": "string", "from_ref": "string", "checkout": "boolean"},
            "allowed": ["name", "from_ref", "checkout"],
        },
        "description": "Create a git branch. `name` (str, required), `from_ref` (str, optional), `checkout` (bool, optional).",
        "asks": [
            "Create branch feature/opd-rollout-fix from main and check it out",
            "Make a branch called hotfix/cache-leak",
        ],
    },
]


SYSTEM_PROMPT_TMPL = """You are a coding agent. Tools are invoked by emitting a single JSON object with the tool's arguments, NOTHING ELSE (no prose, no code fences, no commentary).

Tool: {name}
{description}

Emit ONLY the JSON object. Do not wrap it in markdown. Do not explain."""


def make_prompt(tool: dict, ask: str) -> dict:
    sys_p = SYSTEM_PROMPT_TMPL.format(name=tool["name"], description=tool["description"])
    return {
        "id": f"{tool['name']}-{hash(ask) & 0xffff:04x}",
        "schema": tool["schema"],
        "messages": [
            {"role": "system", "content": sys_p},
            {"role": "user", "content": ask},
            {"role": "assistant", "content": "{}"},  # dummy; student samples its own
        ],
    }


def main() -> None:
    workdir = Path(__file__).resolve().parent
    train_prompts = []
    eval_prompts = []

    # Use a deterministic split: per tool, first 3 asks → train, rest → eval
    for tool in TOOLS:
        asks = tool["asks"]
        random.shuffle(asks)
        train = asks[:max(1, len(asks) - 1)]
        eval_ = asks[max(1, len(asks) - 1):]
        for ask in train:
            train_prompts.append(make_prompt(tool, ask))
        for ask in eval_:
            eval_prompts.append(make_prompt(tool, ask))

    # Pad eval to ~30 examples by re-asking with slight variations
    while len(eval_prompts) < 30:
        tool = random.choice(TOOLS)
        ask = random.choice(tool["asks"]) + (" please" if random.random() < 0.5 else "")
        p = make_prompt(tool, ask)
        eval_prompts.append(p)

    random.shuffle(train_prompts)
    random.shuffle(eval_prompts)

    # Write
    (workdir / "prompts").mkdir(exist_ok=True)
    (workdir / "datasets").mkdir(exist_ok=True)

    with (workdir / "prompts/h1-r16-6ep.jsonl").open("w") as f:
        for p in train_prompts:
            f.write(json.dumps(p) + "\n")
    with (workdir / "datasets/train.opd.jsonl").open("w") as f:
        for p in train_prompts:
            f.write(json.dumps(p) + "\n")
    with (workdir / "datasets/eval.jsonl").open("w") as f:
        for p in eval_prompts:
            f.write(json.dumps(p) + "\n")

    print(f"train: {len(train_prompts)} prompts")
    print(f"eval:  {len(eval_prompts)} prompts")
    print(f"tools: {len(TOOLS)}")


if __name__ == "__main__":
    main()
