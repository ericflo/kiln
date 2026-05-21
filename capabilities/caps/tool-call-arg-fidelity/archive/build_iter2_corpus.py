"""Build iter 2 prompts (H9 asymmetric) — same student-side prompts as
iter 1, but each gets a `teacher_extra_messages` prefix containing 3
pristine tool-call exemplars per tool shape.

The teacher sees these exemplars + the user prompt; the student sees
only the user prompt. Reverse-KL pulls the student toward the sharp
teacher distribution that knows what good tool calls look like.
"""
from __future__ import annotations

import json
from pathlib import Path

WORKDIR = Path(__file__).resolve().parent
IN_PATH = WORKDIR / "prompts/h1-r16-6ep.jsonl"
OUT_PATH = WORKDIR / "prompts/h9-asym.jsonl"

# 3 pristine exemplars per tool shape, all the EXACT correct form.
# These never appear in the eval (different specifics) but teach the
# canonical shape.
EXEMPLARS = {
    "file_read": [
        ("Read the entire file src/main.rs",
         '{"path": "src/main.rs"}'),
        ("Open lines 1-100 of crates/utils/Cargo.toml",
         '{"path": "crates/utils/Cargo.toml", "offset": 1, "limit": 100}'),
        ("Read from line 500 of /etc/nginx/nginx.conf",
         '{"path": "/etc/nginx/nginx.conf", "offset": 500}'),
    ],
    "file_write": [
        ("Save the content 'hello' to test.txt",
         '{"path": "test.txt", "content": "hello"}'),
        ("Append the line 'export X=1' to ~/.bashrc",
         '{"path": "/root/.bashrc", "content": "export X=1\\n", "mode": "append"}'),
        ("Create config.json with content {\"key\":\"value\"}",
         '{"path": "config.json", "content": "{\\"key\\":\\"value\\"}", "mode": "overwrite"}'),
    ],
    "run_bash": [
        ("Show git log",
         '{"command": "git log --oneline -10"}'),
        ("Build the project with a 10 minute timeout",
         '{"command": "cargo build --release", "timeout": 600}'),
        ("Start the watcher in the background",
         '{"command": "cargo watch -x check", "background": true}'),
    ],
    "web_search": [
        ("Search for the latest Rust release",
         '{"query": "latest Rust release notes"}'),
        ("Find 3 results about WebGPU adoption",
         '{"query": "WebGPU adoption", "max_results": 3}'),
        ("Look up how to install Docker on Ubuntu",
         '{"query": "install Docker Ubuntu 22.04"}'),
    ],
    "list_files": [
        ("List the workspace directory",
         '{"directory": "/workspace"}'),
        ("List all .rs files under crates/ recursively",
         '{"directory": "crates", "recursive": true, "pattern": "*.rs"}'),
        ("Show me the contents of src/",
         '{"directory": "src"}'),
    ],
    "git_diff": [
        ("Show the working tree diff",
         '{}'),
        ("Show staged changes for src/lib.rs",
         '{"paths": ["src/lib.rs"], "staged": true}'),
        ("Diff between main and HEAD",
         '{"base": "main"}'),
    ],
    "http_request": [
        ("GET https://api.example.com/status",
         '{"url": "https://api.example.com/status", "method": "GET"}'),
        ("POST {\"x\":1} to https://api.example.com/items as JSON",
         '{"url": "https://api.example.com/items", "method": "POST", "headers": {"Content-Type": "application/json"}, "body": "{\\"x\\":1}"}'),
        ("DELETE /users/99 with a 30s timeout",
         '{"url": "https://api.example.com/users/99", "method": "DELETE", "timeout": 30}'),
    ],
    "schedule_task": [
        ("Run backup nightly at 2am",
         '{"task": "backup", "when": "0 2 * * *"}'),
        ("Schedule deploy for 2026-06-15T09:00:00Z with high priority",
         '{"task": "deploy", "when": "2026-06-15T09:00:00Z", "priority": "high"}'),
        ("Run health check every hour with 3 retries",
         '{"task": "health_check", "when": "0 * * * *", "retry_count": 3}'),
    ],
    "code_search": [
        ("Grep for 'TODO' in *.rs files",
         '{"query": "TODO", "file_pattern": "*.rs"}'),
        ("Case-insensitive search for ERROR, max 10 matches",
         '{"query": "ERROR", "case_sensitive": false, "max_matches": 10}'),
        ("Find uses of fetch_logprobs",
         '{"query": "fetch_logprobs"}'),
    ],
    "delete_file": [
        ("Delete /tmp/output.log",
         '{"path": "/tmp/output.log"}'),
        ("Force-delete the dist/ directory recursively",
         '{"path": "dist", "force": true, "recursive": true}'),
        ("Remove the file old.bak",
         '{"path": "old.bak"}'),
    ],
    "set_env": [
        ("Set PATH to /usr/local/bin for this session",
         '{"name": "PATH", "value": "/usr/local/bin"}'),
        ("Persistently set RUST_LOG to debug",
         '{"name": "RUST_LOG", "value": "debug", "scope": "persistent"}'),
        ("Set DATABASE_URL for the session",
         '{"name": "DATABASE_URL", "value": "postgres://localhost/dev", "scope": "session"}'),
    ],
    "create_branch": [
        ("Create branch feature/foo from main",
         '{"name": "feature/foo", "from_ref": "main"}'),
        ("Create and check out hotfix/leak",
         '{"name": "hotfix/leak", "checkout": true}'),
        ("Make branch experiment/x",
         '{"name": "experiment/x"}'),
    ],
}


def build_teacher_prefix(tool_name: str) -> list[dict]:
    """Return a list of ChatMessage dicts to prepend on the teacher's side."""
    exemplars = EXEMPLARS.get(tool_name, [])
    if not exemplars:
        return []
    body = "Here are 3 pristine examples of correct tool-call JSON (visible only to you, the teacher — not the student):\n\n"
    for i, (user_ask, response) in enumerate(exemplars, 1):
        body += f"Example {i}:\nUser: {user_ask}\nCorrect JSON: {response}\n\n"
    body += "Use these to inform the distribution at every position of the actual response below."
    return [{"role": "system", "content": body}]


def main() -> None:
    out_rows = []
    with IN_PATH.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            # Derive tool name from id prefix (e.g. "file_read-a924")
            tool_name = row["id"].rsplit("-", 1)[0]
            row["teacher_extra_messages"] = build_teacher_prefix(tool_name)
            out_rows.append(row)
    with OUT_PATH.open("w") as f:
        for row in out_rows:
            f.write(json.dumps(row) + "\n")
    n_asym = sum(1 for r in out_rows if r.get("teacher_extra_messages"))
    avg_extra_len = sum(
        len(m["content"]) for r in out_rows for m in r.get("teacher_extra_messages", [])
    ) / max(n_asym, 1)
    print(f"prompts: {len(out_rows)}")
    print(f"with teacher_extra_messages: {n_asym}")
    print(f"avg teacher prefix chars: {avg_extra_len:.0f}")


if __name__ == "__main__":
    main()
