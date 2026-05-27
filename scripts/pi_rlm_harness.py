#!/usr/bin/env python3
"""OpenAI-compatible Pi RLM harness proxy.

The proxy makes one Pi provider turn behave like an RLM turn:

1. Store Pi's full `messages` + `tools` payload as external environment state.
2. Ask an upstream OpenAI-compatible model for bounded JSON actions.
3. Execute internal inspect/search/slice/subcall/subrlm actions against state.
4. Return exactly one normal assistant response or OpenAI tool call to Pi.

The model-facing contract deliberately uses fixed-size windows:

- 8192 tokens reserved for the stable RLM system/prefix/tool protocol.
- 4096 tokens for the dynamic environment summary and observations.
- 4096 tokens for each model output.

Token limits are enforced approximately by characters in this Python harness.
Kiln-side training/eval should enforce them with the real tokenizer.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib import error, request


DEFAULT_LISTEN = "127.0.0.1:8421"
DEFAULT_UPSTREAM = "http://127.0.0.1:8420/v1"
DEFAULT_MODEL = "Qwen3.5-4B"
DEFAULT_STATE_DIR = ".kiln/pi-rlm-harness"

PREFIX_TOKENS = 8192
INPUT_TOKENS = 4096
OUTPUT_TOKENS = 4096
CHARS_PER_TOKEN = 4
MAX_OBSERVATION_CHARS = INPUT_TOKENS * CHARS_PER_TOKEN

RLM_SYSTEM_PROMPT = f"""You are the root controller for a Recursive Language Model (RLM) harness.

The user's full Pi conversation and tool catalog are stored outside your context
as an environment. You do not need to read the whole transcript at once. Use
bounded JSON actions to inspect the environment, search it, slice exact regions,
ask bounded semantic sub-calls, or finish with the next assistant response Pi
should receive.

Budget contract:
- Stable system/prefix/tool protocol budget: {PREFIX_TOKENS} tokens.
- Dynamic environment input budget: {INPUT_TOKENS} tokens.
- Your output budget: {OUTPUT_TOKENS} tokens.

Return one JSON object and no prose. Valid actions:

{{"action":"inspect","target":"summary"}}
{{"action":"inspect","target":"message","index":0}}
{{"action":"inspect","target":"tools"}}
{{"action":"search","query":"needle","regex":false,"max_matches":8}}
{{"action":"slice","index":0,"start":0,"length":4096}}
{{"action":"subcall","prompt":"bounded prompt","system":"optional system","max_tokens":1024}}
{{"action":"subrlm","prompt":"bounded prompt","max_tokens":1024}}
{{"action":"finish","content":"assistant text for Pi"}}
{{"action":"finish","tool_call":{{"name":"Bash","arguments":{{"cmd":"..."}}}}}}

Rules:
- First inspect/search before asking broad semantic subcalls.
- Use code/tool calls only in the final `finish.tool_call`; internal actions are
  harness actions, not Pi-visible tools.
- For Pi tools, preserve the recorded OpenAI function-call shape: tool name plus
  JSON arguments object.
- Stop as soon as the next Pi-visible assistant response or tool call is clear.
"""


def approx_token_chars(tokens: int) -> int:
    return tokens * CHARS_PER_TOKEN


def clip_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head - 80
    return (
        text[:head]
        + f"\n...[clipped {len(text) - max_chars} chars from middle]...\n"
        + text[-tail:]
    )


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def pretty_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def first_json_object(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object from a model response."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(stripped):
        if ch != "{":
            continue
        try:
            value, _ = decoder.raw_decode(stripped[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def parse_listen(value: str) -> tuple[str, int]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("listen address must be HOST:PORT")
    host, port_s = value.rsplit(":", 1)
    try:
        port = int(port_s)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("listen port must be an integer") from exc
    return host, port


class UpstreamClient:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = dict(payload)
        body.setdefault("model", self.model)
        data = json.dumps(body).encode("utf-8")
        req = request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers={
                "content-type": "application/json",
                "authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"upstream returned {exc.code}: {detail}") from exc

    def first_text(self, payload: dict[str, Any]) -> str:
        resp = self.chat(payload)
        choices = resp.get("choices") or []
        if not choices:
            return ""
        msg = choices[0].get("message") or {}
        if msg.get("tool_calls"):
            return json_dumps({"tool_calls": msg["tool_calls"]})
        return msg.get("content") or ""


@dataclass
class RlmResult:
    content: str = ""
    tool_call: dict[str, Any] | None = None
    trace: list[dict[str, Any]] = field(default_factory=list)
    fallback_used: bool = False


@dataclass
class RlmEnvironment:
    request_id: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    upstream_request: dict[str, Any]
    state_dir: Path
    observations: list[dict[str, Any]] = field(default_factory=list)

    def persist(self, result: RlmResult | None = None) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "request_id": self.request_id,
            "created_at_unix": time.time(),
            "messages": self.messages,
            "tools": self.tools,
            "observations": self.observations,
        }
        if result is not None:
            payload["result"] = {
                "content": result.content,
                "tool_call": result.tool_call,
                "fallback_used": result.fallback_used,
                "trace": result.trace,
            }
        path = self.state_dir / f"{self.request_id}.json"
        path.write_text(pretty_json(payload) + "\n", encoding="utf-8")

    def summary(self) -> dict[str, Any]:
        inventory = []
        for index, msg in enumerate(self.messages):
            content = message_text(msg)
            inventory.append(
                {
                    "index": index,
                    "role": msg.get("role", ""),
                    "chars": len(content),
                    "preview": clip_text(content, 240),
                    "has_tool_calls": bool(msg.get("tool_calls")),
                    "tool_call_id": msg.get("tool_call_id"),
                    "name": msg.get("name"),
                }
            )
        latest_user = next(
            (message_text(m) for m in reversed(self.messages) if m.get("role") == "user"),
            "",
        )
        tool_names = [tool_name(t) for t in self.tools if tool_name(t)]
        return {
            "request_id": self.request_id,
            "message_count": len(self.messages),
            "tool_count": len(self.tools),
            "tool_names": tool_names,
            "latest_user_preview": clip_text(latest_user, 1200),
            "messages": inventory,
        }

    def observe(self, action: dict[str, Any], output: Any) -> dict[str, Any]:
        obs = {
            "action": action,
            "output": output,
        }
        self.observations.append(obs)
        return obs

    def bounded_observations(self) -> str:
        raw = pretty_json(self.observations[-8:])
        return clip_text(raw, MAX_OBSERVATION_CHARS)

    def inspect(self, action: dict[str, Any]) -> Any:
        target = action.get("target", "summary")
        if target == "summary":
            return self.summary()
        if target == "tools":
            return self.tools
        if target == "message":
            index = int(action.get("index", -1))
            if index < 0 or index >= len(self.messages):
                return {"error": f"message index {index} out of range"}
            msg = self.messages[index]
            text = message_text(msg)
            return {
                "index": index,
                "role": msg.get("role", ""),
                "chars": len(text),
                "content_preview": clip_text(text, 4000),
                "tool_calls": msg.get("tool_calls"),
                "tool_call_id": msg.get("tool_call_id"),
                "name": msg.get("name"),
            }
        if target == "observations":
            return self.observations
        return {"error": f"unknown inspect target {target!r}"}

    def search(self, action: dict[str, Any]) -> Any:
        query = str(action.get("query", ""))
        if not query:
            return {"error": "empty query"}
        use_regex = bool(action.get("regex", False))
        max_matches = int(action.get("max_matches", 8))
        matches = []
        for index, msg in enumerate(self.messages):
            text = message_text(msg)
            if use_regex:
                try:
                    found = list(re.finditer(query, text))
                except re.error as exc:
                    return {"error": f"invalid regex: {exc}"}
                spans = [m.span() for m in found[: max_matches - len(matches)]]
            else:
                spans = []
                start = 0
                while len(spans) + len(matches) < max_matches:
                    pos = text.find(query, start)
                    if pos < 0:
                        break
                    spans.append((pos, pos + len(query)))
                    start = pos + max(1, len(query))
            for start, end in spans:
                lo = max(0, start - 160)
                hi = min(len(text), end + 160)
                matches.append(
                    {
                        "message_index": index,
                        "role": msg.get("role", ""),
                        "start": start,
                        "end": end,
                        "context": text[lo:hi],
                    }
                )
                if len(matches) >= max_matches:
                    return matches
        return matches

    def slice(self, action: dict[str, Any]) -> Any:
        index = int(action.get("index", -1))
        if index < 0 or index >= len(self.messages):
            return {"error": f"message index {index} out of range"}
        text = message_text(self.messages[index])
        start = max(0, int(action.get("start", 0)))
        length = min(max(1, int(action.get("length", 4096))), MAX_OBSERVATION_CHARS)
        return {
            "message_index": index,
            "start": start,
            "end": min(len(text), start + length),
            "chars_total": len(text),
            "content": text[start : start + length],
        }


class RlmController:
    def __init__(
        self,
        upstream: UpstreamClient,
        state_dir: Path,
        max_iters: int,
        max_depth: int,
    ):
        self.upstream = upstream
        self.state_dir = state_dir
        self.max_iters = max_iters
        self.max_depth = max_depth

    def run(self, payload: dict[str, Any], depth: int = 0) -> RlmResult:
        request_id = request_id_from_payload(payload)
        env = RlmEnvironment(
            request_id=request_id,
            messages=list(payload.get("messages") or []),
            tools=list(payload.get("tools") or []),
            upstream_request=payload,
            state_dir=self.state_dir,
        )
        env.persist()
        trace: list[dict[str, Any]] = []
        for step in range(self.max_iters):
            root_payload = {
                "model": payload.get("model") or self.upstream.model,
                "messages": [
                    {"role": "system", "content": RLM_SYSTEM_PROMPT},
                    {"role": "user", "content": self.root_user_prompt(env, depth, step)},
                ],
                "temperature": 0.0,
                "max_tokens": OUTPUT_TOKENS,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            raw = self.upstream.first_text(root_payload)
            action = first_json_object(raw)
            if action is None:
                action = {
                    "action": "finish",
                    "content": raw,
                    "warning": "model did not emit JSON action",
                }
            trace.append({"step": step, "raw": raw, "action": action})
            kind = str(action.get("action", "")).lower()
            if kind == "finish":
                result = self.finish(action, trace)
                env.persist(result)
                return result
            output = self.execute(env, action, depth)
            obs = env.observe(action, output)
            trace[-1]["observation"] = obs
            env.persist()
        result = self.direct_fallback(payload, trace)
        env.persist(result)
        return result

    def root_user_prompt(self, env: RlmEnvironment, depth: int, step: int) -> str:
        body = {
            "depth": depth,
            "step": step,
            "environment_summary": env.summary(),
            "recent_observations": env.observations[-8:],
        }
        return clip_text(pretty_json(body), approx_token_chars(INPUT_TOKENS))

    def execute(self, env: RlmEnvironment, action: dict[str, Any], depth: int) -> Any:
        kind = str(action.get("action", "")).lower()
        if kind == "inspect":
            return env.inspect(action)
        if kind == "search":
            return env.search(action)
        if kind == "slice":
            return env.slice(action)
        if kind == "subcall":
            return self.subcall(action)
        if kind == "subrlm":
            if depth >= self.max_depth:
                return self.subcall(action)
            prompt = str(action.get("prompt", ""))
            sub_payload = {
                "model": env.upstream_request.get("model") or self.upstream.model,
                "messages": [{"role": "user", "content": prompt}],
                "tools": env.tools,
                "max_tokens": min(int(action.get("max_tokens", 1024)), OUTPUT_TOKENS),
                "temperature": 0.0,
            }
            result = self.run(sub_payload, depth + 1)
            return {
                "content": result.content,
                "tool_call": result.tool_call,
                "fallback_used": result.fallback_used,
            }
        return {"error": f"unknown action {kind!r}"}

    def subcall(self, action: dict[str, Any]) -> Any:
        prompt = str(action.get("prompt", ""))
        system = action.get("system")
        messages = []
        if isinstance(system, str) and system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": clip_text(prompt, approx_token_chars(INPUT_TOKENS))})
        max_tokens = min(int(action.get("max_tokens", 1024)), OUTPUT_TOKENS)
        text = self.upstream.first_text(
            {
                "model": action.get("model") or self.upstream.model,
                "messages": messages,
                "temperature": float(action.get("temperature", 0.0)),
                "max_tokens": max_tokens,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        )
        return {"content": text}

    def finish(self, action: dict[str, Any], trace: list[dict[str, Any]]) -> RlmResult:
        tool_call = action.get("tool_call")
        if isinstance(tool_call, dict) and tool_call.get("name"):
            return RlmResult(tool_call=normalize_tool_call(tool_call), trace=trace)
        return RlmResult(content=str(action.get("content", "")), trace=trace)

    def direct_fallback(self, payload: dict[str, Any], trace: list[dict[str, Any]]) -> RlmResult:
        fallback = dict(payload)
        fallback["stream"] = False
        fallback["max_tokens"] = min(int(fallback.get("max_tokens", OUTPUT_TOKENS)), OUTPUT_TOKENS)
        resp = self.upstream.chat(fallback)
        choices = resp.get("choices") or []
        if not choices:
            return RlmResult(content="", trace=trace, fallback_used=True)
        msg = choices[0].get("message") or {}
        tool_calls = msg.get("tool_calls") or []
        if tool_calls:
            first = tool_calls[0]
            fn = first.get("function") or {}
            return RlmResult(
                tool_call=normalize_tool_call(
                    {
                        "id": first.get("id"),
                        "name": fn.get("name"),
                        "arguments": parse_arguments(fn.get("arguments")),
                    }
                ),
                trace=trace,
                fallback_used=True,
            )
        return RlmResult(content=msg.get("content") or "", trace=trace, fallback_used=True)


def message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if isinstance(block.get("text"), str):
                    parts.append(block["text"])
                elif isinstance(block.get("content"), str):
                    parts.append(block["content"])
        return "".join(parts)
    return json_dumps(content)


def tool_name(tool: dict[str, Any]) -> str | None:
    fn = tool.get("function")
    if isinstance(fn, dict) and isinstance(fn.get("name"), str):
        return fn["name"]
    if isinstance(tool.get("name"), str):
        return tool["name"]
    return None


def parse_arguments(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    if value is None:
        return {}
    return value


def normalize_tool_call(tool_call: dict[str, Any]) -> dict[str, Any]:
    name = str(tool_call.get("name", ""))
    arguments = parse_arguments(tool_call.get("arguments", {}))
    call_id = str(tool_call.get("id") or f"call_{uuid.uuid4().hex[:12]}")
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json_dumps(arguments),
        },
    }


def request_id_from_payload(payload: dict[str, Any]) -> str:
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        for key in ("trace_id", "session_id", "request_id"):
            if isinstance(metadata.get(key), str) and metadata[key]:
                return sanitize_id(metadata[key])
    return f"rlm_{uuid.uuid4().hex}"


def sanitize_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:120]


class ProxyServer(ThreadingHTTPServer):
    controller: RlmController
    upstream: UpstreamClient


class Handler(BaseHTTPRequestHandler):
    server: ProxyServer

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("pi-rlm-harness: " + fmt % args + "\n")

    def do_GET(self) -> None:
        if self.path.rstrip("/") == "/v1/models":
            body = {
                "object": "list",
                "data": [
                    {
                        "id": self.server.upstream.model,
                        "object": "model",
                        "owned_by": "kiln",
                    }
                ],
            }
            self.send_json(200, body)
            return
        self.send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/v1/chat/completions":
            self.send_json(404, {"error": "not found"})
            return
        try:
            payload = self.read_json()
            stream = bool(payload.get("stream"))
            payload["stream"] = False
            result = self.server.controller.run(payload)
            if stream:
                self.send_stream(result, payload)
            else:
                self.send_json(200, completion_response(result, payload))
        except Exception as exc:  # pragma: no cover - last-resort HTTP boundary
            self.send_json(500, {"error": str(exc)})

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(length)
        value = json.loads(body.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("request body must be a JSON object")
        return value

    def send_json(self, status: int, value: Any) -> None:
        data = (pretty_json(value) + "\n").encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_stream(self, result: RlmResult, payload: dict[str, Any]) -> None:
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.end_headers()
        chunk = stream_chunk(result, payload)
        self.wfile.write(f"data: {json_dumps(chunk)}\n\n".encode("utf-8"))
        self.wfile.write(b"data: [DONE]\n\n")


def completion_response(result: RlmResult, payload: dict[str, Any]) -> dict[str, Any]:
    msg: dict[str, Any] = {
        "role": "assistant",
        "content": result.content if result.tool_call is None else "",
    }
    finish_reason = "stop"
    if result.tool_call is not None:
        msg["tool_calls"] = [result.tool_call]
        finish_reason = "tool_calls"
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": payload.get("model") or DEFAULT_MODEL,
        "choices": [
            {
                "index": 0,
                "message": msg,
                "finish_reason": finish_reason,
            }
        ],
    }


def stream_chunk(result: RlmResult, payload: dict[str, Any]) -> dict[str, Any]:
    delta: dict[str, Any] = {"role": "assistant"}
    finish_reason = "stop"
    if result.tool_call is not None:
        delta["tool_calls"] = [result.tool_call]
        finish_reason = "tool_calls"
    else:
        delta["content"] = result.content
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": payload.get("model") or DEFAULT_MODEL,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


class FakeUpstream(UpstreamClient):
    def __init__(self, responses: list[str]):
        self.responses = responses
        self.model = DEFAULT_MODEL

    def first_text(self, payload: dict[str, Any]) -> str:
        if self.responses:
            return self.responses.pop(0)
        return '{"action":"finish","content":"done"}'

    def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        text = self.first_text(payload)
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": text,
                    }
                }
            ]
        }


def self_test() -> None:
    assert first_json_object('```json\n{"action":"inspect"}\n```') == {"action": "inspect"}
    tmp = Path("/tmp/pi-rlm-harness-self-test")
    upstream = FakeUpstream(
        [
            '{"action":"inspect","target":"summary"}',
            '{"action":"search","query":"failing doctest","max_matches":2}',
            '{"action":"finish","tool_call":{"name":"Bash","arguments":{"cmd":"pytest -q"}}}',
        ]
    )
    controller = RlmController(upstream, tmp, max_iters=4, max_depth=1)
    result = controller.run(
        {
            "model": DEFAULT_MODEL,
            "messages": [
                {"role": "system", "content": "You are pi."},
                {"role": "user", "content": "Fix the failing doctest."},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "Bash",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            "metadata": {"trace_id": "self-test"},
        }
    )
    assert result.tool_call is not None
    assert result.tool_call["function"]["name"] == "Bash"
    assert json.loads(result.tool_call["function"]["arguments"])["cmd"] == "pytest -q"
    assert (tmp / "self-test.json").exists()
    print("pi_rlm_harness self-test passed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen", default=DEFAULT_LISTEN, help="HOST:PORT to serve")
    parser.add_argument("--upstream", default=DEFAULT_UPSTREAM, help="upstream OpenAI-compatible /v1 base URL")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="upstream model id")
    parser.add_argument("--api-key", default="kiln", help="upstream API key")
    parser.add_argument("--state-dir", default=DEFAULT_STATE_DIR, help="directory for external RLM state")
    parser.add_argument("--max-iters", type=int, default=8, help="max root RLM actions per Pi turn")
    parser.add_argument("--max-depth", type=int, default=1, help="max recursive subrlm depth")
    parser.add_argument("--self-test", action="store_true", help="run local parser/controller smoke test")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return 0

    host, port = parse_listen(args.listen)
    upstream = UpstreamClient(args.upstream, args.api_key, args.model)
    controller = RlmController(upstream, Path(args.state_dir), args.max_iters, args.max_depth)
    server = ProxyServer((host, port), Handler)
    server.controller = controller
    server.upstream = upstream
    print(
        f"pi-rlm-harness listening on http://{host}:{port}/v1 "
        f"upstream={args.upstream} model={args.model}",
        flush=True,
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
