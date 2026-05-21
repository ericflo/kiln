"""Generate the prompt corpus for faithful-code-summarization.

Produces:
  datasets/eval.jsonl     — the blind eval set (do NOT read after generation)
  datasets/train.opd.jsonl — the OPD training prompts (visible; smaller subset)

Each line:
  {"id": str, "code": str, "messages": [{"role":"system",...}, {"role":"user", "content": "Summarize this code:\\n\\n<code>"}, {"role":"assistant", "content":"<dummy>"}]}

The "code" field is what the rubric scores against; the "messages" field is
what the model is asked. Eval and train sets are disjoint and pulled from
distinct generator seeds — the eval set is blind to the agent.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable


# A library of small code snippets, organized by language.
# Each generator returns a (code: str, label: str) tuple.

def py_simple_function(rng: random.Random) -> tuple[str, str]:
    verbs = ["fetch", "load", "parse", "compute", "render", "validate", "extract"]
    nouns = ["user", "record", "config", "payload", "token", "session", "result"]
    verb = rng.choice(verbs)
    noun = rng.choice(nouns)
    code = f"""def {verb}_{noun}({noun}_id):
    cache = _get_cache()
    if {noun}_id in cache:
        return cache[{noun}_id]
    value = _load_from_store({noun}_id)
    cache[{noun}_id] = value
    return value
"""
    return code, f"py-{verb}-{noun}"


def py_class_with_methods(rng: random.Random) -> tuple[str, str]:
    name = rng.choice(["UserService", "OrderManager", "ConfigLoader", "EventDispatcher"])
    code = f"""class {name}:
    def __init__(self, store):
        self.store = store
        self._cache = {{}}

    def get(self, key):
        if key not in self._cache:
            self._cache[key] = self.store.fetch(key)
        return self._cache[key]

    def invalidate(self, key):
        self._cache.pop(key, None)
"""
    return code, f"py-class-{name}"


def py_dataclass(rng: random.Random) -> tuple[str, str]:
    name = rng.choice(["Point", "Vector", "Rect", "Range", "Interval"])
    code = f"""from dataclasses import dataclass

@dataclass
class {name}:
    x: float
    y: float

    def magnitude(self) -> float:
        return (self.x ** 2 + self.y ** 2) ** 0.5

    def scaled(self, factor: float) -> "{name}":
        return {name}(self.x * factor, self.y * factor)
"""
    return code, f"py-dataclass-{name}"


def py_recursive(rng: random.Random) -> tuple[str, str]:
    name = rng.choice(["factorial", "fibonacci", "ackermann", "tree_depth", "count_leaves"])
    if name == "factorial":
        code = """def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
"""
    elif name == "fibonacci":
        code = """def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
"""
    else:
        code = f"""def {name}(node):
    if node is None:
        return 0
    return 1 + max({name}(node.left), {name}(node.right))
"""
    return code, f"py-recursive-{name}"


def rust_function(rng: random.Random) -> tuple[str, str]:
    name = rng.choice(["normalize", "transform", "encode", "decode", "compress"])
    code = f"""pub fn {name}(input: &str) -> Result<String, ParseError> {{
    let trimmed = input.trim();
    if trimmed.is_empty() {{
        return Err(ParseError::Empty);
    }}
    let cleaned = trimmed.to_lowercase();
    Ok(cleaned.replace(' ', "_"))
}}
"""
    return code, f"rust-{name}"


def rust_struct(rng: random.Random) -> tuple[str, str]:
    name = rng.choice(["Config", "Settings", "Options", "State"])
    code = f"""pub struct {name} {{
    pub timeout: u64,
    pub retries: u32,
    name: String,
}}

impl {name} {{
    pub fn new(name: String) -> Self {{
        {name} {{ timeout: 30, retries: 3, name }}
    }}

    pub fn with_timeout(mut self, t: u64) -> Self {{
        self.timeout = t;
        self
    }}
}}
"""
    return code, f"rust-struct-{name}"


def go_function(rng: random.Random) -> tuple[str, str]:
    name = rng.choice(["ReadFile", "WriteFile", "ParseJSON", "ValidateInput"])
    code = f"""func {name}(path string) ([]byte, error) {{
    data, err := os.ReadFile(path)
    if err != nil {{
        return nil, fmt.Errorf("{name}: %w", err)
    }}
    if len(data) == 0 {{
        return nil, errors.New("{name}: empty file")
    }}
    return data, nil
}}
"""
    return code, f"go-{name}"


def js_arrow(rng: random.Random) -> tuple[str, str]:
    name = rng.choice(["debounce", "throttle", "memoize", "compose"])
    code = f"""const {name} = (fn, delay) => {{
    let timer = null;
    return (...args) => {{
        clearTimeout(timer);
        timer = setTimeout(() => fn(...args), delay);
    }};
}};
"""
    return code, f"js-{name}"


def js_class(rng: random.Random) -> tuple[str, str]:
    name = rng.choice(["EventEmitter", "Stream", "Queue", "Stack"])
    code = f"""class {name} {{
    constructor() {{
        this.items = [];
    }}

    push(item) {{
        this.items.push(item);
        return this.items.length;
    }}

    pop() {{
        return this.items.pop();
    }}

    get size() {{
        return this.items.length;
    }}
}}
"""
    return code, f"js-class-{name}"


def py_async(rng: random.Random) -> tuple[str, str]:
    name = rng.choice(["fetch_remote", "poll_status", "stream_events"])
    code = f"""import asyncio

async def {name}(url, timeout=30):
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=timeout) as resp:
            resp.raise_for_status()
            return await resp.json()
"""
    return code, f"py-async-{name}"


def py_generator(rng: random.Random) -> tuple[str, str]:
    name = rng.choice(["chunks", "batched", "windowed"])
    code = f"""def {name}(iterable, size):
    buffer = []
    for item in iterable:
        buffer.append(item)
        if len(buffer) >= size:
            yield buffer
            buffer = []
    if buffer:
        yield buffer
"""
    return code, f"py-gen-{name}"


def py_decorator(rng: random.Random) -> tuple[str, str]:
    name = rng.choice(["log_calls", "retry_on_error", "timed"])
    code = f"""def {name}(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.time()
        try:
            return fn(*args, **kwargs)
        finally:
            elapsed = time.time() - start
            print(f"{{fn.__name__}}: {{elapsed:.3f}}s")
    return wrapper
"""
    return code, f"py-deco-{name}"


GENERATORS: list[Callable[[random.Random], tuple[str, str]]] = [
    py_simple_function, py_class_with_methods, py_dataclass, py_recursive,
    py_async, py_generator, py_decorator,
    rust_function, rust_struct,
    go_function,
    js_arrow, js_class,
]


SYSTEM_PROMPT = (
    "You are a careful code assistant. When shown a code snippet, "
    "write a brief (2–4 sentence, under 150 words) summary of what it does. "
    "Mention the function and class names from the code by name. "
    "Do not invent names that are not in the code. Do not describe behavior "
    "that is not in the snippet."
)


def build_prompt(code: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Summarize this code:\n\n```\n{code}```"},
        {"role": "assistant", "content": ""},  # dummy; student samples real
    ]


def main() -> None:
    out_dir = Path("datasets")
    out_dir.mkdir(exist_ok=True)

    # Eval set: seed 42, 50 prompts. AGENT MUST NOT READ THIS FILE.
    rng = random.Random(42)
    eval_rows = []
    seen_ids: set[str] = set()
    while len(eval_rows) < 50:
        gen = rng.choice(GENERATORS)
        code, label = gen(rng)
        eid = f"eval-{label}-{len(eval_rows):03d}"
        if eid in seen_ids:
            continue
        seen_ids.add(eid)
        eval_rows.append({"id": eid, "code": code, "messages": build_prompt(code)})

    # Train set: seed 4242 (different), 40 prompts. Visible to agent.
    rng = random.Random(4242)
    train_rows = []
    seen_ids = set()
    while len(train_rows) < 40:
        gen = rng.choice(GENERATORS)
        code, label = gen(rng)
        tid = f"train-{label}-{len(train_rows):03d}"
        if tid in seen_ids:
            continue
        seen_ids.add(tid)
        train_rows.append({"id": tid, "code": code, "messages": build_prompt(code)})

    with open(out_dir / "eval.jsonl", "w") as f:
        for row in eval_rows:
            f.write(json.dumps(row) + "\n")
    with open(out_dir / "train.opd.jsonl", "w") as f:
        for row in train_rows:
            f.write(json.dumps(row) + "\n")

    print(f"wrote {len(eval_rows)} eval prompts → datasets/eval.jsonl")
    print(f"wrote {len(train_rows)} train prompts → datasets/train.opd.jsonl")
    print()
    print("REMINDER: do not read datasets/eval.jsonl from this point on.")
    print("The eval set is blind. Use only the oracle's SCORE= output.")


if __name__ == "__main__":
    main()
