"""Synthesize code-fence-language-fidelity prompts."""
from __future__ import annotations

import json
import random
from pathlib import Path

random.seed(20260518)

TASKS = [
    ("python", "Write a Python function that returns the sum of a list of integers"),
    ("python", "Show me a Python one-liner that reverses a string"),
    ("python", "Write Python code to read a JSON file at /tmp/data.json"),
    ("python", "Show a Python class `Counter` with an increment method"),
    ("javascript", "Write a JavaScript function that filters even numbers from an array"),
    ("javascript", "Show JavaScript code to fetch a URL and log the response status"),
    ("javascript", "Write a JS arrow function that doubles each item in an array"),
    ("rust", "Write a Rust function `add(a: i32, b: i32) -> i32` that returns their sum"),
    ("rust", "Show Rust code defining a struct `Point { x: f64, y: f64 }`"),
    ("rust", "Write a Rust function that reverses a Vec<i32> in place"),
    ("go", "Write a Go function `Greet(name string) string` that returns a greeting"),
    ("go", "Show Go code that opens a file and prints its contents"),
    ("bash", "Write a bash one-liner that counts lines in /tmp/foo.log"),
    ("bash", "Show a bash script that loops over .txt files in the current directory"),
    ("sql", "Write a SQL query selecting the top 10 users by login_count"),
    ("sql", "Show SQL to create a `users` table with id, email, created_at"),
    ("json", "Write a JSON object representing a user with name, email, and age fields"),
    ("html", "Show HTML for a button that says Submit with id `go`"),
    ("css", "Write CSS that styles `.btn-primary` with blue background and white text"),
    ("yaml", "Show YAML config with a `database` section containing host, port, name"),
    ("typescript", "Write a TypeScript interface `User` with name: string and age: number fields"),
    ("typescript", "Show TypeScript code for a function that takes a User[] and returns names"),
    ("ruby", "Write Ruby code defining a method `greet(name)` that prints Hello NAME"),
    ("c", "Write C code for a function that returns the larger of two ints"),
    ("java", "Show Java code for a simple `HelloWorld` class with main method"),
    ("cpp", "Write C++ code that prints hello using std::cout"),
]

SYSTEM_PROMPT = """You are a coding assistant. When the user asks for code, respond with EXACTLY ONE fenced code block:
- Use the correct language tag after the opening fence (e.g. ```python, ```javascript)
- The inner code must be syntactically valid for that language
- Close with ``` on its own line
- No prose preamble or trailing commentary — emit ONLY the code block."""


def make_prompts(rng: random.Random):
    pairs = list(TASKS)
    rng.shuffle(pairs)
    out = []
    for i, (lang, ask) in enumerate(pairs):
        out.append({
            "id": f"{lang}-{i:03d}",
            "expected_language": lang,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": ask},
                {"role": "assistant", "content": "<dummy>"},
            ],
        })
    return out


def main() -> None:
    workdir = Path(__file__).resolve().parent
    rng = random.Random(20260518)
    train = make_prompts(rng)
    eval_prompts = []
    rephrasings = [("Show me a", "Could you show a"), ("Write a", "Please write a"), ("Write", "Please show")]
    for p in train:
        new_ask = p["messages"][1]["content"]
        for old, new in rephrasings:
            if old in new_ask:
                new_ask = new_ask.replace(old, new, 1)
                break
        else:
            new_ask = "Please show: " + new_ask
        eval_prompts.append({
            "id": f"eval-{p['id']}",
            "expected_language": p["expected_language"],
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": new_ask},
                {"role": "assistant", "content": "<dummy>"},
            ],
        })
    extra = [
        ("python", "Show Python code to compute fibonacci(10)"),
        ("javascript", "Show JS code that sums an array using reduce"),
        ("bash", "Show a bash command that lists files modified in the last 24 hours"),
        ("rust", "Show Rust code that prints hello world"),
    ]
    for lang, ask in extra:
        eval_prompts.append({
            "id": f"eval-extra-{lang}-{len(eval_prompts)}",
            "expected_language": lang,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": ask},
                {"role": "assistant", "content": "<dummy>"},
            ],
        })
    rng2 = random.Random(20260519)
    rng2.shuffle(eval_prompts)

    (workdir / "prompts").mkdir(exist_ok=True)
    (workdir / "datasets").mkdir(exist_ok=True)
    with (workdir / "prompts/h1-r16-2ep.jsonl").open("w") as f:
        for p in train: f.write(json.dumps(p) + "\n")
    with (workdir / "datasets/train.opd.jsonl").open("w") as f:
        for p in train: f.write(json.dumps(p) + "\n")
    with (workdir / "datasets/eval.jsonl").open("w") as f:
        for p in eval_prompts: f.write(json.dumps(p) + "\n")
    print(f"train: {len(train)} prompts")
    print(f"eval:  {len(eval_prompts)} prompts")
    print(f"languages: {len(set(p['expected_language'] for p in train))}")


if __name__ == "__main__":
    main()
