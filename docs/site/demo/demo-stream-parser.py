#!/usr/bin/env python3
"""Tiny SSE parser for the streaming demo casts.

Reads OpenAI-compatible `text/event-stream` chunks from stdin and prints the
delta content tokens to stdout as they arrive, so the asciicast captures
real first-token latency and per-token cadence.

Supports reasoning models by also outputting reasoning_content tokens with
appropriate transitions (mode switches get "\n\n", reasoning gets a "Reasoning: " prefix).
"""
import json
import sys

mode = "init"  # one of: init, reasoning, content

for line in sys.stdin:
    line = line.strip()
    if not line.startswith("data: "):
        continue
    payload = line[6:]
    if payload == "[DONE]":
        break
    try:
        delta = json.loads(payload)["choices"][0].get("delta", {})
    except (json.JSONDecodeError, KeyError, IndexError):
        continue

    reasoning = delta.get("reasoning_content")
    content = delta.get("content")

    if reasoning:
        if mode == "init":
            print("Reasoning: ", end="", flush=True)
        elif mode == "content":
            print("\n\nReasoning: ", end="", flush=True)
        # mode == "reasoning" — no prefix, just append tokens
        print(reasoning, end="", flush=True)
        mode = "reasoning"

    if content:
        if mode == "reasoning":
            print("\n\n", end="", flush=True)
        # mode == "init" or "content" — no prefix, just append tokens
        print(content, end="", flush=True)
        mode = "content"

print()
