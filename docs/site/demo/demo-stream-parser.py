#!/usr/bin/env python3
"""Tiny SSE parser for the streaming demo casts.

Reads OpenAI-compatible `text/event-stream` chunks from stdin and prints the
delta content tokens to stdout as they arrive, so the asciicast captures
real first-token latency and per-token cadence.
"""
import json
import sys

for line in sys.stdin:
    line = line.strip()
    if not line.startswith("data: "):
        continue
    payload = line[6:]
    if payload == "[DONE]":
        break
    try:
        delta = json.loads(payload)["choices"][0].get("delta", {}).get("content")
    except (json.JSONDecodeError, KeyError, IndexError):
        continue
    if delta:
        print(delta, end="", flush=True)
print()
