"""OpenAI Python client drop-in demo for Kiln.

Run via demo-openai.sh, which records the asciicast. The script itself just
points the official `openai` client at http://localhost:8420/v1 and streams a
chat completion — no Kiln-specific code at all.
"""
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8420/v1", api_key="kiln")

stream = client.chat.completions.create(
    model="Qwen3.5-4B",
    messages=[
        {"role": "user", "content": "In two short sentences, why does single-GPU LLM inference matter?"},
    ],
    max_tokens=96,
    temperature=0.3,
    seed=5,
    stream=True,
)

for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
print()
