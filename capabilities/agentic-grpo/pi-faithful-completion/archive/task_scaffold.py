"""Task-scaffold builder for pi-faithful-completion.

Each task is a single-turn text completion: the prompt embeds all needed
context directly. There is no pi tool loop in v0; what we are training is
the model's FINAL-TURN discipline (format + no-question + no-soft-punt +
honest-failure).

A task dict (produced by build_corpus.py) has these fields:
  - task_id            str
  - task_kind          str
  - format_kind        str
  - system_prompt      str
  - user_prompt        str
  - format_regex       str         re pattern with one capture group
  - expected_value     str | None  None ⇒ failure task
  - value_normalizer   str         key into rubric.VALUE_NORMALIZERS
  - is_failure         bool
"""

from __future__ import annotations

from typing import Any


DEFAULT_SYSTEM_PROMPT = (
    "You are a precise, autonomous task-execution assistant. You must:\n"
    "  - Execute the task without asking the user for clarification.\n"
    "  - Provide a single final answer in the EXACT OUTPUT FORMAT requested.\n"
    "  - If the task cannot be completed, output the literal token "
    "`precondition_failed: <reason>` and stop.\n"
    "  - Never ask questions, never soft-punt (\"let me know\", \"should I\"), "
    "never defer to the user.\n"
    "  - Keep responses terse. The final OUTPUT FORMAT line is what matters."
)


def build_messages(task: dict[str, Any]) -> list[dict[str, str]]:
    """Build the chat-completion messages for kiln from a task spec."""
    return [
        {"role": "system", "content": task.get("system_prompt", DEFAULT_SYSTEM_PROMPT)},
        {"role": "user", "content": task["user_prompt"]},
    ]
