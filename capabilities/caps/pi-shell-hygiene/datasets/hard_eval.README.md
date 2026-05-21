# hard_eval.tasks.jsonl — pi-shell-hygiene hard-eval pool

Tasks where the 4B base is most likely to fall into the `until ssh
... kill -0` / `while sleep 5` trap.

## How to build it

1. Run base on standard eval; mark tasks where the model used a known
   bad pattern.
2. Hand-construct adversarial cases:
   - **Prompt hints at polling** — e.g. "check every N seconds"; agent
     must resist the polling phrasing and use wait-file instead.
   - **Multi-process scenario** — launch 3+ background processes with
     proper cleanup; one EXIT trap kills them all on first tool-call
     return.
   - **Mixed timing** — task has a 30-min total budget; agent must
     allocate sleeps appropriately (not all in cache window).
   - **Cross-platform** — what works on Linux differs subtly from
     macOS bash; agent should detect.

File is gitignored.
