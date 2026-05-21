# hard_eval.tasks.jsonl — pi-test-interpretation hard-eval pool

Tasks where the 4B is most likely to confuse warmup with real signal
or mean with median.

## How to build it

1. Run base; mark tasks where the model reported mean or first-run
   only.
2. Hand-construct adversarial:
   - **Subtle warmup** — first run 5% slower than 2-5; tempting to
     report it as the "median".
   - **Bimodal distribution** — runs cluster at 10ms and 20ms;
     mean misleads; median picks one mode.
   - **Real-but-rare-fail** — test fails 1/10 runs; agent must
     classify as "intermittent" not "flake" or "pass".
   - **Cargo+pytest mixed** — different runners output different
     summary formats; agent must normalize.

File is gitignored.
