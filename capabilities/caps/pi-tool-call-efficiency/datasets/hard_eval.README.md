# hard_eval.tasks.jsonl — pi-tool-call-efficiency hard-eval pool

NOTE: eval-only cap. The "hard-eval pool" here means tasks where
multi-step tool use is unavoidable, so any reduction in n_tool_calls
is a real efficiency gain (vs. shallow tasks where 0-calls is trivially
optimal).

## How to build it

1. Sample tasks from sister caps' eval pools where the base model used
   >=8 tool calls (the wasteful zone). These are the high-leverage
   tasks for measurement.
2. Avoid tasks where 0-1 tool calls is the optimum (those don't
   discriminate).

File is gitignored.
