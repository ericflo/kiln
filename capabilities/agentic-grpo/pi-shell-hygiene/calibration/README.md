# calibration/ — pi-shell-hygiene rubric sanity fixtures

Paired good/bad shell patterns from the clouderic kiln-skill
anti-pattern doc.

## What "good" looks like

| Pattern | Why good |
|---------|----------|
| `nohup <cmd> &` + `wait-file --timeout` | Proper background launch with bounded wait |
| `sleep 270 && check` | Single substantial sleep instead of polling loop |
| `trap 'cleanup' ERR INT TERM` | Cleanup on failure (not EXIT) |
| `timeout 1200 <cmd>` | Bounded wall-clock |

## What "bad" looks like — kiln-skill anti-patterns

| Anti-pattern | bad.jsonl id |
|--------------|--------------|
| `until ssh <pod> 'test -f /tmp/done'; do sleep 5; done` | `calib_bad_until_ssh` |
| `sleep 5;` polling | `calib_bad_short_sleep_poll` |
| `trap 'cleanup' EXIT` (kills pod on every tool-call shell exit) | `calib_bad_exit_trap` |
| `while [ ! -f /tmp/done ]; do sleep 5; done` | `calib_bad_while_poll` |
| No timeout on background process | `calib_bad_no_timeout` |

## Refreshing

After changing `../rubric.py`, run `python3 ../rubric_sanity.py`.

## Current calibration state

  good min=0.83, max=1.00
  bad  min=0.00, max=0.63
  separation: +0.20 (at margin; consider tightening if a real iter
              produces a regression here)
