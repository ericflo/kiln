# H35 H33 Confirmation

## Hypothesis

H33 cleared one `LIMIT=8` larger gate, but H34's failed chain made it important
to check whether H33 itself is stable. H35 reruns the H33 adapter on the same
blind aggregate size with fresh stochastic rollouts.

No training is performed in this iteration. This is a confirmation gate for the
existing adapter `pi-doctest-h33-hardneg-g2-noecho-r4a8`.

## Eval

`LIMIT=8 SEEDS=1`, paired against `/tmp/pi-doctest-h19-promo-base8.json`.

| metric | base8 | H33 original gate | H35 fresh H33 |
| --- | ---: | ---: | ---: |
| composite | 0.832812 | 0.892188 | 0.721875 |
| delta vs base8 | | +0.059375 | -0.110937 |
| outcome | 0.875000 | 1.000000 | 0.750000 |
| tested_before_done | 1.000000 | 1.000000 | 1.000000 |
| format_compliance | 1.000000 | 1.000000 | 1.000000 |
| tool_call_efficiency | 0.781250 | 0.640625 | 0.843750 |
| mean tool calls | 5.25 | 6.75 | 4.625 |
| mean thinking chars | 3486.0 | 3313.5 | 3096.25 |
| zero rollouts | 1 | 0 | 2 |
| mean wall-clock s | 49.62 | 62.65 | 50.20 |

## Verdict

Rejected as a confirmation. H33's original larger-gate composite gain did not
reproduce. The fresh run was more efficient by tool calls, but lost too much
outcome reliability and produced two zero rollouts.

Lessons:

- H33 should be treated as an unstable candidate, not a shipped stage.
- The apparent outcome-reliability gain from one `LIMIT=8` run is within the
  current eval variance for this adapter family.
- Do not chain further from H33 until it passes a stronger confirmation gate.
  Future experiments should either use broader data from base or run a
  multi-seed promotion gate before any chain stage.
