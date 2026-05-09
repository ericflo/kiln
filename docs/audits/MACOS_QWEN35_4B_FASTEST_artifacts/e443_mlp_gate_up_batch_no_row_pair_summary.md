E443 MLP gate/up row-pair counter-order comparison

Same-source Qwen3.5 MLP gate/up decode-batch bench, default versus
`KILN_DISABLE_METAL_MLP_GATE_UP_ROW_PAIR=1`.

| batch | default1 | no row pair1 | no row pair2 | default2 | verdict |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `1676.208` | `1624.977` | `1801.737` | `1654.338` | env does not change policy; noise |
| 2 | `1953.688` | `1889.987` | `1962.617` | `1964.735` | no source change; tie/noise |
| 3 | `1973.442` | `3128.746` | `3119.140` | `1917.848` | row-pair clearly favored |
| 4 | `2012.179` | `3598.829` | `3593.256` | `2025.869` | row-pair clearly favored |
| 8 | `2368.365` | `6895.800` | `6786.956` | `2228.929` | row-pair clearly favored |

Batch 2 remained the only tempting rowwise case, but the repeat reduced the
margin to a tie. Keep the current row-pair selector; do not add a rows==2
exception from this evidence.
