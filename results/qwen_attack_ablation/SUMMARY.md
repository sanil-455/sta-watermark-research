# Ablation comparison (items 5, 6, 8): 12 spans, same subset across every config

Each config's own internal break threshold is shown where it differs from the paper's default (2.0); a span's "broken" status always means z_final <= that config's own target.

| config | finished | broken | avg z_final | avg queries | avg rewrites |
|---|---|---|---|---|---|
| default (original run) | 12 of 12 | 8 | 1.93 | 126 | 3.0 |
| n_cands=20 | 12 of 12 | 8 | 2.13 | 57 | 3.1 |
| seed=777 | 12 of 12 | 8 | 1.88 | 145 | 2.8 |
| n_cands=90 | 12 of 12 | 9 | 1.86 | 181 | 2.9 |
| preset=strict | 12 of 12 | 6 | 2.22 | 107 | 3.0 |
| target z<=1.5 (target 1.5) | 12 of 12 | 7 | 1.76 | 159 | 3.1 |

## Per span (only spans every finished config has reached)

| span | default (original run) | n_cands=20 | seed=777 | n_cands=90 | preset=strict | target z<=1.5 |
|---|---|---|---|---|---|---|
| baseline_extra:20 | z 1.90 (broken, 45q) | z 1.97 (broken, 34q) | z 1.84 (broken, 46q) | z 1.62 (broken, 60q) | z 1.76 (broken, 31q) | z 1.61 (stalled, 171q) |
| baseline_extra:22 | z 1.35 (broken, 97q) | z 1.92 (broken, 33q) | z 1.71 (broken, 100q) | z 1.22 (broken, 137q) | z 2.09 (stalled, 187q) | z 1.35 (broken, 97q) |
| baseline_extra:23 | z 2.78 (stalled, 203q) | z 3.02 (stalled, 53q) | z 2.10 (stalled, 248q) | z 2.65 (stalled, 290q) | z 3.17 (stalled, 226q) | z 2.78 (stalled, 203q) |
| baseline_extra:24 | z 0.46 (broken, 189q) | z 1.29 (broken, 66q) | z 0.89 (broken, 184q) | z 1.99 (broken, 199q) | z 1.28 (broken, 120q) | z 0.46 (broken, 189q) |
| baseline_extra:25 | z 2.60 (stalled, 95q) | z 2.67 (stalled, 38q) | z 2.32 (stalled, 156q) | z 2.60 (stalled, 122q) | z 2.80 (stalled, 84q) | z 2.60 (stalled, 95q) |
| baseline_extra:26 | z 1.13 (broken, 54q) | z 1.35 (broken, 20q) | z 1.34 (broken, 53q) | z 1.13 (broken, 79q) | z 1.13 (broken, 41q) | z 1.13 (broken, 54q) |
| baseline_extra:27 | z 1.74 (broken, 186q) | z 1.54 (broken, 127q) | z 1.99 (broken, 148q) | z 1.68 (broken, 257q) | z 1.81 (broken, 155q) | z 1.15 (broken, 322q) |
| baseline_safe:0 | z 2.56 (stalled, 167q) | z 2.70 (stalled, 100q) | z 2.56 (stalled, 258q) | z 2.56 (stalled, 231q) | z 2.83 (stalled, 121q) | z 2.56 (stalled, 167q) |
| baseline_safe:1 | z 3.41 (stalled, 205q) | z 3.54 (stalled, 74q) | z 3.05 (stalled, 283q) | z 1.70 (broken, 400q) | z 3.60 (stalled, 100q) | z 3.41 (stalled, 205q) |
| baseline_safe:2 | z 1.56 (broken, 140q) | z 1.86 (broken, 52q) | z 1.37 (broken, 136q) | z 1.56 (broken, 206q) | z 2.49 (stalled, 129q) | z 1.48 (broken, 195q) |
| baseline_safe:3 | z 1.72 (broken, 71q) | z 1.93 (broken, 35q) | z 1.77 (broken, 74q) | z 1.72 (broken, 97q) | z 1.85 (broken, 47q) | z 1.32 (broken, 115q) |
| baseline_safe:4 | z 1.92 (broken, 61q) | z 1.77 (broken, 47q) | z 1.62 (broken, 54q) | z 1.92 (broken, 89q) | z 1.80 (broken, 48q) | z 1.31 (broken, 90q) |
