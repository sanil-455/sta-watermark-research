# Repair summary (strict gates)

| span | z0 | z after putting originals back | z final | result | approved | skipped | rejected |
|---|---|---|---|---|---|---|---|
| baseline_extra:22 | 4.67 | 1.78 | 1.78 | broken | 0 | 0 | 0 |
| baseline_safe:2 | 5.37 | 3.03 | 3.03 | not broken: no accepted proposal got z under 2.0 | 0 | 1 | 2 |
| baseline_safe:3 | 3.68 | 2.52 | 1.85 | broken | 1 | 0 | 0 |
| samples_len200:36 | 5.09 | 4.30 | 2.27 | not broken: no accepted proposal got z under 2.0 | 2 | 0 | 0 |
| samples_len200:42 | 3.68 | 3.68 | 1.45 | broken | 1 | 0 | 0 |
| samples_len200:43 | 6.79 | 4.06 | 4.06 | not broken: no accepted proposal got z under 2.0 | 0 | 2 | 0 |
| samples_len200:61 | 5.09 | 2.07 | 0.89 | broken | 1 | 0 | 0 |
| samples_len200:64 | 4.10 | 2.04 | 1.90 | broken | 1 | 2 | 0 |
| samples_len200:66 | 5.80 | 2.06 | 1.15 | broken | 1 | 0 | 0 |
| samples_len200:73 | 3.96 | 2.96 | 1.78 | broken | 2 | 0 | 0 |
| samples_len200:74 | 2.97 | 2.48 | 1.57 | broken | 1 | 0 | 0 |
