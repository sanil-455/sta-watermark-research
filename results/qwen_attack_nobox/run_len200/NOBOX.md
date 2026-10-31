# No-box baseline (paraphrase-and-check-once, zero detector feedback)

Same spans, paraphraser and quality gates as the adaptive attack; the only difference is the detector gives no feedback during rewriting (query budget = 1, used only to measure the final z). Compare the broken rate here against the adaptive attack's on the same spans: a large gap is evidence that detector guidance specifically helps on this watermark.

- spans attacked: 46 | broken (z <= 2.0): 4 (9% [3%, 20%])

| span | z0 | z_final | status | sentences rewritten |
|---|---|---|---|---|
| baseline_safe:0 | 3.82 | 3.92 | not broken | 2 of 2 |
| baseline_safe:1 | 6.51 | 5.63 | not broken | 3 of 4 |
| baseline_safe:2 | 5.37 | 4.63 | not broken | 3 of 3 |
| baseline_safe:3 | 3.68 | 2.85 | not broken | 3 of 3 |
| baseline_safe:4 | 3.54 | 1.86 | broken | 5 of 6 |
| baseline_extra:20 | 3.39 | 2.49 | not broken | 2 of 2 |
| baseline_extra:22 | 4.67 | 3.90 | not broken | 6 of 6 |
| baseline_extra:23 | 5.23 | 4.46 | not broken | 4 of 4 |
| baseline_extra:24 | 5.09 | 3.92 | not broken | 4 of 4 |
| baseline_extra:25 | 3.82 | 3.52 | not broken | 2 of 2 |
| baseline_extra:26 | 2.83 | 2.44 | not broken | 4 of 4 |
| baseline_extra:27 | 6.22 | 4.72 | not broken | 6 of 6 |
| samples_len200:28 | 5.37 | 3.63 | not broken | 5 of 5 |
| samples_len200:29 | 4.81 | 3.69 | not broken | 4 of 5 |
| samples_len200:30 | 4.10 | 2.47 | not broken | 3 of 3 |
| samples_len200:31 | 5.09 | 4.10 | not broken | 3 of 3 |
| samples_len200:34 | 3.25 | 3.49 | not broken | 5 of 5 |
| samples_len200:35 | 5.94 | 3.32 | not broken | 8 of 8 |
| samples_len200:36 | 5.09 | 5.08 | not broken | 3 of 3 |
| samples_len200:40 | 4.10 | 4.10 | not broken | 0 of 1 |
| samples_len200:42 | 3.68 | 2.03 | not broken | 1 of 1 |
| samples_len200:43 | 6.79 | 5.17 | not broken | 4 of 5 |
| samples_len200:44 | 5.09 | 1.90 | broken | 6 of 6 |
| samples_len200:46 | 2.12 | 2.01 | not broken | 3 of 3 |
| samples_len200:47 | 5.52 | 2.67 | not broken | 4 of 4 |
| samples_len200:51 | 3.39 | 3.43 | not broken | 1 of 1 |
| samples_len200:52 | 5.23 | 3.22 | not broken | 10 of 10 |
| samples_len200:56 | 3.54 | 2.17 | not broken | 6 of 6 |
| samples_len200:60 | 5.52 | 4.24 | not broken | 3 of 3 |
| samples_len200:61 | 5.09 | 3.90 | not broken | 5 of 5 |
| samples_len200:63 | 6.65 | 5.69 | not broken | 4 of 4 |
| samples_len200:64 | 4.10 | 3.96 | not broken | 5 of 6 |
| samples_len200:65 | 2.26 | 2.40 | not broken | 1 of 2 |
| samples_len200:66 | 5.80 | 3.84 | not broken | 4 of 4 |
| samples_len200:69 | 3.11 | 2.27 | not broken | 4 of 4 |
| samples_len200:70 | 4.67 | 4.08 | not broken | 2 of 2 |
| samples_len200:71 | 4.95 | 4.95 | not broken | 0 of 0 |
| samples_len200:72 | 5.23 | 4.78 | not broken | 3 of 3 |
| samples_len200:73 | 3.96 | 2.76 | not broken | 4 of 4 |
| samples_len200:74 | 2.97 | 1.95 | broken | 4 of 4 |
| samples_len200:78 | 4.95 | 2.98 | not broken | 6 of 6 |
| samples_len200:81 | 5.23 | 3.59 | not broken | 6 of 6 |
| samples_len200:84 | 2.55 | 1.29 | broken | 8 of 8 |
| samples_len200:85 | 3.96 | 2.86 | not broken | 5 of 6 |
| samples_len200:86 | 3.39 | 2.97 | not broken | 5 of 5 |
| samples_len200:87 | 4.81 | 4.00 | not broken | 3 of 3 |
