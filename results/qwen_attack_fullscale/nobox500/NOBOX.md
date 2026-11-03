# No-box baseline (paraphrase-and-check-once, zero detector feedback)

Same spans, paraphraser and quality gates as the adaptive attack; the only difference is the detector gives no feedback during rewriting (query budget = 1, used only to measure the final z). Compare the broken rate here against the adaptive attack's on the same spans: a large gap is evidence that detector guidance specifically helps on this watermark.

- spans attacked: 21 | broken (z <= 2.0): 1 (5% [1%, 23%])

| span | z0 | z_final | status | sentences rewritten |
|---|---|---|---|---|
| samples_len500:28 | 7.87 | 5.57 | not broken | 9 of 9 |
| samples_len500:30 | 7.33 | 5.32 | not broken | 12 of 12 |
| samples_len500:34 | 6.89 | 5.29 | not broken | 16 of 16 |
| samples_len500:43 | 7.60 | 6.31 | not broken | 8 of 11 |
| samples_len500:44 | 6.89 | 3.14 | not broken | 16 of 16 |
| samples_len500:46 | 6.17 | 4.70 | not broken | 8 of 8 |
| samples_len500:47 | 8.77 | 3.77 | not broken | 10 of 10 |
| samples_len500:51 | 6.26 | 5.26 | not broken | 6 of 6 |
| samples_len500:52 | 7.60 | 5.85 | not broken | 18 of 19 |
| samples_len500:56 | 7.69 | 4.08 | not broken | 16 of 16 |
| samples_len500:62 | 4.20 | 1.34 | broken | 9 of 9 |
| samples_len500:64 | 6.71 | 4.78 | not broken | 14 of 16 |
| samples_len500:65 | 6.44 | 6.53 | not broken | 1 of 2 |
| samples_len500:66 | 7.51 | 5.53 | not broken | 8 of 9 |
| samples_len500:69 | 6.53 | 5.69 | not broken | 13 of 13 |
| samples_len500:71 | 7.51 | 7.50 | not broken | 1 of 1 |
| samples_len500:72 | 7.92 | 5.19 | not broken | 11 of 11 |
| samples_len500:74 | 6.08 | 4.64 | not broken | 11 of 11 |
| samples_len500:78 | 8.68 | 3.89 | not broken | 15 of 15 |
| samples_len500:81 | 7.78 | 5.08 | not broken | 15 of 15 |
| samples_len500:86 | 6.80 | 5.36 | not broken | 14 of 14 |
