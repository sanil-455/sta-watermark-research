# No-box baseline (paraphrase-and-check-once, zero detector feedback)

Same spans, paraphraser and quality gates as the adaptive attack; the only difference is the detector gives no feedback during rewriting (query budget = 1, used only to measure the final z). Compare the broken rate here against the adaptive attack's on the same spans: a large gap is evidence that detector guidance specifically helps on this watermark.

- spans attacked: 53 | broken (z <= 2.0): 0 (0% [-0%, 7%])

| span | z0 | z_final | status | sentences rewritten |
|---|---|---|---|---|
| sta16_len500:29 | 16.16 | 15.65 | not broken | 1 of 1 |
| sta16_len500:31 | 15.56 | 12.46 | not broken | 10 of 11 |
| sta16_len500:32 | 18.07 | 16.93 | not broken | 3 of 3 |
| sta16_len500:33 | 17.35 | 13.05 | not broken | 12 of 13 |
| sta16_len500:35 | 16.64 | 8.09 | not broken | 14 of 14 |
| sta16_len500:37 | 17.44 | 11.34 | not broken | 16 of 17 |
| sta16_len500:44 | 15.65 | 9.53 | not broken | 10 of 10 |
| sta16_len500:47 | 16.55 | 9.60 | not broken | 12 of 13 |
| sta16_len500:50 | 19.05 | 12.46 | not broken | 8 of 9 |
| sta16_len500:51 | 14.13 | 11.58 | not broken | 4 of 4 |
| sta16_len500:54 | 19.05 | 14.32 | not broken | 11 of 13 |
| sta16_len500:55 | 15.95 | 11.42 | not broken | 12 of 12 |
| sta16_len500:57 | 14.49 | 11.48 | not broken | 7 of 7 |
| sta16_len500:59 | 14.94 | 12.14 | not broken | 7 of 7 |
| sta16_len500:60 | 16.10 | 11.20 | not broken | 7 of 7 |
| sta16_len500:61 | 16.43 | 10.27 | not broken | 16 of 16 |
| sta16_len500:72 | 14.64 | 13.35 | not broken | 4 of 4 |
| sta16_len500:74 | 14.85 | 13.06 | not broken | 6 of 6 |
| sta16_len500:75 | 16.64 | 14.91 | not broken | 10 of 10 |
| sta16_len500:76 | 15.03 | 11.74 | not broken | 4 of 4 |
| sta16_len500:78 | 16.64 | 12.23 | not broken | 17 of 17 |
| sta16_len500:80 | 18.78 | 8.09 | not broken | 19 of 19 |
| sta16_len500:86 | 16.70 | 9.95 | not broken | 16 of 16 |
| sta16_len500:89 | 15.98 | 10.04 | not broken | 15 of 16 |
| sta16_len500:97 | 15.74 | 11.72 | not broken | 9 of 9 |
| sta16_len500:100 | 16.90 | 13.68 | not broken | 4 of 4 |
| sta16_len500:102 | 18.69 | 13.40 | not broken | 13 of 13 |
| sta16_len500:104 | 16.19 | 11.06 | not broken | 7 of 7 |
| sta16_len500:106 | 16.84 | 8.78 | not broken | 13 of 13 |
| sta16_len500:107 | 13.86 | 7.69 | not broken | 10 of 10 |
| sta16_len500:108 | 15.74 | 9.38 | not broken | 8 of 8 |
| sta16_len500:109 | 15.83 | 13.87 | not broken | 5 of 5 |
| sta16_len500:110 | 17.89 | 11.44 | not broken | 14 of 14 |
| sta16_len500:114 | 15.65 | 13.33 | not broken | 9 of 9 |
| sta16_len500:115 | 14.76 | 13.98 | not broken | 3 of 3 |
| sta16_len500:120 | 14.85 | 10.31 | not broken | 16 of 17 |
| sta16_len500:123 | 14.76 | 12.09 | not broken | 8 of 9 |
| sta16_len500:125 | 17.62 | 13.67 | not broken | 5 of 5 |
| sta16_len500:128 | 16.28 | 13.12 | not broken | 6 of 6 |
| sta16_len500:132 | 11.72 | 8.39 | not broken | 10 of 10 |
| sta16_len500:134 | 18.78 | 13.93 | not broken | 10 of 10 |
| sta16_len500:135 | 17.80 | 11.45 | not broken | 9 of 9 |
| sta16_len500:136 | 18.25 | 10.17 | not broken | 16 of 16 |
| sta16_len500:138 | 15.65 | 14.10 | not broken | 2 of 2 |
| sta16_len500:139 | 16.28 | 16.01 | not broken | 1 of 1 |
| sta16_len500:140 | 18.78 | 18.78 | not broken | 0 of 0 |
| sta16_len500:141 | 18.87 | 9.21 | not broken | 15 of 15 |
| sta16_len500:143 | 15.29 | 9.99 | not broken | 8 of 8 |
| sta16_len500:144 | 15.83 | 12.48 | not broken | 10 of 10 |
| sta16_len500:148 | 17.08 | 13.68 | not broken | 14 of 14 |
| sta16_len500:152 | 16.16 | 11.49 | not broken | 13 of 14 |
| sta16_len500:154 | 16.10 | 13.12 | not broken | 8 of 8 |
| sta16_len500:156 | 17.71 | 11.11 | not broken | 20 of 20 |
