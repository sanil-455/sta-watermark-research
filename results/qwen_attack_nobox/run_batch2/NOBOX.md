# No-box baseline (paraphrase-and-check-once, zero detector feedback)

Same spans, paraphraser and quality gates as the adaptive attack; the only difference is the detector gives no feedback during rewriting (query budget = 1, used only to measure the final z). Compare the broken rate here against the adaptive attack's on the same spans: a large gap is evidence that detector guidance specifically helps on this watermark.

- spans attacked: 241 | broken (z <= 2.0): 30 (12% [9%, 17%])

| span | z0 | z_final | status | sentences rewritten |
|---|---|---|---|---|
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
| samples_len200:89 | 7.21 | 5.60 | not broken | 7 of 7 |
| samples_len200:90 | 5.09 | 4.04 | not broken | 3 of 3 |
| samples_len200:91 | 3.33 | 3.33 | not broken | 0 of 0 |
| samples_len200:94 | 3.96 | 3.47 | not broken | 4 of 5 |
| samples_len200:95 | 2.40 | 1.37 | broken | 3 of 3 |
| samples_len200:97 | 4.67 | 2.22 | not broken | 3 of 4 |
| samples_len200:98 | 2.55 | 1.22 | broken | 3 of 3 |
| samples_len200:99 | 2.26 | 0.99 | broken | 3 of 3 |
| samples_len200:101 | 3.25 | 3.28 | not broken | 3 of 4 |
| samples_len200:104 | 5.52 | 4.04 | not broken | 5 of 5 |
| samples_len200:105 | 4.67 | 4.06 | not broken | 1 of 1 |
| samples_len200:106 | 3.82 | 2.92 | not broken | 4 of 5 |
| samples_len200:108 | 5.37 | 2.12 | not broken | 2 of 2 |
| samples_len200:109 | 5.52 | 3.86 | not broken | 2 of 2 |
| samples_len200:110 | 4.38 | 2.39 | not broken | 5 of 5 |
| samples_len200:112 | 5.09 | 4.12 | not broken | 1 of 1 |
| samples_len200:113 | 5.23 | 0.79 | broken | 8 of 8 |
| samples_len200:115 | 2.12 | 2.47 | not broken | 4 of 4 |
| samples_len200:116 | 3.39 | 2.09 | not broken | 4 of 4 |
| samples_len200:117 | 5.94 | 5.06 | not broken | 5 of 5 |
| samples_len200:118 | 5.66 | 3.58 | not broken | 2 of 2 |
| samples_len200:119 | 3.82 | 3.46 | not broken | 4 of 4 |
| samples_len200:122 | 5.37 | 5.03 | not broken | 1 of 1 |
| samples_len200:123 | 4.10 | 4.16 | not broken | 6 of 6 |
| samples_len200:124 | 5.85 | 4.64 | not broken | 5 of 5 |
| samples_len200:126 | 3.39 | 2.03 | not broken | 4 of 4 |
| samples_len200:127 | 3.68 | 3.33 | not broken | 5 of 5 |
| samples_len200:128 | 3.96 | 3.61 | not broken | 3 of 3 |
| samples_len200:129 | 4.24 | 2.79 | not broken | 2 of 2 |
| samples_len200:130 | 6.22 | 5.63 | not broken | 2 of 2 |
| samples_len200:131 | 3.96 | 3.90 | not broken | 3 of 4 |
| samples_len200:134 | 4.81 | 3.70 | not broken | 3 of 4 |
| samples_len200:135 | 4.81 | 3.90 | not broken | 6 of 6 |
| samples_len200:136 | 4.38 | 2.62 | not broken | 10 of 10 |
| samples_len200:137 | 4.67 | 2.95 | not broken | 4 of 4 |
| samples_len200:138 | 3.25 | 2.19 | not broken | 1 of 1 |
| samples_len200:139 | 5.37 | 4.64 | not broken | 2 of 2 |
| samples_len200:140 | 5.37 | 5.37 | not broken | 0 of 2 |
| samples_len200:141 | 4.53 | 3.51 | not broken | 8 of 8 |
| samples_len200:142 | 4.53 | 2.81 | not broken | 8 of 8 |
| samples_len200:143 | 3.96 | 3.88 | not broken | 1 of 1 |
| samples_len200:144 | 3.54 | 2.46 | not broken | 6 of 6 |
| samples_len200:146 | 4.75 | 4.10 | not broken | 4 of 4 |
| samples_len200:148 | 4.95 | 4.97 | not broken | 4 of 4 |
| samples_len200:150 | 5.09 | 4.50 | not broken | 3 of 3 |
| samples_len200:151 | 5.80 | 3.51 | not broken | 5 of 5 |
| samples_len200:152 | 4.44 | 3.11 | not broken | 4 of 4 |
| samples_len200:153 | 3.68 | 2.50 | not broken | 4 of 4 |
| samples_len200:154 | 2.97 | 3.11 | not broken | 2 of 2 |
| samples_len200:155 | 4.95 | 3.35 | not broken | 3 of 3 |
| samples_len200:157 | 2.97 | 2.06 | not broken | 2 of 2 |
| samples_len200:158 | 4.10 | 2.30 | not broken | 5 of 5 |
| samples_len200:160 | 3.11 | 3.19 | not broken | 6 of 6 |
| samples_len200:161 | 6.08 | 4.67 | not broken | 6 of 7 |
| samples_len200:162 | 2.83 | 1.01 | broken | 4 of 4 |
| samples_len200:164 | 4.67 | 2.21 | not broken | 6 of 6 |
| samples_len200:167 | 4.24 | 3.90 | not broken | 4 of 4 |
| samples_len200:168 | 5.52 | 4.96 | not broken | 4 of 4 |
| samples_len200:169 | 6.93 | 6.22 | not broken | 2 of 2 |
| samples_len200:170 | 2.55 | 1.36 | broken | 3 of 4 |
| samples_len200:171 | 2.26 | 1.51 | broken | 1 of 1 |
| samples_len200:172 | 4.38 | 3.52 | not broken | 3 of 3 |
| samples_len200:173 | 5.09 | 3.24 | not broken | 4 of 5 |
| samples_len200:174 | 4.53 | 3.11 | not broken | 6 of 6 |
| samples_len200:176 | 2.83 | 2.64 | not broken | 4 of 4 |
| samples_len200:179 | 4.95 | 3.47 | not broken | 4 of 4 |
| samples_len200:180 | 4.53 | 2.38 | not broken | 7 of 7 |
| samples_len200:183 | 3.54 | 2.98 | not broken | 2 of 2 |
| samples_len200:185 | 3.54 | 2.13 | not broken | 4 of 4 |
| samples_len200:186 | 4.67 | 4.12 | not broken | 3 of 3 |
| samples_len200:187 | 2.83 | 1.85 | broken | 2 of 2 |
| samples_len200:188 | 4.53 | 3.16 | not broken | 5 of 5 |
| samples_len200:189 | 3.68 | 2.20 | not broken | 4 of 4 |
| samples_len200:191 | 4.24 | 3.00 | not broken | 3 of 3 |
| samples_len200:192 | 6.08 | 4.74 | not broken | 2 of 2 |
| samples_len200:194 | 5.37 | 4.42 | not broken | 5 of 5 |
| samples_len200:197 | 2.55 | 2.11 | not broken | 3 of 3 |
| samples_len200:198 | 4.38 | 4.21 | not broken | 2 of 2 |
| samples_len200:199 | 6.22 | 5.37 | not broken | 3 of 3 |
| samples_len200:200 | 4.81 | 3.02 | not broken | 4 of 4 |
| samples_len200:201 | 5.09 | 3.94 | not broken | 2 of 2 |
| samples_len200:202 | 3.11 | 1.73 | broken | 3 of 3 |
| samples_len200:204 | 6.51 | 4.87 | not broken | 2 of 2 |
| samples_len200:205 | 5.37 | 2.64 | not broken | 6 of 6 |
| samples_len200:206 | 3.54 | 3.54 | not broken | 1 of 1 |
| samples_len200:207 | 5.80 | 5.17 | not broken | 3 of 3 |
| samples_len200:209 | 6.22 | 4.74 | not broken | 4 of 4 |
| samples_len200:210 | 2.97 | 2.69 | not broken | 2 of 3 |
| samples_len200:211 | 4.67 | 3.33 | not broken | 4 of 4 |
| samples_len200:212 | 3.68 | 2.83 | not broken | 1 of 1 |
| samples_len200:213 | 4.95 | 4.38 | not broken | 6 of 6 |
| samples_len200:214 | 3.39 | 3.39 | not broken | 0 of 0 |
| samples_len200:215 | 3.54 | 2.76 | not broken | 3 of 3 |
| samples_len200:216 | 3.11 | 2.19 | not broken | 3 of 3 |
| samples_len200:217 | 5.52 | 1.93 | broken | 6 of 6 |
| samples_len200:219 | 4.10 | 3.35 | not broken | 3 of 3 |
| samples_len200:220 | 4.24 | 3.06 | not broken | 4 of 4 |
| samples_len200:224 | 3.62 | 3.62 | not broken | 0 of 0 |
| samples_len200:225 | 4.81 | 3.71 | not broken | 3 of 3 |
| samples_len200:228 | 4.38 | 3.27 | not broken | 3 of 3 |
| samples_len200:229 | 4.67 | 3.17 | not broken | 4 of 4 |
| samples_len200:230 | 4.38 | 2.61 | not broken | 4 of 4 |
| samples_len200:231 | 5.23 | 3.01 | not broken | 3 of 3 |
| samples_len200:233 | 6.22 | 5.88 | not broken | 2 of 2 |
| samples_len200:234 | 3.54 | 0.21 | broken | 3 of 3 |
| samples_len200:235 | 5.37 | 4.61 | not broken | 3 of 3 |
| samples_len200:236 | 5.37 | 3.96 | not broken | 6 of 6 |
| samples_len200:237 | 5.23 | 3.57 | not broken | 2 of 2 |
| samples_len200:238 | 3.25 | 1.60 | broken | 5 of 5 |
| samples_len200:239 | 7.35 | 5.83 | not broken | 5 of 5 |
| samples_len200:240 | 3.25 | 2.40 | not broken | 2 of 2 |
| samples_len200:241 | 3.96 | 3.33 | not broken | 10 of 11 |
| samples_len200:242 | 5.23 | 3.41 | not broken | 5 of 6 |
| samples_len200:244 | 5.37 | 5.07 | not broken | 2 of 2 |
| samples_len200:245 | 5.09 | 3.21 | not broken | 3 of 3 |
| samples_len200:246 | 5.37 | 4.38 | not broken | 3 of 3 |
| samples_len200:247 | 2.26 | 1.93 | broken | 3 of 4 |
| samples_len200:248 | 4.10 | 4.10 | not broken | 0 of 0 |
| samples_len200:249 | 3.25 | 3.26 | not broken | 3 of 3 |
| samples_len200:250 | 4.53 | 4.08 | not broken | 1 of 1 |
| samples_len200:252 | 4.67 | 3.14 | not broken | 5 of 5 |
| samples_len200:253 | 3.25 | 2.47 | not broken | 5 of 5 |
| samples_len200:254 | 3.68 | 2.60 | not broken | 5 of 5 |
| samples_len200:256 | 4.95 | 3.43 | not broken | 6 of 6 |
| samples_len200:257 | 5.37 | 4.69 | not broken | 7 of 7 |
| samples_len200:259 | 6.93 | 4.97 | not broken | 4 of 4 |
| samples_len200:261 | 5.80 | 5.34 | not broken | 5 of 5 |
| samples_len200:262 | 5.37 | 2.97 | not broken | 6 of 6 |
| samples_len200:264 | 4.24 | 2.11 | not broken | 3 of 3 |
| samples_len200:265 | 6.22 | 3.51 | not broken | 4 of 4 |
| samples_len200:266 | 5.37 | 5.14 | not broken | 5 of 5 |
| samples_len200:268 | 5.37 | 4.35 | not broken | 5 of 5 |
| samples_len200:269 | 5.66 | 4.16 | not broken | 3 of 3 |
| samples_len200:270 | 3.68 | 1.92 | broken | 3 of 3 |
| samples_len200:271 | 4.53 | 2.97 | not broken | 4 of 4 |
| samples_len200:272 | 2.83 | 1.18 | broken | 5 of 5 |
| samples_len200:273 | 5.66 | 4.02 | not broken | 9 of 9 |
| samples_len200:274 | 3.82 | 2.01 | not broken | 7 of 7 |
| samples_len200:275 | 4.10 | 3.16 | not broken | 3 of 3 |
| samples_len200:276 | 4.81 | 2.65 | not broken | 3 of 3 |
| samples_len200:277 | 4.24 | 4.06 | not broken | 1 of 2 |
| samples_len200:278 | 5.80 | 4.35 | not broken | 3 of 3 |
| samples_len200:279 | 3.25 | 1.89 | broken | 6 of 6 |
| samples_len200:280 | 2.83 | 2.62 | not broken | 1 of 2 |
| samples_len200:281 | 2.69 | 2.24 | not broken | 2 of 2 |
| samples_len200:282 | 5.09 | 3.90 | not broken | 5 of 5 |
| samples_len200:285 | 5.15 | 1.84 | broken | 6 of 6 |
| samples_len200:286 | 3.39 | 1.65 | broken | 3 of 3 |
| samples_len200:288 | 4.53 | 3.26 | not broken | 3 of 3 |
| samples_len200:289 | 2.83 | 2.27 | not broken | 5 of 5 |
| samples_len200:291 | 4.95 | 2.89 | not broken | 8 of 8 |
| samples_len200:292 | 2.83 | 3.06 | not broken | 4 of 4 |
| samples_len200:294 | 3.39 | 3.13 | not broken | 5 of 5 |
| samples_len200:295 | 3.96 | 1.63 | broken | 5 of 5 |
| samples_len200:296 | 4.38 | 4.38 | not broken | 0 of 0 |
| samples_len200:297 | 2.83 | 1.45 | broken | 5 of 5 |
| samples_len200:298 | 4.67 | 3.03 | not broken | 3 of 3 |
| samples_len200:300 | 3.96 | 3.92 | not broken | 1 of 1 |
| samples_len200:301 | 5.17 | 3.63 | not broken | 9 of 9 |
| samples_len200:302 | 4.24 | 3.76 | not broken | 4 of 4 |
| samples_len200:303 | 3.82 | 3.58 | not broken | 1 of 1 |
| samples_len200:304 | 2.69 | 1.00 | broken | 5 of 5 |
| samples_len200:307 | 4.24 | 2.82 | not broken | 5 of 5 |
| samples_len200:308 | 5.23 | 3.28 | not broken | 3 of 3 |
| samples_len200:310 | 5.09 | 4.18 | not broken | 2 of 2 |
| samples_len200:311 | 4.81 | 2.27 | not broken | 5 of 5 |
| samples_len200:314 | 3.11 | 3.32 | not broken | 1 of 1 |
| samples_len200:316 | 5.66 | 3.55 | not broken | 4 of 4 |
| samples_len200:317 | 3.96 | 3.26 | not broken | 5 of 5 |
| samples_len200:318 | 3.96 | 3.43 | not broken | 10 of 10 |
| samples_len200:319 | 4.81 | 2.80 | not broken | 4 of 4 |
| samples_len200:321 | 5.09 | 3.61 | not broken | 4 of 4 |
| samples_len200:322 | 4.53 | 4.29 | not broken | 2 of 2 |
| samples_len200:323 | 2.69 | 3.51 | not broken | 5 of 5 |
| samples_len200:324 | 2.40 | 1.84 | broken | 5 of 5 |
| samples_len200:325 | 2.83 | 1.49 | broken | 5 of 5 |
| samples_len200:326 | 4.10 | 3.58 | not broken | 5 of 5 |
| samples_len200:327 | 4.81 | 4.50 | not broken | 4 of 4 |
| samples_len200:328 | 3.82 | 2.28 | not broken | 3 of 3 |
| samples_len200:329 | 3.82 | 1.87 | broken | 5 of 5 |
| samples_len200:332 | 5.23 | 3.41 | not broken | 8 of 8 |
| samples_len200:333 | 4.53 | 1.24 | broken | 4 of 4 |
| samples_len200:334 | 4.24 | 3.50 | not broken | 6 of 6 |
| samples_len200:335 | 3.82 | 3.74 | not broken | 1 of 1 |
| samples_len200:336 | 3.82 | 3.08 | not broken | 3 of 4 |
| samples_len200:337 | 5.80 | 4.88 | not broken | 3 of 3 |
| samples_len200:339 | 4.53 | 3.98 | not broken | 1 of 1 |
| samples_len200:340 | 2.40 | 1.41 | broken | 3 of 3 |
| samples_len200:341 | 5.52 | 3.94 | not broken | 4 of 4 |
| samples_len200:342 | 5.37 | 4.69 | not broken | 2 of 2 |
| samples_len200:343 | 4.53 | 2.95 | not broken | 5 of 6 |
| samples_len200:344 | 4.81 | 4.20 | not broken | 4 of 4 |
| samples_len200:345 | 4.10 | 2.83 | not broken | 2 of 2 |
| samples_len200:347 | 4.81 | 3.73 | not broken | 3 of 3 |
| samples_len200:348 | 4.67 | 3.35 | not broken | 4 of 4 |
| samples_len200:349 | 4.24 | 4.12 | not broken | 6 of 6 |
| samples_len200:350 | 4.38 | 3.05 | not broken | 3 of 3 |
| samples_len200:351 | 3.96 | 4.18 | not broken | 2 of 2 |
| samples_len200:352 | 4.38 | 3.10 | not broken | 3 of 3 |
| samples_len200:355 | 3.25 | 2.47 | not broken | 2 of 2 |
| samples_len200:356 | 3.11 | 3.25 | not broken | 1 of 1 |
| samples_len200:357 | 4.38 | 3.49 | not broken | 4 of 4 |
| samples_len200:358 | 4.95 | 2.77 | not broken | 3 of 3 |
| samples_len200:359 | 2.26 | 1.40 | broken | 5 of 5 |
| samples_len200:360 | 4.67 | 3.83 | not broken | 6 of 6 |
| samples_len200:361 | 4.81 | 3.47 | not broken | 5 of 5 |
| samples_len200:362 | 3.68 | 2.25 | not broken | 6 of 6 |
