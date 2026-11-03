# Qwen paraphrase attack on the STA watermark: final report

Run: `adaptive500`. Started 2026-11-02 09:05:42, ended 2026-11-03 08:14:38. Git commit `ad8f7e292088dcb1011d5c6e2189a697e8b5d8cc`.

**What was attacked.** Only the generated span of each sample, never the prompt. The detector was asked only about span text. **Success** means the span's z-score is 2.0 or below under the same detector. Preset: **lenient**. Quality is judged by reading (see READING.md); the gates below are filters, not proof of faithfulness.

## 1. Data

- spans loaded: 135 | flagged by the detector (z0 > 2): 135 | not flagged: 0
- not flagged (not attacked): none
- 200-token spans: 0 loaded, 0 flagged
- 500-token spans: 135 loaded, 135 flagged
- generation run: 250 kept, 61 stopped before 195 tokens, 24 skipped for prompt length, 0 out-of-memory events. Both span sets are selections (the model had to keep generating).
- the 500-token spans are extensions of the 200-token spans of the same prompts, so the two groups are never pooled.

## 2. Models and environment

- **Generator of the watermarked text:** Llama-2-7B base, 6,738,415,616 parameters (computed from its published configuration, not re-measured in this run), fp16
- **Paraphraser:** Qwen/Qwen2.5-7B-Instruct, revision `a09a35458c702b33eeacc393d103063234e8bc28`, **7,615,616,512 parameters** (computed from the architecture; transformers reports 7,615,616,512; agree: True), 28 layers, hidden size 3584, 28 attention heads (4 key/value), vocabulary 152064, context 32768 tokens
  - 4-bit nf4 with double quantisation, float16 compute; stored elements by dtype: {'torch.bfloat16': 1090328064, 'torch.uint8': 3262644224}; model memory footprint 5.07 GiB; GPU allocated after load 5.26 GiB; load time 5.6 s
- **Similarity model:** sentence-transformers/all-MiniLM-L6-v2, 22,713,216 parameters
- **Detector:** the STA detector of the paper, queried as a black box that returns z-scores; the key is never used by the attack
- GPU: Quadro RTX 5000 | python 3.11.16 | torch 2.7.1+cu128 | transformers 5.17.0 | bitsandbytes 0.50.2 | accelerate 1.15.0 | WordNet available: True
- peak GPU memory during the run: allocated 5.54 GiB, reserved 5.83 GiB

## 3. Settings

- gates (lenient): {'semantic': 0.75, 'fluency': 1.8, 'min_for_fluency': 3, 'spelling': True, 'max_unknown': 0, 'check_numbers': True, 'check_quotes': True, 'check_negations': True, 'check_names': False}
- sampling: {'temperature': 1.5, 'top_p': 0.98, 'top_k': 20, 'repetition_penalty': 1.05, 'batch': 4, 'length_factor': 2.0, 'extra_tokens': 32}
- attack: {'n_cands': 60, 'seed': 0, 'max_minutes': 0} | segmentation: {'min_chars': 30, 'quote_words': 4, 'abbreviations': 50}

## 4. Results (never pooled across lengths)

Columns marked mean / median / max.

### 500-token spans: 134 attacked, **100 broken (75%)**

- all attacked (n=134): z0 7.3 / 7.2 / 10.4 | z final 2.3 / 1.9 / 7.9 | sentences rewritten % of eligible 75.8 / 77.8 / 100.0 | tokens not preserved % 23.4 / 23.1 / 47.1 | tokens that are new % 22.8 / 23.0 / 42.8 | words not preserved % 29.5 / 29.3 / 54.2 | characters not preserved % 18.6 / 17.9 / 41.8 | length change (tokens) -4.2 / -2.0 / 41.0 | queries 543.2 / 354.0 / 2422.0 | minutes 10.5 / 6.1 / 45.3
- broken only (n=100): z0 7.1 / 7.0 / 9.8 | z final 1.8 / 1.8 / 2.0 | sentences rewritten % of eligible 68.9 / 66.7 / 100.0 | tokens not preserved % 25.6 / 24.7 / 47.1 | tokens that are new % 24.8 / 24.0 / 42.8 | words not preserved % 32.3 / 31.3 / 54.2 | characters not preserved % 20.2 / 19.2 / 41.8 | length change (tokens) -5.6 / -3.5 / 41.0 | queries 376.1 / 311.0 / 2422.0 | minutes 7.0 / 5.8 / 34.9
- outcomes: {'broken': 100, 'improved but not enough (stalled)': 34}

## 5. Per-sample table

tokens = span length; eligible = share of characters that may be rewritten; needed = share that must be rewritten at chance level (1 - 2/z0); delta needed = selection gain (how much less green than chance the rewrites must be) if eligible < needed; delta achieved = 0.5 - green fraction of the new token pairs. Percentages of tokens/characters are of the ORIGINAL span (not preserved) and of the FINAL span (new).

| sample | tokens | z0 | eligible | needed | delta needed | outcome | z final | rewritten | tokens not kept % | tokens new % | chars not kept % | length change | queries | min | delta achieved |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| samples_len500:104 | 500 | 8.68 | 61% | 77% | 0.05 | broken | 1.99 | 11 of 11 (100%) | 25.3 | 26.7 | 20.8 | +9 | 2422 | 34.9 | 0.20 |
| samples_len500:108 | 500 | 8.94 | 86% | 78% | 0.00 | broken | 1.20 | 7 of 9 (78%) | 34.3 | 30.1 | 27.1 | -30 | 264 | 6.9 | 0.14 |
| samples_len500:116 | 500 | 5.63 | 84% | 65% | 0.00 | broken | 1.72 | 5 of 11 (45%) | 21.4 | 19.1 | 17.9 | -14 | 206 | 4.5 | 0.11 |
| samples_len500:118 | 500 | 6.89 | 87% | 71% | 0.00 | broken | 1.67 | 5 of 10 (50%) | 27.1 | 21.5 | 19.4 | -36 | 141 | 4.9 | 0.12 |
| samples_len500:120 | 500 | 3.22 | 82% | 38% | 0.00 | broken | 1.64 | 2 of 15 (13%) | 6.4 | 8.0 | 4.7 | +9 | 96 | 1.4 | 0.14 |
| samples_len500:122 | 500 | 10.29 | 27% | 81% | 0.46 | stalled (improved but not enough (stalled)) | 7.92 | 8 of 8 (100%) | 9.6 | 9.4 | 5.4 | -1 | 250 | 8.0 | 0.13 |
| samples_len500:126 | 500 | 5.99 | 82% | 67% | 0.00 | broken | 1.98 | 9 of 13 (69%) | 15.8 | 18.5 | 16.0 | +17 | 361 | 5.9 | 0.21 |
| samples_len500:127 | 500 | 5.72 | 61% | 65% | 0.01 | broken | 1.73 | 6 of 10 (60%) | 16.4 | 17.5 | 11.6 | +7 | 260 | 3.7 | 0.16 |
| samples_len500:129 | 500 | 7.87 | 54% | 75% | 0.07 | stalled (improved but not enough (stalled)) | 2.66 | 7 of 7 (100%) | 22.0 | 20.9 | 17.5 | -7 | 1516 | 34.4 | 0.13 |
| samples_len500:134 | 500 | 8.85 | 66% | 77% | 0.03 | stalled (improved but not enough (stalled)) | 2.62 | 8 of 9 (89%) | 22.8 | 21.2 | 17.4 | -10 | 2194 | 45.3 | 0.19 |
| samples_len500:135 | 500 | 8.68 | 96% | 77% | 0.00 | broken | 1.74 | 14 of 19 (74%) | 31.9 | 32.6 | 24.8 | +5 | 384 | 8.1 | 0.10 |
| samples_len500:139 | 500 | 8.41 | 29% | 76% | 0.30 | stalled (improved but not enough (stalled)) | 5.92 | 3 of 3 (100%) | 15.8 | 15.4 | 13.7 | -2 | 426 | 6.7 | 0.17 |
| samples_len500:140 | 500 | 8.05 | 30% | 75% | 0.27 | stalled (improved but not enough (stalled)) | 6.75 | 3 of 5 (60%) | 4.0 | 4.2 | 3.1 | +1 | 282 | 11.0 | 0.19 |
| samples_len500:143 | 500 | 5.72 | 70% | 65% | 0.00 | broken | 1.75 | 9 of 9 (100%) | 22.0 | 21.2 | 15.6 | -5 | 292 | 6.0 | 0.11 |
| samples_len500:144 | 500 | 6.98 | 82% | 71% | 0.00 | broken | 1.77 | 9 of 19 (47%) | 22.6 | 20.2 | 16.6 | -15 | 326 | 4.8 | 0.18 |
| samples_len500:148 | 500 | 8.85 | 86% | 77% | 0.00 | broken | 1.98 | 13 of 15 (87%) | 29.5 | 28.7 | 21.2 | -6 | 561 | 10.6 | 0.15 |
| samples_len500:150 | 500 | 7.24 | 87% | 72% | 0.00 | broken | 1.97 | 7 of 11 (64%) | 23.2 | 22.8 | 19.5 | -2 | 299 | 5.8 | 0.09 |
| samples_len500:151 | 500 | 8.05 | 91% | 75% | 0.00 | broken | 1.65 | 7 of 13 (54%) | 21.4 | 21.5 | 16.8 | +1 | 340 | 5.5 | 0.22 |
| samples_len500:153 | 500 | 6.44 | 83% | 69% | 0.00 | broken | 1.55 | 7 of 15 (47%) | 19.8 | 21.5 | 14.2 | +11 | 310 | 4.5 | 0.16 |
| samples_len500:154 | 500 | 8.14 | 51% | 75% | 0.09 | stalled (improved but not enough (stalled)) | 2.19 | 6 of 6 (100%) | 25.3 | 25.5 | 19.9 | +1 | 1102 | 29.8 | 0.15 |
| samples_len500:157 | 500 | 6.80 | 67% | 71% | 0.01 | broken | 1.72 | 8 of 8 (100%) | 24.4 | 22.8 | 18.1 | -10 | 337 | 5.7 | 0.15 |
| samples_len500:169 | 500 | 8.77 | 46% | 77% | 0.13 | stalled (improved but not enough (stalled)) | 3.76 | 7 of 7 (100%) | 15.4 | 17.2 | 12.4 | +11 | 1296 | 28.8 | 0.19 |
| samples_len500:172 | 500 | 7.33 | 76% | 73% | 0.00 | broken | 1.75 | 9 of 9 (100%) | 20.2 | 19.4 | 16.4 | -5 | 317 | 6.1 | 0.20 |
| samples_len500:173 | 500 | 6.17 | 88% | 68% | 0.00 | broken | 1.96 | 8 of 14 (57%) | 21.4 | 22.3 | 14.4 | +6 | 303 | 6.0 | 0.13 |
| samples_len500:174 | 500 | 6.53 | 65% | 69% | 0.01 | broken | 1.77 | 9 of 10 (90%) | 22.6 | 24.4 | 14.6 | +12 | 261 | 5.5 | 0.14 |
| samples_len500:176 | 500 | 6.35 | 81% | 69% | 0.00 | broken | 1.91 | 8 of 11 (73%) | 23.4 | 24.1 | 18.1 | +5 | 369 | 6.0 | 0.16 |
| samples_len500:179 | 500 | 8.94 | 93% | 78% | 0.00 | broken | 1.60 | 11 of 13 (85%) | 31.7 | 32.3 | 29.4 | +4 | 322 | 8.2 | 0.13 |
| samples_len500:180 | 500 | 8.23 | 90% | 76% | 0.00 | broken | 1.52 | 12 of 17 (71%) | 28.7 | 28.7 | 24.3 | +0 | 437 | 6.8 | 0.15 |
| samples_len500:185 | 500 | 6.08 | 89% | 67% | 0.00 | broken | 1.95 | 6 of 12 (50%) | 20.8 | 22.0 | 15.5 | +8 | 235 | 5.0 | 0.13 |
| samples_len500:187 | 500 | 4.56 | 91% | 56% | 0.00 | broken | 1.55 | 3 of 6 (50%) | 22.0 | 19.0 | 17.2 | -18 | 96 | 5.3 | 0.07 |
| samples_len500:188 | 500 | 6.98 | 83% | 71% | 0.00 | broken | 1.87 | 7 of 11 (64%) | 25.7 | 23.1 | 21.6 | -17 | 210 | 5.2 | 0.11 |
| samples_len500:189 | 500 | 6.44 | 54% | 69% | 0.04 | stalled (improved but not enough (stalled)) | 2.59 | 7 of 7 (100%) | 18.8 | 19.1 | 15.9 | +2 | 1582 | 28.6 | 0.14 |
| samples_len500:191 | 500 | 6.08 | 60% | 67% | 0.02 | broken | 1.48 | 7 of 9 (78%) | 18.8 | 18.3 | 17.5 | -3 | 282 | 4.6 | 0.18 |
| samples_len500:192 | 500 | 8.68 | 45% | 77% | 0.14 | stalled (improved but not enough (stalled)) | 4.64 | 5 of 5 (100%) | 18.6 | 15.9 | 13.7 | -16 | 803 | 19.0 | 0.18 |
| samples_len500:194 | 500 | 9.39 | 76% | 79% | 0.01 | broken | 2.00 | 14 of 14 (100%) | 26.7 | 30.9 | 19.3 | +30 | 878 | 20.3 | 0.15 |
| samples_len500:197 | 500 | 6.35 | 61% | 69% | 0.02 | broken | 1.77 | 8 of 10 (80%) | 25.0 | 23.0 | 17.0 | -13 | 360 | 4.8 | 0.13 |
| samples_len500:198 | 500 | 7.33 | 84% | 73% | 0.00 | broken | 1.86 | 8 of 8 (100%) | 47.1 | 42.8 | 41.8 | -38 | 654 | 13.9 | 0.04 |
| samples_len500:199 | 500 | 9.12 | 88% | 78% | 0.00 | broken | 1.94 | 10 of 11 (91%) | 32.7 | 31.2 | 22.6 | -11 | 414 | 7.2 | 0.12 |
| samples_len500:200 | 500 | 7.96 | 58% | 75% | 0.05 | stalled (improved but not enough (stalled)) | 3.24 | 6 of 6 (100%) | 19.0 | 20.1 | 19.8 | +7 | 1224 | 22.2 | 0.15 |
| samples_len500:201 | 500 | 7.33 | 56% | 73% | 0.05 | broken | 1.94 | 6 of 7 (86%) | 25.3 | 27.7 | 25.6 | +16 | 509 | 10.5 | 0.10 |
| samples_len500:202 | 500 | 6.53 | 66% | 69% | 0.01 | stalled (improved but not enough (stalled)) | 2.30 | 8 of 8 (100%) | 31.1 | 27.4 | 26.0 | -26 | 1081 | 31.6 | 0.11 |
| samples_len500:205 | 500 | 7.60 | 88% | 74% | 0.00 | broken | 1.65 | 8 of 16 (50%) | 29.1 | 25.6 | 21.3 | -24 | 353 | 4.5 | 0.17 |
| samples_len500:206 | 500 | 7.60 | 16% | 74% | impossible | stalled (improved but not enough (stalled)) | 6.35 | 2 of 2 (100%) | 5.8 | 7.1 | 3.9 | +7 | 369 | 5.8 | 0.09 |
| samples_len500:209 | 500 | 9.03 | 83% | 78% | 0.00 | broken | 1.90 | 11 of 11 (100%) | 35.3 | 30.5 | 27.4 | -35 | 453 | 7.2 | 0.10 |
| samples_len500:212 | 500 | 6.26 | 70% | 68% | 0.00 | broken | 1.93 | 5 of 8 (62%) | 20.2 | 19.4 | 16.0 | -5 | 196 | 4.7 | 0.13 |
| samples_len500:213 | 500 | 8.41 | 92% | 76% | 0.00 | broken | 1.63 | 8 of 16 (50%) | 35.3 | 33.7 | 25.9 | -12 | 365 | 5.6 | 0.13 |
| samples_len500:214 | 500 | 6.98 | 39% | 71% | 0.13 | stalled (improved but not enough (stalled)) | 2.30 | 6 of 6 (100%) | 19.4 | 18.2 | 13.9 | -7 | 726 | 17.0 | 0.17 |
| samples_len500:216 | 500 | 6.17 | 82% | 68% | 0.00 | broken | 1.46 | 5 of 9 (56%) | 26.5 | 23.5 | 21.9 | -20 | 209 | 5.0 | 0.05 |
| samples_len500:217 | 500 | 7.24 | 81% | 72% | 0.00 | broken | 1.79 | 8 of 13 (62%) | 28.9 | 28.9 | 21.7 | +0 | 257 | 5.6 | 0.13 |
| samples_len500:219 | 500 | 4.65 | 75% | 57% | 0.00 | broken | 1.89 | 4 of 8 (50%) | 19.8 | 18.8 | 14.0 | -6 | 208 | 3.9 | 0.12 |
| samples_len500:224 | 500 | 6.85 | 40% | 71% | 0.12 | stalled (improved but not enough (stalled)) | 5.52 | 3 of 4 (75%) | 6.2 | 5.8 | 4.0 | -2 | 286 | 12.2 | 0.14 |
| samples_len500:225 | 500 | 8.41 | 46% | 76% | 0.12 | stalled (improved but not enough (stalled)) | 2.79 | 6 of 6 (100%) | 19.2 | 18.2 | 15.5 | -6 | 1463 | 20.1 | 0.20 |
| samples_len500:228 | 500 | 6.44 | 98% | 69% | 0.00 | broken | 1.47 | 6 of 11 (55%) | 33.1 | 29.5 | 26.3 | -26 | 265 | 6.5 | 0.11 |
| samples_len500:230 | 500 | 5.55 | 82% | 64% | 0.00 | broken | 1.94 | 6 of 9 (67%) | 27.1 | 26.1 | 20.7 | -7 | 283 | 5.8 | 0.07 |
| samples_len500:234 | 500 | 6.44 | 88% | 69% | 0.00 | broken | 1.99 | 5 of 14 (36%) | 19.6 | 21.0 | 15.7 | +9 | 240 | 4.0 | 0.15 |
| samples_len500:236 | 500 | 6.94 | 76% | 71% | 0.00 | broken | 1.92 | 10 of 14 (71%) | 23.2 | 23.8 | 17.7 | +4 | 428 | 5.5 | 0.13 |
| samples_len500:238 | 500 | 5.63 | 83% | 65% | 0.00 | broken | 1.90 | 4 of 12 (33%) | 20.6 | 22.3 | 19.2 | +11 | 116 | 3.5 | 0.08 |
| samples_len500:239 | 500 | 9.84 | 75% | 80% | 0.01 | broken | 1.98 | 11 of 14 (79%) | 34.3 | 36.7 | 26.1 | +19 | 359 | 6.4 | 0.08 |
| samples_len500:242 | 500 | 9.57 | 85% | 79% | 0.00 | broken | 1.74 | 12 of 13 (92%) | 37.7 | 34.9 | 23.1 | -22 | 424 | 12.1 | 0.11 |
| samples_len500:244 | 500 | 8.32 | 54% | 76% | 0.07 | stalled (improved but not enough (stalled)) | 3.08 | 9 of 9 (100%) | 19.4 | 17.4 | 15.2 | -12 | 1032 | 19.2 | 0.20 |
| samples_len500:245 | 500 | 8.59 | 72% | 77% | 0.01 | broken | 1.80 | 12 of 12 (100%) | 29.1 | 28.6 | 26.9 | -4 | 378 | 5.7 | 0.14 |
| samples_len500:246 | 500 | 7.78 | 67% | 74% | 0.02 | stalled (improved but not enough (stalled)) | 2.24 | 10 of 10 (100%) | 23.0 | 25.5 | 19.9 | +17 | 1659 | 36.9 | 0.13 |
| samples_len500:250 | 500 | 6.17 | 41% | 68% | 0.09 | stalled (improved but not enough (stalled)) | 3.11 | 5 of 5 (100%) | 17.8 | 16.3 | 14.5 | -9 | 500 | 9.6 | 0.19 |
| samples_len500:254 | 500 | 6.62 | 44% | 70% | 0.09 | stalled (improved but not enough (stalled)) | 3.28 | 7 of 7 (100%) | 15.6 | 16.9 | 14.1 | +8 | 1211 | 24.6 | 0.14 |
| samples_len500:256 | 500 | 7.60 | 89% | 74% | 0.00 | broken | 1.96 | 11 of 17 (65%) | 23.2 | 19.8 | 18.4 | -21 | 359 | 5.8 | 0.17 |
| samples_len500:259 | 500 | 9.57 | 62% | 79% | 0.06 | stalled (improved but not enough (stalled)) | 2.75 | 9 of 9 (100%) | 28.7 | 29.9 | 26.6 | +8 | 904 | 17.0 | 0.13 |
| samples_len500:262 | 500 | 7.83 | 84% | 74% | 0.00 | broken | 1.96 | 9 of 17 (53%) | 27.0 | 28.0 | 18.7 | +7 | 273 | 5.3 | 0.12 |
| samples_len500:264 | 500 | 5.63 | 94% | 65% | 0.00 | broken | 1.95 | 4 of 11 (36%) | 22.8 | 16.4 | 18.3 | -38 | 178 | 3.3 | 0.08 |
| samples_len500:265 | 500 | 6.53 | 45% | 69% | 0.08 | broken | 1.95 | 11 of 11 (100%) | 18.4 | 19.6 | 15.9 | +8 | 636 | 11.7 | 0.15 |
| samples_len500:268 | 500 | 8.85 | 95% | 77% | 0.00 | broken | 1.93 | 9 of 16 (56%) | 35.3 | 34.9 | 25.5 | -3 | 340 | 6.1 | 0.14 |
| samples_len500:269 | 500 | 8.50 | 58% | 76% | 0.06 | broken | 1.90 | 7 of 7 (100%) | 25.1 | 26.8 | 19.1 | +11 | 2196 | 33.1 | 0.21 |
| samples_len500:270 | 500 | 6.71 | 52% | 70% | 0.05 | broken | 1.92 | 7 of 7 (100%) | 23.6 | 23.7 | 17.8 | +1 | 1116 | 15.1 | 0.14 |
| samples_len500:271 | 500 | 7.24 | 54% | 72% | 0.06 | broken | 1.98 | 5 of 5 (100%) | 26.5 | 25.4 | 22.9 | -8 | 369 | 8.1 | 0.18 |
| samples_len500:272 | 500 | 7.16 | 70% | 72% | 0.01 | broken | 1.95 | 9 of 11 (82%) | 22.2 | 23.4 | 17.0 | +8 | 445 | 6.1 | 0.15 |
| samples_len500:273 | 500 | 8.94 | 80% | 78% | 0.00 | broken | 1.48 | 14 of 22 (64%) | 30.9 | 30.8 | 22.6 | -1 | 447 | 6.6 | 0.17 |
| samples_len500:275 | 500 | 7.16 | 42% | 72% | 0.12 | stalled (improved but not enough (stalled)) | 3.61 | 5 of 5 (100%) | 9.8 | 10.3 | 8.9 | +3 | 754 | 12.8 | 0.20 |
| samples_len500:277 | 500 | 6.35 | 22% | 69% | 0.30 | stalled (improved but not enough (stalled)) | 4.66 | 3 of 5 (60%) | 5.0 | 6.3 | 3.6 | +7 | 232 | 6.1 | 0.22 |
| samples_len500:278 | 500 | 9.21 | 79% | 78% | 0.00 | broken | 1.99 | 7 of 7 (100%) | 38.1 | 30.3 | 33.5 | -56 | 392 | 8.9 | 0.08 |
| samples_len500:28 | 500 | 7.87 | 47% | 75% | 0.10 | stalled (improved but not enough (stalled)) | 3.23 | 9 of 9 (100%) | 17.2 | 16.8 | 14.2 | -2 | 1113 | 20.0 | 0.22 |
| samples_len500:282 | 500 | 6.09 | 89% | 67% | 0.00 | broken | 1.72 | 9 of 12 (75%) | 13.2 | 15.8 | 11.2 | +15 | 338 | 6.9 | 0.20 |
| samples_len500:286 | 500 | 5.99 | 92% | 67% | 0.00 | broken | 1.90 | 6 of 11 (55%) | 20.8 | 19.1 | 17.7 | -10 | 234 | 5.7 | 0.10 |
| samples_len500:294 | 500 | 6.80 | 75% | 71% | 0.00 | broken | 1.59 | 11 of 11 (100%) | 30.9 | 29.1 | 30.8 | -13 | 380 | 5.9 | 0.15 |
| samples_len500:30 | 500 | 7.33 | 70% | 73% | 0.01 | broken | 1.99 | 10 of 12 (83%) | 21.8 | 23.4 | 16.6 | +11 | 325 | 5.9 | 0.15 |
| samples_len500:300 | 500 | 5.90 | 51% | 66% | 0.04 | broken | 1.96 | 6 of 6 (100%) | 18.2 | 18.5 | 14.1 | +2 | 327 | 6.6 | 0.16 |
| samples_len500:301 | 500 | 6.67 | 86% | 70% | 0.00 | broken | 1.80 | 5 of 18 (28%) | 27.2 | 26.8 | 22.3 | -3 | 210 | 4.1 | 0.15 |
| samples_len500:303 | 500 | 7.16 | 41% | 72% | 0.12 | stalled (improved but not enough (stalled)) | 3.35 | 8 of 8 (100%) | 13.6 | 15.9 | 9.8 | +14 | 1770 | 24.8 | 0.17 |
| samples_len500:304 | 500 | 4.38 | 84% | 54% | 0.00 | broken | 1.81 | 5 of 14 (36%) | 22.2 | 19.9 | 17.6 | -14 | 217 | 4.0 | 0.15 |
| samples_len500:307 | 500 | 7.69 | 94% | 74% | 0.00 | broken | 1.78 | 6 of 12 (50%) | 29.1 | 26.0 | 23.7 | -21 | 266 | 5.6 | 0.17 |
| samples_len500:310 | 500 | 8.85 | 44% | 77% | 0.15 | stalled (improved but not enough (stalled)) | 3.94 | 6 of 7 (86%) | 17.0 | 17.0 | 16.3 | +0 | 1068 | 18.3 | 0.19 |
| samples_len500:317 | 500 | 6.98 | 51% | 71% | 0.06 | stalled (improved but not enough (stalled)) | 2.22 | 8 of 8 (100%) | 17.8 | 18.7 | 12.6 | +6 | 1412 | 26.6 | 0.17 |
| samples_len500:318 | 500 | 7.33 | 89% | 73% | 0.00 | broken | 1.76 | 11 of 25 (44%) | 19.6 | 25.6 | 15.0 | +41 | 391 | 5.1 | 0.18 |
| samples_len500:319 | 500 | 5.19 | 71% | 61% | 0.00 | broken | 1.62 | 6 of 9 (67%) | 21.0 | 20.0 | 15.4 | -6 | 233 | 5.7 | 0.13 |
| samples_len500:322 | 500 | 7.07 | 70% | 72% | 0.00 | stalled (improved but not enough (stalled)) | 2.56 | 8 of 8 (100%) | 21.6 | 23.4 | 17.0 | +12 | 1835 | 33.8 | 0.13 |
| samples_len500:323 | 495 | 6.16 | 91% | 68% | 0.00 | broken | 1.86 | 7 of 15 (47%) | 31.7 | 30.2 | 25.8 | -10 | 216 | 6.0 | 0.10 |
| samples_len500:325 | 500 | 6.08 | 80% | 67% | 0.00 | broken | 1.77 | 7 of 12 (58%) | 21.6 | 19.1 | 15.8 | -15 | 257 | 4.3 | 0.17 |
| samples_len500:332 | 500 | 7.51 | 94% | 73% | 0.00 | broken | 1.86 | 8 of 16 (50%) | 39.1 | 34.1 | 27.2 | -38 | 256 | 6.4 | 0.04 |
| samples_len500:333 | 500 | 6.89 | 86% | 71% | 0.00 | broken | 1.27 | 4 of 9 (44%) | 35.1 | 33.5 | 25.1 | -12 | 116 | 4.6 | 0.07 |
| samples_len500:334 | 500 | 7.07 | 91% | 72% | 0.00 | broken | 1.99 | 6 of 12 (50%) | 24.6 | 23.0 | 19.4 | -10 | 299 | 5.9 | 0.18 |
| samples_len500:335 | 500 | 7.87 | 69% | 75% | 0.02 | broken | 2.00 | 7 of 7 (100%) | 25.5 | 26.6 | 21.2 | +7 | 427 | 8.6 | 0.17 |
| samples_len500:336 | 500 | 5.99 | 76% | 67% | 0.00 | broken | 1.85 | 6 of 11 (55%) | 18.8 | 17.3 | 16.0 | -9 | 254 | 4.1 | 0.20 |
| samples_len500:337 | 500 | 9.30 | 79% | 78% | 0.00 | stalled (improved but not enough (stalled)) | 3.13 | 8 of 8 (100%) | 17.4 | 19.8 | 12.6 | +15 | 1644 | 31.6 | 0.24 |
| samples_len500:339 | 500 | 6.80 | 45% | 71% | 0.09 | stalled (improved but not enough (stalled)) | 2.69 | 5 of 5 (100%) | 20.6 | 19.9 | 13.8 | -4 | 1688 | 33.5 | 0.16 |
| samples_len500:34 | 500 | 6.89 | 87% | 71% | 0.00 | broken | 1.62 | 10 of 16 (62%) | 24.6 | 23.3 | 19.5 | -8 | 377 | 5.8 | 0.14 |
| samples_len500:341 | 500 | 10.38 | 66% | 81% | 0.05 | stalled (improved but not enough (stalled)) | 3.52 | 9 of 9 (100%) | 28.3 | 27.2 | 25.9 | -8 | 1410 | 34.7 | 0.16 |
| samples_len500:343 | 495 | 7.60 | 88% | 74% | 0.00 | broken | 1.80 | 7 of 14 (50%) | 30.6 | 23.4 | 23.9 | -47 | 265 | 4.8 | 0.15 |
| samples_len500:345 | 500 | 8.50 | 79% | 76% | 0.00 | broken | 1.96 | 8 of 11 (73%) | 30.5 | 31.4 | 22.9 | +6 | 355 | 5.9 | 0.11 |
| samples_len500:349 | 500 | 8.50 | 88% | 76% | 0.00 | broken | 1.95 | 6 of 13 (46%) | 32.7 | 27.5 | 26.2 | -36 | 239 | 4.7 | 0.15 |
| samples_len500:355 | 500 | 7.16 | 52% | 72% | 0.06 | broken | 1.96 | 6 of 6 (100%) | 18.4 | 18.7 | 13.6 | +2 | 569 | 9.8 | 0.20 |
| samples_len500:357 | 500 | 6.89 | 80% | 71% | 0.00 | broken | 1.52 | 8 of 12 (67%) | 25.5 | 25.3 | 21.3 | -2 | 311 | 5.6 | 0.15 |
| samples_len500:358 | 500 | 8.23 | 85% | 76% | 0.00 | broken | 1.76 | 8 of 10 (80%) | 27.7 | 26.7 | 19.9 | -7 | 262 | 6.8 | 0.14 |
| samples_len500:360 | 500 | 7.96 | 81% | 75% | 0.00 | broken | 1.77 | 12 of 17 (71%) | 24.6 | 25.7 | 20.2 | +8 | 397 | 6.5 | 0.13 |
| samples_len500:361 | 500 | 7.51 | 48% | 73% | 0.09 | stalled (improved but not enough (stalled)) | 3.07 | 10 of 10 (100%) | 21.6 | 20.0 | 16.7 | -10 | 1671 | 24.6 | 0.12 |
| samples_len500:43 | 500 | 7.60 | 84% | 74% | 0.00 | broken | 1.84 | 10 of 11 (91%) | 42.5 | 39.4 | 34.0 | -26 | 217 | 6.8 | 0.08 |
| samples_len500:44 | 500 | 6.89 | 99% | 71% | 0.00 | broken | 1.75 | 7 of 16 (44%) | 23.4 | 23.2 | 17.4 | -1 | 246 | 4.4 | 0.12 |
| samples_len500:46 | 500 | 6.17 | 56% | 68% | 0.03 | broken | 1.96 | 8 of 8 (100%) | 25.7 | 26.3 | 24.2 | +4 | 577 | 14.1 | 0.11 |
| samples_len500:47 | 500 | 8.77 | 91% | 77% | 0.00 | broken | 1.48 | 7 of 10 (70%) | 41.5 | 37.8 | 32.4 | -30 | 275 | 6.6 | 0.09 |
| samples_len500:51 | 500 | 6.26 | 50% | 68% | 0.05 | broken | 1.97 | 6 of 6 (100%) | 21.0 | 20.6 | 17.8 | -2 | 857 | 17.2 | 0.13 |
| samples_len500:52 | 500 | 7.60 | 69% | 74% | 0.01 | broken | 1.70 | 14 of 19 (74%) | 24.8 | 24.8 | 18.0 | +0 | 360 | 5.0 | 0.16 |
| samples_len500:56 | 500 | 7.69 | 92% | 74% | 0.00 | broken | 1.72 | 14 of 16 (88%) | 31.7 | 30.1 | 24.9 | -12 | 405 | 7.1 | 0.13 |
| samples_len500:62 | 500 | 4.20 | 71% | 52% | 0.00 | broken | 1.80 | 2 of 9 (22%) | 10.6 | 9.5 | 9.8 | -6 | 107 | 1.9 | 0.16 |
| samples_len500:64 | 500 | 6.71 | 88% | 70% | 0.00 | broken | 1.65 | 8 of 16 (50%) | 24.8 | 25.2 | 18.9 | +3 | 263 | 5.9 | 0.08 |
| samples_len500:65 | 500 | 6.44 | 11% | 69% | impossible | stalled (improved but not enough (stalled)) | 5.36 | 2 of 2 (100%) | 4.4 | 4.8 | 3.1 | +2 | 36 | 3.7 | 0.25 |
| samples_len500:66 | 500 | 7.51 | 64% | 73% | 0.03 | broken | 1.91 | 8 of 9 (89%) | 22.2 | 22.9 | 16.8 | +5 | 291 | 5.3 | 0.15 |
| samples_len500:69 | 500 | 6.53 | 84% | 69% | 0.00 | broken | 1.44 | 6 of 13 (46%) | 23.6 | 22.9 | 16.2 | -4 | 271 | 5.1 | 0.16 |
| samples_len500:71 | 500 | 7.51 | 7% | 73% | impossible | stalled (improved but not enough (stalled)) | 7.16 | 1 of 1 (100%) | 2.4 | 2.4 | 2.4 | +0 | 160 | 1.6 | 0.08 |
| samples_len500:72 | 500 | 7.92 | 69% | 75% | 0.02 | broken | 1.80 | 11 of 11 (100%) | 26.0 | 25.3 | 22.9 | -5 | 558 | 10.1 | 0.20 |
| samples_len500:74 | 500 | 6.08 | 67% | 67% | 0.00 | broken | 1.43 | 7 of 11 (64%) | 21.0 | 20.6 | 15.7 | -2 | 311 | 4.3 | 0.13 |
| samples_len500:78 | 500 | 8.68 | 86% | 77% | 0.00 | broken | 1.66 | 7 of 15 (47%) | 28.7 | 28.3 | 27.1 | -3 | 218 | 5.4 | 0.13 |
| samples_len500:81 | 500 | 7.78 | 91% | 74% | 0.00 | broken | 1.92 | 10 of 15 (67%) | 24.2 | 24.3 | 19.4 | +1 | 393 | 6.3 | 0.13 |
| samples_len500:86 | 500 | 6.80 | 62% | 71% | 0.02 | broken | 1.97 | 10 of 14 (71%) | 19.6 | 19.6 | 12.9 | +0 | 304 | 4.2 | 0.20 |
| samples_len500:90 | 500 | 8.94 | 38% | 78% | 0.21 | stalled (improved but not enough (stalled)) | 5.61 | 5 of 5 (100%) | 18.2 | 18.8 | 17.5 | +4 | 475 | 11.4 | 0.11 |
| samples_len500:94 | 500 | 6.17 | 61% | 68% | 0.01 | broken | 1.97 | 13 of 14 (93%) | 24.0 | 23.6 | 18.1 | -2 | 408 | 5.9 | 0.15 |
| samples_len500:97 | 500 | 8.85 | 95% | 77% | 0.00 | broken | 1.59 | 8 of 12 (67%) | 29.1 | 27.0 | 22.1 | -15 | 277 | 7.5 | 0.16 |
| samples_len500:98 | 500 | 5.72 | 89% | 65% | 0.00 | broken | 1.85 | 4 of 9 (44%) | 18.0 | 16.8 | 15.3 | -7 | 223 | 4.2 | 0.16 |

## 6. Where it breaks

- 500-token spans, broken spans: sentences rewritten at the break 8.0 / 8.0 / 14.0 | queries at the break 376.1 / 311.0 / 2422.0
- 500-token spans: 75 need no selection gain (eligible text enough at chance level), 56 need delta > 0, of which the achieved delta reached the need in 51; 3 are impossible by rewriting (eligible share too small even for delta 0.5)

## 7. Gates: where candidates are lost

- candidates generated 134520 -> after clean 100647 -> spelling 94283 -> content 81097 -> semantic 77334 -> fluency 73792
- rejected by reason: {'duplicate': 27238, 'negation count changed': 9955, 'unknown word': 6001, 'CJK text': 5289, 'meaning too different': 3763, 'perplexity too high': 3542, 'quotation changed': 2162, 'numbers changed': 1069, 'several lines': 721, 'preamble': 477, 'glued punctuation': 363, 'hit length limit': 96, 'identical to original': 50, 'too short': 2}

## 8. Resources and problems

- time (hours): attack 23.48 | candidate generation 21.83 | gates 1.64 | detector queries 0.01
- detector queries: 72787 in total (screening queries not included)
- out-of-memory events during rewriting: 0 | rewrites that hit the length limit: 172 | samples that raised an error: 1 (samples_len500:351)
- HTML entities (such as &apos;) removed by rewriting: 0 across all spans. Rewriting turns them into plain characters, which changes tokenisation without paraphrasing; this effect is not separated here.

## 9. Limitations

- Sentence boundaries are a heuristic; sentences ending in an abbreviation merge with the next one.
- About a third of each span is not attackable by design (the mid-sentence start, the cut-off end, quotations, short or non-prose pieces).
- The gates cannot see all meaning changes (for example income -> revenue). Faithfulness is established only by reading READING.md.
- The necessary-condition bound uses the eligible character share as a stand-in for the share of token pairs.
- The spans are a selection: the model had to continue for 195 tokens (500-token set: 495). The two lengths are not independent samples.
- Spans are model-generated at temperature 1.0 and already contain glued words and HTML entities copied from C4.
- One detector (the STA detector with its published key) and one paraphrasing model; results do not transfer automatically to other models.

## 10. Reproduction

Command: `research/qwen_attack/run_attack.py --files results/raw/samples_len500.json --resume results/qwen_attack_fullscale/adaptive500`. Files: `results/raw/samples_len500.json`. Gate config and seeds are in run.json; every sample's steps, funnels and rewrites are in `samples/`.
