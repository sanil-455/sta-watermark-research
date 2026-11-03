# Qwen paraphrase attack on the STA watermark: final report

Run: `sta16_attack`. Started 2026-11-01 19:55:39, ended 2026-11-02 04:58:55. Git commit `ad8f7e292088dcb1011d5c6e2189a697e8b5d8cc`.

**What was attacked.** Only the generated span of each sample, never the prompt. The detector was asked only about span text. **Success** means the span's z-score is 2.0 or below under the same detector. Preset: **lenient**. Quality is judged by reading (see READING.md); the gates below are filters, not proof of faithfulness.

## 1. Data

- spans loaded: 100 | flagged by the detector (z0 > 2): 99 | not flagged: 1
- not flagged (not attacked): sta16_len200:45 (z0 0.707)
- 200-token spans: 100 loaded, 99 flagged
- 500-token spans: 0 loaded, 0 flagged
- generation run: 250 kept, 61 stopped before 195 tokens, 24 skipped for prompt length, 0 out-of-memory events. Both span sets are selections (the model had to keep generating).
- the 500-token spans are extensions of the 200-token spans of the same prompts, so the two groups are never pooled.

## 2. Models and environment

- **Generator of the watermarked text:** Llama-2-7B base, 6,738,415,616 parameters (computed from its published configuration, not re-measured in this run), fp16
- **Paraphraser:** Qwen/Qwen2.5-7B-Instruct, revision `a09a35458c702b33eeacc393d103063234e8bc28`, **7,615,616,512 parameters** (computed from the architecture; transformers reports 7,615,616,512; agree: True), 28 layers, hidden size 3584, 28 attention heads (4 key/value), vocabulary 152064, context 32768 tokens
  - 4-bit nf4 with double quantisation, float16 compute; stored elements by dtype: {'torch.bfloat16': 1090328064, 'torch.uint8': 3262644224}; model memory footprint 5.07 GiB; GPU allocated after load 5.26 GiB; load time 4.7 s
- **Similarity model:** sentence-transformers/all-MiniLM-L6-v2, 22,713,216 parameters
- **Detector:** the STA detector of the paper, queried as a black box that returns z-scores; the key is never used by the attack
- GPU: Quadro RTX 5000 | python 3.11.16 | torch 2.7.1+cu128 | transformers 5.17.0 | bitsandbytes 0.50.2 | accelerate 1.15.0 | WordNet available: True
- peak GPU memory during the run: allocated 5.50 GiB, reserved 5.55 GiB

## 3. Settings

- gates (lenient): {'semantic': 0.75, 'fluency': 1.8, 'min_for_fluency': 3, 'spelling': True, 'max_unknown': 0, 'check_numbers': True, 'check_quotes': True, 'check_negations': True, 'check_names': False}
- sampling: {'temperature': 1.5, 'top_p': 0.98, 'top_k': 20, 'repetition_penalty': 1.05, 'batch': 4, 'length_factor': 2.0, 'extra_tokens': 32}
- attack: {'n_cands': 60, 'seed': 0, 'max_minutes': 0} | segmentation: {'min_chars': 30, 'quote_words': 4, 'abbreviations': 50}

## 4. Results (never pooled across lengths)

Columns marked mean / median / max.

### 200-token spans: 99 attacked, **32 broken (32%)**

- all attacked (n=99): z0 9.7 / 9.8 / 12.3 | z final 4.2 / 3.1 / 11.3 | sentences rewritten % of eligible 90.6 / 100.0 / 100.0 | tokens not preserved % 24.6 / 25.9 / 50.7 | tokens that are new % 25.0 / 27.9 / 47.3 | words not preserved % 31.6 / 34.4 / 57.1 | characters not preserved % 19.1 / 21.0 / 39.6 | length change (tokens) 1.1 / 1.0 / 38.0 | queries 286.3 / 224.0 / 1459.0 | minutes 5.5 / 4.9 / 25.6
- broken only (n=32): z0 9.6 / 9.8 / 12.3 | z final 1.7 / 1.7 / 2.0 | sentences rewritten % of eligible 92.8 / 100.0 / 100.0 | tokens not preserved % 35.2 / 36.1 / 50.7 | tokens that are new % 35.6 / 35.0 / 47.3 | words not preserved % 44.7 / 45.8 / 57.1 | characters not preserved % 27.1 / 27.6 / 34.6 | length change (tokens) 1.2 / -2.0 / 38.0 | queries 283.2 / 211.5 / 1459.0 | minutes 4.5 / 3.1 / 19.1
- outcomes: {'improved but not enough (stalled)': 60, 'broken': 32, 'no eligible text': 4, 'candidates passed but none lowered z': 1, 'no candidate passed the gates': 2}

## 5. Per-sample table

tokens = span length; eligible = share of characters that may be rewritten; needed = share that must be rewritten at chance level (1 - 2/z0); delta needed = selection gain (how much less green than chance the rewrites must be) if eligible < needed; delta achieved = 0.5 - green fraction of the new token pairs. Percentages of tokens/characters are of the ORIGINAL span (not preserved) and of the FINAL span (new).

| sample | tokens | z0 | eligible | needed | delta needed | outcome | z final | rewritten | tokens not kept % | tokens new % | chars not kept % | length change | queries | min | delta achieved |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sta16_len200:100 | 200 | 10.18 | 59% | 80% | 0.13 | stalled (improved but not enough (stalled)) | 3.98 | 2 of 2 (100%) | 33.8 | 33.2 | 26.8 | -2 | 178 | 6.8 | 0.11 |
| sta16_len200:101 | 200 | 7.78 | 92% | 74% | 0.00 | stalled (improved but not enough (stalled)) | 2.88 | 4 of 4 (100%) | 21.4 | 29.5 | 21.5 | +23 | 229 | 10.3 | 0.11 |
| sta16_len200:102 | 200 | 11.74 | 83% | 83% | 0.00 | broken | 1.61 | 7 of 8 (88%) | 40.3 | 41.2 | 24.0 | +3 | 209 | 3.0 | 0.15 |
| sta16_len200:103 | 200 | 10.56 | 76% | 81% | 0.02 | stalled (improved but not enough (stalled)) | 4.77 | 6 of 6 (100%) | 23.0 | 26.7 | 18.8 | +10 | 406 | 10.2 | 0.15 |
| sta16_len200:104 | 200 | 9.48 | 75% | 79% | 0.02 | stalled (improved but not enough (stalled)) | 2.55 | 4 of 4 (100%) | 31.8 | 27.9 | 23.4 | -11 | 297 | 5.4 | 0.25 |
| sta16_len200:105 | 200 | 10.75 | 9% | 81% | impossible | stalled (improved but not enough (stalled)) | 9.90 | 1 of 1 (100%) | 3.0 | 3.0 | 2.4 | +0 | 43 | 0.7 | 0.25 |
| sta16_len200:106 | 200 | 10.37 | 93% | 81% | 0.00 | broken | 1.92 | 5 of 5 (100%) | 34.2 | 32.8 | 28.3 | -4 | 198 | 3.5 | 0.14 |
| sta16_len200:107 | 200 | 8.77 | 74% | 77% | 0.01 | stalled (improved but not enough (stalled)) | 2.09 | 4 of 4 (100%) | 25.4 | 27.5 | 17.3 | +6 | 277 | 5.6 | 0.18 |
| sta16_len200:108 | 200 | 9.62 | 69% | 79% | 0.05 | stalled (improved but not enough (stalled)) | 3.23 | 3 of 3 (100%) | 42.3 | 38.0 | 27.6 | -14 | 199 | 7.6 | 0.06 |
| sta16_len200:109 | 200 | 10.47 | 0% | 81% | impossible | no eligible text (no eligible text) | 10.47 | 0 of 0 (0%) | 0.0 | 0.0 | 0.0 | +0 | 1 | 0.0 | - |
| sta16_len200:110 | 200 | 10.89 | 93% | 82% | 0.00 | broken | 1.11 | 5 of 6 (83%) | 39.8 | 41.5 | 33.5 | +6 | 234 | 3.1 | 0.15 |
| sta16_len200:112 | 200 | 5.94 | 35% | 66% | 0.19 | stalled (improved but not enough (stalled)) | 4.44 | 2 of 2 (100%) | 9.0 | 9.4 | 9.5 | +1 | 140 | 3.4 | 0.11 |
| sta16_len200:113 | 200 | 11.03 | 75% | 82% | 0.04 | stalled (improved but not enough (stalled)) | 2.78 | 4 of 4 (100%) | 48.3 | 38.5 | 39.6 | -32 | 514 | 7.4 | 0.14 |
| sta16_len200:114 | 200 | 9.05 | 91% | 78% | 0.00 | stalled (improved but not enough (stalled)) | 2.18 | 5 of 5 (100%) | 23.9 | 25.0 | 23.2 | +3 | 1216 | 25.6 | 0.24 |
| sta16_len200:115 | 200 | 8.34 | 15% | 76% | impossible | stalled (improved but not enough (stalled)) | 7.30 | 1 of 1 (100%) | 9.0 | 8.5 | 5.9 | -1 | 80 | 1.6 | 0.05 |
| sta16_len200:116 | 200 | 10.47 | 75% | 81% | 0.03 | broken | 1.87 | 6 of 6 (100%) | 29.4 | 32.4 | 23.2 | +9 | 479 | 6.3 | 0.21 |
| sta16_len200:118 | 200 | 10.18 | 54% | 80% | 0.18 | stalled (improved but not enough (stalled)) | 5.49 | 2 of 2 (100%) | 24.4 | 25.1 | 19.4 | +2 | 229 | 8.6 | 0.05 |
| sta16_len200:119 | 200 | 10.04 | 67% | 80% | 0.07 | broken | 1.95 | 3 of 3 (100%) | 37.8 | 34.9 | 27.3 | -9 | 422 | 8.0 | 0.17 |
| sta16_len200:120 | 200 | 8.77 | 71% | 77% | 0.03 | broken | 1.69 | 7 of 7 (100%) | 22.9 | 35.1 | 17.2 | +38 | 466 | 7.6 | 0.13 |
| sta16_len200:121 | 200 | 9.62 | 52% | 79% | 0.18 | stalled (improved but not enough (stalled)) | 5.34 | 3 of 3 (100%) | 23.9 | 22.7 | 21.3 | -3 | 308 | 4.9 | 0.11 |
| sta16_len200:123 | 200 | 7.92 | 29% | 75% | 0.45 | stalled (improved but not enough (stalled)) | 6.02 | 2 of 2 (100%) | 8.0 | 11.9 | 8.1 | +9 | 185 | 2.8 | 0.22 |
| sta16_len200:124 | 200 | 8.91 | 79% | 78% | 0.00 | broken | 1.62 | 5 of 5 (100%) | 22.4 | 29.4 | 16.8 | +20 | 130 | 3.1 | 0.17 |
| sta16_len200:125 | 200 | 10.61 | 77% | 81% | 0.02 | stalled (improved but not enough (stalled)) | 3.13 | 3 of 3 (100%) | 35.3 | 34.7 | 25.3 | -2 | 369 | 9.6 | 0.13 |
| sta16_len200:126 | 200 | 7.50 | 43% | 73% | 0.19 | stalled (improved but not enough (stalled)) | 3.30 | 2 of 2 (100%) | 21.9 | 23.0 | 15.8 | +3 | 447 | 7.6 | 0.14 |
| sta16_len200:127 | 200 | 10.32 | 6% | 81% | impossible | stalled (candidates passed but none lowered z) | 10.32 | 0 of 1 (0%) | 0.0 | 0.0 | 0.0 | +0 | 2 | 0.2 | - |
| sta16_len200:128 | 200 | 11.17 | 89% | 82% | 0.00 | broken | 1.71 | 5 of 5 (100%) | 36.3 | 35.0 | 30.0 | -4 | 214 | 4.5 | 0.21 |
| sta16_len200:129 | 200 | 11.17 | 25% | 82% | impossible | stalled (no candidate passed the gates) | 11.17 | 0 of 1 (0%) | 0.0 | 0.0 | 0.0 | +0 | 1 | 0.8 | - |
| sta16_len200:130 | 200 | 10.32 | 26% | 81% | impossible | stalled (no candidate passed the gates) | 10.32 | 0 of 1 (0%) | 0.0 | 0.0 | 0.0 | +0 | 1 | 1.0 | - |
| sta16_len200:131 | 200 | 8.63 | 85% | 77% | 0.00 | stalled (improved but not enough (stalled)) | 3.03 | 3 of 3 (100%) | 21.9 | 22.3 | 19.0 | +1 | 429 | 10.8 | 0.19 |
| sta16_len200:132 | 200 | 7.07 | 71% | 72% | 0.00 | stalled (improved but not enough (stalled)) | 2.55 | 5 of 5 (100%) | 22.9 | 22.9 | 16.9 | +0 | 503 | 10.5 | 0.18 |
| sta16_len200:134 | 200 | 12.30 | 73% | 84% | 0.07 | stalled (improved but not enough (stalled)) | 4.90 | 3 of 3 (100%) | 28.9 | 30.2 | 19.6 | +4 | 154 | 11.2 | 0.17 |
| sta16_len200:135 | 200 | 11.60 | 58% | 83% | 0.17 | stalled (improved but not enough (stalled)) | 5.71 | 4 of 4 (100%) | 29.4 | 27.9 | 21.7 | -4 | 600 | 12.2 | 0.14 |
| sta16_len200:136 | 200 | 10.89 | 89% | 82% | 0.00 | broken | 1.10 | 5 of 7 (71%) | 37.8 | 40.8 | 29.2 | +10 | 212 | 2.8 | 0.16 |
| sta16_len200:137 | 200 | 9.05 | 76% | 78% | 0.01 | broken | 1.54 | 4 of 4 (100%) | 40.8 | 36.7 | 28.8 | -13 | 98 | 2.4 | 0.12 |
| sta16_len200:138 | 200 | 8.49 | 31% | 76% | 0.44 | stalled (improved but not enough (stalled)) | 6.79 | 1 of 1 (100%) | 8.5 | 8.5 | 8.7 | +0 | 20 | 2.7 | 0.06 |
| sta16_len200:139 | 200 | 11.03 | 0% | 82% | impossible | no eligible text (no eligible text) | 11.03 | 0 of 0 (0%) | 0.0 | 0.0 | 0.0 | +0 | 1 | 0.0 | - |
| sta16_len200:140 | 200 | 11.31 | 0% | 82% | impossible | no eligible text (no eligible text) | 11.31 | 0 of 0 (0%) | 0.0 | 0.0 | 0.0 | +0 | 1 | 0.0 | - |
| sta16_len200:141 | 200 | 10.18 | 91% | 80% | 0.00 | broken | 1.74 | 5 of 5 (100%) | 40.8 | 37.7 | 28.0 | -10 | 148 | 3.1 | 0.14 |
| sta16_len200:142 | 200 | 6.79 | 76% | 71% | 0.00 | broken | 1.66 | 4 of 6 (67%) | 23.9 | 26.8 | 20.6 | +8 | 167 | 2.0 | 0.18 |
| sta16_len200:143 | 200 | 9.19 | 61% | 78% | 0.09 | stalled (improved but not enough (stalled)) | 2.69 | 3 of 3 (100%) | 23.4 | 23.4 | 22.0 | +0 | 396 | 6.7 | 0.16 |
| sta16_len200:144 | 200 | 11.03 | 0% | 82% | impossible | no eligible text (no eligible text) | 11.03 | 0 of 0 (0%) | 0.0 | 0.0 | 0.0 | +0 | 1 | 0.0 | - |
| sta16_len200:146 | 200 | 9.05 | 61% | 78% | 0.09 | stalled (improved but not enough (stalled)) | 2.21 | 4 of 4 (100%) | 23.4 | 22.2 | 20.7 | -3 | 496 | 8.4 | 0.29 |
| sta16_len200:148 | 200 | 9.48 | 90% | 79% | 0.00 | broken | 1.64 | 6 of 9 (67%) | 39.3 | 38.4 | 32.5 | -3 | 211 | 2.7 | 0.14 |
| sta16_len200:150 | 200 | 7.92 | 61% | 75% | 0.06 | stalled (improved but not enough (stalled)) | 3.03 | 2 of 2 (100%) | 27.9 | 28.2 | 20.8 | +1 | 289 | 6.6 | 0.11 |
| sta16_len200:151 | 200 | 10.61 | 98% | 81% | 0.00 | broken | 1.51 | 9 of 9 (100%) | 37.8 | 41.3 | 27.1 | +12 | 273 | 4.0 | 0.11 |
| sta16_len200:152 | 200 | 10.32 | 57% | 81% | 0.15 | stalled (improved but not enough (stalled)) | 3.30 | 5 of 5 (100%) | 33.3 | 34.3 | 25.8 | +3 | 1127 | 12.5 | 0.16 |
| sta16_len200:153 | 195 | 7.66 | 73% | 74% | 0.00 | stalled (improved but not enough (stalled)) | 2.49 | 5 of 5 (100%) | 28.6 | 25.1 | 20.9 | -9 | 511 | 9.6 | 0.10 |
| sta16_len200:154 | 200 | 9.19 | 38% | 78% | 0.34 | stalled (improved but not enough (stalled)) | 6.65 | 2 of 2 (100%) | 12.9 | 12.9 | 11.1 | +0 | 67 | 3.7 | 0.08 |
| sta16_len200:156 | 200 | 9.48 | 82% | 79% | 0.00 | broken | 1.88 | 7 of 7 (100%) | 36.3 | 42.6 | 33.0 | +22 | 224 | 3.5 | 0.10 |
| sta16_len200:157 | 200 | 8.20 | 88% | 76% | 0.00 | broken | 1.93 | 4 of 5 (80%) | 30.8 | 34.1 | 26.1 | +10 | 174 | 2.6 | 0.12 |
| sta16_len200:29 | 200 | 9.62 | 17% | 79% | impossible | stalled (improved but not enough (stalled)) | 8.34 | 1 of 1 (100%) | 5.5 | 4.0 | 4.6 | -3 | 52 | 1.2 | 0.12 |
| sta16_len200:30 | 200 | 9.33 | 47% | 79% | 0.22 | stalled (improved but not enough (stalled)) | 5.49 | 3 of 3 (100%) | 21.4 | 22.2 | 15.1 | +2 | 402 | 7.1 | 0.15 |
| sta16_len200:31 | 200 | 9.76 | 92% | 80% | 0.00 | broken | 0.50 | 6 of 6 (100%) | 43.3 | 43.0 | 33.0 | -1 | 297 | 8.2 | 0.14 |
| sta16_len200:32 | 200 | 10.04 | 19% | 80% | impossible | stalled (improved but not enough (stalled)) | 8.29 | 1 of 1 (100%) | 10.0 | 9.5 | 7.4 | -1 | 128 | 1.8 | 0.15 |
| sta16_len200:33 | 200 | 10.47 | 71% | 81% | 0.05 | stalled (improved but not enough (stalled)) | 3.52 | 3 of 3 (100%) | 34.3 | 35.0 | 23.0 | +2 | 338 | 7.9 | 0.13 |
| sta16_len200:35 | 200 | 9.76 | 92% | 80% | 0.00 | broken | 2.00 | 5 of 6 (83%) | 37.8 | 32.1 | 34.6 | -17 | 153 | 3.0 | 0.15 |
| sta16_len200:36 | 200 | 8.91 | 88% | 78% | 0.00 | broken | 1.95 | 4 of 4 (100%) | 37.8 | 39.6 | 32.7 | +6 | 81 | 2.9 | 0.05 |
| sta16_len200:37 | 200 | 10.47 | 74% | 81% | 0.04 | broken | 1.92 | 6 of 6 (100%) | 33.8 | 32.8 | 29.8 | -3 | 359 | 4.6 | 0.24 |
| sta16_len200:41 | 200 | 9.62 | 59% | 79% | 0.11 | stalled (improved but not enough (stalled)) | 4.04 | 2 of 2 (100%) | 28.9 | 33.2 | 21.6 | +13 | 255 | 6.4 | 0.09 |
| sta16_len200:44 | 200 | 10.18 | 64% | 80% | 0.10 | stalled (improved but not enough (stalled)) | 2.75 | 3 of 3 (100%) | 30.8 | 31.2 | 24.8 | +1 | 277 | 4.9 | 0.14 |
| sta16_len200:46 | 200 | 8.20 | 79% | 76% | 0.00 | broken | 1.65 | 5 of 7 (71%) | 30.8 | 29.1 | 23.4 | -5 | 157 | 2.3 | 0.22 |
| sta16_len200:47 | 200 | 10.89 | 66% | 82% | 0.09 | stalled (improved but not enough (stalled)) | 3.28 | 4 of 4 (100%) | 32.3 | 34.0 | 29.5 | +5 | 348 | 6.9 | 0.15 |
| sta16_len200:50 | 200 | 12.16 | 34% | 84% | impossible | stalled (improved but not enough (stalled)) | 8.74 | 2 of 2 (100%) | 13.4 | 17.9 | 10.1 | +11 | 168 | 3.7 | 0.08 |
| sta16_len200:51 | 200 | 9.62 | 21% | 79% | impossible | stalled (improved but not enough (stalled)) | 8.40 | 1 of 1 (100%) | 5.0 | 6.8 | 3.5 | +4 | 41 | 1.6 | 0.17 |
| sta16_len200:52 | 200 | 9.48 | 83% | 79% | 0.00 | broken | 1.74 | 6 of 6 (100%) | 34.8 | 31.4 | 26.7 | -10 | 193 | 2.9 | 0.18 |
| sta16_len200:53 | 200 | 8.49 | 90% | 76% | 0.00 | stalled (improved but not enough (stalled)) | 2.46 | 4 of 4 (100%) | 29.4 | 34.0 | 21.0 | +14 | 269 | 6.3 | 0.08 |
| sta16_len200:54 | 200 | 10.89 | 77% | 82% | 0.02 | stalled (improved but not enough (stalled)) | 2.11 | 7 of 7 (100%) | 36.8 | 37.4 | 24.2 | +2 | 761 | 13.7 | 0.21 |
| sta16_len200:55 | 200 | 9.48 | 64% | 79% | 0.08 | stalled (improved but not enough (stalled)) | 3.76 | 7 of 7 (100%) | 25.9 | 28.0 | 22.2 | +6 | 343 | 8.2 | 0.10 |
| sta16_len200:56 | 200 | 9.90 | 81% | 80% | 0.00 | broken | 1.95 | 7 of 7 (100%) | 33.3 | 30.2 | 26.4 | -9 | 373 | 6.1 | 0.18 |
| sta16_len200:57 | 200 | 9.05 | 32% | 78% | 0.47 | stalled (improved but not enough (stalled)) | 6.08 | 2 of 2 (100%) | 12.9 | 12.9 | 13.0 | +0 | 219 | 3.0 | 0.14 |
| sta16_len200:59 | 200 | 10.32 | 55% | 81% | 0.17 | stalled (improved but not enough (stalled)) | 4.28 | 3 of 3 (100%) | 27.9 | 28.9 | 22.6 | +3 | 299 | 5.6 | 0.14 |
| sta16_len200:60 | 200 | 9.48 | 33% | 79% | 0.47 | stalled (improved but not enough (stalled)) | 7.16 | 2 of 2 (100%) | 11.4 | 11.0 | 10.3 | -1 | 133 | 3.3 | 0.12 |
| sta16_len200:61 | 200 | 11.17 | 48% | 82% | 0.28 | stalled (improved but not enough (stalled)) | 3.90 | 6 of 6 (100%) | 22.9 | 25.1 | 16.1 | +6 | 344 | 6.3 | 0.26 |
| sta16_len200:62 | 200 | 7.64 | 62% | 74% | 0.05 | stalled (improved but not enough (stalled)) | 4.04 | 3 of 3 (100%) | 25.4 | 25.0 | 21.5 | -1 | 356 | 9.3 | 0.13 |
| sta16_len200:63 | 195 | 10.38 | 19% | 81% | impossible | stalled (improved but not enough (stalled)) | 8.29 | 1 of 1 (100%) | 12.2 | 12.7 | 10.8 | +1 | 122 | 2.4 | 0.10 |
| sta16_len200:65 | 200 | 7.21 | 72% | 72% | 0.00 | broken | 1.81 | 4 of 4 (100%) | 23.9 | 26.1 | 14.5 | +6 | 124 | 2.5 | 0.16 |
| sta16_len200:66 | 200 | 8.63 | 86% | 77% | 0.00 | stalled (improved but not enough (stalled)) | 3.10 | 4 of 4 (100%) | 17.4 | 21.7 | 14.8 | +11 | 472 | 9.4 | 0.21 |
| sta16_len200:68 | 200 | 10.61 | 10% | 81% | impossible | stalled (improved but not enough (stalled)) | 9.62 | 1 of 1 (100%) | 3.0 | 3.0 | 2.9 | +0 | 50 | 0.7 | 0.39 |
| sta16_len200:69 | 200 | 9.05 | 39% | 78% | 0.32 | stalled (improved but not enough (stalled)) | 4.82 | 2 of 2 (100%) | 19.4 | 16.5 | 14.8 | -7 | 175 | 5.9 | 0.26 |
| sta16_len200:71 | 200 | 11.03 | 62% | 82% | 0.13 | stalled (improved but not enough (stalled)) | 4.34 | 4 of 4 (100%) | 29.4 | 30.7 | 19.4 | +4 | 475 | 6.4 | 0.12 |
| sta16_len200:72 | 200 | 10.04 | 14% | 80% | impossible | stalled (improved but not enough (stalled)) | 8.44 | 1 of 1 (100%) | 9.0 | 9.9 | 6.6 | +2 | 170 | 2.6 | 0.12 |
| sta16_len200:74 | 200 | 7.50 | 63% | 73% | 0.04 | stalled (improved but not enough (stalled)) | 3.20 | 4 of 4 (100%) | 14.9 | 17.4 | 10.6 | +6 | 394 | 6.9 | 0.27 |
| sta16_len200:75 | 200 | 9.76 | 31% | 80% | impossible | stalled (improved but not enough (stalled)) | 7.65 | 2 of 2 (100%) | 8.0 | 9.3 | 5.8 | +3 | 409 | 5.2 | 0.16 |
| sta16_len200:76 | 200 | 9.19 | 33% | 78% | 0.43 | stalled (improved but not enough (stalled)) | 6.31 | 1 of 1 (100%) | 14.9 | 8.6 | 10.9 | -14 | 58 | 3.7 | 0.28 |
| sta16_len200:77 | 200 | 11.46 | 42% | 83% | 0.38 | stalled (improved but not enough (stalled)) | 7.55 | 3 of 3 (100%) | 11.9 | 16.9 | 9.5 | +12 | 507 | 7.8 | 0.18 |
| sta16_len200:78 | 200 | 9.76 | 63% | 80% | 0.09 | broken | 2.00 | 6 of 6 (100%) | 35.3 | 38.7 | 25.5 | +11 | 1459 | 19.1 | 0.20 |
| sta16_len200:80 | 200 | 12.30 | 84% | 84% | 0.00 | broken | 1.97 | 7 of 7 (100%) | 50.7 | 47.3 | 31.8 | -13 | 425 | 6.0 | 0.12 |
| sta16_len200:84 | 200 | 9.76 | 65% | 80% | 0.07 | broken | 1.96 | 5 of 5 (100%) | 32.8 | 34.1 | 23.6 | +4 | 652 | 8.6 | 0.21 |
| sta16_len200:85 | 200 | 7.78 | 69% | 74% | 0.02 | broken | 1.79 | 4 of 4 (100%) | 25.9 | 30.0 | 21.0 | +12 | 275 | 4.3 | 0.17 |
| sta16_len200:86 | 200 | 9.19 | 91% | 78% | 0.00 | broken | 1.32 | 5 of 5 (100%) | 48.8 | 44.9 | 32.9 | -14 | 137 | 2.8 | 0.10 |
| sta16_len200:88 | 200 | 9.90 | 89% | 80% | 0.00 | broken | 1.45 | 3 of 4 (75%) | 41.8 | 38.7 | 33.7 | -10 | 111 | 2.3 | 0.10 |
| sta16_len200:89 | 200 | 8.77 | 77% | 77% | 0.00 | broken | 1.15 | 5 of 6 (83%) | 29.9 | 26.9 | 22.5 | -8 | 152 | 2.8 | 0.23 |
| sta16_len200:90 | 200 | 10.47 | 14% | 81% | impossible | stalled (improved but not enough (stalled)) | 7.97 | 1 of 1 (100%) | 10.9 | 11.4 | 6.4 | +1 | 157 | 1.7 | 0.21 |
| sta16_len200:92 | 200 | 10.04 | 47% | 80% | 0.25 | stalled (improved but not enough (stalled)) | 5.15 | 2 of 2 (100%) | 23.4 | 23.8 | 18.6 | +1 | 183 | 4.8 | 0.14 |
| sta16_len200:93 | 200 | 10.47 | 73% | 81% | 0.04 | stalled (improved but not enough (stalled)) | 4.28 | 3 of 3 (100%) | 35.8 | 30.3 | 28.2 | -16 | 192 | 4.5 | 0.06 |
| sta16_len200:94 | 200 | 10.85 | 45% | 82% | 0.31 | stalled (improved but not enough (stalled)) | 4.81 | 5 of 5 (100%) | 18.5 | 18.9 | 16.6 | +1 | 682 | 9.9 | 0.30 |
| sta16_len200:97 | 200 | 8.91 | 70% | 78% | 0.03 | stalled (improved but not enough (stalled)) | 3.05 | 3 of 3 (100%) | 29.9 | 29.5 | 20.1 | -1 | 591 | 9.2 | 0.15 |
| sta16_len200:98 | 200 | 9.19 | 82% | 78% | 0.00 | broken | 1.87 | 3 of 3 (100%) | 35.8 | 33.8 | 30.9 | -6 | 255 | 4.9 | 0.14 |
| sta16_len200:99 | 200 | 9.05 | 77% | 78% | 0.00 | stalled (improved but not enough (stalled)) | 3.12 | 2 of 2 (100%) | 27.4 | 33.3 | 21.4 | +18 | 197 | 5.6 | 0.11 |

## 6. Where it breaks

- 200-token spans, broken spans: sentences rewritten at the break 5.2 / 5.0 / 9.0 | queries at the break 283.2 / 211.5 / 1459.0
- 200-token spans: 27 need no selection gain (eligible text enough at chance level), 54 need delta > 0, of which the achieved delta reached the need in 33; 18 are impossible by rewriting (eligible share too small even for delta 0.5)

## 7. Gates: where candidates are lost

- candidates generated 56040 -> after clean 40376 -> spelling 37607 -> content 34019 -> semantic 31252 -> fluency 29298
- rejected by reason: {'duplicate': 13316, 'meaning too different': 2767, 'unknown word': 2550, 'negation count changed': 2394, 'CJK text': 1979, 'perplexity too high': 1954, 'quotation changed': 811, 'numbers changed': 383, 'several lines': 281, 'glued punctuation': 219, 'preamble': 47, 'hit length limit': 26, 'identical to original': 15}

## 8. Resources and problems

- time (hours): attack 9.07 | candidate generation 8.42 | gates 0.65 | detector queries 0.00
- detector queries: 28346 in total (screening queries not included)
- out-of-memory events during rewriting: 0 | rewrites that hit the length limit: 53 | samples that raised an error: 0
- HTML entities (such as &apos;) removed by rewriting: 1 across all spans. Rewriting turns them into plain characters, which changes tokenisation without paraphrasing; this effect is not separated here.

## 9. Limitations

- Sentence boundaries are a heuristic; sentences ending in an abbreviation merge with the next one.
- About a third of each span is not attackable by design (the mid-sentence start, the cut-off end, quotations, short or non-prose pieces).
- The gates cannot see all meaning changes (for example income -> revenue). Faithfulness is established only by reading READING.md.
- The necessary-condition bound uses the eligible character share as a stand-in for the share of token pairs.
- The spans are a selection: the model had to continue for 195 tokens (500-token set: 495). The two lengths are not independent samples.
- Spans are model-generated at temperature 1.0 and already contain glued words and HTML entities copied from C4.
- One detector (the STA detector with its published key) and one paraphrasing model; results do not transfer automatically to other models.

## 10. Reproduction

Command: `research/qwen_attack/run_attack.py --files results/raw/sta16_len200.json --resume results/qwen_attack_fullscale/sta16_attack`. Files: `results/raw/sta16_len200.json`. Gate config and seeds are in run.json; every sample's steps, funnels and rewrites are in `samples/`.
