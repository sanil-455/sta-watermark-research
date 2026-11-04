# Qwen paraphrase attack on the STA watermark: final report

Run: `attack`. Started 2026-11-03 08:51:04, ended 2026-11-04 09:15:35. Git commit `fd6c899966a2af30c3d1ce27bd59a8d276df771b`.

**What was attacked.** Only the generated span of each sample, never the prompt. The detector was asked only about span text. **Success** means the span's z-score is 2.0 or below under the same detector. Preset: **lenient**. Quality is judged by reading (see READING.md); the gates below are filters, not proof of faithfulness.

## 1. Data

- spans loaded: 53 | flagged by the detector (z0 > 2): 53 | not flagged: 0
- not flagged (not attacked): none
- 200-token spans: 0 loaded, 0 flagged
- 500-token spans: 53 loaded, 53 flagged
- generation run: 250 kept, 61 stopped before 195 tokens, 24 skipped for prompt length, 0 out-of-memory events. Both span sets are selections (the model had to keep generating).
- the 500-token spans are extensions of the 200-token spans of the same prompts, so the two groups are never pooled.

## 2. Models and environment

- **Generator of the watermarked text:** Llama-2-7B base, 6,738,415,616 parameters (computed from its published configuration, not re-measured in this run), fp16
- **Paraphraser:** Qwen/Qwen2.5-7B-Instruct, revision `a09a35458c702b33eeacc393d103063234e8bc28`, **7,615,616,512 parameters** (computed from the architecture; transformers reports 7,615,616,512; agree: True), 28 layers, hidden size 3584, 28 attention heads (4 key/value), vocabulary 152064, context 32768 tokens
  - 4-bit nf4 with double quantisation, float16 compute; stored elements by dtype: {'torch.bfloat16': 1090328064, 'torch.uint8': 3262644224}; model memory footprint 5.07 GiB; GPU allocated after load 5.26 GiB; load time 4.6 s
- **Similarity model:** sentence-transformers/all-MiniLM-L6-v2, 22,713,216 parameters
- **Detector:** the STA detector of the paper, queried as a black box that returns z-scores; the key is never used by the attack
- GPU: Quadro RTX 5000 | python 3.11.16 | torch 2.7.1+cu128 | transformers 5.17.0 | bitsandbytes 0.50.2 | accelerate 1.15.0 | WordNet available: True
- peak GPU memory during the run: allocated 5.51 GiB, reserved 5.56 GiB

## 3. Settings

- gates (lenient): {'semantic': 0.75, 'fluency': 1.8, 'min_for_fluency': 3, 'spelling': True, 'max_unknown': 0, 'check_numbers': True, 'check_quotes': True, 'check_negations': True, 'check_names': False}
- sampling: {'temperature': 1.5, 'top_p': 0.98, 'top_k': 20, 'repetition_penalty': 1.05, 'batch': 4, 'length_factor': 2.0, 'extra_tokens': 32}
- attack: {'n_cands': 60, 'seed': 0, 'max_minutes': 0} | segmentation: {'min_chars': 30, 'quote_words': 4, 'abbreviations': 50}

## 4. Results (never pooled across lengths)

Columns marked mean / median / max.

### 500-token spans: 53 attacked, **16 broken (30%)**

- all attacked (n=53): z0 16.4 / 16.2 / 19.1 | z final 6.4 / 6.1 / 18.8 | sentences rewritten % of eligible 96.7 / 100.0 / 100.0 | tokens not preserved % 28.6 / 29.5 / 49.3 | tokens that are new % 29.1 / 29.4 / 50.7 | words not preserved % 35.6 / 37.4 / 57.7 | characters not preserved % 22.1 / 23.5 / 37.6 | length change (tokens) 4.0 / 3.0 / 53.0 | queries 1383.6 / 1087.0 / 4319.0 | minutes 27.7 / 21.9 / 81.8
- broken only (n=16): z0 16.8 / 16.7 / 18.9 | z final 1.9 / 1.9 / 2.0 | sentences rewritten % of eligible 98.1 / 100.0 / 100.0 | tokens not preserved % 42.4 / 42.1 / 49.3 | tokens that are new % 43.3 / 44.3 / 50.7 | words not preserved % 51.1 / 51.6 / 57.7 | characters not preserved % 31.8 / 31.8 / 37.6 | length change (tokens) 8.9 / 9.5 / 53.0 | queries 1339.9 / 872.5 / 3229.0 | minutes 23.3 / 16.8 / 55.6
- outcomes: {'improved but not enough (stalled)': 36, 'broken': 16, 'no eligible text': 1}

## 5. Per-sample table

tokens = span length; eligible = share of characters that may be rewritten; needed = share that must be rewritten at chance level (1 - 2/z0); delta needed = selection gain (how much less green than chance the rewrites must be) if eligible < needed; delta achieved = 0.5 - green fraction of the new token pairs. Percentages of tokens/characters are of the ORIGINAL span (not preserved) and of the FINAL span (new).

| sample | tokens | z0 | eligible | needed | delta needed | outcome | z final | rewritten | tokens not kept % | tokens new % | chars not kept % | length change | queries | min | delta achieved |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sta16_len500:100 | 500 | 16.90 | 39% | 88% | 0.48 | stalled (improved but not enough (stalled)) | 10.33 | 4 of 4 (100%) | 20.4 | 21.0 | 16.1 | +4 | 347 | 10.7 | 0.11 |
| sta16_len500:102 | 500 | 18.69 | 67% | 89% | 0.14 | stalled (improved but not enough (stalled)) | 4.70 | 13 of 13 (100%) | 35.7 | 35.6 | 25.6 | -1 | 1143 | 25.0 | 0.18 |
| sta16_len500:104 | 500 | 16.19 | 65% | 88% | 0.13 | stalled (improved but not enough (stalled)) | 5.77 | 7 of 7 (100%) | 29.3 | 29.2 | 21.2 | -1 | 1945 | 54.2 | 0.17 |
| sta16_len500:106 | 500 | 16.84 | 91% | 88% | 0.00 | stalled (improved but not enough (stalled)) | 2.20 | 13 of 13 (100%) | 38.4 | 35.5 | 33.3 | -23 | 4319 | 81.8 | 0.20 |
| sta16_len500:107 | 500 | 13.86 | 84% | 86% | 0.01 | broken | 1.93 | 10 of 10 (100%) | 39.7 | 36.7 | 29.8 | -24 | 824 | 16.8 | 0.13 |
| sta16_len500:108 | 500 | 15.74 | 71% | 87% | 0.08 | stalled (improved but not enough (stalled)) | 4.73 | 8 of 8 (100%) | 39.5 | 36.3 | 28.3 | -25 | 753 | 19.4 | 0.11 |
| sta16_len500:109 | 500 | 15.83 | 38% | 87% | 0.46 | stalled (improved but not enough (stalled)) | 10.34 | 5 of 5 (100%) | 18.0 | 17.8 | 13.7 | -1 | 1087 | 15.8 | 0.12 |
| sta16_len500:110 | 500 | 17.89 | 91% | 89% | 0.00 | broken | 1.70 | 14 of 14 (100%) | 42.7 | 45.6 | 33.1 | +27 | 557 | 8.9 | 0.15 |
| sta16_len500:114 | 500 | 15.65 | 54% | 87% | 0.21 | stalled (improved but not enough (stalled)) | 8.06 | 9 of 9 (100%) | 15.6 | 18.0 | 13.7 | +15 | 2143 | 41.0 | 0.24 |
| sta16_len500:115 | 500 | 14.76 | 23% | 86% | impossible | stalled (improved but not enough (stalled)) | 11.59 | 3 of 3 (100%) | 11.2 | 12.4 | 7.9 | +7 | 462 | 10.4 | 0.08 |
| sta16_len500:120 | 500 | 14.85 | 92% | 87% | 0.00 | broken | 1.69 | 17 of 17 (100%) | 35.3 | 39.6 | 28.9 | +35 | 2178 | 33.6 | 0.15 |
| sta16_len500:123 | 500 | 14.76 | 72% | 86% | 0.07 | stalled (improved but not enough (stalled)) | 6.05 | 9 of 9 (100%) | 29.5 | 28.1 | 26.3 | -10 | 826 | 21.9 | 0.09 |
| sta16_len500:125 | 500 | 17.62 | 52% | 89% | 0.27 | stalled (improved but not enough (stalled)) | 10.01 | 5 of 5 (100%) | 24.4 | 22.3 | 17.4 | -13 | 420 | 14.1 | 0.10 |
| sta16_len500:128 | 500 | 16.28 | 36% | 88% | impossible | stalled (improved but not enough (stalled)) | 9.61 | 6 of 6 (100%) | 16.4 | 15.7 | 13.7 | -4 | 660 | 13.1 | 0.22 |
| sta16_len500:132 | 500 | 11.72 | 85% | 83% | 0.00 | stalled (improved but not enough (stalled)) | 2.60 | 10 of 10 (100%) | 29.9 | 34.1 | 25.7 | +32 | 3082 | 80.8 | 0.14 |
| sta16_len500:134 | 500 | 18.78 | 69% | 89% | 0.12 | stalled (improved but not enough (stalled)) | 7.25 | 10 of 10 (100%) | 31.5 | 28.8 | 24.7 | -19 | 1761 | 39.4 | 0.18 |
| sta16_len500:135 | 500 | 17.80 | 80% | 89% | 0.04 | stalled (improved but not enough (stalled)) | 7.47 | 8 of 9 (89%) | 29.5 | 30.4 | 22.5 | +6 | 2048 | 61.2 | 0.15 |
| sta16_len500:136 | 500 | 18.25 | 86% | 89% | 0.01 | broken | 1.89 | 15 of 16 (94%) | 47.3 | 46.9 | 34.8 | -4 | 501 | 7.9 | 0.12 |
| sta16_len500:138 | 500 | 15.65 | 30% | 87% | impossible | stalled (improved but not enough (stalled)) | 12.04 | 2 of 2 (100%) | 10.8 | 12.5 | 8.9 | +10 | 150 | 9.7 | 0.08 |
| sta16_len500:139 | 500 | 16.28 | 5% | 88% | impossible | stalled (improved but not enough (stalled)) | 15.74 | 1 of 1 (100%) | 2.0 | 2.6 | 1.7 | +3 | 4 | 1.4 | 0.03 |
| sta16_len500:140 | 500 | 18.78 | 0% | 89% | impossible | no eligible text (no eligible text) | 18.78 | 0 of 0 (0%) | 0.0 | 0.0 | 0.0 | +0 | 1 | 0.0 | - |
| sta16_len500:141 | 500 | 18.87 | 93% | 89% | 0.00 | broken | 1.90 | 15 of 15 (100%) | 47.1 | 50.7 | 35.0 | +36 | 669 | 16.9 | 0.11 |
| sta16_len500:143 | 500 | 15.29 | 76% | 87% | 0.05 | stalled (improved but not enough (stalled)) | 3.90 | 8 of 8 (100%) | 29.5 | 27.5 | 23.7 | -14 | 1074 | 19.8 | 0.17 |
| sta16_len500:144 | 500 | 15.83 | 55% | 87% | 0.20 | stalled (improved but not enough (stalled)) | 6.65 | 10 of 10 (100%) | 24.6 | 27.7 | 19.3 | +22 | 2158 | 30.8 | 0.16 |
| sta16_len500:148 | 500 | 17.08 | 57% | 88% | 0.21 | stalled (improved but not enough (stalled)) | 6.62 | 14 of 14 (100%) | 26.5 | 29.5 | 22.1 | +21 | 1825 | 26.3 | 0.17 |
| sta16_len500:152 | 500 | 16.16 | 77% | 88% | 0.05 | stalled (improved but not enough (stalled)) | 3.25 | 13 of 14 (93%) | 34.0 | 34.5 | 29.7 | +4 | 2618 | 40.2 | 0.17 |
| sta16_len500:154 | 500 | 16.10 | 50% | 88% | 0.27 | stalled (improved but not enough (stalled)) | 8.89 | 8 of 8 (100%) | 22.4 | 23.3 | 16.6 | +6 | 1338 | 33.4 | 0.11 |
| sta16_len500:156 | 500 | 17.71 | 91% | 89% | 0.00 | broken | 2.00 | 20 of 20 (100%) | 42.1 | 47.7 | 34.9 | +53 | 1280 | 20.1 | 0.14 |
| sta16_len500:29 | 500 | 16.16 | 7% | 88% | impossible | stalled (improved but not enough (stalled)) | 15.36 | 1 of 1 (100%) | 2.2 | 1.6 | 1.9 | -3 | 52 | 1.2 | 0.12 |
| sta16_len500:31 | 500 | 15.56 | 76% | 87% | 0.05 | stalled (improved but not enough (stalled)) | 4.56 | 11 of 11 (100%) | 32.5 | 31.3 | 27.1 | -9 | 1652 | 34.9 | 0.14 |
| sta16_len500:32 | 500 | 18.07 | 22% | 89% | impossible | stalled (improved but not enough (stalled)) | 14.43 | 3 of 3 (100%) | 12.2 | 12.4 | 9.9 | +1 | 449 | 7.8 | 0.08 |
| sta16_len500:33 | 500 | 17.35 | 91% | 88% | 0.00 | broken | 1.94 | 13 of 13 (100%) | 43.5 | 45.0 | 30.7 | +14 | 1541 | 29.3 | 0.16 |
| sta16_len500:35 | 500 | 16.64 | 91% | 88% | 0.00 | broken | 1.98 | 14 of 14 (100%) | 40.9 | 40.0 | 31.9 | -8 | 886 | 16.7 | 0.16 |
| sta16_len500:37 | 500 | 17.44 | 85% | 89% | 0.02 | broken | 1.95 | 17 of 17 (100%) | 49.3 | 48.0 | 32.5 | -13 | 2824 | 38.2 | 0.14 |
| sta16_len500:44 | 500 | 15.65 | 88% | 87% | 0.00 | broken | 1.56 | 10 of 10 (100%) | 41.5 | 38.8 | 31.7 | -22 | 391 | 7.5 | 0.17 |
| sta16_len500:47 | 500 | 16.55 | 90% | 88% | 0.00 | broken | 1.69 | 12 of 13 (92%) | 46.1 | 46.7 | 37.6 | +6 | 643 | 14.0 | 0.14 |
| sta16_len500:50 | 500 | 19.05 | 61% | 90% | 0.20 | stalled (improved but not enough (stalled)) | 8.40 | 8 of 9 (89%) | 32.9 | 35.1 | 23.5 | +17 | 825 | 36.7 | 0.08 |
| sta16_len500:51 | 500 | 14.13 | 39% | 86% | 0.38 | stalled (improved but not enough (stalled)) | 8.53 | 4 of 4 (100%) | 16.6 | 16.7 | 12.7 | +1 | 993 | 21.2 | 0.16 |
| sta16_len500:54 | 500 | 19.05 | 72% | 90% | 0.10 | stalled (improved but not enough (stalled)) | 7.09 | 11 of 13 (85%) | 28.9 | 29.4 | 24.6 | +3 | 3028 | 73.3 | 0.22 |
| sta16_len500:55 | 500 | 15.95 | 67% | 87% | 0.11 | stalled (improved but not enough (stalled)) | 5.40 | 12 of 12 (100%) | 29.9 | 32.3 | 22.6 | +18 | 2274 | 44.0 | 0.13 |
| sta16_len500:57 | 500 | 14.49 | 64% | 86% | 0.11 | stalled (improved but not enough (stalled)) | 7.42 | 7 of 7 (100%) | 26.5 | 28.3 | 18.5 | +12 | 1868 | 32.9 | 0.07 |
| sta16_len500:59 | 500 | 14.94 | 44% | 87% | 0.33 | stalled (improved but not enough (stalled)) | 8.37 | 7 of 7 (100%) | 21.0 | 20.8 | 17.6 | -1 | 1381 | 23.0 | 0.17 |
| sta16_len500:60 | 500 | 16.10 | 50% | 88% | 0.27 | stalled (improved but not enough (stalled)) | 7.90 | 7 of 7 (100%) | 23.0 | 22.3 | 20.6 | -4 | 1707 | 25.8 | 0.16 |
| sta16_len500:61 | 500 | 16.43 | 75% | 88% | 0.07 | broken | 1.98 | 16 of 16 (100%) | 34.4 | 36.4 | 26.5 | +16 | 1885 | 30.4 | 0.23 |
| sta16_len500:72 | 500 | 14.64 | 37% | 86% | 0.43 | stalled (improved but not enough (stalled)) | 9.53 | 4 of 4 (100%) | 14.4 | 15.2 | 11.9 | +5 | 688 | 15.3 | 0.16 |
| sta16_len500:74 | 500 | 14.85 | 47% | 87% | 0.28 | stalled (improved but not enough (stalled)) | 9.76 | 6 of 6 (100%) | 12.8 | 13.3 | 10.2 | +3 | 653 | 13.0 | 0.20 |
| sta16_len500:75 | 500 | 16.64 | 64% | 88% | 0.14 | stalled (improved but not enough (stalled)) | 9.17 | 10 of 10 (100%) | 17.6 | 21.3 | 12.7 | +24 | 2546 | 46.3 | 0.14 |
| sta16_len500:76 | 500 | 15.03 | 41% | 87% | 0.38 | stalled (improved but not enough (stalled)) | 8.58 | 4 of 4 (100%) | 18.0 | 16.3 | 13.5 | -10 | 689 | 19.8 | 0.18 |
| sta16_len500:78 | 500 | 16.64 | 79% | 88% | 0.04 | broken | 1.99 | 17 of 17 (100%) | 42.1 | 43.6 | 30.8 | +13 | 3229 | 55.6 | 0.18 |
| sta16_len500:80 | 500 | 18.78 | 93% | 89% | 0.00 | broken | 1.67 | 17 of 19 (89%) | 47.5 | 46.8 | 33.2 | -7 | 582 | 8.3 | 0.12 |
| sta16_len500:86 | 499 | 16.70 | 97% | 88% | 0.00 | broken | 1.87 | 16 of 16 (100%) | 40.4 | 40.8 | 29.1 | +3 | 859 | 13.8 | 0.17 |
| sta16_len500:89 | 500 | 15.98 | 80% | 87% | 0.04 | broken | 1.94 | 15 of 16 (94%) | 38.4 | 40.4 | 28.2 | +17 | 2590 | 55.1 | 0.17 |
| sta16_len500:97 | 500 | 15.74 | 68% | 87% | 0.10 | stalled (improved but not enough (stalled)) | 5.90 | 9 of 9 (100%) | 28.9 | 27.9 | 21.3 | -7 | 2925 | 46.8 | 0.15 |

## 6. Where it breaks

- 500-token spans, broken spans: sentences rewritten at the break 14.9 / 15.0 / 20.0 | queries at the break 1339.9 / 872.5 / 3229.0
- 500-token spans: 12 need no selection gain (eligible text enough at chance level), 34 need delta > 0, of which the achieved delta reached the need in 19; 7 are impossible by rewriting (eligible share too small even for delta 0.5)

## 7. Gates: where candidates are lost

- candidates generated 150120 -> after clean 111443 -> spelling 105021 -> content 89910 -> semantic 79464 -> fluency 74485
- rejected by reason: {'duplicate': 32078, 'meaning too different': 10446, 'negation count changed': 10378, 'unknown word': 6250, 'CJK text': 5281, 'perplexity too high': 4979, 'quotation changed': 3662, 'numbers changed': 1071, 'several lines': 733, 'preamble': 402, 'glued punctuation': 172, 'hit length limit': 110, 'identical to original': 72, 'too short': 1}

## 8. Resources and problems

- time (hours): attack 24.43 | candidate generation 22.76 | gates 1.66 | detector queries 0.01
- detector queries: 73333 in total (screening queries not included)
- out-of-memory events during rewriting: 0 | rewrites that hit the length limit: 193 | samples that raised an error: 0
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

Command: `research/qwen_attack/run_attack.py --files results/raw/sta16_len500.json --resume results/qwen_attack_sta16_500/attack`. Files: `results/raw/sta16_len500.json`. Gate config and seeds are in run.json; every sample's steps, funnels and rewrites are in `samples/`.
