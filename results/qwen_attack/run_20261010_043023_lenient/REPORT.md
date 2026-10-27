# Qwen paraphrase attack on the STA watermark: final report

Run: `run_20261010_043023_lenient`. Started 2026-10-10 04:30:32, ended 2026-10-10 06:16:56. Git commit `adf6cc65c7fdc046f185d04c94ff87d1ed91573d`.

**What was attacked.** Only the generated span of each sample, never the prompt. The detector was asked only about span text. **Success** means the span's z-score is 2.0 or below under the same detector. Preset: **lenient**. Quality is judged by reading (see READING.md); the gates below are filters, not proof of faithfulness.

## 1. Data

- spans loaded: 50 | flagged by the detector (z0 > 2): 46 | not flagged: 4
- not flagged (not attacked): baseline_extra:21 (z0 0.424), samples_len200:45 (z0 0.707), samples_len200:55 (z0 1.838), samples_len200:62 (z0 1.556)
- 200-token spans: 50 loaded, 46 flagged
- 500-token spans: 0 loaded, 0 flagged
- generation run: 37 kept, 13 stopped before 195 tokens, 11 skipped for prompt length, 0 out-of-memory events. Both span sets are selections (the model had to keep generating).
- the 500-token spans are extensions of the 200-token spans of the same prompts, so the two groups are never pooled.

## 2. Models and environment

- **Generator of the watermarked text:** Llama-2-7B base, 6,738,415,616 parameters (computed from its published configuration, not re-measured in this run), fp16
- **Paraphraser:** Qwen/Qwen2.5-7B-Instruct, revision `a09a35458c702b33eeacc393d103063234e8bc28`, **7,615,616,512 parameters** (computed from the architecture; transformers reports 7,615,616,512; agree: True), 28 layers, hidden size 3584, 28 attention heads (4 key/value), vocabulary 152064, context 32768 tokens
  - 4-bit nf4 with double quantisation, float16 compute; stored elements by dtype: {'torch.bfloat16': 1090328064, 'torch.uint8': 3262644224}; model memory footprint 5.07 GiB; GPU allocated after load 5.26 GiB; load time 4.9 s
- **Similarity model:** sentence-transformers/all-MiniLM-L6-v2, 22,713,216 parameters
- **Detector:** the STA detector of the paper, queried as a black box that returns z-scores; the key is never used by the attack
- GPU: Quadro RTX 5000 | python 3.11.16 | torch 2.7.1+cu128 | transformers 5.17.0 | bitsandbytes 0.50.2 | accelerate 1.15.0 | WordNet available: True
- peak GPU memory during the run: allocated 5.48 GiB, reserved 5.55 GiB

## 3. Settings

- gates (lenient): {'semantic': 0.75, 'fluency': 1.8, 'min_for_fluency': 3, 'spelling': True, 'max_unknown': 0, 'check_numbers': True, 'check_quotes': True, 'check_negations': True, 'check_names': False}
- sampling: {'temperature': 1.5, 'top_p': 0.98, 'top_k': 20, 'repetition_penalty': 1.05, 'batch': 4, 'length_factor': 2.0, 'extra_tokens': 32}
- attack: {'n_cands': 60, 'seed': 0, 'max_minutes': 0} | segmentation: {'min_chars': 30, 'quote_words': 4, 'abbreviations': 50}

## 4. Results (never pooled across lengths)

Columns marked mean / median / max.

### 200-token spans: 46 attacked, **35 broken (76%)**

- all attacked (n=46): z0 4.5 / 4.7 / 6.8 | z final 2.0 / 1.8 / 4.9 | sentences rewritten % of eligible 69.9 / 70.8 / 100.0 | tokens not preserved % 18.7 / 17.4 / 43.3 | tokens that are new % 17.7 / 15.9 / 37.7 | words not preserved % 24.1 / 24.2 / 51.8 | characters not preserved % 15.4 / 15.2 / 34.8 | length change (tokens) -2.4 / -2.0 / 19.0 | queries 121.2 / 104.5 / 457.0 | minutes 2.3 / 2.0 / 8.4
- broken only (n=35): z0 4.4 / 4.7 / 6.8 | z final 1.6 / 1.7 / 2.0 | sentences rewritten % of eligible 67.8 / 66.7 / 100.0 | tokens not preserved % 21.1 / 19.9 / 43.3 | tokens that are new % 19.5 / 18.3 / 37.7 | words not preserved % 26.9 / 24.8 / 51.8 | characters not preserved % 17.1 / 16.8 / 34.8 | length change (tokens) -3.9 / -3.0 / 11.0 | queries 106.9 / 101.0 / 252.0 | minutes 1.9 / 2.0 / 4.7
- outcomes: {'broken': 35, 'improved but not enough (stalled)': 9, 'no candidate passed the gates': 1, 'no eligible text': 1}

## 5. Per-sample table

tokens = span length; eligible = share of characters that may be rewritten; needed = share that must be rewritten at chance level (1 - 2/z0); delta needed = selection gain (how much less green than chance the rewrites must be) if eligible < needed; delta achieved = 0.5 - green fraction of the new token pairs. Percentages of tokens/characters are of the ORIGINAL span (not preserved) and of the FINAL span (new).

| sample | tokens | z0 | eligible | needed | delta needed | outcome | z final | rewritten | tokens not kept % | tokens new % | chars not kept % | length change | queries | min | delta achieved |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_extra:20 | 200 | 3.39 | 32% | 41% | 0.03 | broken | 1.90 | 2 of 2 (100%) | 11.4 | 12.7 | 9.7 | +3 | 45 | 1.1 | 0.18 |
| baseline_extra:22 | 200 | 4.67 | 77% | 57% | 0.00 | broken | 1.35 | 4 of 6 (67%) | 29.4 | 28.3 | 21.9 | -3 | 97 | 2.1 | 0.15 |
| baseline_extra:23 | 200 | 5.23 | 47% | 62% | 0.06 | stalled (improved but not enough (stalled)) | 2.78 | 4 of 4 (100%) | 19.4 | 25.7 | 25.8 | +17 | 203 | 5.6 | 0.12 |
| baseline_extra:24 | 200 | 5.09 | 75% | 61% | 0.00 | broken | 0.46 | 4 of 4 (100%) | 43.3 | 34.1 | 34.8 | -28 | 189 | 2.3 | 0.13 |
| baseline_extra:25 | 200 | 3.82 | 37% | 48% | 0.04 | stalled (improved but not enough (stalled)) | 2.60 | 2 of 2 (100%) | 9.0 | 10.3 | 4.9 | +3 | 95 | 2.7 | 0.12 |
| baseline_extra:26 | 200 | 2.83 | 60% | 29% | 0.00 | broken | 1.13 | 1 of 4 (25%) | 7.0 | 7.9 | 5.8 | +2 | 54 | 0.5 | 0.29 |
| baseline_extra:27 | 200 | 6.22 | 94% | 68% | 0.00 | broken | 1.74 | 6 of 6 (100%) | 32.8 | 29.3 | 28.3 | -10 | 186 | 3.1 | 0.14 |
| baseline_safe:0 | 200 | 3.82 | 29% | 48% | 0.09 | stalled (improved but not enough (stalled)) | 2.56 | 2 of 2 (100%) | 13.4 | 12.6 | 16.5 | -2 | 167 | 1.6 | 0.15 |
| baseline_safe:1 | 200 | 6.51 | 51% | 69% | 0.08 | stalled (improved but not enough (stalled)) | 3.41 | 3 of 4 (75%) | 16.9 | 16.1 | 12.1 | -2 | 205 | 5.8 | 0.19 |
| baseline_safe:2 | 200 | 5.37 | 72% | 63% | 0.00 | broken | 1.56 | 3 of 3 (100%) | 37.8 | 37.2 | 29.3 | -2 | 140 | 2.3 | 0.06 |
| baseline_safe:3 | 170 | 3.68 | 48% | 46% | 0.00 | broken | 1.72 | 3 of 3 (100%) | 16.4 | 13.3 | 15.9 | -6 | 71 | 1.3 | 0.19 |
| baseline_safe:4 | 200 | 3.54 | 83% | 43% | 0.00 | broken | 1.92 | 2 of 6 (33%) | 15.9 | 14.6 | 10.7 | -3 | 61 | 2.0 | 0.13 |
| samples_len200:28 | 200 | 5.37 | 46% | 63% | 0.07 | stalled (improved but not enough (stalled)) | 2.23 | 5 of 5 (100%) | 20.4 | 17.5 | 16.5 | -7 | 457 | 8.4 | 0.20 |
| samples_len200:29 | 200 | 4.81 | 79% | 58% | 0.00 | broken | 1.79 | 2 of 5 (40%) | 28.9 | 27.0 | 23.2 | -5 | 98 | 1.3 | 0.14 |
| samples_len200:30 | 200 | 4.10 | 58% | 51% | 0.00 | broken | 1.71 | 3 of 3 (100%) | 15.4 | 14.6 | 12.0 | -2 | 173 | 2.8 | 0.13 |
| samples_len200:31 | 200 | 5.09 | 50% | 61% | 0.04 | broken | 1.71 | 3 of 3 (100%) | 18.4 | 17.6 | 19.4 | -2 | 147 | 2.2 | 0.15 |
| samples_len200:34 | 200 | 3.25 | 75% | 39% | 0.00 | broken | 1.62 | 3 of 5 (60%) | 14.9 | 15.3 | 9.9 | +1 | 114 | 1.8 | 0.13 |
| samples_len200:35 | 200 | 5.94 | 87% | 66% | 0.00 | broken | 1.35 | 4 of 8 (50%) | 26.4 | 26.0 | 18.9 | -1 | 150 | 2.0 | 0.17 |
| samples_len200:36 | 200 | 5.09 | 46% | 61% | 0.06 | broken | 1.78 | 3 of 3 (100%) | 21.9 | 20.7 | 13.0 | -3 | 252 | 4.5 | 0.19 |
| samples_len200:40 | 200 | 4.10 | 21% | 51% | 0.20 | stalled (no candidate passed the gates) | 4.10 | 0 of 1 (0%) | 0.0 | 0.0 | 0.0 | +0 | 1 | 0.6 | - |
| samples_len200:42 | 200 | 3.68 | 34% | 46% | 0.05 | broken | 1.95 | 1 of 1 (100%) | 17.9 | 14.1 | 11.9 | -9 | 32 | 2.0 | 0.08 |
| samples_len200:43 | 200 | 6.79 | 68% | 71% | 0.01 | broken | 2.00 | 5 of 5 (100%) | 34.3 | 37.7 | 27.2 | +11 | 154 | 2.6 | 0.12 |
| samples_len200:44 | 200 | 5.09 | 90% | 61% | 0.00 | broken | 1.09 | 3 of 6 (50%) | 28.9 | 25.5 | 21.0 | -9 | 101 | 1.6 | 0.15 |
| samples_len200:46 | 200 | 2.12 | 39% | 6% | 0.00 | broken | 1.79 | 1 of 3 (33%) | 8.5 | 6.1 | 7.7 | -5 | 2 | 0.8 | 0.15 |
| samples_len200:47 | 200 | 5.52 | 75% | 64% | 0.00 | broken | 1.16 | 3 of 4 (75%) | 30.3 | 26.7 | 26.3 | -10 | 136 | 2.4 | 0.14 |
| samples_len200:51 | 200 | 3.39 | 20% | 41% | 0.13 | stalled (improved but not enough (stalled)) | 2.42 | 1 of 1 (100%) | 7.5 | 6.5 | 5.2 | -2 | 94 | 1.3 | 0.05 |
| samples_len200:52 | 200 | 5.23 | 88% | 62% | 0.00 | broken | 1.56 | 5 of 10 (50%) | 25.9 | 25.9 | 18.7 | +0 | 111 | 1.9 | 0.11 |
| samples_len200:56 | 200 | 3.54 | 73% | 43% | 0.00 | broken | 1.60 | 2 of 6 (33%) | 16.9 | 11.6 | 16.8 | -12 | 68 | 1.2 | 0.16 |
| samples_len200:60 | 200 | 5.52 | 73% | 64% | 0.00 | broken | 1.58 | 2 of 3 (67%) | 25.4 | 15.7 | 23.5 | -23 | 97 | 1.5 | 0.28 |
| samples_len200:61 | 200 | 5.09 | 83% | 61% | 0.00 | broken | 0.98 | 4 of 5 (80%) | 29.4 | 30.7 | 23.9 | +4 | 174 | 2.6 | 0.14 |
| samples_len200:63 | 200 | 6.65 | 54% | 70% | 0.07 | stalled (improved but not enough (stalled)) | 3.31 | 4 of 4 (100%) | 15.9 | 23.2 | 15.3 | +19 | 394 | 6.8 | 0.17 |
| samples_len200:64 | 200 | 4.10 | 69% | 51% | 0.00 | broken | 1.80 | 4 of 6 (67%) | 19.9 | 23.0 | 12.5 | +8 | 175 | 4.7 | 0.08 |
| samples_len200:65 | 200 | 2.26 | 27% | 12% | 0.00 | broken | 1.77 | 2 of 2 (100%) | 8.5 | 8.0 | 6.2 | -1 | 11 | 0.9 | 0.05 |
| samples_len200:66 | 200 | 5.80 | 69% | 66% | 0.00 | broken | 0.99 | 4 of 4 (100%) | 32.3 | 31.7 | 25.1 | -2 | 119 | 2.3 | 0.13 |
| samples_len200:69 | 200 | 3.11 | 75% | 36% | 0.00 | broken | 1.78 | 1 of 4 (25%) | 10.9 | 9.6 | 8.0 | -3 | 54 | 1.0 | 0.16 |
| samples_len200:70 | 200 | 4.67 | 56% | 57% | 0.00 | broken | 1.80 | 2 of 2 (100%) | 24.9 | 22.2 | 26.8 | -7 | 82 | 2.0 | 0.09 |
| samples_len200:71 | 200 | 4.95 | 0% | 60% | impossible | no eligible text (no eligible text) | 4.95 | 0 of 0 (0%) | 0.0 | 0.0 | 0.0 | +0 | 1 | 0.0 | - |
| samples_len200:72 | 200 | 5.23 | 28% | 62% | 0.22 | stalled (improved but not enough (stalled)) | 3.76 | 3 of 3 (100%) | 10.4 | 13.0 | 8.5 | +6 | 204 | 3.9 | 0.07 |
| samples_len200:73 | 200 | 3.96 | 62% | 49% | 0.00 | broken | 1.98 | 2 of 4 (50%) | 12.4 | 12.4 | 11.9 | +0 | 108 | 1.8 | 0.13 |
| samples_len200:74 | 200 | 2.97 | 51% | 33% | 0.00 | broken | 1.64 | 2 of 4 (50%) | 10.9 | 9.6 | 7.1 | -3 | 74 | 1.1 | 0.11 |
| samples_len200:78 | 200 | 4.95 | 65% | 60% | 0.00 | broken | 1.71 | 3 of 6 (50%) | 19.9 | 18.3 | 19.6 | -4 | 112 | 1.8 | 0.15 |
| samples_len200:81 | 200 | 5.23 | 78% | 62% | 0.00 | broken | 1.66 | 4 of 6 (67%) | 21.9 | 19.1 | 18.0 | -7 | 131 | 2.0 | 0.19 |
| samples_len200:84 | 200 | 2.55 | 94% | 21% | 0.00 | broken | 1.73 | 1 of 8 (12%) | 10.0 | 6.2 | 9.7 | -8 | 53 | 0.6 | 0.22 |
| samples_len200:85 | 200 | 3.96 | 88% | 49% | 0.00 | broken | 1.61 | 3 of 6 (50%) | 17.9 | 19.1 | 15.1 | +3 | 88 | 2.2 | 0.11 |
| samples_len200:86 | 200 | 3.39 | 56% | 41% | 0.00 | broken | 1.77 | 2 of 5 (40%) | 11.4 | 11.0 | 7.2 | -1 | 81 | 0.9 | 0.21 |
| samples_len200:87 | 200 | 4.81 | 40% | 58% | 0.08 | stalled (improved but not enough (stalled)) | 4.23 | 2 of 3 (67%) | 9.5 | 7.1 | 7.6 | -5 | 15 | 2.3 | 0.08 |

## 6. Where it breaks

- 200-token spans, broken spans: sentences rewritten at the break 2.8 / 3.0 / 6.0 | queries at the break 106.9 / 101.0 / 252.0
- 200-token spans: 29 need no selection gain (eligible text enough at chance level), 16 need delta > 0, of which the achieved delta reached the need in 13; 1 are impossible by rewriting (eligible share too small even for delta 0.5)

## 7. Gates: where candidates are lost

- candidates generated 12180 -> after clean 8731 -> spelling 8209 -> content 6491 -> semantic 5960 -> fluency 5615
- rejected by reason: {'duplicate': 3082, 'negation count changed': 1246, 'meaning too different': 531, 'unknown word': 492, 'perplexity too high': 345, 'CJK text': 322, 'quotation changed': 316, 'numbers changed': 156, 'several lines': 37, 'glued punctuation': 30, 'hit length limit': 8}

## 8. Resources and problems

- time (hours): attack 1.77 | candidate generation 1.65 | gates 0.12 | detector queries 0.00
- detector queries: 5576 in total (screening queries not included)
- out-of-memory events during rewriting: 0 | rewrites that hit the length limit: 14 | samples that raised an error: 0
- HTML entities (such as &apos;) removed by rewriting: 3 across all spans. Rewriting turns them into plain characters, which changes tokenisation without paraphrasing; this effect is not separated here.

## 9. Limitations

- Sentence boundaries are a heuristic; sentences ending in an abbreviation merge with the next one.
- About a third of each span is not attackable by design (the mid-sentence start, the cut-off end, quotations, short or non-prose pieces).
- The gates cannot see all meaning changes (for example income -> revenue). Faithfulness is established only by reading READING.md.
- The necessary-condition bound uses the eligible character share as a stand-in for the share of token pairs.
- The spans are a selection: the model had to continue for 195 tokens (500-token set: 495). The two lengths are not independent samples.
- Spans are model-generated at temperature 1.0 and already contain glued words and HTML entities copied from C4.
- One detector (the STA detector with its published key) and one paraphrasing model; results do not transfer automatically to other models.

## 10. Reproduction

Command: `research/qwen_attack/run_attack.py --files results/raw/baseline_safe.json results/raw/baseline_extra.json results/raw/samples_len200.json`. Files: `results/raw/baseline_safe.json`, `results/raw/baseline_extra.json`, `results/raw/samples_len200.json`. Gate config and seeds are in run.json; every sample's steps, funnels and rewrites are in `samples/`.
