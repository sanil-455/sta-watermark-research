# Qwen paraphrase attack on the STA watermark: final report

Run: `ncands90`. Started 2026-10-31 08:14:35, ended 2026-10-31 09:01:04. Git commit `645284db7983152eb7a1ff703b5b02afd9d0d5dc`.

**What was attacked.** Only the generated span of each sample, never the prompt. The detector was asked only about span text. **Success** means the span's z-score is 2.0 or below under the same detector. Preset: **lenient**. Quality is judged by reading (see READING.md); the gates below are filters, not proof of faithfulness.

## 1. Data

- spans loaded: 263 | flagged by the detector (z0 > 2): 253 | not flagged: 10
- not flagged (not attacked): baseline_extra:21 (z0 0.424), samples_len200:45 (z0 0.707), samples_len200:55 (z0 1.838), samples_len200:62 (z0 1.556), samples_len200:107 (z0 1.273), samples_len200:120 (z0 0.990), samples_len200:178 (z0 1.838), samples_len200:299 (z0 1.414), samples_len200:331 (z0 0.283), samples_len200:353 (z0 1.131)
- 200-token spans: 263 loaded, 253 flagged
- 500-token spans: 0 loaded, 0 flagged
- generation run: 250 kept, 61 stopped before 195 tokens, 24 skipped for prompt length, 0 out-of-memory events. Both span sets are selections (the model had to keep generating).
- the 500-token spans are extensions of the 200-token spans of the same prompts, so the two groups are never pooled.

## 2. Models and environment

- **Generator of the watermarked text:** Llama-2-7B base, 6,738,415,616 parameters (computed from its published configuration, not re-measured in this run), fp16
- **Paraphraser:** Qwen/Qwen2.5-7B-Instruct, revision `a09a35458c702b33eeacc393d103063234e8bc28`, **7,615,616,512 parameters** (computed from the architecture; transformers reports 7,615,616,512; agree: True), 28 layers, hidden size 3584, 28 attention heads (4 key/value), vocabulary 152064, context 32768 tokens
  - 4-bit nf4 with double quantisation, float16 compute; stored elements by dtype: {'torch.bfloat16': 1090328064, 'torch.uint8': 3262644224}; model memory footprint 5.07 GiB; GPU allocated after load 5.26 GiB; load time 4.8 s
- **Similarity model:** sentence-transformers/all-MiniLM-L6-v2, 22,713,216 parameters
- **Detector:** the STA detector of the paper, queried as a black box that returns z-scores; the key is never used by the attack
- GPU: Quadro RTX 5000 | python 3.11.16 | torch 2.7.1+cu128 | transformers 5.17.0 | bitsandbytes 0.50.2 | accelerate 1.15.0 | WordNet available: True
- peak GPU memory during the run: allocated 5.48 GiB, reserved 5.55 GiB

## 3. Settings

- gates (lenient): {'semantic': 0.75, 'fluency': 1.8, 'min_for_fluency': 3, 'spelling': True, 'max_unknown': 0, 'check_numbers': True, 'check_quotes': True, 'check_negations': True, 'check_names': False}
- sampling: {'temperature': 1.5, 'top_p': 0.98, 'top_k': 20, 'repetition_penalty': 1.05, 'batch': 4, 'length_factor': 2.0, 'extra_tokens': 32}
- attack: {'n_cands': 90, 'seed': 0, 'max_minutes': 0} | segmentation: {'min_chars': 30, 'quote_words': 4, 'abbreviations': 50}

## 4. Results (never pooled across lengths)

Columns marked mean / median / max.

### 200-token spans: 12 attacked, **9 broken (75%)**

- all attacked (n=12): z0 4.5 / 4.2 / 6.5 | z final 1.9 / 1.7 / 2.6 | sentences rewritten % of eligible 81.9 / 100.0 / 100.0 | tokens not preserved % 20.4 / 17.1 / 37.8 | tokens that are new % 19.6 / 17.5 / 37.2 | words not preserved % 26.6 / 24.7 / 47.7 | characters not preserved % 17.9 / 16.2 / 30.5 | length change (tokens) -2.1 / -2.0 / 17.0 | queries 180.6 / 168.0 / 400.0 | minutes 3.9 / 3.1 / 10.4
- broken only (n=9): z0 4.6 / 4.7 / 6.5 | z final 1.6 / 1.7 / 2.0 | sentences rewritten % of eligible 75.9 / 83.3 / 100.0 | tokens not preserved % 22.7 / 20.4 / 37.8 | tokens that are new % 20.9 / 20.4 / 37.2 | words not preserved % 28.7 / 23.7 / 47.7 | characters not preserved % 18.8 / 15.9 / 30.5 | length change (tokens) -4.8 / -3.0 / 2.0 | queries 169.3 / 137.0 / 400.0 | minutes 3.5 / 3.0 / 10.4
- outcomes: {'broken': 9, 'improved but not enough (stalled)': 3}

## 5. Per-sample table

tokens = span length; eligible = share of characters that may be rewritten; needed = share that must be rewritten at chance level (1 - 2/z0); delta needed = selection gain (how much less green than chance the rewrites must be) if eligible < needed; delta achieved = 0.5 - green fraction of the new token pairs. Percentages of tokens/characters are of the ORIGINAL span (not preserved) and of the FINAL span (new).

| sample | tokens | z0 | eligible | needed | delta needed | outcome | z final | rewritten | tokens not kept % | tokens new % | chars not kept % | length change | queries | min | delta achieved |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_extra:20 | 200 | 3.39 | 32% | 41% | 0.03 | broken | 1.62 | 2 of 2 (100%) | 10.9 | 11.4 | 9.9 | +1 | 60 | 1.6 | 0.24 |
| baseline_extra:22 | 200 | 4.67 | 77% | 57% | 0.00 | broken | 1.22 | 4 of 6 (67%) | 30.8 | 28.4 | 23.9 | -7 | 137 | 3.2 | 0.17 |
| baseline_extra:23 | 200 | 5.23 | 47% | 62% | 0.06 | stalled (improved but not enough (stalled)) | 2.65 | 4 of 4 (100%) | 17.9 | 24.3 | 24.1 | +17 | 290 | 8.4 | 0.12 |
| baseline_extra:24 | 200 | 5.09 | 75% | 61% | 0.00 | broken | 1.99 | 3 of 4 (75%) | 32.8 | 27.4 | 27.9 | -15 | 199 | 2.7 | 0.08 |
| baseline_extra:25 | 200 | 3.82 | 37% | 48% | 0.04 | stalled (improved but not enough (stalled)) | 2.60 | 2 of 2 (100%) | 9.0 | 10.3 | 4.9 | +3 | 122 | 4.1 | 0.12 |
| baseline_extra:26 | 200 | 2.83 | 60% | 29% | 0.00 | broken | 1.13 | 1 of 4 (25%) | 7.0 | 7.9 | 5.8 | +2 | 79 | 0.8 | 0.29 |
| baseline_extra:27 | 200 | 6.22 | 94% | 68% | 0.00 | broken | 1.68 | 5 of 6 (83%) | 31.8 | 27.1 | 30.5 | -13 | 257 | 4.3 | 0.14 |
| baseline_safe:0 | 200 | 3.82 | 29% | 48% | 0.09 | stalled (improved but not enough (stalled)) | 2.56 | 2 of 2 (100%) | 13.4 | 12.6 | 16.5 | -2 | 231 | 2.4 | 0.15 |
| baseline_safe:1 | 200 | 6.51 | 51% | 69% | 0.08 | broken | 1.70 | 4 of 4 (100%) | 20.4 | 20.4 | 15.7 | +0 | 400 | 10.4 | 0.29 |
| baseline_safe:2 | 200 | 5.37 | 72% | 63% | 0.00 | broken | 1.56 | 3 of 3 (100%) | 37.8 | 37.2 | 29.3 | -2 | 206 | 3.6 | 0.06 |
| baseline_safe:3 | 170 | 3.68 | 48% | 46% | 0.00 | broken | 1.72 | 3 of 3 (100%) | 16.4 | 13.3 | 15.9 | -6 | 97 | 2.0 | 0.19 |
| baseline_safe:4 | 200 | 3.54 | 83% | 43% | 0.00 | broken | 1.92 | 2 of 6 (33%) | 15.9 | 14.6 | 10.7 | -3 | 89 | 3.0 | 0.13 |

## 6. Where it breaks

- 200-token spans, broken spans: sentences rewritten at the break 3.0 / 3.0 / 5.0 | queries at the break 169.3 / 137.0 / 400.0
- 200-token spans: 7 need no selection gain (eligible text enough at chance level), 5 need delta > 0, of which the achieved delta reached the need in 5; 0 are impossible by rewriting (eligible share too small even for delta 0.5)

## 7. Gates: where candidates are lost

- candidates generated 5130 -> after clean 3324 -> spelling 3160 -> content 2493 -> semantic 2305 -> fluency 2188
- rejected by reason: {'duplicate': 1668, 'negation count changed': 487, 'meaning too different': 188, 'numbers changed': 180, 'unknown word': 133, 'CJK text': 118, 'perplexity too high': 117, 'glued punctuation': 31, 'several lines': 18, 'hit length limit': 2}

## 8. Resources and problems

- time (hours): attack 0.77 | candidate generation 0.73 | gates 0.05 | detector queries 0.00
- detector queries: 2167 in total (screening queries not included)
- out-of-memory events during rewriting: 0 | rewrites that hit the length limit: 4 | samples that raised an error: 0
- HTML entities (such as &apos;) removed by rewriting: 2 across all spans. Rewriting turns them into plain characters, which changes tokenisation without paraphrasing; this effect is not separated here.

## 9. Limitations

- Sentence boundaries are a heuristic; sentences ending in an abbreviation merge with the next one.
- About a third of each span is not attackable by design (the mid-sentence start, the cut-off end, quotations, short or non-prose pieces).
- The gates cannot see all meaning changes (for example income -> revenue). Faithfulness is established only by reading READING.md.
- The necessary-condition bound uses the eligible character share as a stand-in for the share of token pairs.
- The spans are a selection: the model had to continue for 195 tokens (500-token set: 495). The two lengths are not independent samples.
- Spans are model-generated at temperature 1.0 and already contain glued words and HTML entities copied from C4.
- One detector (the STA detector with its published key) and one paraphrasing model; results do not transfer automatically to other models.

## 10. Reproduction

Command: `research/qwen_attack/run_attack.py --files results/raw/baseline_safe.json results/raw/baseline_extra.json results/raw/samples_len200.json --only baseline_extra:20 baseline_extra:22 baseline_extra:23 baseline_extra:24 baseline_extra:25 baseline_extra:26 baseline_extra:27 baseline_safe:0 baseline_safe:1 baseline_safe:2 baseline_safe:3 baseline_safe:4 --n-cands 90 --resume results/qwen_attack_ablation/ncands90`. Files: `results/raw/baseline_safe.json`, `results/raw/baseline_extra.json`, `results/raw/samples_len200.json`. Gate config and seeds are in run.json; every sample's steps, funnels and rewrites are in `samples/`.
