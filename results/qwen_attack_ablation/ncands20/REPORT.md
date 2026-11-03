# Qwen paraphrase attack on the STA watermark: final report

Run: `ncands20`. Started 2026-10-31 07:26:15, ended 2026-10-31 07:37:26. Git commit `645284db7983152eb7a1ff703b5b02afd9d0d5dc`.

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
  - 4-bit nf4 with double quantisation, float16 compute; stored elements by dtype: {'torch.bfloat16': 1090328064, 'torch.uint8': 3262644224}; model memory footprint 5.07 GiB; GPU allocated after load 5.26 GiB; load time 4.5 s
- **Similarity model:** sentence-transformers/all-MiniLM-L6-v2, 22,713,216 parameters
- **Detector:** the STA detector of the paper, queried as a black box that returns z-scores; the key is never used by the attack
- GPU: Quadro RTX 5000 | python 3.11.16 | torch 2.7.1+cu128 | transformers 5.17.0 | bitsandbytes 0.50.2 | accelerate 1.15.0 | WordNet available: True
- peak GPU memory during the run: allocated 5.48 GiB, reserved 5.55 GiB

## 3. Settings

- gates (lenient): {'semantic': 0.75, 'fluency': 1.8, 'min_for_fluency': 3, 'spelling': True, 'max_unknown': 0, 'check_numbers': True, 'check_quotes': True, 'check_negations': True, 'check_names': False}
- sampling: {'temperature': 1.5, 'top_p': 0.98, 'top_k': 20, 'repetition_penalty': 1.05, 'batch': 4, 'length_factor': 2.0, 'extra_tokens': 32}
- attack: {'n_cands': 20, 'seed': 0, 'max_minutes': 0} | segmentation: {'min_chars': 30, 'quote_words': 4, 'abbreviations': 50}

## 4. Results (never pooled across lengths)

Columns marked mean / median / max.

### 200-token spans: 12 attacked, **8 broken (67%)**

- all attacked (n=12): z0 4.5 / 4.2 / 6.5 | z final 2.1 / 1.9 / 3.5 | sentences rewritten % of eligible 84.7 / 100.0 / 100.0 | tokens not preserved % 20.8 / 15.5 / 44.3 | tokens that are new % 19.6 / 15.8 / 37.6 | words not preserved % 25.8 / 19.5 / 47.7 | characters not preserved % 17.4 / 13.1 / 34.4 | length change (tokens) -3.8 / -1.5 / 3.0 | queries 56.6 / 49.5 / 127.0 | minutes 0.9 / 0.8 / 1.9
- broken only (n=8): z0 4.3 / 4.2 / 6.2 | z final 1.7 / 1.8 / 2.0 | sentences rewritten % of eligible 80.2 / 100.0 / 100.0 | tokens not preserved % 24.2 / 22.9 / 44.3 | tokens that are new % 22.2 / 22.1 / 37.6 | words not preserved % 29.6 / 27.8 / 47.7 | characters not preserved % 20.1 / 17.7 / 34.4 | length change (tokens) -6.0 / -3.0 / 2.0 | queries 51.8 / 41.0 / 127.0 | minutes 0.8 / 0.7 / 1.7
- outcomes: {'broken': 8, 'improved but not enough (stalled)': 4}

## 5. Per-sample table

tokens = span length; eligible = share of characters that may be rewritten; needed = share that must be rewritten at chance level (1 - 2/z0); delta needed = selection gain (how much less green than chance the rewrites must be) if eligible < needed; delta achieved = 0.5 - green fraction of the new token pairs. Percentages of tokens/characters are of the ORIGINAL span (not preserved) and of the FINAL span (new).

| sample | tokens | z0 | eligible | needed | delta needed | outcome | z final | rewritten | tokens not kept % | tokens new % | chars not kept % | length change | queries | min | delta achieved |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_extra:20 | 200 | 3.39 | 32% | 41% | 0.03 | broken | 1.97 | 2 of 2 (100%) | 10.4 | 11.3 | 10.3 | +2 | 34 | 0.6 | 0.14 |
| baseline_extra:22 | 200 | 4.67 | 77% | 57% | 0.00 | broken | 1.92 | 4 of 6 (67%) | 28.9 | 27.8 | 22.6 | -3 | 33 | 0.7 | 0.10 |
| baseline_extra:23 | 200 | 5.23 | 47% | 62% | 0.06 | stalled (improved but not enough (stalled)) | 3.02 | 4 of 4 (100%) | 13.9 | 15.2 | 12.1 | +3 | 53 | 1.3 | 0.15 |
| baseline_extra:24 | 200 | 5.09 | 75% | 61% | 0.00 | broken | 1.29 | 4 of 4 (100%) | 44.3 | 36.4 | 34.4 | -25 | 66 | 0.8 | 0.07 |
| baseline_extra:25 | 200 | 3.82 | 37% | 48% | 0.04 | stalled (improved but not enough (stalled)) | 2.67 | 2 of 2 (100%) | 10.9 | 11.8 | 7.8 | +2 | 38 | 0.9 | 0.08 |
| baseline_extra:26 | 200 | 2.83 | 60% | 29% | 0.00 | broken | 1.35 | 1 of 4 (25%) | 7.0 | 6.5 | 5.7 | -1 | 20 | 0.2 | 0.30 |
| baseline_extra:27 | 200 | 6.22 | 94% | 68% | 0.00 | broken | 1.54 | 6 of 6 (100%) | 33.3 | 28.7 | 30.3 | -13 | 127 | 1.7 | 0.16 |
| baseline_safe:0 | 200 | 3.82 | 29% | 48% | 0.09 | stalled (improved but not enough (stalled)) | 2.70 | 2 of 2 (100%) | 12.4 | 11.6 | 14.6 | -2 | 100 | 0.8 | 0.16 |
| baseline_safe:1 | 200 | 6.51 | 51% | 69% | 0.08 | stalled (improved but not enough (stalled)) | 3.54 | 3 of 4 (75%) | 18.9 | 18.9 | 13.3 | +0 | 74 | 1.9 | 0.15 |
| baseline_safe:2 | 200 | 5.37 | 72% | 63% | 0.00 | broken | 1.86 | 3 of 3 (100%) | 38.8 | 37.6 | 33.0 | -4 | 52 | 0.8 | 0.06 |
| baseline_safe:3 | 170 | 3.68 | 48% | 46% | 0.00 | broken | 1.93 | 3 of 3 (100%) | 14.0 | 12.5 | 11.8 | -3 | 35 | 0.5 | 0.22 |
| baseline_safe:4 | 200 | 3.54 | 83% | 43% | 0.00 | broken | 1.77 | 3 of 6 (50%) | 16.9 | 16.5 | 12.9 | -1 | 47 | 0.9 | 0.06 |

## 6. Where it breaks

- 200-token spans, broken spans: sentences rewritten at the break 3.2 / 3.0 / 6.0 | queries at the break 51.8 / 41.0 / 127.0
- 200-token spans: 7 need no selection gain (eligible text enough at chance level), 5 need delta > 0, of which the achieved delta reached the need in 5; 0 are impossible by rewriting (eligible share too small even for delta 0.5)

## 7. Gates: where candidates are lost

- candidates generated 1260 -> after clean 985 -> spelling 930 -> content 770 -> semantic 715 -> fluency 677
- rejected by reason: {'duplicate': 238, 'negation count changed': 113, 'meaning too different': 55, 'numbers changed': 47, 'unknown word': 38, 'perplexity too high': 38, 'CJK text': 31, 'glued punctuation': 17, 'several lines': 4, 'hit length limit': 2}

## 8. Resources and problems

- time (hours): attack 0.19 | candidate generation 0.17 | gates 0.01 | detector queries 0.00
- detector queries: 679 in total (screening queries not included)
- out-of-memory events during rewriting: 0 | rewrites that hit the length limit: 2 | samples that raised an error: 0
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

Command: `research/qwen_attack/run_attack.py --files results/raw/baseline_safe.json results/raw/baseline_extra.json results/raw/samples_len200.json --only baseline_extra:20 baseline_extra:22 baseline_extra:23 baseline_extra:24 baseline_extra:25 baseline_extra:26 baseline_extra:27 baseline_safe:0 baseline_safe:1 baseline_safe:2 baseline_safe:3 baseline_safe:4 --n-cands 20 --resume results/qwen_attack_ablation/ncands20`. Files: `results/raw/baseline_safe.json`, `results/raw/baseline_extra.json`, `results/raw/samples_len200.json`. Gate config and seeds are in run.json; every sample's steps, funnels and rewrites are in `samples/`.
