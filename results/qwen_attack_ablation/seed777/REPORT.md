# Qwen paraphrase attack on the STA watermark: final report

Run: `seed777`. Started 2026-10-31 07:37:39, ended 2026-10-31 08:14:22. Git commit `645284db7983152eb7a1ff703b5b02afd9d0d5dc`.

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
- attack: {'n_cands': 60, 'seed': 777, 'max_minutes': 0} | segmentation: {'min_chars': 30, 'quote_words': 4, 'abbreviations': 50}

## 4. Results (never pooled across lengths)

Columns marked mean / median / max.

### 200-token spans: 12 attacked, **8 broken (67%)**

- all attacked (n=12): z0 4.5 / 4.2 / 6.5 | z final 1.9 / 1.8 / 3.0 | sentences rewritten % of eligible 80.6 / 100.0 / 100.0 | tokens not preserved % 20.5 / 17.7 / 40.3 | tokens that are new % 20.2 / 17.7 / 34.4 | words not preserved % 26.2 / 21.6 / 46.8 | characters not preserved % 17.7 / 13.0 / 33.4 | length change (tokens) -1.1 / -1.0 / 17.0 | queries 145.0 / 142.0 / 283.0 | minutes 3.1 / 2.3 / 7.7
- broken only (n=8): z0 4.3 / 4.2 / 6.2 | z final 1.6 / 1.7 / 2.0 | sentences rewritten % of eligible 74.0 / 83.3 / 100.0 | tokens not preserved % 22.9 / 23.4 / 40.3 | tokens that are new % 21.7 / 23.2 / 34.4 | words not preserved % 28.7 / 28.8 / 46.8 | characters not preserved % 18.9 / 19.0 / 33.4 | length change (tokens) -3.8 / -1.5 / 1.0 | queries 99.4 / 87.0 / 184.0 | minutes 1.7 / 1.8 / 2.5
- outcomes: {'broken': 8, 'improved but not enough (stalled)': 4}

## 5. Per-sample table

tokens = span length; eligible = share of characters that may be rewritten; needed = share that must be rewritten at chance level (1 - 2/z0); delta needed = selection gain (how much less green than chance the rewrites must be) if eligible < needed; delta achieved = 0.5 - green fraction of the new token pairs. Percentages of tokens/characters are of the ORIGINAL span (not preserved) and of the FINAL span (new).

| sample | tokens | z0 | eligible | needed | delta needed | outcome | z final | rewritten | tokens not kept % | tokens new % | chars not kept % | length change | queries | min | delta achieved |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_extra:20 | 200 | 3.39 | 32% | 41% | 0.03 | broken | 1.84 | 2 of 2 (100%) | 10.9 | 10.9 | 8.3 | +0 | 46 | 1.1 | 0.18 |
| baseline_extra:22 | 200 | 4.67 | 77% | 57% | 0.00 | broken | 1.71 | 4 of 6 (67%) | 30.8 | 29.4 | 25.3 | -4 | 100 | 2.0 | 0.11 |
| baseline_extra:23 | 200 | 5.23 | 47% | 62% | 0.06 | stalled (improved but not enough (stalled)) | 2.10 | 4 of 4 (100%) | 17.9 | 24.3 | 24.0 | +17 | 248 | 7.5 | 0.12 |
| baseline_extra:24 | 200 | 5.09 | 75% | 61% | 0.00 | broken | 0.89 | 4 of 4 (100%) | 40.3 | 34.4 | 33.4 | -18 | 184 | 2.3 | 0.14 |
| baseline_extra:25 | 200 | 3.82 | 37% | 48% | 0.04 | stalled (improved but not enough (stalled)) | 2.32 | 2 of 2 (100%) | 14.4 | 15.7 | 12.6 | +3 | 156 | 5.5 | 0.10 |
| baseline_extra:26 | 200 | 2.83 | 60% | 29% | 0.00 | broken | 1.34 | 1 of 4 (25%) | 7.0 | 7.4 | 6.0 | +1 | 53 | 0.5 | 0.24 |
| baseline_extra:27 | 200 | 6.22 | 94% | 68% | 0.00 | broken | 1.99 | 4 of 6 (67%) | 29.4 | 28.6 | 28.6 | -2 | 148 | 2.5 | 0.10 |
| baseline_safe:0 | 200 | 3.82 | 29% | 48% | 0.09 | stalled (improved but not enough (stalled)) | 2.56 | 2 of 2 (100%) | 12.9 | 12.1 | 11.5 | -2 | 258 | 2.4 | 0.15 |
| baseline_safe:1 | 200 | 6.51 | 51% | 69% | 0.08 | stalled (improved but not enough (stalled)) | 3.05 | 3 of 4 (75%) | 17.9 | 17.5 | 12.8 | -1 | 283 | 7.7 | 0.23 |
| baseline_safe:2 | 200 | 5.37 | 72% | 63% | 0.00 | broken | 1.37 | 3 of 3 (100%) | 32.8 | 30.4 | 24.8 | -7 | 136 | 2.4 | 0.13 |
| baseline_safe:3 | 170 | 3.68 | 48% | 46% | 0.00 | broken | 1.77 | 3 of 3 (100%) | 14.6 | 14.1 | 12.0 | -1 | 74 | 1.4 | 0.14 |
| baseline_safe:4 | 200 | 3.54 | 83% | 43% | 0.00 | broken | 1.62 | 2 of 6 (33%) | 17.4 | 17.8 | 13.2 | +1 | 54 | 1.5 | 0.17 |

## 6. Where it breaks

- 200-token spans, broken spans: sentences rewritten at the break 2.9 / 3.0 / 4.0 | queries at the break 99.4 / 87.0 / 184.0
- 200-token spans: 7 need no selection gain (eligible text enough at chance level), 5 need delta > 0, of which the achieved delta reached the need in 5; 0 are impossible by rewriting (eligible share too small even for delta 0.5)

## 7. Gates: where candidates are lost

- candidates generated 4140 -> after clean 2650 -> spelling 2534 -> content 2014 -> semantic 1884 -> fluency 1797
- rejected by reason: {'duplicate': 1377, 'negation count changed': 353, 'numbers changed': 166, 'meaning too different': 130, 'CJK text': 90, 'perplexity too high': 87, 'unknown word': 81, 'glued punctuation': 35, 'several lines': 22, 'hit length limit': 1, 'quotation changed': 1}

## 8. Resources and problems

- time (hours): attack 0.61 | candidate generation 0.57 | gates 0.04 | detector queries 0.00
- detector queries: 1740 in total (screening queries not included)
- out-of-memory events during rewriting: 0 | rewrites that hit the length limit: 6 | samples that raised an error: 0
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

Command: `research/qwen_attack/run_attack.py --files results/raw/baseline_safe.json results/raw/baseline_extra.json results/raw/samples_len200.json --only baseline_extra:20 baseline_extra:22 baseline_extra:23 baseline_extra:24 baseline_extra:25 baseline_extra:26 baseline_extra:27 baseline_safe:0 baseline_safe:1 baseline_safe:2 baseline_safe:3 baseline_safe:4 --seed 777 --resume results/qwen_attack_ablation/seed777`. Files: `results/raw/baseline_safe.json`, `results/raw/baseline_extra.json`, `results/raw/samples_len200.json`. Gate config and seeds are in run.json; every sample's steps, funnels and rewrites are in `samples/`.
