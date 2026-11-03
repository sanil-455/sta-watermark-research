# Qwen paraphrase attack on the STA watermark: final report

Run: `strict`. Started 2026-10-31 09:01:16, ended 2026-10-31 09:58:10. Git commit `645284db7983152eb7a1ff703b5b02afd9d0d5dc`.

**What was attacked.** Only the generated span of each sample, never the prompt. The detector was asked only about span text. **Success** means the span's z-score is 2.0 or below under the same detector. Preset: **strict**. Quality is judged by reading (see READING.md); the gates below are filters, not proof of faithfulness.

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
  - 4-bit nf4 with double quantisation, float16 compute; stored elements by dtype: {'torch.bfloat16': 1090328064, 'torch.uint8': 3262644224}; model memory footprint 5.07 GiB; GPU allocated after load 5.26 GiB; load time 4.7 s
- **Similarity model:** sentence-transformers/all-MiniLM-L6-v2, 22,713,216 parameters
- **Detector:** the STA detector of the paper, queried as a black box that returns z-scores; the key is never used by the attack
- GPU: Quadro RTX 5000 | python 3.11.16 | torch 2.7.1+cu128 | transformers 5.17.0 | bitsandbytes 0.50.2 | accelerate 1.15.0 | WordNet available: True
- peak GPU memory during the run: allocated 5.48 GiB, reserved 5.55 GiB

## 3. Settings

- gates (strict): {'semantic': 0.88, 'fluency': 1.3, 'min_for_fluency': 3, 'spelling': True, 'max_unknown': 0, 'check_numbers': True, 'check_quotes': True, 'check_negations': True, 'check_names': True}
- sampling: {'temperature': 1.5, 'top_p': 0.98, 'top_k': 20, 'repetition_penalty': 1.05, 'batch': 4, 'length_factor': 2.0, 'extra_tokens': 32}
- attack: {'n_cands': 60, 'seed': 0, 'max_minutes': 0} | segmentation: {'min_chars': 30, 'quote_words': 4, 'abbreviations': 50}

## 4. Results (never pooled across lengths)

Columns marked mean / median / max.

### 200-token spans: 12 attacked, **6 broken (50%)**

- all attacked (n=12): z0 4.5 / 4.2 / 6.5 | z final 2.2 / 2.0 / 3.6 | sentences rewritten % of eligible 83.3 / 100.0 / 100.0 | tokens not preserved % 18.8 / 17.2 / 40.3 | tokens that are new % 17.8 / 16.2 / 32.6 | words not preserved % 24.3 / 20.2 / 44.7 | characters not preserved % 14.7 / 12.3 / 32.2 | length change (tokens) -2.8 / -1.0 / 10.0 | queries 107.4 / 110.0 / 226.0 | minutes 4.7 / 2.5 / 18.0
- broken only (n=6): z0 4.1 / 3.6 / 6.2 | z final 1.6 / 1.8 / 1.9 | sentences rewritten % of eligible 76.4 / 100.0 / 100.0 | tokens not preserved % 19.7 / 14.1 / 40.3 | tokens that are new % 17.4 / 12.4 / 32.6 | words not preserved % 24.6 / 18.1 / 44.7 | characters not preserved % 15.9 / 10.7 / 32.2 | length change (tokens) -6.3 / -4.5 / 2.0 | queries 73.7 / 47.5 / 155.0 | minutes 1.9 / 2.2 / 3.1
- outcomes: {'broken': 6, 'improved but not enough (stalled)': 6}

## 5. Per-sample table

tokens = span length; eligible = share of characters that may be rewritten; needed = share that must be rewritten at chance level (1 - 2/z0); delta needed = selection gain (how much less green than chance the rewrites must be) if eligible < needed; delta achieved = 0.5 - green fraction of the new token pairs. Percentages of tokens/characters are of the ORIGINAL span (not preserved) and of the FINAL span (new).

| sample | tokens | z0 | eligible | needed | delta needed | outcome | z final | rewritten | tokens not kept % | tokens new % | chars not kept % | length change | queries | min | delta achieved |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_extra:20 | 200 | 3.39 | 32% | 41% | 0.03 | broken | 1.76 | 2 of 2 (100%) | 10.9 | 11.4 | 9.4 | +1 | 31 | 2.2 | 0.21 |
| baseline_extra:22 | 200 | 4.67 | 77% | 57% | 0.00 | stalled (improved but not enough (stalled)) | 2.09 | 4 of 6 (67%) | 22.4 | 19.6 | 16.5 | -7 | 187 | 18.0 | 0.12 |
| baseline_extra:23 | 200 | 5.23 | 47% | 62% | 0.06 | stalled (improved but not enough (stalled)) | 3.17 | 4 of 4 (100%) | 21.4 | 25.1 | 14.8 | +10 | 226 | 7.2 | 0.08 |
| baseline_extra:24 | 200 | 5.09 | 75% | 61% | 0.00 | broken | 1.28 | 4 of 4 (100%) | 40.3 | 32.6 | 32.2 | -23 | 120 | 2.3 | 0.10 |
| baseline_extra:25 | 200 | 3.82 | 37% | 48% | 0.04 | stalled (improved but not enough (stalled)) | 2.80 | 2 of 2 (100%) | 12.4 | 14.1 | 8.6 | +4 | 84 | 2.7 | 0.03 |
| baseline_extra:26 | 200 | 2.83 | 60% | 29% | 0.00 | broken | 1.13 | 1 of 4 (25%) | 7.0 | 7.9 | 5.8 | +2 | 41 | 0.5 | 0.29 |
| baseline_extra:27 | 200 | 6.22 | 94% | 68% | 0.00 | broken | 1.81 | 6 of 6 (100%) | 31.8 | 28.6 | 26.7 | -9 | 155 | 3.1 | 0.16 |
| baseline_safe:0 | 200 | 3.82 | 29% | 48% | 0.09 | stalled (improved but not enough (stalled)) | 2.83 | 2 of 2 (100%) | 9.5 | 9.5 | 7.5 | +0 | 121 | 2.3 | 0.18 |
| baseline_safe:1 | 200 | 6.51 | 51% | 69% | 0.08 | stalled (improved but not enough (stalled)) | 3.60 | 3 of 4 (75%) | 17.9 | 18.3 | 12.9 | +1 | 100 | 3.8 | 0.14 |
| baseline_safe:2 | 200 | 5.37 | 72% | 63% | 0.00 | stalled (improved but not enough (stalled)) | 2.49 | 3 of 3 (100%) | 23.9 | 22.7 | 20.1 | -3 | 129 | 11.3 | 0.11 |
| baseline_safe:3 | 170 | 3.68 | 48% | 46% | 0.00 | broken | 1.85 | 3 of 3 (100%) | 11.7 | 10.7 | 9.8 | -2 | 47 | 1.3 | 0.22 |
| baseline_safe:4 | 200 | 3.54 | 83% | 43% | 0.00 | broken | 1.80 | 2 of 6 (33%) | 16.4 | 13.4 | 11.7 | -7 | 48 | 2.3 | 0.08 |

## 6. Where it breaks

- 200-token spans, broken spans: sentences rewritten at the break 3.0 / 2.5 / 6.0 | queries at the break 73.7 / 47.5 / 155.0
- 200-token spans: 7 need no selection gain (eligible text enough at chance level), 5 need delta > 0, of which the achieved delta reached the need in 4; 0 are impossible by rewriting (eligible share too small even for delta 0.5)

## 7. Gates: where candidates are lost

- candidates generated 6780 -> after clean 4747 -> spelling 4559 -> content 3684 -> semantic 1611 -> fluency 1297
- rejected by reason: {'meaning too different': 2073, 'duplicate': 1821, 'negation count changed': 644, 'perplexity too high': 314, 'numbers changed': 179, 'CJK text': 178, 'unknown word': 172, 'name changed': 45, 'several lines': 30, 'glued punctuation': 16, 'quotation changed': 7, 'hit length limit': 4}

## 8. Resources and problems

- time (hours): attack 0.95 | candidate generation 0.91 | gates 0.03 | detector queries 0.00
- detector queries: 1289 in total (screening queries not included)
- out-of-memory events during rewriting: 0 | rewrites that hit the length limit: 7 | samples that raised an error: 0
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

Command: `research/qwen_attack/run_attack.py --files results/raw/baseline_safe.json results/raw/baseline_extra.json results/raw/samples_len200.json --only baseline_extra:20 baseline_extra:22 baseline_extra:23 baseline_extra:24 baseline_extra:25 baseline_extra:26 baseline_extra:27 baseline_safe:0 baseline_safe:1 baseline_safe:2 baseline_safe:3 baseline_safe:4 --preset strict --resume results/qwen_attack_ablation/strict`. Files: `results/raw/baseline_safe.json`, `results/raw/baseline_extra.json`, `results/raw/samples_len200.json`. Gate config and seeds are in run.json; every sample's steps, funnels and rewrites are in `samples/`.
