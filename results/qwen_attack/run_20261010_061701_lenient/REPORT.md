# Qwen paraphrase attack on the STA watermark: final report

Run: `run_20261010_061701_lenient`. Started 2026-10-10 06:17:10, ended 2026-10-10 08:44:15. Git commit `adf6cc65c7fdc046f185d04c94ff87d1ed91573d`.

**What was attacked.** Only the generated span of each sample, never the prompt. The detector was asked only about span text. **Success** means the span's z-score is 2.0 or below under the same detector. Preset: **lenient**. Quality is judged by reading (see READING.md); the gates below are filters, not proof of faithfulness.

## 1. Data

- spans loaded: 21 | flagged by the detector (z0 > 2): 21 | not flagged: 0
- not flagged (not attacked): none
- 200-token spans: 0 loaded, 0 flagged
- 500-token spans: 21 loaded, 21 flagged
- generation run: 37 kept, 13 stopped before 195 tokens, 11 skipped for prompt length, 0 out-of-memory events. Both span sets are selections (the model had to keep generating).
- the 500-token spans are extensions of the 200-token spans of the same prompts, so the two groups are never pooled.

## 2. Models and environment

- **Generator of the watermarked text:** Llama-2-7B base, 6,738,415,616 parameters (computed from its published configuration, not re-measured in this run), fp16
- **Paraphraser:** Qwen/Qwen2.5-7B-Instruct, revision `a09a35458c702b33eeacc393d103063234e8bc28`, **7,615,616,512 parameters** (computed from the architecture; transformers reports 7,615,616,512; agree: True), 28 layers, hidden size 3584, 28 attention heads (4 key/value), vocabulary 152064, context 32768 tokens
  - 4-bit nf4 with double quantisation, float16 compute; stored elements by dtype: {'torch.bfloat16': 1090328064, 'torch.uint8': 3262644224}; model memory footprint 5.07 GiB; GPU allocated after load 5.26 GiB; load time 5.4 s
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

### 500-token spans: 21 attacked, **18 broken (86%)**

- all attacked (n=21): z0 7.1 / 7.3 / 8.8 | z final 2.3 / 1.8 / 7.2 | sentences rewritten % of eligible 74.6 / 73.7 / 100.0 | tokens not preserved % 22.9 / 23.6 / 42.5 | tokens that are new % 22.5 / 23.3 / 39.4 | words not preserved % 29.5 / 30.6 / 50.4 | characters not preserved % 18.3 / 17.8 / 34.0 | length change (tokens) -3.6 / -2.0 / 11.0 | queries 365.0 / 304.0 / 1113.0 | minutes 7.0 / 5.8 / 20.1
- broken only (n=18): z0 7.1 / 7.1 / 8.8 | z final 1.8 / 1.8 / 2.0 | sentences rewritten % of eligible 70.4 / 70.7 / 100.0 | tokens not preserved % 25.4 / 24.4 / 42.5 | tokens that are new % 24.9 / 23.9 / 39.4 | words not preserved % 32.7 / 32.1 / 50.4 | characters not preserved % 20.3 / 18.4 / 34.0 | length change (tokens) -4.2 / -2.0 / 11.0 | queries 353.1 / 307.5 / 857.0 | minutes 6.8 / 5.9 / 17.2
- outcomes: {'improved but not enough (stalled)': 3, 'broken': 18}

## 5. Per-sample table

tokens = span length; eligible = share of characters that may be rewritten; needed = share that must be rewritten at chance level (1 - 2/z0); delta needed = selection gain (how much less green than chance the rewrites must be) if eligible < needed; delta achieved = 0.5 - green fraction of the new token pairs. Percentages of tokens/characters are of the ORIGINAL span (not preserved) and of the FINAL span (new).

| sample | tokens | z0 | eligible | needed | delta needed | outcome | z final | rewritten | tokens not kept % | tokens new % | chars not kept % | length change | queries | min | delta achieved |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| samples_len500:28 | 500 | 7.87 | 47% | 75% | 0.10 | stalled (improved but not enough (stalled)) | 3.23 | 9 of 9 (100%) | 17.2 | 16.8 | 14.2 | -2 | 1113 | 20.1 | 0.22 |
| samples_len500:30 | 500 | 7.33 | 70% | 73% | 0.01 | broken | 1.99 | 10 of 12 (83%) | 21.8 | 23.4 | 16.6 | +11 | 325 | 5.9 | 0.15 |
| samples_len500:34 | 500 | 6.89 | 87% | 71% | 0.00 | broken | 1.62 | 10 of 16 (62%) | 24.6 | 23.3 | 19.5 | -8 | 377 | 5.8 | 0.14 |
| samples_len500:43 | 500 | 7.60 | 84% | 74% | 0.00 | broken | 1.84 | 10 of 11 (91%) | 42.5 | 39.4 | 34.0 | -26 | 217 | 6.8 | 0.08 |
| samples_len500:44 | 500 | 6.89 | 99% | 71% | 0.00 | broken | 1.75 | 7 of 16 (44%) | 23.4 | 23.2 | 17.4 | -1 | 246 | 4.4 | 0.12 |
| samples_len500:46 | 500 | 6.17 | 56% | 68% | 0.03 | broken | 1.96 | 8 of 8 (100%) | 25.7 | 26.3 | 24.2 | +4 | 577 | 14.2 | 0.11 |
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

## 6. Where it breaks

- 500-token spans, broken spans: sentences rewritten at the break 8.6 / 8.0 / 14.0 | queries at the break 353.1 / 307.5 / 857.0
- 500-token spans: 11 need no selection gain (eligible text enough at chance level), 8 need delta > 0, of which the achieved delta reached the need in 8; 2 are impossible by rewriting (eligible share too small even for delta 0.5)

## 7. Gates: where candidates are lost

- candidates generated 14940 -> after clean 10935 -> spelling 10404 -> content 8508 -> semantic 8084 -> fluency 7699
- rejected by reason: {'duplicate': 3340, 'negation count changed': 1213, 'quotation changed': 626, 'CJK text': 543, 'unknown word': 518, 'meaning too different': 424, 'perplexity too high': 385, 'several lines': 65, 'numbers changed': 57, 'preamble': 45, 'glued punctuation': 13, 'hit length limit': 12}

## 8. Resources and problems

- time (hours): attack 2.45 | candidate generation 2.28 | gates 0.17 | detector queries 0.00
- detector queries: 7664 in total (screening queries not included)
- out-of-memory events during rewriting: 0 | rewrites that hit the length limit: 22 | samples that raised an error: 0
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

Command: `research/qwen_attack/run_attack.py --files results/raw/samples_len500.json`. Files: `results/raw/samples_len500.json`. Gate config and seeds are in run.json; every sample's steps, funnels and rewrites are in `samples/`.
