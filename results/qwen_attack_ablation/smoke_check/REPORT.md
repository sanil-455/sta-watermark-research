# Qwen paraphrase attack on the STA watermark: final report

Run: `smoke_check`. Started 2026-10-31 07:24:58, ended 2026-10-31 07:26:02. Git commit `645284db7983152eb7a1ff703b5b02afd9d0d5dc`.

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
  - 4-bit nf4 with double quantisation, float16 compute; stored elements by dtype: {'torch.bfloat16': 1090328064, 'torch.uint8': 3262644224}; model memory footprint 5.07 GiB; GPU allocated after load 5.26 GiB; load time 4.9 s
- **Similarity model:** sentence-transformers/all-MiniLM-L6-v2, 22,713,216 parameters
- **Detector:** the STA detector of the paper, queried as a black box that returns z-scores; the key is never used by the attack
- GPU: Quadro RTX 5000 | python 3.11.16 | torch 2.7.1+cu128 | transformers 5.17.0 | bitsandbytes 0.50.2 | accelerate 1.15.0 | WordNet available: True
- peak GPU memory during the run: allocated 5.46 GiB, reserved 5.53 GiB

## 3. Settings

- gates (lenient): {'semantic': 0.75, 'fluency': 1.8, 'min_for_fluency': 3, 'spelling': True, 'max_unknown': 0, 'check_numbers': True, 'check_quotes': True, 'check_negations': True, 'check_names': False}
- sampling: {'temperature': 1.5, 'top_p': 0.98, 'top_k': 20, 'repetition_penalty': 1.05, 'batch': 4, 'length_factor': 2.0, 'extra_tokens': 32}
- attack: {'n_cands': 60, 'seed': 0, 'max_minutes': 0} | segmentation: {'min_chars': 30, 'quote_words': 4, 'abbreviations': 50}

## 4. Results (never pooled across lengths)

Columns marked mean / median / max.

### 200-token spans: 1 attacked, **1 broken (100%)**

- all attacked (n=1): z0 3.4 / 3.4 / 3.4 | z final 1.9 / 1.9 / 1.9 | sentences rewritten % of eligible 100.0 / 100.0 / 100.0 | tokens not preserved % 11.4 / 11.4 / 11.4 | tokens that are new % 12.7 / 12.7 / 12.7 | words not preserved % 14.6 / 14.6 / 14.6 | characters not preserved % 9.7 / 9.7 / 9.7 | length change (tokens) 3.0 / 3.0 / 3.0 | queries 45.0 / 45.0 / 45.0 | minutes 1.1 / 1.1 / 1.1
- broken only (n=1): z0 3.4 / 3.4 / 3.4 | z final 1.9 / 1.9 / 1.9 | sentences rewritten % of eligible 100.0 / 100.0 / 100.0 | tokens not preserved % 11.4 / 11.4 / 11.4 | tokens that are new % 12.7 / 12.7 / 12.7 | words not preserved % 14.6 / 14.6 / 14.6 | characters not preserved % 9.7 / 9.7 / 9.7 | length change (tokens) 3.0 / 3.0 / 3.0 | queries 45.0 / 45.0 / 45.0 | minutes 1.1 / 1.1 / 1.1
- outcomes: {'broken': 1}

## 5. Per-sample table

tokens = span length; eligible = share of characters that may be rewritten; needed = share that must be rewritten at chance level (1 - 2/z0); delta needed = selection gain (how much less green than chance the rewrites must be) if eligible < needed; delta achieved = 0.5 - green fraction of the new token pairs. Percentages of tokens/characters are of the ORIGINAL span (not preserved) and of the FINAL span (new).

| sample | tokens | z0 | eligible | needed | delta needed | outcome | z final | rewritten | tokens not kept % | tokens new % | chars not kept % | length change | queries | min | delta achieved |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_extra:20 | 200 | 3.39 | 32% | 41% | 0.03 | broken | 1.90 | 2 of 2 (100%) | 11.4 | 12.7 | 9.7 | +3 | 45 | 1.1 | 0.18 |

## 6. Where it breaks

- 200-token spans, broken spans: sentences rewritten at the break 2.0 / 2.0 / 2.0 | queries at the break 45.0 / 45.0 / 45.0
- 200-token spans: 0 need no selection gain (eligible text enough at chance level), 1 need delta > 0, of which the achieved delta reached the need in 1; 0 are impossible by rewriting (eligible share too small even for delta 0.5)

## 7. Gates: where candidates are lost

- candidates generated 120 -> after clean 93 -> spelling 70 -> content 46 -> semantic 46 -> fluency 44
- rejected by reason: {'unknown word': 23, 'duplicate': 23, 'numbers changed': 23, 'CJK text': 2, 'perplexity too high': 2, 'several lines': 1, 'hit length limit': 1, 'negation count changed': 1}

## 8. Resources and problems

- time (hours): attack 0.02 | candidate generation 0.02 | gates 0.00 | detector queries 0.00
- detector queries: 45 in total (screening queries not included)
- out-of-memory events during rewriting: 0 | rewrites that hit the length limit: 1 | samples that raised an error: 0
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

Command: `research/qwen_attack/run_attack.py --files results/raw/baseline_safe.json results/raw/baseline_extra.json results/raw/samples_len200.json --only baseline_extra:20 --resume results/qwen_attack_ablation/smoke_check`. Files: `results/raw/baseline_safe.json`, `results/raw/baseline_extra.json`, `results/raw/samples_len200.json`. Gate config and seeds are in run.json; every sample's steps, funnels and rewrites are in `samples/`.
