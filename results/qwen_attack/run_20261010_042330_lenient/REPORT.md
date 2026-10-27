# Qwen paraphrase attack on the STA watermark: final report

Run: `run_20261010_042330_lenient`. Started 2026-10-10 04:23:40, ended 2026-10-10 04:26:07. Git commit `adf6cc65c7fdc046f185d04c94ff87d1ed91573d`.

**What was attacked.** Only the generated span of each sample, never the prompt. The detector was asked only about span text. **Success** means the span's z-score is 2.0 or below under the same detector. Preset: **lenient**. Quality is judged by reading (see READING.md); the gates below are filters, not proof of faithfulness.

## 1. Data

- spans loaded: 71 | flagged by the detector (z0 > 2): 67 | not flagged: 4
- not flagged (not attacked): baseline_extra:21 (z0 0.424), samples_len200:45 (z0 0.707), samples_len200:55 (z0 1.838), samples_len200:62 (z0 1.556)
- 200-token spans: 50 loaded, 46 flagged
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
- attack: {'n_cands': 20, 'seed': 0, 'max_minutes': 0} | segmentation: {'min_chars': 30, 'quote_words': 4, 'abbreviations': 50}

## 4. Results (never pooled across lengths)

Columns marked mean / median / max.

### 200-token spans: 2 attacked, **1 broken (50%)**

- all attacked (n=2): z0 5.8 / 5.8 / 6.5 | z final 2.4 / 2.4 / 3.5 | sentences rewritten % of eligible 62.5 / 62.5 / 75.0 | tokens not preserved % 23.4 / 23.4 / 27.9 | tokens that are new % 22.5 / 22.5 / 26.0 | words not preserved % 28.4 / 28.4 / 37.7 | characters not preserved % 16.9 / 16.9 / 20.5 | length change (tokens) -2.5 / -2.5 / 0.0 | queries 56.5 / 56.5 / 74.0 | minutes 1.2 / 1.2 / 1.9
- broken only (n=1): z0 5.1 / 5.1 / 5.1 | z final 1.4 / 1.4 / 1.4 | sentences rewritten % of eligible 50.0 / 50.0 / 50.0 | tokens not preserved % 27.9 / 27.9 / 27.9 | tokens that are new % 26.0 / 26.0 / 26.0 | words not preserved % 37.7 / 37.7 / 37.7 | characters not preserved % 20.5 / 20.5 / 20.5 | length change (tokens) -5.0 / -5.0 / -5.0 | queries 39.0 / 39.0 / 39.0 | minutes 0.6 / 0.6 / 0.6
- outcomes: {'improved but not enough (stalled)': 1, 'broken': 1}

## 5. Per-sample table

tokens = span length; eligible = share of characters that may be rewritten; needed = share that must be rewritten at chance level (1 - 2/z0); delta needed = selection gain (how much less green than chance the rewrites must be) if eligible < needed; delta achieved = 0.5 - green fraction of the new token pairs. Percentages of tokens/characters are of the ORIGINAL span (not preserved) and of the FINAL span (new).

| sample | tokens | z0 | eligible | needed | delta needed | outcome | z final | rewritten | tokens not kept % | tokens new % | chars not kept % | length change | queries | min | delta achieved |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_safe:1 | 200 | 6.51 | 51% | 69% | 0.08 | stalled (improved but not enough (stalled)) | 3.54 | 3 of 4 (75%) | 18.9 | 18.9 | 13.3 | +0 | 74 | 1.9 | 0.15 |
| samples_len200:44 | 200 | 5.09 | 90% | 61% | 0.00 | broken | 1.36 | 3 of 6 (50%) | 27.9 | 26.0 | 20.5 | -5 | 39 | 0.6 | 0.14 |

## 6. Where it breaks

- 200-token spans, broken spans: sentences rewritten at the break 3.0 / 3.0 / 3.0 | queries at the break 39.0 / 39.0 / 39.0
- 200-token spans: 1 need no selection gain (eligible text enough at chance level), 1 need delta > 0, of which the achieved delta reached the need in 1; 0 are impossible by rewriting (eligible share too small even for delta 0.5)

## 7. Gates: where candidates are lost

- candidates generated 300 -> after clean 249 -> spelling 227 -> content 154 -> semantic 124 -> fluency 114
- rejected by reason: {'negation count changed': 73, 'duplicate': 46, 'meaning too different': 30, 'glued punctuation': 16, 'perplexity too high': 10, 'unknown word': 6, 'CJK text': 5}

## 8. Resources and problems

- time (hours): attack 0.04 | candidate generation 0.04 | gates 0.00 | detector queries 0.00
- detector queries: 113 in total (screening queries not included)
- out-of-memory events during rewriting: 0 | rewrites that hit the length limit: 0 | samples that raised an error: 0
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

Command: `research/qwen_attack/run_attack.py --only baseline_safe:1 samples_len200:44 --n-cands 20`. Files: `results/raw/baseline_safe.json`, `results/raw/baseline_extra.json`, `results/raw/samples_len200.json`, `results/raw/samples_len500.json`. Gate config and seeds are in run.json; every sample's steps, funnels and rewrites are in `samples/`.
