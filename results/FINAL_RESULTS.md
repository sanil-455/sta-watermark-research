# FINAL RESULTS -- STA watermark adaptive attack

All numbers read directly from result folders. 'broken' = span z <= 2.0 under the same detector. Intervals are Wilson 95%. Each attack sees only the generated span, never the prompt (enforced by an assertion guard).

## Attack success

| setting | broken | n | 95% interval | errors | note |
|---|---|---|---|---|---|
| STA-1 200-tok, adaptive (batch1) | 35 | 46 | 76% [62%, 86%] | 0 |  |
| STA-1 500-tok, adaptive (batch1) | 18 | 21 | 86% [65%, 95%] | 0 | n=21, small |
| STA-1 500-tok, adaptive (scaled) | 100 | 134 | 75% [67%, 81%] | 1 | 1 span excluded by prompt-leakage guard |
| STA-1 200-tok, no-box (batch2) | 30 | 241 | 12% [9%, 17%] | 0 |  |
| STA-1 500-tok, no-box | 1 | 21 | 5% [1%, 23%] | 0 | n=21, small |
| STA-16 200-tok, adaptive | 32 | 99 | 32% [24%, 42%] | 0 | robust variant |
| STA-16 200-tok, no-box | 0 | 99 | 0% [0%, 4%] | 0 | robust variant |

## Headline: detector guidance vs blind paraphrase (same spans, same gates)

| watermark / length | adaptive | no-box | n(adaptive/nobox) |
|---|---|---|---|
| STA-1 200-tok | 76% [62-86] | 12% [9-17] | 46/241 |
| STA-1 500-tok | 75% [67-81] | 5% [1-23] | 134/21 |
| STA-16 (robust) 200-tok | 32% [24-42] | 0% [0-4] | 99/99 |

## Supporting analyses (full detail in the named files)

- Bound (predicts breakability): results/qwen_attack/BOUND_CHECK.md
- Query-budget defence: results/qwen_attack/QUERY_BUDGET.md
- Copy-paste baseline (paper-faithful): results/qwen_attack/COPY_PASTE_FAITHFUL.md
- Ablations (seed/n_cands/preset/target): results/qwen_attack_ablation/SUMMARY.md
- ROC/AUC/TPR vs controls: results/qwen_attack/ROC_EVAL.md

## Methodology notes

- 1 span (samples_len500:351) excluded from scaled 500-tok adaptive: the generated text echoed its prompt at char 1020, tripping the prompt-leakage guard. 134 clean spans reported.
- Quality of rewrites judged by AI reading + author repair; independent human rating NOT yet done (stated limitation).
- 500-tok no-box and STA-1 500-tok batch1 are n=21 (small); report with their wide intervals.

