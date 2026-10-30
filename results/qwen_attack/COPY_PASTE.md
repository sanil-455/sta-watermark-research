# Copy-paste baseline attack (standard literature setup, CPU-only, no model)

A block of human-written C4 text is prepended before the watermarked span, at several ratios of human-to-total words, and the WHOLE combined text is rescored. This is the one standard non-adaptive baseline that needs no paraphraser model. It runs on the exact same flagged spans the adaptive attack ran on, so the two are directly comparable. 0%% is the unattacked span itself (should match the original z0 count). Lengths are never pooled.

## 200-token spans (46 flagged spans)

| copy ratio (human words : total) | still detected (z>2.0) | detection rate (95%% interval) |
|---|---|---|
| 0% | 46 of 46 | 100% [92%, 100%] |
| 10% | 45 of 46 | 98% [89%, 100%] |
| 25% | 43 of 46 | 93% [82%, 98%] |
| 50% | 17 of 46 | 37% [25%, 51%] |

## 500-token spans (21 flagged spans)

| copy ratio (human words : total) | still detected (z>2.0) | detection rate (95%% interval) |
|---|---|---|
| 0% | 21 of 21 | 100% [85%, 100%] |
| 10% | 21 of 21 | 100% [85%, 100%] |
| 25% | 21 of 21 | 100% [85%, 100%] |
| 50% | 20 of 21 | 95% [77%, 99%] |

