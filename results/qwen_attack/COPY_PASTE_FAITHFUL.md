# Copy-paste attack, faithful to the paper's own code (attack/copy_paste_attack.py, single_insertion)

Reimplemented from the paper's source, not a guess at 'standard practice': random positions within the first 100 tokens of the watermarked span are replaced with the SAME-PROMPT unwatermarked control's tokens at those positions (not generic corpus text, and not a prepended block -- see COPY_PASTE.md for that separate, differently-defined experiment). Pairs where either text has fewer than 100 tokens are skipped entirely, exactly as the paper's code does. Random positions use Python's RNG (seed 42), not torch's, so counts will differ run to run from the paper's own numbers even under the same algorithm; only spans whose prompt also has a usable unwatermarked control (from generate_controls.py) can be tested here. Lengths are never pooled.

## 200-token spans (46 flagged spans; 24 have a same-prompt unwatermarked control to draw from)

| insertion ratio | min_attack_length | tested | too-short pairs skipped | still detected (z>2.0) | detection rate (95%% interval) |
|---|---|---|---|---|---|
| 10% | 100 | 24 | 0 | 24 of 24 | 100% [86%, 100%] |
| 25% | 100 | 24 | 0 | 18 of 24 | 75% [55%, 88%] |
| 50% | 100 | 24 | 0 | 10 of 24 | 42% [24%, 61%] |

*Only 24 of 46 flagged spans had a same-prompt control; the rest are not covered by this experiment.*

## 500-token spans (21 flagged spans; 10 have a same-prompt unwatermarked control to draw from)

| insertion ratio | min_attack_length | tested | too-short pairs skipped | still detected (z>2.0) | detection rate (95%% interval) |
|---|---|---|---|---|---|
| 10% | 100 | 10 | 0 | 10 of 10 | 100% [72%, 100%] |
| 25% | 100 | 10 | 0 | 10 of 10 | 100% [72%, 100%] |
| 50% | 100 | 10 | 0 | 10 of 10 | 100% [72%, 100%] |

*Only 10 of 21 flagged spans had a same-prompt control; the rest are not covered by this experiment.*

