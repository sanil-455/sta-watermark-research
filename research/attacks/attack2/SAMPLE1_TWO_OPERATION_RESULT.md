# Sample 1: two-operation Attack 1 + Attack 2 search

This experiment exhaustively evaluated the defined Sample 1 combinations consisting of:

- one valid Attack 1 word substitution
- one valid Attack 2 insertion or deletion

The search contained 742 valid combinations.

## Result

- Combinations evaluated: 742
- Successful evaluations: 742
- Errors: 0
- Baseline STA-1 z: 3.363765
- Detection threshold: 2.0
- Lowest attacked z: 2.894737
- Best delta z: -0.469028
- Watermark breaks: 0

The strongest combination was:

`might -> could` + deletion of `you` at Attack 2 position 206.

Its STA-1 score changed from:

`z = 3.363765` to `z = 2.894737`

with:

- green-count change: -5
- pair-count change: -1
- global semantic similarity: 0.9979
- local semantic similarity: 0.9980

No evaluated combination crossed the STA-1 detection threshold.

## Interpretation

For Sample 1, within this complete 742-combination attack space, the two-operation attack did not break STA-1 detection.

The strongest attack reduced the detection statistic substantially while preserving very high semantic similarity, but the resulting z-score remained above the paper's detection threshold of 2.0.

This result is specific to Sample 1 and to the defined two-operation attack space. It does not by itself establish robustness of STA-1 in general.

## Reproducibility

The scored records are stored separately from this research note:

`results/raw/attack12_scored/sample_1_scored.jsonl`

The corresponding summary is:

`results/raw/attack12_scored/sample_1_scored_summary.json`

The scoring implementation uses the Llama-2 tokenizer and computes the STA-1 pairwise-token statistic directly.
