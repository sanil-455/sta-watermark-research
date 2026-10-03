"""
Test the independence assumption on real watermarked samples.

Everything before this validated the tools. This points them
at actual STA-1 output.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "attacks_core"))

from sta_core import load_tokenizer, text_to_ids, GAMMA
from covariance import green_sequence, mean, lag_correlation, correlation_profile
from variance_compare import report, compare_variance

BASELINE = Path("results/raw/baseline_safe.json")


def main():
    tok = load_tokenizer()
    rows = json.load(BASELINE.open())

    print("=" * 60)
    print("INDEPENDENCE ASSUMPTION, TESTED ON REAL STA-1 OUTPUT")
    print("=" * 60)
    print(f"gamma = {GAMMA}")
    print()

    all_sequences = []

    for row in rows:
        ids = text_to_ids(tok, row["watermarked_text"])
        seq = green_sequence(ids)
        all_sequences.append(seq)
        report(seq, f"prompt {row['prompt_id']}")

    pooled = [g for seq in all_sequences for g in seq]

    print("=" * 60)
    print("POOLED ACROSS ALL SAMPLES")
    print("=" * 60)
    report(pooled, "pooled")

    out = Path("results/raw/covariance_measurement.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "gamma": GAMMA,
        "pooled": compare_variance(pooled),
        "pooled_profile": {
            str(k): v for k, v in correlation_profile(pooled).items()
        },
        "per_sample": [compare_variance(s) for s in all_sequences],
    }, indent=2))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()

