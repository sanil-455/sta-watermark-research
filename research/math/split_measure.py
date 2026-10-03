"""
Measure the green fraction over the prompt and the generated
span separately. Only generated tokens passed through STA-1's
accept/resample step, so only they should show inflation.
"""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "attacks_core"))

from sta_core import load_tokenizer, text_to_ids, GAMMA
from covariance import green_sequence, mean

BASELINE = Path("results/raw/baseline_safe.json")


def z_from(p, T):
    """
    z-score computed from a green fraction rather than a raw count.

    Since p = G/T, substituting G = pT into the detector's
    formula gives z = (pT - gamma*T) / sqrt(gamma(1-gamma)T).

    Returns nan rather than crashing when a span is empty, so a
    document with no prompt or no generation still reports.
    """
    if T <= 0 or p is None:
        return float("nan")
    return (p * T - GAMMA * T) / math.sqrt(GAMMA * (1 - GAMMA) * T)

def split_stats(ids, n_prompt):
    """
    Green fraction over the prompt pairs and generated pairs
    separately.

    Pair i covers tokens (i, i+1). A pair counts as generated
    when its SECOND token came from the model, since that is the
    token the accept/resample step acted on. The first generated
    token sits at index n_prompt, so the first generated pair is
    i = n_prompt - 1, which straddles the boundary and correctly
    belongs to the generated side.
    """
    seq = green_sequence(ids)
    boundary = max(0, n_prompt - 1)

    return {
        "total_pairs": len(seq),
        "prompt_pairs": len(seq[:boundary]),
        "gen_pairs": len(seq[boundary:]),
        "p_all": mean(seq),
        "p_prompt": mean(seq[:boundary]),
        "p_gen": mean(seq[boundary:]),
    }

def main():
    tok = load_tokenizer()
    rows = json.load(BASELINE.open())

    header = "{:>3s} {:>6s} {:>7s} {:>5s} {:>8s} {:>8s} {:>8s} {:>7s}"
    print(header.format("id", "ntok", "prompt", "gen",
                        "p_prompt", "p_gen", "z_all", "z_gen"))
    print("-" * 64)

    gen_fractions = []

    for row in rows:
        ids = text_to_ids(tok, row["watermarked_text"])
        s = split_stats(ids, row["prompt_tokens"])
        gen_fractions.append(s["p_gen"])

        line = "{:3d} {:6d} {:7d} {:5d} {:8.4f} {:8.4f} {:7.3f} {:7.3f}"
        print(line.format(
            row["prompt_id"],
            len(ids),
            s["prompt_pairs"],
            s["gen_pairs"],
            s["p_prompt"],
            s["p_gen"],
            z_from(s["p_all"], s["total_pairs"]),
            z_from(s["p_gen"], s["gen_pairs"]),
        ))

    print()
    print("mean p over generated spans: {:.4f}".format(mean(gen_fractions)))
    print("gamma                      : {}".format(GAMMA))


if __name__ == "__main__":
    main()
