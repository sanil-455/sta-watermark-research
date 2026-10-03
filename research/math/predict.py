"""
Forward prediction: does eta measured on one document predict
edits on another?

If eta is a property of the method rather than the text, this
works. If not, the cost law describes history and predicts
nothing.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from cost_law import edits_required, eta_observed


def cross_predict(runs):
    """
    For each pair, measure eta on one run and predict the other.

    runs: list of (name, p, T, actual_edits)
    """
    out = []
    for src_name, sp, sT, se in runs:
        eta = eta_observed(sp, sT, se)
        for dst_name, dp, dT, de in runs:
            if src_name == dst_name:
                continue
            pred = edits_required(dp, dT, eta)
            out.append({
                "eta_from": src_name,
                "predicting": dst_name,
                "eta": eta,
                "predicted": pred,
                "actual": de,
                "error": pred - de,
            })
    return out


def query_cost(pool_size, edits):
    """
    Detector queries the black-box attack needs.

    One per candidate for the ranking pass, one for the
    baseline, one per accepted edit to confirm it in
    combination. Rejected trials inside the greedy loop are
    cached, so they cost nothing extra.
    """
    return pool_size + 1 + edits
