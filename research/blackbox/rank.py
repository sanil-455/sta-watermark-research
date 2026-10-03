"""
Rank candidates by measured effect, using only detector queries.

The white-box attack reads `drop` straight from the key. Here
the attacker buys the same information: apply one candidate,
ask the detector, record how far z fell.

Cost is one query per candidate, plus one for the baseline.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "attacks_core"))

from edit_ops import edits_to_text


def rank_candidates(oracle, tokenizer, base_ids, pool, log=print):
    """
    Measure every candidate alone and sort by how much it helps.

    Returns a list of (edit, z_after, delta) sorted so the most
    damaging candidate comes first. delta is negative when the
    edit lowers z, which is what the attacker wants.

    Requires oracle.mode == "score". A binary oracle returns the
    same answer for every single edit, since one edit never
    flips a flagged document to unflagged, so there is no
    signal to rank on.
    """
    if oracle.mode != "score":
        raise ValueError(
            "ranking needs z-scores; a binary oracle gives no gradient"
        )

    base_text = tokenizer.decode(base_ids, skip_special_tokens=True)
    base_z = oracle.query(base_text)

    log(f"baseline z = {base_z:.6f} (1 query)")

    measured = []
    for edit in pool:
        text = edits_to_text(tokenizer, base_ids, [edit])
        z = oracle.query(text)
        measured.append((edit, z, z - base_z))

    measured.sort(key=lambda row: row[2])

    helpful = sum(1 for _, _, d in measured if d < 0)
    log(f"measured {len(pool)} candidates in {oracle.queries} queries")
    log(f"{helpful} of {len(pool)} lower z")

    return base_z, measured
