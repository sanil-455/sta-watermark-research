"""
Greedy black-box attack.

Apply the best-ranked candidate, re-measure the real combined
text, repeat until the document escapes detection or the
candidates run out.

Greedy rather than beam search because every state costs a
query here, where the white-box attack got states for free.
Distant edits do not interact, so the initial ranking stays
largely valid as edits accumulate; the per-step re-query
catches the cases where it does not.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "attacks_core"))

from edit_ops import edits_conflict, edits_to_text, describe


def greedy_attack(oracle, tokenizer, base_ids, ranked,
                  max_edits=40, log=print):
    """
    Add candidates one at a time, best first.

    `ranked` comes from rank_candidates and is already sorted
    by measured effect. Each accepted edit costs one query to
    confirm its real effect in combination with the edits
    already applied.

    Stops when the oracle reports the text is no longer
    flagged, or when no candidate remains that lowers z.
    """
    applied = []
    base_text = tokenizer.decode(base_ids, skip_special_tokens=True)
    current_z = oracle.query(base_text)

    log(f"start z = {current_z:.6f}")

    for step in range(max_edits):
        best = None

        for edit, _, _ in ranked:
            if any(edits_conflict(edit, a) for a in applied):
                continue

            trial = applied + [edit]
            text = edits_to_text(tokenizer, base_ids, trial)
            z = oracle.query(text)

            if z < current_z:
                best = (edit, z, text)
                break

        if best is None:
            log("no remaining candidate lowers z")
            break

        edit, z, text = best
        applied.append(edit)
        current_z = z

        log(f"  edit {len(applied):2d}: z={z:.6f}  {edit['label']}")

        if oracle.is_broken(text):
            log(f"BROKEN after {len(applied)} edits, "
                f"{oracle.queries} queries")
            return applied, current_z, text

    log(f"stopped at z={current_z:.6f} after {len(applied)} edits")
    return applied, current_z, None

