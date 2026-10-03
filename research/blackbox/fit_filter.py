"""
Contextual-fit filter for key-free candidates.

context_fit.filter_pool sorts by `drop`, which black-box
candidates do not have. Same scoring, sorted by fit alone.
"""

from context_fit import BLOCKED


def filter_by_fit(pool, base_ids, fit, max_drop=7.0, log=print):
    """Keep candidates whose replacement fits the context."""
    kept, dropped = [], []

    for e in pool:
        if e["type"] != "replace":
            kept.append(e)
            continue
        if e["label"] in BLOCKED:
            dropped.append((e, float("inf")))
            continue

        score = fit.score(base_ids, e["pos"], e["new_id"])
        e["fit_drop"] = score
        (kept if score <= max_drop else dropped).append(
            e if score <= max_drop else (e, score)
        )

    kept.sort(key=lambda e: e.get("fit_drop", 0))
    log(f"contextual fit: {len(kept)} kept, {len(dropped)} rejected")
    return kept
