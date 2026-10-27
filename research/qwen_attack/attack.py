"""
Modules 5 and 6 of the Qwen paraphrase attack: the attack loop and the
change measures.

The attack sees ONE span of generated text, never the prompt. For each
eligible sentence, longest first, it asks Qwen for candidates, passes
them through the gates, asks the detector about every survivor (the
whole span with just that sentence replaced) and keeps the survivor with
the lowest z if that lowers the current z. It stops when z <= 2.0, or
after a whole pass in which nothing improved. There is no cap on
sentences, edits, passes or queries (an optional time budget exists and
is off by default). Candidates are always generated from, and gated
against, the ORIGINAL sentence.
"""

import difflib
import math
import re
import time

Z_THRESHOLD = 2.0
ENTITY = re.compile(r"&(?:[a-zA-Z]+|#\d+);")


class SpanOracle:
    """Counts the queries of one attack and refuses to see anything but the span."""

    def __init__(self, oracle, forbidden=""):
        self.oracle, self.forbidden, self.asked = oracle, forbidden, 0

    def query(self, text):
        if self.forbidden and self.forbidden in text:
            raise AssertionError("the source prompt reached the detector")
        before = self.oracle.queries
        z = self.oracle.query(text)
        self.asked += self.oracle.queries - before          # a repeated text is served from cache and not counted
        return z


def feasibility(z0, eligible_share, tokens):
    """Best case: every eligible character is rewritten with text of green fraction 0.5 - delta.

    Returns the share that must be rewritten at chance level (1 - 2/z0) and the selection gain
    delta needed when the eligible share is smaller than that (None = impossible, delta > 0.5).
    """
    needed_share = max(0.0, 1 - 2 / z0) if z0 > 0 else 0.0
    if eligible_share <= 0:
        return needed_share, None
    delta = ((1 - eligible_share) * z0 - 2) / (2 * eligible_share * math.sqrt(tokens))
    delta = max(0.0, delta)
    return needed_share, (None if delta > 0.5 else delta)


def attack_span(seg, paraphraser, gates, oracle, cfg, log=print):
    started = time.time()
    eligible = sorted(seg.eligible(), key=lambda g: (-len(g.text), g.index))
    z0 = oracle.query(seg.span)
    z, repl, steps = z0, {}, []
    clock = {"generate": 0.0, "gates": 0.0, "detector": 0.0}
    passes, status = 0, None
    if z0 <= Z_THRESHOLD:
        status = "not flagged"
    elif not eligible:
        status = "no eligible text"
    while status is None:
        passes += 1
        improved = False
        for g in eligible:
            if z <= Z_THRESHOLD:
                break
            if cfg.max_minutes and time.time() - started > 60 * cfg.max_minutes:
                status = "time budget"
                break
            t = time.time()
            cands, rec = paraphraser.rewrite(g.text, n=cfg.n_cands, seed=cfg.seed + 7919 * passes + 31 * g.index)
            clock["generate"] += time.time() - t
            t = time.time()
            survivors, funnel = gates.run(g.text, cands)
            clock["gates"] += time.time() - t
            t = time.time()
            best_z, best = None, None
            for text in survivors:
                zt = oracle.query(seg.reassemble({**repl, g.index: text}))
                if best_z is None or zt < best_z:
                    best_z, best = zt, text
            clock["detector"] += time.time() - t
            accepted = best is not None and best_z < z - 1e-9
            steps.append({"pass": passes, "segment": g.index, "chars": len(g.text), "z_before": z,
                          "z_after": best_z, "accepted": accepted, "survivors": len(survivors),
                          "funnel": funnel, "queries_total": oracle.asked,
                          "hit_limit": rec["hit_limit"], "oom_events": rec["oom_events"],
                          "seconds_generate": rec["seconds"]})
            if accepted:
                repl[g.index], z, improved = best, best_z, True
        if status is None:
            if z <= Z_THRESHOLD:
                status = "broken"
            elif not improved:
                status = "stalled"
    if status == "broken":
        reason = None
    elif status in ("no eligible text", "not flagged", "time budget"):
        reason = status
    elif not any(s["survivors"] for s in steps):
        reason = "no candidate passed the gates"
    elif not any(s["accepted"] for s in steps):
        reason = "candidates passed but none lowered z"
    else:
        reason = "improved but not enough (stalled)"
    by_index = {s.index: s for s in seg.segments}
    return {
        "z0": z0, "z_final": z, "status": status, "reason": reason, "passes": passes,
        "queries": oracle.asked, "seconds": round(time.time() - started, 1),
        "seconds_by_phase": {k: round(v, 1) for k, v in clock.items()}, "steps": steps,
        "rewrites": {str(i): {"original": by_index[i].text, "rewrite": t} for i, t in repl.items()},
        "final_span": seg.reassemble(repl),
        "break_point": ({"step": len(steps) - 1, "sentences_rewritten": len(repl),
                         "queries": steps[-1]["queries_total"]} if status == "broken" else None),
    }


def _shares(a, b):
    """(percent of a that is not in b, percent of b that is not in a), by longest matching blocks."""
    kept = sum(m.size for m in difflib.SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks())
    return 100 * (1 - kept / max(1, len(a))), 100 * (1 - kept / max(1, len(b)))


def change_measures(seg, repl, final_span, ids_fn, green_fn):
    """How much of the span changed, three ways, plus the green fraction the rewrites reached."""
    orig = seg.span
    ia, ib = ids_fn(orig), ids_fn(final_span)
    blocks = [b for b in difflib.SequenceMatcher(None, ia, ib, autojunk=False).get_matching_blocks() if b.size]
    unchanged = sum(b.size - 1 for b in blocks)                      # pairs that are the same pair in both texts
    g_unchanged = sum(green_fn(ia[b.a + t], ia[b.a + t + 1]) for b in blocks for t in range(b.size - 1))

    def greens(ids):
        return sum(green_fn(ids[k], ids[k + 1]) for k in range(len(ids) - 1))

    ga, gb = greens(ia), greens(ib)
    changed_a, changed_b = len(ia) - 1 - unchanged, len(ib) - 1 - unchanged
    q_new = (gb - g_unchanged) / changed_b if changed_b > 0 else None
    n_elig, n_seg = len(seg.eligible()), len(seg.segments)
    wa, wb = orig.split(), final_span.split()
    (tl, tn), (wl, wn), (cl, cn) = _shares(ia, ib), _shares(wa, wb), _shares(orig, final_span)
    return {
        "sentences": {"rewritten": len(repl), "eligible": n_elig, "segments": n_seg,
                      "pct_of_eligible": 100 * len(repl) / n_elig if n_elig else 0.0,
                      "pct_of_segments": 100 * len(repl) / n_seg if n_seg else 0.0},
        "tokens": {"original": len(ia), "final": len(ib), "length_change": len(ib) - len(ia),
                   "pct_not_preserved": tl, "pct_new": tn},
        "words": {"original": len(wa), "final": len(wb), "length_change": len(wb) - len(wa),
                  "pct_not_preserved": wl, "pct_new": wn},
        "characters": {"original": len(orig), "final": len(final_span), "length_change": len(final_span) - len(orig),
                       "pct_not_preserved": cl, "pct_new": cn},
        "entities": {"before": len(ENTITY.findall(orig)), "after": len(ENTITY.findall(final_span))},
        "pairs": {"green_before": ga, "green_after": gb, "pairs_before": len(ia) - 1, "pairs_after": len(ib) - 1,
                  "pairs_changed_before": changed_a, "pairs_changed_after": changed_b,
                  "green_fraction_of_new_pairs": q_new,
                  "delta_achieved": None if q_new is None else 0.5 - q_new},
    }
