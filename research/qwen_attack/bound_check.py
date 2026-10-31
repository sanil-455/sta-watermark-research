"""
Does the necessary-condition bound predict which spans can be broken?

  python research/qwen_attack/bound_check.py                                       (CPU only; the two main run folders, each length on its own)
  python research/qwen_attack/bound_check.py --batch results/qwen_attack_scale/batch2_len200   (adds the new spans of batch 2 to the 200-token group)

The bound (gamma = 0.5, T tokens, z0 the starting z): rewriting a share r of the pairs at chance level gives z = (1 - r) * z0, so z <= 2 needs
r >= 1 - 2/z0. Only the eligible sentences (character share e) may change; when e is smaller than that, the new pairs must be green LESS often
than chance by delta = ((1 - e) * z0 - 2) / (2 * e * sqrt(T)). The attack stores it as delta_needed (None: more than 0.5 or no eligible text,
i.e. impossible). This table shows how often each class of span really broke. Writes results/qwen_attack/BOUND_CHECK.md.
"""

import argparse
import json
import math
import os
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CLASSES = [("chance-level rewriting is enough (delta <= 0)", lambda d: d is not None and d <= 0),
           ("needs selection gain in (0, 0.05]", lambda d: d is not None and 0 < d <= 0.05),
           ("needs selection gain in (0.05, 0.10]", lambda d: d is not None and 0.05 < d <= 0.10),
           ("needs selection gain in (0.10, 0.20]", lambda d: d is not None and 0.10 < d <= 0.20),
           ("needs selection gain above 0.20", lambda d: d is not None and d > 0.20),
           ("impossible by the bound (gain above 0.5 or no eligible text)", lambda d: d is None)]


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * p, 100 * (c - h), 100 * (c + h)


def load(folder, skip=()):
    out = []
    for p in sorted((Path(folder) / "samples").glob("*.json")):
        s = json.load(open(p))
        if "attack" in s and s["sample_id"] not in skip:
            out.append({"id": s["sample_id"], "broken": s["attack"]["status"] == "broken", "dn": s["delta_needed"], "e": s["eligible_share"], "need": s["needed_share"],
                        "da": s["measures"]["pairs"]["delta_achieved"], "z0": s["attack"]["z0"]})
    return out


def analyse(label, spans):
    L = ["## %s (%d attacked spans, %d broken)\n" % (label, len(spans), sum(s["broken"] for s in spans)), "| class of span, by the bound | spans | broken | break rate (95% interval) |", "|---|---|---|---|"]
    for name, test in CLASSES:
        sel = [s for s in spans if test(s["dn"])]
        k = sum(s["broken"] for s in sel)
        p, lo, hi = wilson(k, len(sel))
        L.append("| %s | %d | %d | %s |" % (name, len(sel), k, ("%.0f%% [%.0f%%, %.0f%%]" % (p, lo, hi)) if sel else "n/a"))
    pos, neg = [s["dn"] for s in spans if s["broken"]], [s["dn"] for s in spans if not s["broken"]]
    val = lambda d: math.inf if d is None else d
    wins = sum((val(a) < val(b)) + 0.5 * (val(a) == val(b)) for a in pos for b in neg)
    L.append("")
    if pos and neg:
        finite = sorted({s["dn"] for s in spans if s["dn"] is not None})
        acc = lambda t: sum(((s["dn"] is not None and s["dn"] <= t) == s["broken"]) for s in spans) / len(spans)
        best = max(finite, key=lambda t: (acc(t), -t))
        L += ["- a lower delta_needed predicts a break: AUC %.3f (1.0 = every broken span needed less gain than every unbroken one, 0.5 = no information)" % (wins / (len(pos) * len(neg))),
              "- the rule \"predicted to break when delta_needed <= t\" is right for %.0f%% of spans at its best threshold t = %.3f, and for %.0f%% at t = 0 (chance-level rewriting only)" % (100 * acc(best), best, 100 * acc(0.0)),
              "- largest delta_needed among spans that broke: %.3f; smallest among spans that did not (finite ones): %s" % (max(d for d in pos if d is not None), ("%.3f" % min(d for d in neg if d is not None)) if any(d is not None for d in neg) else "n/a")]
    need = [s for s in spans if s["dn"] is not None and s["dn"] > 0 and s["da"] is not None]
    hit = [s for s in need if s["da"] >= s["dn"]]
    got = [s["da"] for s in spans if s["da"] is not None]
    L.append("- among spans that needed a gain above chance: the attack's measured gain reached it for %d of %d (broken: %d of %d)" % (
        len(hit), len(need), sum(s["broken"] for s in hit), len(hit)))
    if got:
        L.append("- the attack's measured selection gain (delta_achieved) over all spans, mean / median / max: %.3f / %.3f / %.3f" % (statistics.mean(got), statistics.median(got), max(got)))
    return L + [""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default=None)
    args = ap.parse_args()
    os.chdir(ROOT)
    groups = {}
    for d in sorted(Path("results/qwen_attack").glob("run_*")):
        try:
            run = json.load(open(d / "run.json"))
        except Exception:
            continue
        if not run["args"].get("only"):
            groups["500" if any("len500" in f for f in run["args"].get("files", [])) else "200"] = d
    doc = ["# Bound check: does the necessary condition predict which spans can be broken?\n", ("The bound" + __doc__.split("The bound", 1)[1].split("Writes")[0]).strip() + "\n"]
    for key in ("200", "500"):
        if key not in groups:
            continue
        spans = load(groups[key])
        if key == "200" and args.batch:
            spans += load(args.batch, skip={s["id"] for s in spans})
        doc += analyse("%s-token spans%s" % (key, " (batch 1 + batch 2)" if key == "200" and args.batch else ""), spans)
    (ROOT / "results" / "qwen_attack" / "BOUND_CHECK.md").write_text("\n".join(doc) + "\n")
    print("\n".join(doc))


if __name__ == "__main__":
    main()
