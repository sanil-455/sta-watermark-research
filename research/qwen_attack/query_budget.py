"""
Item 9 (realistic threat model): how much does the result depend on detector access?

Reconstructs, from the already-saved per-step logs, what z each attacked span would be
stuck at if the attacker could only query the detector N times (instead of however many
it actually took). This needs no GPU and no new attack run -- every number comes from the
run folders you already have. Budgets include the query cost of the initial z0 check.

  python research/qwen_attack/query_budget.py

Writes results/qwen_attack/QUERY_BUDGET.md and .csv.
"""

import bisect
import csv
import json
import math
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BUDGETS = [5, 10, 20, 50, 100, 200, 500]


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * p, 100 * (c - h), 100 * (c + h)


def trajectory(a):
    """[(cumulative_queries, z_in_effect_after_that_many_queries), ...], starting from the z0 query."""
    z0 = a["z0"]
    points = [(1, z0)]          # the single query that produced z0
    running = z0
    for s in a["steps"]:
        if s["accepted"]:
            running = s["z_after"]
        points.append((s["queries_total"], running))
    return points


def z_at_budget(points, budget):
    qs = [p[0] for p in points]
    i = bisect.bisect_right(qs, budget) - 1
    return points[i][1] if i >= 0 else points[0][1]   # budget < 1 query: can't even measure z0; treat as z0 itself


def latest_full_runs(root):
    found = {}
    for d in sorted(Path(root).glob("run_*")):
        try:
            run = json.load(open(d / "run.json"))
        except Exception:
            continue
        if not run["args"].get("only"):
            found["500" if any("len500" in f for f in run["args"].get("files", [])) else "200"] = d
    return found


def analyse(label, d):
    flagged = {x["sample_id"] for x in json.load(open(d / "screening.json")) if x["flagged"]}
    spans = []
    for p in sorted((d / "samples").glob("*.json")):
        s = json.load(open(p))
        if s["sample_id"] in flagged and "attack" in s:
            spans.append(s["attack"])
    n = len(spans)
    trajs = [trajectory(a) for a in spans]
    actual_used = [a["queries"] for a in spans]
    L = ["## %s (%d flagged, attacked spans)\n" % (label, n), "| query budget | broken (z<=2.0) | break rate (95%% interval) |", "|---|---|---|"]
    rows = []
    for b in BUDGETS + ["unlimited (actual)"]:
        # "unlimited" uses the SAME trajectory logic as every finite budget (the trajectory's
        # last point, i.e. z_final) rather than the recorded status string, so there is exactly
        # one definition of "broken" throughout -- no edge case can make the two disagree.
        k = sum(1 for t in trajs if (t[-1][1] if b == "unlimited (actual)" else z_at_budget(t, b)) <= 2.0)
        p, lo, hi = wilson(k, n)
        L.append("| %s | %d of %d | %.0f%% [%.0f%%, %.0f%%] |" % (b, k, n, p, lo, hi))
        rows.append([label, b, k, n, "%.1f" % p, "%.1f" % lo, "%.1f" % hi])
    broken_actual = [q for q, a in zip(actual_used, spans) if a["status"] == "broken"]
    if broken_actual:
        s = sorted(broken_actual)
        median = s[len(s) // 2] if len(s) % 2 else (s[len(s) // 2 - 1] + s[len(s) // 2]) / 2
        L.append("\n- of the spans the attack actually broke, queries needed: min %d, median %.0f, max %d" % (min(s), median, max(s)))
        for pct in (50, 75, 90):
            idx = min(len(s) - 1, int(math.ceil(pct / 100 * len(s))) - 1)
            L.append("  - %d%% of eventually-broken spans needed %d queries or fewer" % (pct, s[idx]))
    return L + [""], rows


def main():
    os.chdir(ROOT)
    groups = latest_full_runs(ROOT / "results" / "qwen_attack")
    doc = ["# Query budget: how much does the attack rely on detector access?\n",
           "Reconstructed from the saved per-step logs of the lenient attack (no new attack run). A query budget of N means: what z would the attacker be stuck "
           "at if the detector refused all queries past the Nth (the first query is always the z0 check). Lengths are never pooled.\n"]
    rows = []
    for key in ("200", "500"):
        if key not in groups:
            continue
        L, r = analyse("%s-token spans" % key, groups[key])
        doc += L
        rows += r
    out = ROOT / "results" / "qwen_attack"
    (out / "QUERY_BUDGET.md").write_text("\n".join(doc) + "\n")
    with open(out / "QUERY_BUDGET.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["length", "budget", "broken", "n", "rate_pct", "ci_low", "ci_high"])
        w.writerows(rows)
    print("\n".join(doc))
    print("written:", out / "QUERY_BUDGET.md")


if __name__ == "__main__":
    main()
