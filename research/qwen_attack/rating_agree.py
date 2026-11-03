"""
Agreement between raters, and between raters and the reading.

  python research/qwen_attack/rating_agree.py          (CPU only; reads results/rating/rating_*.csv and answer_key.csv)

Ratings: F faithful, M minor drift, C meaning changed, U unreadable; "acceptable" = F or M. Cohen's kappa = (observed - chance) / (1 - chance).
A pair's consensus is the majority of its raters on acceptable versus not; a tie counts as "no majority" and is left out of the rates.
Control pairs are shown separately and never enter the rates. Writes results/rating/RATING_REPORT.md.
"""

import csv
import math
import os
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CODES = ("F", "M", "C", "U")
ACCEPT = ("F", "M")


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * p, 100 * (c - h), 100 * (c + h)


def kappa(a, b):
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in set(a) | set(b))
    return 1.0 if pe == 1 and po == 1 else (0.0 if pe == 1 else (po - pe) / (1 - pe))


def read_ratings(folder):
    raters, invalid = {}, {}
    for p in sorted(Path(folder).glob("rating_*.csv")):
        name, r = p.stem[len("rating_"):], {}
        invalid[name] = 0
        for row in csv.DictReader(open(p, newline="")):
            v = (row.get("rating") or "").strip().upper()
            if v in CODES:
                r[row["pair_id"]] = v
            elif v:
                invalid[name] += 1
        raters[name] = r
    return raters, invalid


def consensus(raters, pid):
    votes = [r[pid] in ACCEPT for r in raters.values() if pid in r]
    if len(votes) < 2 or votes.count(True) == votes.count(False):
        return None
    return votes.count(True) > votes.count(False)


def main():
    os.chdir(ROOT)
    folder = ROOT / "results" / "rating"
    raters, invalid = read_ratings(folder)
    key = {r["pair_id"]: r for r in csv.DictReader(open(folder / "answer_key.csv", newline=""))}
    L = ["# Rating report\n", "Raters: %s. Cells that were not F, M, C or U were ignored: %s.\n" % (", ".join(raters) or "none", invalid)]
    if len(raters) < 2:
        L.append("At least two returned sheets are needed for agreement; found %d." % len(raters))
        (folder / "RATING_REPORT.md").write_text("\n".join(L) + "\n")
        print("\n".join(L))
        return
    L += ["## Agreement between raters (all pairs, controls included)\n", "| raters | pairs both rated | exact agreement | kappa (F/M/C/U) | kappa (acceptable vs not) |", "|---|---|---|---|---|"]
    for a, b in combinations(raters, 2):
        both = sorted(set(raters[a]) & set(raters[b]))
        x, y = [raters[a][p] for p in both], [raters[b][p] for p in both]
        xb, yb = [v in ACCEPT for v in x], [v in ACCEPT for v in y]
        L.append("| %s and %s | %d | %.0f%% | %.3f | %.3f |" % (a, b, len(both), 100 * sum(i == j for i, j in zip(x, y)) / max(1, len(both)), kappa(x, y), kappa(xb, yb)))
    L.append("")
    cons = {p: consensus(raters, p) for p in key}
    for kind in ("control_known_meaning_change", "control_identical"):
        ids = [p for p, r in key.items() if r["kind"] == kind and cons[p] is not None]
        want = (kind == "control_identical")
        ok = sum(cons[p] == want for p in ids)
        L.append("- %s: %d of %d controls with a majority were rated as expected (%s)" % (kind, ok, len(ids), "acceptable" if want else "not acceptable"))
    L += ["", "## The final edits (controls excluded), by length\n", "| length | edits rated | with a majority | judged NOT acceptable (95% interval) | tied |", "|---|---|---|---|---|"]
    for g in ("200", "500"):
        ids = [p for p, r in key.items() if r["kind"] == "final" and r["group"] == g]
        got = [p for p in ids if cons[p] is not None]
        bad = sum(1 for p in got if cons[p] is False)
        p_, lo, hi = wilson(bad, len(got))
        L.append("| %s-token | %d | %d | %d (%.0f%% [%.0f%%, %.0f%%]) | %d |" % (g, len(ids), len(got), bad, p_, lo, hi, len(ids) - len(got)))
    both = [(p, cons[p]) for p, r in key.items() if r["kind"] == "final" and cons[p] is not None and r["reading_verdict"] in ("faithful", "minor drift", "meaning changed")]
    if both:
        mine = [key[p]["reading_verdict"] in ("faithful", "minor drift") for p, _ in both]
        theirs = [c for _, c in both]
        L += ["", "## Raters against the reading (final edits that have a reading verdict)\n",
              "- pairs: %d, agreement on acceptable vs not: %.0f%%, kappa: %.3f" % (len(both), 100 * sum(a == b for a, b in zip(mine, theirs)) / len(both), kappa(mine, theirs)),
              "- pairs where the raters' majority and the reading disagree: %s" % (", ".join(p for (p, c), m in zip(both, mine) if c != m) or "none")]
    (folder / "RATING_REPORT.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
