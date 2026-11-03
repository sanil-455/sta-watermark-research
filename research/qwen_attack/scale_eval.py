"""
Pooled evaluation of the scaled experiment (200-token spans): the verified batch 1 and the new batch 2.
Lenient attack only (batch 2 has no repair step and, until you read a sample, no quality check).

  python research/qwen_attack/scale_eval.py                      (CPU only; works on a partly finished batch 2 and says so)
  python research/qwen_attack/scale_eval.py --export-sample 25   (after batch 2 is complete: a random sample of broken spans for reading)

Batch 2 spans are those of the batch-2 folder that are not in batch 1. A flagged span without a result file is "pending" and is left out
of every number (keeping its starting z would make the attack look weaker than it is). Writes results/qwen_attack_scale/SCALE_EVAL.md and .csv.
"""

import argparse
import csv
import html
import json
import math
import os
import random
import statistics
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MAIN = "results/qwen_attack/run_20261010_043023_lenient"
BATCH = "results/qwen_attack_scale/batch2_len200"
THRESHOLDS = (2.0, 1.9, 1.5)
ALPHAS = (0.01, 0.05, 0.10)


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * p, 100 * (c - h), 100 * (c + h)


def stat(values):
    return "n/a" if not values else "%.1f / %.1f / %.1f" % (statistics.mean(values), statistics.median(values), max(values))


def load_batch(d, exclude=()):
    d = Path(d)
    z0 = {x["sample_id"]: x["z0"] for x in json.load(open(d / "screening.json")) if x["sample_id"] not in exclude}
    flagged = {x["sample_id"] for x in json.load(open(d / "screening.json")) if x["flagged"] and x["sample_id"] in z0}
    samples, errors = {}, []
    for p in sorted((d / "samples").glob("*.json")) if (d / "samples").exists() else []:
        s = json.load(open(p))
        if s["sample_id"] in z0:
            (samples.__setitem__(s["sample_id"], s) if "attack" in s else errors.append(s["sample_id"]))
    pending = sorted(flagged - set(samples) - set(errors))
    ids = sorted(i for i in z0 if i not in pending and i not in errors)
    return {"ids": ids, "before": [z0[i] for i in ids], "after": [samples[i]["attack"]["z_final"] if i in samples else z0[i] for i in ids],
            "samples": {i: samples[i] for i in ids if i in samples}, "pending": pending, "errors": errors, "z0": z0}


def attack_lines(name, b):
    s = b["samples"]
    n = len(s)
    broken = [i for i, x in s.items() if x["attack"]["status"] == "broken"]
    cells = ["%d spans" % len(b["ids"]), "%d" % n]
    for t in THRESHOLDS:
        k = sum(1 for x in s.values() if x["attack"]["z_final"] <= t)
        p, lo, hi = wilson(k, n)
        cells.append("%d (%.0f%% [%.0f%%, %.0f%%])" % (k, p, lo, hi))
    cells.append("%d stalled, %d with no eligible text" % (sum(1 for x in s.values() if x["attack"]["status"] == "stalled"), sum(1 for x in s.values() if x["attack"]["status"] == "no eligible text")))
    return "| %s | %s |" % (name, " | ".join(cells)), broken


def edit_line(name, samples, broken):
    m = [samples[i]["measures"] for i in broken]
    return ("- %s (%d broken spans), mean / median / max: sentences rewritten %% of eligible %s | tokens not kept %% %s | words not kept %% %s | characters not kept %% %s | queries %s | minutes %s"
            % (name, len(broken), stat([x["sentences"]["pct_of_eligible"] for x in m]), stat([x["tokens"]["pct_not_preserved"] for x in m]), stat([x["words"]["pct_not_preserved"] for x in m]),
               stat([x["characters"]["pct_not_preserved"] for x in m]), stat([samples[i]["attack"]["queries"] for i in broken]), stat([samples[i]["attack"]["seconds"] / 60 for i in broken])))


def detection_tables(R, negsets, pos, rows):
    names = ["before (all spans)", "after: batch 1", "after: batch 2", "after: both batches"]
    keys = ["before", "b1", "b2", "pooled"]
    L = []
    for nm, neg in negsets.items():
        if len(neg) < 20:
            L.append("### negatives: %s: only %d, too few to use\n" % (nm, len(neg)))
            continue
        n = np.asarray(neg, dtype=float)
        L += ["### negatives: %s (n = %d, mean %.2f, std %.2f, max %.2f)\n" % (nm, len(n), n.mean(), n.std(), n.max()),
              "false-positive rate of the paper's fixed thresholds on these negatives: " + ", ".join("z > %s: %.1f%%" % (t, 100 * float(np.mean(n > t))) for t in THRESHOLDS) + "\n",
              "| measure | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]

        def row(label, fn, pct=False):
            cells = []
            for k, name in zip(keys, names):
                if not pos[k]:
                    cells.append("n/a")
                    continue
                est, (lo, hi) = fn(pos[k], n), R.interval(fn, pos[k], n, None)
                cells.append(("%.0f%% [%.0f%%, %.0f%%]" % (100 * est, 100 * lo, 100 * hi)) if pct else "%.3f [%.3f, %.3f]" % (est, lo, hi))
                rows.append([nm, label, name, "%.4f" % est, "%.4f" % lo, "%.4f" % hi])
            L.append("| %s | %s |" % (label, " | ".join(cells)))
        row("AUC (rank-based, 95% interval)", R.auc)
        for a in ALPHAS:
            row("TPR at FPR %.0f%% (threshold z > %.2f)" % (100 * a, R.threshold(n, a)), lambda p, q, a=a: R.tpr(p, q, a), pct=True)
        for t in THRESHOLDS:
            cells = []
            for k, name in zip(keys, names):
                m = R.paper_fixed(pos[k], n, t) if pos[k] else None
                cells.append("TPR %.0f%%, auc %.3f, F1 %.3f" % (100 * m["tpr"], m["auc"], m["f1"]) if m else "n/a")
                if m:
                    rows.append([nm, "paper z > %s TPR" % t, name, "%.4f" % m["tpr"], "", ""])
            L.append("| paper-style z > %s (FPR %.1f%%) | %s |" % (t, 100 * float(np.mean(n > t)), " | ".join(cells)))
        L.append("")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default=MAIN)
    ap.add_argument("--batch", default=BATCH)
    ap.add_argument("--export-sample", type=int, default=0)
    args = ap.parse_args()
    os.chdir(ROOT)
    sys.path[:0] = [str(HERE), str(ROOT / "research" / "attacks_core")]
    import roc_eval as R
    if not (Path(args.batch) / "screening.json").exists():
        sys.exit("batch 2 has not started yet (%s/screening.json is missing): wait until scale.log says \"attacking N new 200-token spans\"" % args.batch)
    b1 = load_batch(args.main)
    b2 = load_batch(args.batch, exclude=set(b1["z0"]))
    out = Path(args.batch).parent
    if args.export_sample:
        if b2["pending"]:
            sys.exit("batch 2 is not finished (%d flagged spans pending): export the sample only when it is complete, or the choice would change" % len(b2["pending"]))
        broken = sorted(i for i, s in b2["samples"].items() if s["attack"]["status"] == "broken")
        chosen = sorted(random.Random(0).sample(broken, min(args.export_sample, len(broken))))
        blocks = ["@@ %s | %s\nO: %s\nR: %s\n" % (sid, idx, html.unescape(v["original"]).strip(), html.unescape(v["rewrite"]).strip())
                  for sid in chosen for idx, v in sorted(b2["samples"][sid]["attack"]["rewrites"].items(), key=lambda kv: int(kv[0]))]
        (out / "FOR_REVIEW_SAMPLE.txt").write_text("".join(blocks))
        json.dump(chosen, open(out / "SAMPLE_IDS.json", "w"))
        print("%d of %d broken spans, %d rewrites, %d characters -> %s" % (len(chosen), len(broken), len(blocks), len("".join(blocks)), out / "FOR_REVIEW_SAMPLE.txt"))
        return
    import sta_core
    tok = sta_core.load_tokenizer()
    negsets = {}
    cf = Path("results/raw/controls_len200.json")
    if cf.exists():
        negsets["model controls (unwatermarked)"] = [r["unwatermarked_z"] for r in json.load(open(cf))]
    negsets["human text (C4)"] = R.human_z(json.load(open("data/C4/sampled_dataset_train.json")), tok, lambda t: sta_core.score_text(tok, t)["z"], 200)
    pos = {"before": b1["before"] + b2["before"], "b1": b1["after"], "b2": b2["after"], "pooled": b1["after"] + b2["after"]}
    partial = len(b2["pending"]) > 0
    doc = ["# Scaled experiment, 200-token spans: batch 1 (verified) and batch 2 (new), lenient attack only\n",
           ("**PARTIAL: %d flagged batch-2 spans are not finished yet and are left out of every number below.**\n" % len(b2["pending"])) if partial else "Batch 2 is complete.\n",
           "Batch 2 has no repair step and no quality check yet: the spans counted as broken only reached z <= 2.0 under the lenient gates. Lengths are never pooled; the two batches are, because they share the length, the settings and the controls. "
           "Spans with an error: batch 1 %d, batch 2 %d (left out).\n" % (len(b1["errors"]), len(b2["errors"])),
           "## Attack success among attacked (flagged) spans, with 95% Wilson intervals\n",
           "| batch | spans with a final state | attacked | broken (z <= 2.0) | z <= 1.9 | z <= 1.5 | outcomes of the others |", "|---|---|---|---|---|---|---|"]
    r1, br1 = attack_lines("batch 1", b1)
    r2, br2 = attack_lines("batch 2", b2)
    both = {"ids": b1["ids"] + b2["ids"], "samples": {**b1["samples"], **b2["samples"]}}
    r3, br3 = attack_lines("both", both)
    doc += [r1, r2, r3, "", "## Size of the edits on the broken spans\n", edit_line("batch 1", b1["samples"], br1), edit_line("batch 2", b2["samples"], br2), edit_line("both", both["samples"], br3), "",
            "## Detection against unwatermarked negatives (ROC), before and after the lenient attack\n"]
    rows = []
    doc += detection_tables(R, negsets, pos, rows)
    (out / "SCALE_EVAL.md").write_text("\n".join(doc) + "\n")
    with open(out / "SCALE_EVAL.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["negatives", "metric", "column", "estimate", "ci_low", "ci_high"])
        w.writerows(rows)
    print("\n".join(doc))
    print("written:", out / "SCALE_EVAL.md")


if __name__ == "__main__":
    main()
