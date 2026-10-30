"""
AUC and TPR at fixed false-positive rates, before and after the attack, with 95% bootstrap intervals.

  python research/qwen_attack/roc_eval.py            (CPU only; run final_all.py first)

Negatives (never attacked; "detected" means z > threshold, as in the paper):
  model controls  results/raw/controls_len200.json and controls_len500.json  (generate_controls.py)
  human text      the first L tokens of C4 documents (L = 200 or 500), scored as span text alone
Positives, per length and never pooled, every loaded span of the run folder:
  before          z0
  lenient after   z after the first (lenient) attack
  clean after     z after the repair step (PER_SPAN_FINAL.csv); spans never attacked keep z0
Threshold for a target FPR a: the (floor(a*n)+1)-th largest control, so at most a*n controls exceed it.
Writes results/qwen_attack/ROC_EVAL.md and ROC_EVAL.csv.
"""

import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ALPHAS = (0.01, 0.05, 0.10)
FIXED = (1.5, 1.9, 2.0)
REPS = 1000
HUMAN_START, HUMAN_DOCS = 0, 10 ** 6          # every C4 document in the file: human text needs no prompt


def threshold(neg, alpha):
    s = np.sort(np.asarray(neg, dtype=float))[::-1]
    m = int(math.floor(alpha * len(s)))
    return float(s[m]) if m < len(s) else float("-inf")


def auc(pos, neg):
    neg, pos = np.sort(np.asarray(neg, dtype=float)), np.asarray(pos, dtype=float)
    lo, hi = np.searchsorted(neg, pos, "left"), np.searchsorted(neg, pos, "right")
    return float(np.mean((lo + 0.5 * (hi - lo)) / len(neg)))


def tpr(pos, neg, alpha):
    return float(np.mean(np.asarray(pos) > threshold(neg, alpha)))


def roc_points(pos, neg):
    """The points of sklearn's roc_curve (drop_intermediate=True), without needing sklearn."""
    scores = np.concatenate([pos, neg]).astype(float)
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    order = np.argsort(scores, kind="mergesort")[::-1]
    scores, labels = scores[order], labels[order]
    idx = np.r_[np.where(np.diff(scores))[0], labels.size - 1]
    tps = np.cumsum(labels)[idx]
    fps = 1 + idx - tps
    if len(fps) > 2:
        keep = np.where(np.r_[True, np.logical_or(np.diff(fps, 2), np.diff(tps, 2)), True])[0]
        tps, fps = tps[keep], fps[keep]
    tps, fps = np.r_[0, tps], np.r_[0, fps]
    return fps / fps[-1], tps / tps[-1]


def tpr_closest(pos, neg, alpha):
    """The paper's rule (get_tpr_and_f1_at_fpr): the ROC point whose FPR is closest to the target; the first one on ties."""
    fpr, tpr_ = roc_points(np.asarray(pos, dtype=float), np.asarray(neg, dtype=float))
    i = int(np.argmin(np.abs(fpr - alpha)))
    return float(tpr_[i]), float(fpr[i])


def paper_fixed(pos, neg, t):
    """The paper's fixed-threshold metrics at z > t: its 'auc' is computed on thresholded labels, i.e. (TPR + TNR) / 2; F1 is class-balanced."""
    a, b = float(np.mean(np.asarray(pos) > t)), float(np.mean(np.asarray(neg) > t))
    return {"tpr": a, "fpr": b, "auc": (a + 1 - b) / 2, "f1": 2 * a / (1 + a + b) if a + b > 0 else 0.0}


def interval(fn, pos, neg, rng, reps=REPS):
    pos, neg = np.asarray(pos, dtype=float), np.asarray(neg, dtype=float)
    rng = np.random.default_rng(0)                  # a fresh, fixed stream per interval: the same numbers whatever the order of sections
    vals = [fn(rng.choice(pos, len(pos)), rng.choice(neg, len(neg))) for _ in range(reps)]
    return [float(x) for x in np.percentile(vals, [2.5, 97.5])]


def human_z(docs, tokenizer, score_fn, length):
    out = []
    for doc in docs:
        ids = tokenizer(doc, add_special_tokens=False)["input_ids"]
        if len(ids) >= length:
            out.append(score_fn(tokenizer.decode(ids[:length], skip_special_tokens=True)))
    return out


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


def positives(d):
    z0 = {x["sample_id"]: x["z0"] for x in json.load(open(d / "screening.json"))}
    lenient, clean = dict(z0), dict(z0)
    for p in (d / "samples").glob("*.json"):
        s = json.load(open(p))
        if "attack" in s:
            lenient[s["sample_id"]] = s["attack"]["z_final"]
    final = d / "PER_SPAN_FINAL.csv"
    if not final.exists():
        sys.exit("%s is missing: run final_all.py first" % final)
    for r in csv.DictReader(open(final)):
        clean[r["sample"]] = float(r["z_final"])
    ids = sorted(z0)
    return {"before": [z0[i] for i in ids], "lenient after": [lenient[i] for i in ids], "clean after": [clean[i] for i in ids]}


def evaluate(length, pos, negsets, rng):
    L, rows = ["## %s-token spans (%d spans loaded, never pooled with the other length)\n" % (length, len(pos["before"]))], []
    for name, neg in negsets.items():
        if len(neg) < 20:
            L.append("### negatives: %s: only %d, too few to use\n" % (name, len(neg)))
            continue
        n = np.asarray(neg, dtype=float)
        L += ["### negatives: %s (n = %d, mean %.2f, std %.2f, max %.2f)\n" % (name, len(n), n.mean(), n.std(), n.max()),
              "false-positive rate of the paper's fixed thresholds on these negatives: " + ", ".join("z > %s: %.1f%%" % (t, 100 * float(np.mean(n > t))) for t in FIXED) + "\n",
              "| measure | before attack | after lenient attack | after repair |", "|---|---|---|---|"]
        cells = []
        for key in ("before", "lenient after", "clean after"):
            est, (lo, hi) = auc(pos[key], n), interval(auc, pos[key], n, rng)
            cells.append("%.3f [%.3f, %.3f]" % (est, lo, hi))
            rows.append([length, name, "AUC", "", key, "%.4f" % est, "%.4f" % lo, "%.4f" % hi, "", len(n), len(pos[key])])
        L.append("| AUC (95%% interval) | %s |" % " | ".join(cells))
        for a in ALPHAS:
            t, cells = threshold(n, a), []
            for key in ("before", "lenient after", "clean after"):
                est, (lo, hi) = tpr(pos[key], n, a), interval(lambda p, q: tpr(p, q, a), pos[key], n, rng)
                cells.append("%.0f%% [%.0f%%, %.0f%%]" % (100 * est, 100 * lo, 100 * hi))
                rows.append([length, name, "TPR", "%.0f%%" % (100 * a), key, "%.4f" % est, "%.4f" % lo, "%.4f" % hi, "%.4f" % t, len(n), len(pos[key])])
            note = "" if a * len(n) >= 1 else " (n too small: the threshold is the largest control, so this level is not resolved)"
            L.append("| TPR at FPR %.0f%% (threshold z > %.2f, actual FPR %.1f%%)%s | %s |" % (100 * a, t, 100 * float(np.mean(n > t)), note, " | ".join(cells)))
        L += ["The paper's own definitions on the same negatives (evaluation_metrics/detection.py): its \"auc\" is computed on thresholded labels, i.e. (TPR + TNR) / 2; its F1 is shown class-balanced (equal weight for spans and negatives); "
              "its TPR at a target FPR is read at the ROC point whose FPR is closest to the target.\n", "| paper-style measure | before attack | after lenient attack | after repair |", "|---|---|---|---|"]
        for t in FIXED:
            cells = []
            for key in ("before", "lenient after", "clean after"):
                m = paper_fixed(pos[key], n, t)
                cells.append("TPR %.0f%%, auc %.3f, F1 %.3f" % (100 * m["tpr"], m["auc"], m["f1"]))
                for nm, val in (("TPR", m["tpr"]), ("auc", m["auc"]), ("F1 balanced", m["f1"]), ("FPR", m["fpr"])):
                    rows.append([length, name, "paper z > %s %s" % (t, nm), "", key, "%.4f" % val, "", "", t, len(n), len(pos[key])])
            L.append("| z > %s (FPR %.1f%% on these negatives) | %s |" % (t, 100 * paper_fixed(pos["before"], n, t)["fpr"], " | ".join(cells)))
        for a in ALPHAS:
            cells = []
            for key in ("before", "lenient after", "clean after"):
                est, got = tpr_closest(pos[key], n, a)
                cells.append("%.0f%% (FPR %.1f%%)" % (100 * est, 100 * got))
                rows.append([length, name, "paper-rule TPR", "%.0f%%" % (100 * a), key, "%.4f" % est, "", "", "", len(n), len(pos[key])])
            L.append("| TPR at target FPR %.0f%% (closest ROC point) | %s |" % (100 * a, " | ".join(cells)))
        L.append("")
    return L, rows


def main():
    os.chdir(ROOT)
    sys.path[:0] = [str(HERE), str(ROOT / "research" / "attacks_core")]
    import sta_core
    tok = sta_core.load_tokenizer()
    score = lambda t: sta_core.score_text(tok, t)["z"]
    docs = json.load(open("data/C4/sampled_dataset_train.json"))[HUMAN_START:HUMAN_START + HUMAN_DOCS]
    print("C4 file: %d documents" % len(docs))
    rng = np.random.default_rng(0)
    doc = ["# ROC evaluation: AUC and TPR at fixed false-positive rates\n",
           "Detected means z > threshold. Thresholds come from negatives that were not attacked (the attack only runs on flagged spans). Intervals are 95%% bootstrap percentiles (%d resamples of both the spans and the negatives). "
           "The positives are a small selection (spans where the model kept generating), so the intervals are wide; a 1%% false-positive level needs at least 100 negatives to be resolved at all. "
           "Model controls are unwatermarked generations from the same model, sampling and prompts rule; human text is the start of C4 documents.\n" % REPS]
    csv_rows = []
    for length, d in sorted(latest_full_runs(ROOT / "results" / "qwen_attack").items()):
        negsets = {}
        cf = Path("results/raw/controls_len%s.json" % length)
        if cf.exists():
            negsets["model controls (unwatermarked)"] = [r["unwatermarked_z"] for r in json.load(open(cf))]
        negsets["human text (C4)"] = human_z(docs, tok, score, int(length))
        L, rows = evaluate(length, positives(d), negsets, rng)
        doc += L
        csv_rows += rows
    out = ROOT / "results" / "qwen_attack"
    (out / "ROC_EVAL.md").write_text("\n".join(doc) + "\n")
    with open(out / "ROC_EVAL.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["length", "negatives", "metric", "target_fpr", "positives", "estimate", "ci_low", "ci_high", "threshold", "n_negatives", "n_positives"])
        w.writerows(csv_rows)
    print("\n".join(doc))
    print("written:", out / "ROC_EVAL.md")


if __name__ == "__main__":
    main()
