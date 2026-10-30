"""
Item 3 (partial): the standard copy-paste baseline attack, CPU-only.

Standard literature setup (Kirchenbauer et al. 2023a and later papers): splice human-written
text in with the watermarked span, so the detector scores a mixed document. Here: prepend a
block of human text (from C4, offset well clear of the generation prompts) before the span,
at several "copy ratios" r = (human tokens) / (human + span tokens), and rescore the WHOLE
combined text with the same detector -- exactly what a real detector would see if the
attacker published the two concatenated. No model, no GPU: this is the one standard
baseline attack that needs none.

  python research/qwen_attack/copy_paste_attack.py

Runs on every FLAGGED span of each verified run folder (the same 46 / 21 spans the
adaptive attack was run on), so the two are directly comparable. Writes
results/qwen_attack/COPY_PASTE.md and .csv.
"""

import csv
import json
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RATIOS = (0.0, 0.1, 0.25, 0.5)
HUMAN_START = 500   # the C4 file has 1000 documents; generation prompts used indices ~28-400, so 500: is clear of them and leaves 500 documents to draw from


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * p, 100 * (c - h), 100 * (c + h)


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


def human_prefix_words(docs, needed_words):
    """A single long human passage (several documents concatenated) with at least needed_words words.
    Fails loudly (never silently returns too few words, e.g. from an out-of-range HUMAN_START slicing
    the source list down to nothing) so a configuration mistake cannot pass as a result of zero dilution."""
    words = []
    for doc in docs:
        words += doc.split()
        if len(words) >= needed_words:
            return words
    raise RuntimeError("only %d human words available (needed %d): HUMAN_START leaves too few source documents; "
                        "check it against the real size of data/C4/sampled_dataset_train.json" % (len(words), needed_words))


def analyse(label, d, samples, score_fn, human_words):
    flagged = {x["sample_id"]: x["z0"] for x in json.load(open(d / "screening.json")) if x["flagged"]}
    spans = {s.sample_id: s.span for s in samples if s.sample_id in flagged}
    n = len(spans)
    L = ["## %s (%d flagged spans)\n" % (label, n), "| copy ratio (human words : total) | still detected (z>2.0) | detection rate (95%% interval) |", "|---|---|---|"]
    rows = []
    for r in RATIOS:
        k = 0
        for sid, span in spans.items():
            span_words = len(span.split())
            if r == 0.0:
                combined = span
            else:
                need = int(round(r / (1 - r) * span_words))
                prefix = " ".join(human_words[:need])
                combined = prefix + " " + span
            z = score_fn(combined)
            if z > 2.0:
                k += 1
        p, lo, hi = wilson(k, n)
        L.append("| %.0f%% | %d of %d | %.0f%% [%.0f%%, %.0f%%] |" % (100 * r, k, n, p, lo, hi))
        rows.append([label, r, k, n, "%.1f" % p, "%.1f" % lo, "%.1f" % hi])
    return L + [""], rows


def main():
    os.chdir(ROOT)
    sys.path[:0] = [str(HERE), str(ROOT / "research" / "attacks_core")]
    import samples
    import sta_core
    tok = sta_core.load_tokenizer()
    score_fn = lambda t: sta_core.score_text(tok, t)["z"]
    docs = json.load(open("data/C4/sampled_dataset_train.json"))[HUMAN_START:]
    MAX_NEED = int(round(max(RATIOS) / (1 - max(RATIOS)) * 600))  # a safety margin above the longest span (500 tokens)
    human_words = human_prefix_words(docs, MAX_NEED)
    groups = latest_full_runs(ROOT / "results" / "qwen_attack")
    doc = ["# Copy-paste baseline attack (standard literature setup, CPU-only, no model)\n",
           "A block of human-written C4 text is prepended before the watermarked span, at several ratios of human-to-total words, and the WHOLE "
           "combined text is rescored. This is the one standard non-adaptive baseline that needs no paraphraser model. It runs on the exact same "
           "flagged spans the adaptive attack ran on, so the two are directly comparable. 0%% is the unattacked span itself (should match the original z0 count). Lengths are never pooled.\n"]
    rows = []
    for key in ("200", "500"):
        if key not in groups:
            continue
        files = [f for f in json.load(open(groups[key] / "run.json"))["args"]["files"] if os.path.exists(f)]
        L, r = analyse("%s-token spans" % key, groups[key], samples.load_samples(files), score_fn, human_words)
        doc += L
        rows += r
    out = ROOT / "results" / "qwen_attack"
    (out / "COPY_PASTE.md").write_text("\n".join(doc) + "\n")
    with open(out / "COPY_PASTE.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["length", "ratio", "detected", "n", "rate_pct", "ci_low", "ci_high"])
        w.writerows(rows)
    print("\n".join(doc))
    print("written:", out / "COPY_PASTE.md")


if __name__ == "__main__":
    main()
