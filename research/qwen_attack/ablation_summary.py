"""
Side-by-side comparison of the ablation configs (items 5, 6, 8) against the original
default-config run, on the same subset of spans. CPU-only, read-only.

  python research/qwen_attack/ablation_summary.py

Works on a partly-finished queue: a config whose folder/samples don't exist yet is
shown as "not finished yet", never silently skipped. Writes results/qwen_attack_ablation/SUMMARY.md.
"""

import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MAIN = "results/qwen_attack/run_20261010_043023_lenient"
OUT = "results/qwen_attack_ablation"
CONFIGS = [("default (original run)", MAIN, None), ("n_cands=20", OUT + "/ncands20", None), ("seed=777", OUT + "/seed777", None),
           ("n_cands=90", OUT + "/ncands90", None), ("preset=strict", OUT + "/strict", None), ("target z<=1.5", OUT + "/z15", 1.5)]


def load_samples(d):
    out = {}
    sd = Path(d) / "samples"
    if not sd.exists():
        return None
    for p in sd.glob("*.json"):
        s = json.load(open(p))
        if "attack" in s:
            out[s["sample_id"]] = s["attack"]
    return out


def main():
    os.chdir(ROOT)
    subset_file = Path(OUT) / "subset_ids.txt"
    if not subset_file.exists():
        print("no subset_ids.txt yet: the ablation queue has not started (or has not reached the subset-selection step)")
        return
    ids = subset_file.read_text().split()
    doc = ["# Ablation comparison (items 5, 6, 8): %d spans, same subset across every config\n" % len(ids),
           "Each config's own internal break threshold is shown where it differs from the paper's default (2.0); a span's \"broken\" status always means z_final <= that config's own target.\n",
           "| config | finished | broken | avg z_final | avg queries | avg rewrites |", "|---|---|---|---|---|---|"]
    per_span = {}
    any_missing = False
    for label, folder, target in CONFIGS:
        samp = load_samples(folder)
        if samp is None:
            doc.append("| %s | **not finished yet** | - | - | - | - |" % label)
            any_missing = True
            continue
        rows = [samp[i] for i in ids if i in samp]
        missing_ids = [i for i in ids if i not in samp]
        if missing_ids:
            any_missing = True
        n = len(rows)
        broken = sum(1 for a in rows if a["status"] == "broken")
        avg_z = sum(a["z_final"] for a in rows) / n if n else float("nan")
        avg_q = sum(a["queries"] for a in rows) / n if n else float("nan")
        avg_r = sum(len(a["rewrites"]) for a in rows) / n if n else float("nan")
        note = " (%d of %d spans done)" % (n, len(ids)) if missing_ids else ""
        doc.append("| %s%s | %d of %d%s | %d | %.2f | %.0f | %.1f |" % (label, " (target %.1f)" % target if target else "", n, len(ids), note, broken, avg_z, avg_q, avg_r))
        for i in ids:
            if i in samp:
                per_span.setdefault(i, {})[label] = samp[i]
    doc.append("")
    if any_missing:
        doc.append("**Some configs are not finished yet; the table above only covers what has completed so far.**\n")
    doc += ["## Per span (only spans every finished config has reached)\n", "| span | " + " | ".join(l for l, _, _ in CONFIGS) + " |", "|---|" + "---|" * len(CONFIGS)]
    for i in ids:
        cells = []
        for label, _, _ in CONFIGS:
            a = per_span.get(i, {}).get(label)
            cells.append("z %.2f (%s, %dq)" % (a["z_final"], a["status"], a["queries"]) if a else "-")
        doc.append("| %s | %s |" % (i, " | ".join(cells)))
    out_path = Path(OUT) / "SUMMARY.md"
    out_path.write_text("\n".join(doc) + "\n")
    print("\n".join(doc))
    print("written:", out_path)


if __name__ == "__main__":
    main()
