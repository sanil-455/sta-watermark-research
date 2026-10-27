"""
Check Module 1 on the real files.

Run:  python research/qwen_attack/check_samples.py

Loads every sample file that exists, asks the detector about each span,
and prints the table, the counts, the comparison with the saved z, the
prefix check between lengths, and the state of the generation run.
Loads no language model: only the Llama tokenizer and the detector hash.
"""

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FILES = ["results/raw/baseline_safe.json", "results/raw/baseline_extra.json",
         "results/raw/samples_len200.json", "results/raw/samples_len500.json"]


def report(files, oracle, master_path=None):
    import samples
    ss = samples.load_samples(files)
    flagged, dropped = samples.screen(ss, oracle)
    out = [samples.describe(ss), ""]
    out.append("loaded %d | flagged %d | not flagged %d" % (len(ss), len(flagged), len(dropped)))
    by_file = {}
    for s in ss:
        by_file.setdefault(Path(s.source).stem, []).append(s)
    for tag, group in by_file.items():
        out.append("  %-18s %3d samples, %3d flagged"
                   % (tag, len(group), sum(1 for s in group if s.z0 > samples.Z_THRESHOLD)))
    out.append("not flagged: " + (", ".join("%s (z0 %.3f)" % (s.sample_id, s.z0) for s in dropped) or "none"))
    comp, bad = samples.compared(ss), samples.mismatches(ss)
    out.append("saved z compared with the fresh z0 for %d samples; mismatches: %s"
               % (len(comp), [(s.sample_id, round(s.z0 - s.stored_z, 3)) for s in bad] or "none"))
    if comp:
        out.append("  largest difference: %.4f" % max(abs(s.z0 - s.stored_z) for s in comp))
    pairs, badp = samples.check_prefixes(ss)
    out.append("prefix check (a shorter span must start the longer one): %d pairs, failures: %s"
               % (pairs, badp or "none"))
    if master_path and os.path.exists(master_path):
        m = json.load(open(master_path))
        crashes = m["oom_events"] if "oom_events" in m else m.get("failed", [])
        out.append("generation run: kept %d | stopped early %d | skipped for length %d | out-of-memory %d"
                   % (len(m["done"]), len(m["too_short"]), len(m["skipped_long"]), len(crashes)))
    return "\n".join(out)


def main():
    os.chdir(ROOT)
    sys.path[:0] = [str(HERE), str(ROOT / "research" / "blackbox"),
                    str(ROOT / "research" / "attacks_core")]
    from sta_core import load_tokenizer
    from oracle import DetectorOracle
    files = [p for p in FILES if os.path.exists(p)]
    print("files:", files)
    print(report(files, DetectorOracle(load_tokenizer(), mode="score"),
                 "results/raw/gen_master.json"))


if __name__ == "__main__":
    main()
