"""
Final summary of the whole experiment, built from the run folders. Nothing is
typed in by hand.

  python research/qwen_attack/final_summary.py

Finds the latest full run of each length (quick checks made with --only are
ignored), reads its results, its reading verdicts and its audit, and writes
results/qwen_attack/FINAL_SUMMARY.md. The two lengths are never pooled.
"""

import json
import os
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ACCEPTABLE = ("faithful", "minor drift")


def pick_runs(runs_dir):
    found = {}
    for d in sorted(Path(runs_dir).glob("run_*")):
        try:
            run = json.load(open(d / "run.json"))
        except Exception:
            continue
        if run["args"].get("only"):
            continue
        found["500" if any("len500" in f for f in run["args"].get("files", [])) else "200"] = d
    return found


def stat(values):
    v = [x for x in values if x is not None]
    return "n/a" if not v else "%.1f / %.1f / %.1f" % (statistics.mean(v), statistics.median(v), max(v))


def read_verdicts(reading_text, classify):
    human, sample, idx = {}, None, None
    for line in reading_text.split("\n"):
        m = re.match(r"^## (\S+): z ", line)
        if m:
            sample = m.group(1)
        m = re.match(r"^\*\*\[(\d+)\] rewrite:\*\*", line)
        if m:
            idx = int(m.group(1))
        m = re.match(r"^Verdict:[ \t]*(.*)$", line)
        if m:
            human[(sample, idx)] = classify(m.group(1))
    return human


def audit_vs_reading(audit, human):
    auto = {(s["sample_id"], r["segment"]): r["verdict"] for s in audit["spans"] for r in s["rewrites"]}
    keys = [k for k in human if k in auto and human[k] in ("faithful", "minor drift", "meaning changed", "garbled")]
    if not keys:
        return None
    n = len(keys)
    ok = lambda v: v in ACCEPTABLE
    hb, ab = [ok(human[k]) for k in keys], [ok(auto[k]) for k in keys]
    binary = sum(x == y for x, y in zip(hb, ab)) / n
    pe = sum(((sum(hb) / n) if c else 1 - sum(hb) / n) * ((sum(ab) / n) if c else 1 - sum(ab) / n) for c in (True, False))
    bad = [k for k in keys if not ok(human[k])]
    return {"n": n, "exact": sum(human[k] == auto[k] for k in keys) / n, "binary": binary,
            "kappa": (binary - pe) / (1 - pe) if pe < 1 else 1.0, "meaning_changes": len(bad),
            "caught": sum(1 for k in bad if not ok(auto[k])), "false_alarms": sum(1 for k in keys if ok(human[k]) and not ok(auto[k]))}


def group_section(label, rd, report, tally_reading):
    run, screening, samples = report.load(rd)
    ok = [s for s in samples if "attack" in s]
    broken = [s for s in ok if s["attack"]["status"] == "broken"]
    flagged = sum(1 for x in screening if x["flagged"])
    out = ["## %s (run folder %s)\n" % (label, rd.name)]
    out.append("- spans loaded %d | flagged by the detector (z0 > 2) %d | attacked %d | **broken %d (%.0f%% of attacked)**" % (
        len(screening), flagged, len(ok), len(broken), 100 * len(broken) / max(1, len(ok))))
    reasons = {}
    for s in ok:
        k = "broken" if s["attack"]["status"] == "broken" else s["attack"]["reason"]
        reasons[k] = reasons.get(k, 0) + 1
    out.append("- outcomes: %s" % reasons)
    if broken:
        bp = [s["attack"]["break_point"] for s in broken]
        out.append("- broken spans, mean / median / max: sentences rewritten %% of eligible %s | tokens not preserved %% %s | tokens that are new %% %s | words not preserved %% %s | characters not preserved %% %s | queries %s | minutes %s"
                   % (stat([s["measures"]["sentences"]["pct_of_eligible"] for s in broken]), stat([s["measures"]["tokens"]["pct_not_preserved"] for s in broken]),
                      stat([s["measures"]["tokens"]["pct_new"] for s in broken]), stat([s["measures"]["words"]["pct_not_preserved"] for s in broken]),
                      stat([s["measures"]["characters"]["pct_not_preserved"] for s in broken]), stat([s["attack"]["queries"] for s in broken]),
                      stat([s["attack"]["seconds"] / 60 for s in broken])))
        out.append("- at the break: sentences rewritten %s | queries %s" % (stat([b["sentences_rewritten"] for b in bp]), stat([b["queries"] for b in bp])))
    out.append("- totals over all attacked spans: %d detector queries, %.2f hours of attack time, %d out-of-memory events during rewriting, %d samples with errors" % (
        sum(s["attack"]["queries"] for s in ok), sum(s["attack"]["seconds"] for s in ok) / 3600,
        sum(len(st["oom_events"]) for s in ok for st in s["attack"]["steps"]), len(samples) - len(ok)))
    reading = rd / "READING.md"
    human = {}
    if reading.exists():
        spans = tally_reading.tally(reading.read_text())
        counts = {}
        for s in spans:
            counts[s["status"]] = counts.get(s["status"], 0) + 1
        verdicts = [v for s in spans for v in s["verdicts"]]
        human = read_verdicts(reading.read_text(), tally_reading.classify)
        done = counts.get("unread", 0) == 0 and counts.get("unrecognised", 0) == 0
        out.append("- **quality (reading verdicts)**: %s broken spans on the sheet | clean %d | not clean %d | unread %d | rewrites: %s%s" % (
            len(spans), counts.get("clean", 0), counts.get("not clean", 0), counts.get("unread", 0),
            ", ".join("%s %d" % (k, verdicts.count(k)) for k in ("faithful", "minor drift", "meaning changed", "garbled", "unread")),
            "" if done else " (**reading incomplete: do not quote the clean count yet**)"))
        out.append("  - clean spans are %.0f%% of the %d attacked spans" % (100 * counts.get("clean", 0) / max(1, len(ok)), len(ok)))
        out.append("  - the contested-rewrites review %s" % ("was applied (READING.md.bak2 exists)" if (rd / "READING.md.bak2").exists() else "has not been applied"))
    else:
        out.append("- quality: no READING.md found")
    audit_file = rd / "audit.json"
    if audit_file.exists():
        audit = json.load(open(audit_file))
        na = len(audit["spans"])
        nc = sum(1 for s in audit["spans"] if s["status"] == "clean")
        line = "- automatic audit (%s): %d of %d broken spans judged clean" % (audit["model"], nc, na)
        cmp = audit_vs_reading(audit, human) if human else None
        if cmp:
            line += "; against the reading on %d rewrites: exact agreement %.0f%%, acceptable-vs-not %.0f%%, kappa %.2f, caught %d of %d meaning changes, %d flags on rewrites the reading accepted" % (
                cmp["n"], 100 * cmp["exact"], 100 * cmp["binary"], cmp["kappa"], cmp["caught"], cmp["meaning_changes"], cmp["false_alarms"])
        out.append(line)
    return "\n".join(out) + "\n"


def main():
    os.chdir(ROOT)
    sys.path[:0] = [str(HERE)]
    import report
    import tally_reading
    runs = pick_runs(ROOT / "results" / "qwen_attack")
    if not runs:
        sys.exit("no full run folders found under results/qwen_attack")
    first = runs.get("200") or runs["500"]
    run = json.load(open(first / "run.json"))
    m = run["model"]
    out = ["# Final summary: Qwen paraphrase attack on the STA watermark\n",
           "Every number is read from the run folders. The attack sees only the generated span, never the prompt; success means the span's z-score is 2.0 or below under the same detector. "
           "The 500-token spans extend the 200-token spans of the same prompts, so the two lengths are reported separately and never pooled. "
           "Quality verdicts come from READING.md (see each folder's REPORT.md for the full per-sample tables, settings and limitations).\n",
           "## Models, settings, environment\n",
           "- paraphraser: %s, revision `%s`, %s parameters, 4-bit nf4, %s GiB model memory" % (m["repo"], m["revision"], "{:,}".format(m["parameters_exact"]), m["memory_footprint_gib"]),
           "- text generator: Llama-2-7B base, %s parameters (computed from its published configuration)" % run["generator_parameters"],
           "- similarity model: %s, %s parameters" % (run["minilm"]["name"], "{:,}".format(run["minilm"]["parameters"])),
           "- preset: %s | gates: %s" % (run["args"]["preset"], run["gates"]["config"]),
           "- sampling: %s | attack: %s" % (m["settings"], run["attack_config"]),
           "- environment: %s | git commit %s\n" % (run["environment"], run["environment"].get("git_commit"))]
    for key, label in (("200", "200-token spans"), ("500", "500-token spans")):
        out.append(group_section(label, runs[key], report, tally_reading) if key in runs else "## %s\n\nno completed run folder yet.\n" % label)
    path = ROOT / "results" / "qwen_attack" / "FINAL_SUMMARY.md"
    path.write_text("\n".join(out) + "\n")
    print("\n".join(out))
    print("written:", path)


if __name__ == "__main__":
    main()
