"""
Final tables for both runs, after the repair step. One command, nothing typed in by hand:

  python research/qwen_attack/final_all.py

For each length (never pooled) it writes, inside that run's folder, PER_SPAN_FINAL.csv,
PER_EDIT_FINAL.csv and DETECTION_TABLE_FINAL.csv, and it writes results/qwen_attack/FINAL_REPORT.md.
A span's final state is the repaired one when its repair finished, otherwise the original attack result.
"""

import csv
import json
import os
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PAPER = [1.5, 1.9] + [round(2 + 0.1 * i, 1) for i in range(21)]      # the paper's evaluation thresholds; detected means z > threshold
SHOW = [1.5, 1.9, 2.0, 2.5, 3.0, 3.5, 4.0]
BAD, OK = ("meaning changed", "garbled"), ("faithful", "minor drift")
SPAN_COLS = ["sample", "tokens", "z0", "z_final", "result", "checked_clean", "breaks@2.0", "breaks@1.9", "breaks@1.5", "rewritten", "rewritten%_of_eligible",
             "tokens_not_kept%", "tokens_new%", "words_not_kept%", "chars_not_kept%", "length_change_tokens", "attack_queries", "repair_queries", "attack_min", "repair_min_incl_your_time"]
EDIT_COLS = ["sample", "segment", "origin_of_rewrite", "words_before", "words_after", "words_not_kept%", "words_new%", "tokens_not_kept%", "tokens_new%",
             "chars_not_kept%", "chars_new%", "meaning_similarity"]


def stat(values):
    v = [x for x in values if x is not None]
    return "n/a" if not v else "%.1f / %.1f / %.1f" % (statistics.mean(v), statistics.median(v), max(v))


def write_csv(path, cols, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(rows)


def latest_full_runs(root):
    found = {}
    for d in sorted(Path(root).glob("run_*")):
        try:
            run = json.load(open(d / "run.json"))
        except Exception:
            continue
        if not run["args"].get("only"):
            found["500-token spans" if any("len500" in f for f in run["args"].get("files", [])) else "200-token spans"] = (d, run)
    return found


def process(label, d, run, mods):
    attack, samples, sentences, sim, ids_fn, raw_ids, flagged_from_reading, classify = mods
    shares = attack._shares
    files = [f for f in run["args"]["files"] if os.path.exists(f)]
    by_id = {s.sample_id: s for s in samples.load_samples(files)}
    screening = json.load(open(d / "screening.json"))
    attacked = {}
    for p in sorted((d / "samples").glob("*.json")):
        s = json.load(open(p))
        if "attack" in s:
            attacked[s["sample_id"]] = s
    reading = (d / "READING.md").read_text() if (d / "READING.md").exists() else ""
    verdicts = flagged_from_reading(reading, classify) if reading else {}
    repairs = {}
    if (d / "repair").exists():
        for p in (d / "repair").glob("*.json"):
            try:
                st = json.load(open(p))
                repairs[st["sample_id"]] = st
            except Exception:
                pass
    rows, edits, final_z, info = [], [], {}, {}
    for sid, s in attacked.items():
        a, seg = s["attack"], sentences.split_span(by_id[sid].span)
        orig = {g.index: g.text for g in seg.segments}
        flagged = {i for i, v in verdicts.get(sid, {}).items() if v in BAD}
        unread = reading and not verdicts.get(sid) and a["status"] == "broken"
        rep = repairs.get(sid)
        if rep and rep.get("status") not in (None, "stopped by the user"):
            source, z_final = "repaired", rep["z_final"]
            repl = {int(k): v for k, v in rep["repl"].items()}
            origin = {int(k): v for k, v in rep["provenance"].items()}
            rq, rm = rep.get("queries", 0), rep.get("seconds", 0) / 60
        else:
            source, z_final = "attack", a["z_final"]
            repl = {int(k): v["rewrite"] for k, v in a["rewrites"].items()}
            origin = {i: "attack, accepted by the reading" for i in repl}
            rq, rm = 0, 0
        broken = z_final <= 2.0
        if not broken:
            clean = "-"
        elif source == "repaired":
            clean = "yes (reading + your approvals)"
        elif flagged:
            clean = "NO: a rewrite marked meaning changed is still in it (not repaired)"
        elif unread or not reading:
            clean = "not read"
        else:
            clean = "yes (reading)"
        final = seg.reassemble(repl)
        ia, ib = ids_fn(seg.span), ids_fn(final)
        (tl, tn), (wl, _), (cl, _) = shares(ia, ib), shares(seg.span.split(), final.split()), shares(seg.span, final)
        n_el = len(seg.eligible())
        final_z[sid] = z_final
        info[sid] = {"broken": broken, "clean": clean.startswith("yes"), "z0": a["z0"], "z": z_final, "tl": tl, "wl": wl, "cl": cl, "n": len(repl), "pct": 100 * len(repl) / n_el if n_el else 0.0, "source": source, "flagged": bool(flagged)}
        rows.append([sid, s["tokens"], "%.3f" % a["z0"], "%.3f" % z_final, "broken" if broken else "not broken", clean,
                     *["YES" if z_final <= t else "no" for t in (2.0, 1.9, 1.5)], "%d of %d" % (len(repl), n_el), "%.1f" % info[sid]["pct"],
                     "%.1f" % tl, "%.1f" % tn, "%.1f" % wl, "%.1f" % cl, len(ib) - len(ia), a["queries"], rq, "%.1f" % (a["seconds"] / 60), "%.1f" % rm])
        for i in sorted(repl):
            o, n = orig[i], repl[i]
            w1, w2 = shares(o.split(), n.split())
            t1, t2 = shares(raw_ids(o), raw_ids(n))
            c1, c2 = shares(o, n)
            edits.append([sid, i, origin.get(i, "?"), len(o.split()), len(n.split()), "%.1f" % w1, "%.1f" % w2, "%.1f" % t1, "%.1f" % t2, "%.1f" % c1, "%.1f" % c2, "%.3f" % sim(o, [n])[0]])
    write_csv(d / "PER_SPAN_FINAL.csv", SPAN_COLS, rows)
    write_csv(d / "PER_EDIT_FINAL.csv", EDIT_COLS, edits)
    before = [x["z0"] for x in screening]
    after = [final_z.get(x["sample_id"], x["z0"]) for x in screening]
    n = len(before)
    table = [(t, sum(z > t for z in before), sum(z > t for z in after)) for t in PAPER]
    write_csv(d / "DETECTION_TABLE_FINAL.csv", ["z_threshold", "detected_before", "detected_after", "rate_before%", "rate_after%", "n_spans"],
              [[t, b, a_, "%.1f" % (100 * b / n), "%.1f" % (100 * a_ / n), n] for t, b, a_ in table])

    A = len(attacked)
    lenient = sum(1 for s in attacked.values() if s["attack"]["status"] == "broken")
    read_clean = sum(1 for sid, s in attacked.items() if s["attack"]["status"] == "broken" and verdicts.get(sid) and all(v in OK for v in verdicts[sid].values()))
    need = [sid for sid, s in attacked.items() if s["attack"]["status"] == "broken" and any(v in BAD for v in verdicts.get(sid, {}).values())]
    done = [sid for sid in need if sid in repairs and repairs[sid].get("status") not in (None, "stopped by the user")]
    rep_broken = [sid for sid in done if info[sid]["broken"]]
    clean_all = [sid for sid, i in info.items() if i["clean"]]
    decisions = [x["answer"] for sid in done for x in repairs[sid]["decisions"]]
    L = ["## %s (run folder %s)\n" % (label, d.name),
         "- spans loaded %d | flagged by the detector (z0 > 2) %d | attacked %d" % (n, sum(1 for x in screening if x["flagged"]), A),
         "- lenient attack: **%d broken (%.0f%% of attacked)**; by the reading, %d of those had every rewrite acceptable and %d had at least one rewrite marked meaning changed" % (lenient, 100 * lenient / max(1, A), read_clean, len(need)),
         "- repair (strict gates, every proposal answered by you): %d spans needed it, %d finished (%d broken, %d not broken), %d not finished; your answers: yes %d, no %d, new batch %d, skip %d" % (
             len(need), len(done), len(rep_broken), len(done) - len(rep_broken), len(need) - len(done), decisions.count("y"), decisions.count("n"), decisions.count("r"), decisions.count("s")),
         "- **after repair: %d of %d attacked spans are broken and checked clean (%.0f%%)**; the other %d are still detected at the paper's default threshold or have an unrepaired flagged rewrite" % (len(clean_all), A, 100 * len(clean_all) / max(1, A), A - len(clean_all)),
         "- attacked spans not detected at the paper's other thresholds: z <= 1.9 %d, z <= 1.5 %d (of %d)" % (sum(1 for z in final_z.values() if z <= 1.9), sum(1 for z in final_z.values() if z <= 1.5), A)]
    if clean_all:
        c = [info[s] for s in clean_all]
        L.append("- the %d clean broken spans, mean / median / max: sentences rewritten %% of eligible %s | tokens not kept %% %s | words not kept %% %s | characters not kept %% %s" % (
            len(c), stat([i["pct"] for i in c]), stat([i["tl"] for i in c]), stat([i["wl"] for i in c]), stat([i["cl"] for i in c])))
    L += ["", "Share of spans still detected (z > threshold), paper thresholds, all %d loaded spans, before and after:\n" % n, "| threshold | before | after |", "|---|---|---|"]
    L += ["| z > %s | %d (%.0f%%) | %d (%.0f%%) |" % (t, b, 100 * b / n, a_, 100 * a_ / n) for t, b, a_ in table if t in SHOW]
    L += ["", "Per span (the same rows, with every edit, are in PER_SPAN_FINAL.csv and PER_EDIT_FINAL.csv in the run folder):\n",
          "| span | z0 | z final | result | checked clean | z<=1.9 | z<=1.5 | rewritten | tokens not kept % | words not kept % | chars not kept % |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    L += ["| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (r[0], r[2], r[3], r[4], r[5], r[7], r[8], r[9], r[11], r[13], r[14]) for r in rows]
    return L + [""]


def main():
    os.chdir(ROOT)
    sys.path[:0] = [str(HERE), str(ROOT / "research" / "attacks_core"), str(ROOT / "research" / "blackbox")]
    import attack, gates, samples, sentences, sta_core
    from repair import flagged_from_reading
    from tally_reading import classify
    tok = sta_core.load_tokenizer()
    mods = (attack, samples, sentences, gates.MiniLM(device="cpu").similarity, lambda t: sta_core.text_to_ids(tok, t),
            lambda t: tok(t, add_special_tokens=False)["input_ids"], flagged_from_reading, classify)
    runs = latest_full_runs(ROOT / "results" / "qwen_attack")
    doc = ["# Final results after the repair step\n",
           "Every number is read from the run folders. The attack sees only the generated span; a span counts as broken when its z is 2.0 or below under the same detector. "
           "The two lengths are reported separately and never pooled. \"Checked clean\" means every rewrite in the final span was either judged faithful or minor drift in the reading of READING.md "
           "(an AI reading) or accepted by you during the repair. Percentages of tokens, words and characters are of the ORIGINAL span (not kept) or of the FINAL span (new), measured as in each run's REPORT.md.\n"]
    for label in ("200-token spans", "500-token spans"):
        doc += process(label, *runs[label], mods) if label in runs else ["## %s\n\nno completed run folder.\n" % label]
    out = ROOT / "results" / "qwen_attack" / "FINAL_REPORT.md"
    out.write_text("\n".join(doc) + "\n")
    print("\n".join(doc))
    print("written:", out)


if __name__ == "__main__":
    main()
