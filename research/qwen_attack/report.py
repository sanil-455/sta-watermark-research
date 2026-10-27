"""
The final report, built from a run directory. Every number comes from
the files the run wrote; nothing is typed in by hand.

  python research/qwen_attack/report.py results/qwen_attack/run_...

writes REPORT.md, READING.md (the quality audit sheet) and STATUS_entry.md.
"""

import json
import statistics
import sys
from pathlib import Path


def group(sample_id):
    return "500-token spans" if "len500" in sample_id else "200-token spans"


def load(rundir):
    rd = Path(rundir)
    run = json.load(open(rd / "run.json"))
    screening = json.load(open(rd / "screening.json")) if (rd / "screening.json").exists() else []
    samples = [json.load(open(p)) for p in sorted((rd / "samples").glob("*.json"))]
    return run, screening, samples


def stats(values):
    v = [x for x in values if x is not None]
    if not v:
        return "n/a"
    return "%.1f / %.1f / %.1f" % (statistics.mean(v), statistics.median(v), max(v))


def sample_row(s):
    a, m = s["attack"], s["measures"]
    return ("| %s | %d | %.2f | %.0f%% | %.0f%% | %s | %s | %.2f | %d of %d (%.0f%%) | %.1f | %.1f | %.1f | %+d | %d | %.1f | %s |"
            % (s["sample_id"], s["tokens"], a["z0"], 100 * s["eligible_share"], 100 * s["needed_share"],
               "impossible" if s["delta_needed"] is None and s["eligible_share"] < s["needed_share"] else
               ("%.2f" % s["delta_needed"] if s["delta_needed"] is not None else "-"),
               a["status"] + ("" if a["status"] == "broken" else " (%s)" % a["reason"]), a["z_final"],
               m["sentences"]["rewritten"], m["sentences"]["eligible"], m["sentences"]["pct_of_eligible"],
               m["tokens"]["pct_not_preserved"], m["tokens"]["pct_new"], m["characters"]["pct_not_preserved"],
               m["tokens"]["length_change"], a["queries"], a["seconds"] / 60,
               "-" if m["pairs"]["delta_achieved"] is None else "%.2f" % m["pairs"]["delta_achieved"]))


def build(rundir):
    rd = Path(rundir)
    run, screening, samples = load(rd)
    ok = [s for s in samples if "attack" in s]
    bad = [s for s in samples if "attack" not in s]
    out = []
    w = out.append
    w("# Qwen paraphrase attack on the STA watermark: final report\n")
    w("Run: `%s`. Started %s, ended %s. Git commit `%s`.\n" % (rd.name, run.get("started"), run.get("ended"), run["environment"].get("git_commit")))
    w("**What was attacked.** Only the generated span of each sample, never the prompt. The detector was asked only about span text. "
      "**Success** means the span's z-score is 2.0 or below under the same detector. Preset: **%s**. "
      "Quality is judged by reading (see READING.md); the gates below are filters, not proof of faithfulness.\n" % run["args"]["preset"])

    w("## 1. Data\n")
    n_flag = sum(1 for x in screening if x["flagged"])
    w("- spans loaded: %d | flagged by the detector (z0 > 2): %d | not flagged: %d" % (len(screening), n_flag, len(screening) - n_flag))
    w("- not flagged (not attacked): " + (", ".join("%s (z0 %.3f)" % (x["sample_id"], x["z0"]) for x in screening if not x["flagged"]) or "none"))
    for g in ("200-token spans", "500-token spans"):
        sel = [x for x in screening if group(x["sample_id"]) == g]
        w("- %s: %d loaded, %d flagged" % (g, len(sel), sum(1 for x in sel if x["flagged"])))
    gr = run.get("generation_run")
    if gr:
        w("- generation run: %d kept, %d stopped before 195 tokens, %d skipped for prompt length, %d out-of-memory events. "
          "Both span sets are selections (the model had to keep generating)." % (gr["kept"], gr["too_short"], gr["skipped_long"], gr["oom_events"]))
    w("- the 500-token spans are extensions of the 200-token spans of the same prompts, so the two groups are never pooled.\n")

    w("## 2. Models and environment\n")
    m = run["model"]
    w("- **Generator of the watermarked text:** Llama-2-7B base, %s parameters (computed from its published configuration, not re-measured in this run), fp16" % run["generator_parameters"])
    w("- **Paraphraser:** %s, revision `%s`, **%s parameters** (computed from the architecture; transformers reports %s; agree: %s), %s layers, hidden size %s, %s attention heads (%s key/value), vocabulary %s, context %s tokens"
      % (m["repo"], m["revision"], "{:,}".format(m["parameters_exact"]), "{:,}".format(m["parameters_reported_by_transformers"]), m["parameters_agree"],
         m["layers"], m["hidden_size"], m["attention_heads"], m["kv_heads"], m["vocab_size"], m["context_length"]))
    w("  - 4-bit nf4 with double quantisation, float16 compute; stored elements by dtype: %s; model memory footprint %s GiB; GPU allocated after load %s GiB; load time %s s"
      % (m["stored_elements_by_dtype"], m["memory_footprint_gib"], m["gpu_allocated_gib"], m["load_seconds"]))
    w("- **Similarity model:** %s, %s parameters" % (run["minilm"]["name"], "{:,}".format(run["minilm"]["parameters"])))
    w("- **Detector:** the STA detector of the paper, queried as a black box that returns z-scores; the key is never used by the attack")
    e = run["environment"]
    w("- GPU: %s | python %s | torch %s | transformers %s | bitsandbytes %s | accelerate %s | WordNet available: %s" % (
        e.get("gpu"), e.get("python"), m["versions"].get("torch"), m["versions"].get("transformers"), m["versions"].get("bitsandbytes"), m["versions"].get("accelerate"), run["gates"]["wordnet"]))
    w("- peak GPU memory during the run: %s\n" % run.get("peak_gpu", "not recorded"))

    w("## 3. Settings\n")
    w("- gates (%s): %s" % (run["args"]["preset"], run["gates"]["config"]))
    w("- sampling: %s" % m["settings"])
    w("- attack: %s | segmentation: %s\n" % (run["attack_config"], run["segmentation"]))

    w("## 4. Results (never pooled across lengths)\n")
    w("Columns marked mean / median / max.\n")
    for g in ("200-token spans", "500-token spans"):
        sel = [s for s in ok if group(s["sample_id"]) == g]
        if not sel:
            continue
        broken = [s for s in sel if s["attack"]["status"] == "broken"]
        w("### %s: %d attacked, **%d broken (%.0f%%)**\n" % (g, len(sel), len(broken), 100 * len(broken) / len(sel)))
        for label, grp in (("all attacked", sel), ("broken only", broken)):
            if not grp:
                continue
            w("- %s (n=%d): z0 %s | z final %s | sentences rewritten %% of eligible %s | tokens not preserved %% %s | tokens that are new %% %s | words not preserved %% %s | characters not preserved %% %s | length change (tokens) %s | queries %s | minutes %s"
              % (label, len(grp), stats([s["attack"]["z0"] for s in grp]), stats([s["attack"]["z_final"] for s in grp]),
                 stats([s["measures"]["sentences"]["pct_of_eligible"] for s in grp]), stats([s["measures"]["tokens"]["pct_not_preserved"] for s in grp]),
                 stats([s["measures"]["tokens"]["pct_new"] for s in grp]), stats([s["measures"]["words"]["pct_not_preserved"] for s in grp]),
                 stats([s["measures"]["characters"]["pct_not_preserved"] for s in grp]), stats([s["measures"]["tokens"]["length_change"] for s in grp]),
                 stats([s["attack"]["queries"] for s in grp]), stats([s["attack"]["seconds"] / 60 for s in grp])))
        reasons = {}
        for s in sel:
            k = "broken" if s["attack"]["status"] == "broken" else s["attack"]["reason"]
            reasons[k] = reasons.get(k, 0) + 1
        w("- outcomes: %s\n" % reasons)

    w("## 5. Per-sample table\n")
    w("tokens = span length; eligible = share of characters that may be rewritten; needed = share that must be rewritten at chance level (1 - 2/z0); "
      "delta needed = selection gain (how much less green than chance the rewrites must be) if eligible < needed; delta achieved = 0.5 - green fraction of the new token pairs. "
      "Percentages of tokens/characters are of the ORIGINAL span (not preserved) and of the FINAL span (new).\n")
    w("| sample | tokens | z0 | eligible | needed | delta needed | outcome | z final | rewritten | tokens not kept % | tokens new % | chars not kept % | length change | queries | min | delta achieved |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s in ok:
        w(sample_row(s))
    w("")

    w("## 6. Where it breaks\n")
    for g in ("200-token spans", "500-token spans"):
        sel = [s for s in ok if group(s["sample_id"]) == g and s["attack"]["status"] == "broken"]
        if sel:
            bp = [s["attack"]["break_point"] for s in sel]
            w("- %s, broken spans: sentences rewritten at the break %s | queries at the break %s" % (g, stats([b["sentences_rewritten"] for b in bp]), stats([b["queries"] for b in bp])))
    for g in ("200-token spans", "500-token spans"):
        sel = [s for s in ok if group(s["sample_id"]) == g]
        if sel:
            need = [s for s in sel if s["delta_needed"] is not None and s["delta_needed"] > 0]
            hit = [s for s in need if s["measures"]["pairs"]["delta_achieved"] is not None and s["measures"]["pairs"]["delta_achieved"] >= s["delta_needed"]]
            imp = [s for s in sel if s["delta_needed"] is None and s["eligible_share"] < s["needed_share"]]
            w("- %s: %d need no selection gain (eligible text enough at chance level), %d need delta > 0, of which the achieved delta reached the need in %d; %d are impossible by rewriting (eligible share too small even for delta 0.5)"
              % (g, len(sel) - len(need) - len(imp), len(need), len(hit), len(imp)))
    w("")

    w("## 7. Gates: where candidates are lost\n")
    tot, rej = {"generated": 0, "after_clean": 0, "after_spelling": 0, "after_content": 0, "after_semantic": 0, "after_fluency": 0}, {}
    for s in ok:
        for st in s["attack"]["steps"]:
            f = st["funnel"]
            for k in tot:
                tot[k] += f.get(k, 0)
            for k, v in f.get("rejected", {}).items():
                rej[k] = rej.get(k, 0) + v
    w("- candidates generated %d -> after clean %d -> spelling %d -> content %d -> semantic %d -> fluency %d" % tuple(tot[k] for k in tot))
    w("- rejected by reason: %s\n" % dict(sorted(rej.items(), key=lambda kv: -kv[1])))

    w("## 8. Resources and problems\n")
    w("- time (hours): attack %.2f | candidate generation %.2f | gates %.2f | detector queries %.2f" % (
        sum(s["attack"]["seconds"] for s in ok) / 3600, sum(s["attack"]["seconds_by_phase"]["generate"] for s in ok) / 3600,
        sum(s["attack"]["seconds_by_phase"]["gates"] for s in ok) / 3600, sum(s["attack"]["seconds_by_phase"]["detector"] for s in ok) / 3600))
    w("- detector queries: %d in total (screening queries not included)" % sum(s["attack"]["queries"] for s in ok))
    oom = sum(len(st["oom_events"]) for s in ok for st in s["attack"]["steps"])
    hit = sum(st["hit_limit"] for s in ok for st in s["attack"]["steps"])
    w("- out-of-memory events during rewriting: %d | rewrites that hit the length limit: %d | samples that raised an error: %d%s" % (
        oom, hit, len(bad), "" if not bad else " (" + ", ".join(s["sample_id"] for s in bad) + ")"))
    ent = sum(s["measures"]["entities"]["before"] - s["measures"]["entities"]["after"] for s in ok)
    w("- HTML entities (such as &apos;) removed by rewriting: %d across all spans. Rewriting turns them into plain characters, which changes tokenisation without paraphrasing; this effect is not separated here.\n" % ent)

    w("## 9. Limitations\n")
    for line in (
        "Sentence boundaries are a heuristic; sentences ending in an abbreviation merge with the next one.",
        "About a third of each span is not attackable by design (the mid-sentence start, the cut-off end, quotations, short or non-prose pieces).",
        "The gates cannot see all meaning changes (for example income -> revenue). Faithfulness is established only by reading READING.md.",
        "The necessary-condition bound uses the eligible character share as a stand-in for the share of token pairs.",
        "The spans are a selection: the model had to continue for 195 tokens (500-token set: 495). The two lengths are not independent samples.",
        "Spans are model-generated at temperature 1.0 and already contain glued words and HTML entities copied from C4.",
        "One detector (the STA detector with its published key) and one paraphrasing model; results do not transfer automatically to other models."):
        w("- " + line)
    w("\n## 10. Reproduction\n")
    w("Command: `%s`. Files: `%s`. Gate config and seeds are in run.json; every sample's steps, funnels and rewrites are in `samples/`." % (run["args"].get("command"), "`, `".join(run["args"].get("files", []))))
    (rd / "REPORT.md").write_text("\n".join(out) + "\n")

    r = ["# Reading sheet: judge each rewrite by reading\n", "For every broken span, read each original sentence and its rewrite. Verdict: **faithful**, **minor drift**, **meaning changed**, or **garbled**. A span counts as clean only if every rewrite is faithful or minor drift.\n"]
    for s in ok:
        if s["attack"]["status"] != "broken":
            continue
        r.append("## %s: z %.2f -> %.2f\n" % (s["sample_id"], s["attack"]["z0"], s["attack"]["z_final"]))
        for i, rw in sorted(s["attack"]["rewrites"].items(), key=lambda kv: int(kv[0])):
            r += ["**[%s] original:** %s\n" % (i, rw["original"]), "**[%s] rewrite:** %s\n" % (i, rw["rewrite"]), "Verdict: ______\n"]
    (rd / "READING.md").write_text("\n".join(r) + "\n")

    broken_total = sum(1 for s in ok if s["attack"]["status"] == "broken")
    entry = ["", "## Qwen attack run %s (%s preset)" % (rd.name, run["args"]["preset"]),
             "Attacked %d flagged spans, %d broken (z <= 2.0 on the span alone). Per length: %s." % (
                 len(ok), broken_total, "; ".join("%s %d of %d" % (g, sum(1 for s in ok if group(s["sample_id"]) == g and s["attack"]["status"] == "broken"), sum(1 for s in ok if group(s["sample_id"]) == g)) for g in ("200-token spans", "500-token spans"))),
             "Full report: %s/REPORT.md. Quality audit still to be done by reading %s/READING.md." % (rd, rd)]
    (rd / "STATUS_entry.md").write_text("\n".join(entry) + "\n")
    return rd / "REPORT.md"


if __name__ == "__main__":
    print(build(sys.argv[1]))
