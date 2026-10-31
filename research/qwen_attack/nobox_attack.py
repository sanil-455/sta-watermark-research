"""
No-box baseline: paraphrase-and-check-ONCE, with zero detector feedback during rewriting.

This is the head-to-head baseline for the central claim ("detector guidance helps a lot on STA,
unlike the recent finding that it barely helps on green-list/Gumbel schemes"). It rewrites every
eligible sentence of a span ONCE using the same paraphraser and the same quality gates as the
adaptive attack, picks a gate-passing candidate per sentence WITHOUT ever querying the detector to
choose, reassembles, and scores the final text exactly ONCE. So: same quality bar, same spans,
same paraphraser -- the ONLY difference from the adaptive attack is that the detector gives no
feedback. The one final score is just to measure success, not to steer (query budget = 1).

Candidate choice without the detector: among gate survivors, pick the one the gates already rank
best by semantic similarity to the original (most faithful), breaking ties by lower perplexity.
This is a deliberately attacker-reasonable no-box rule: it maximises faithfulness, not evasion.

  python research/qwen_attack/nobox_attack.py --files ... --only ... --out results/qwen_attack_nobox/run

Writes one JSON per span (same schema spirit as run_attack), plus NOBOX.md / NOBOX.csv.
CPU for the logic, GPU for the paraphraser (same as run_attack). Reads screening.json's z0 per span.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
Z_THRESHOLD = 2.0


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * p, 100 * (c - h), 100 * (c + h)


def pick_candidate(original, survivors, similarity, perplexity):
    """No detector involved: most faithful survivor (highest similarity), tie-break lowest perplexity."""
    if not survivors:
        return None
    sims = similarity(original, survivors)
    best = max(range(len(survivors)), key=lambda i: (round(sims[i], 6), -perplexity(survivors[i])))
    return survivors[best]


def attack_span_nobox(seg, paraphraser, gates, score_text, n_cands=60, seed=0):
    """One rewrite pass, no detector feedback. Returns a result dict."""
    repl, per_sentence = {}, []
    for g in sorted(seg.eligible(), key=lambda g: (-len(g.text), g.index)):
        cands, _ = paraphraser.rewrite(g.text, n=n_cands, seed=seed + g.index)
        survivors, _ = gates.run(g.text, cands)   # gates.run reads c.text/c.hit_limit itself; pass the Candidate objects, not strings
        choice = pick_candidate(g.text, survivors, gates.similarity, paraphraser.perplexity)
        if choice is not None:
            repl[g.index] = choice
        per_sentence.append({"index": g.index, "survivors": len(survivors), "rewritten": choice is not None})
    final = seg.reassemble(repl)
    z_final = score_text(final)                       # the ONLY detector query: to measure, not to steer
    return {"z_final": z_final, "status": "broken" if z_final <= Z_THRESHOLD else "not broken",
            "rewrites": {str(i): {"original": {g.index: g.text for g in seg.segments}[i], "rewrite": t} for i, t in repl.items()},
            "sentences_rewritten": len(repl), "eligible": len(seg.eligible()), "per_sentence": per_sentence,
            "queries": 1, "final_span": final}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*", required=True)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--screening", required=True, help="screening.json of the matching adaptive run, for z0 and the flagged set")
    ap.add_argument("--n-cands", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preset", default="lenient", choices=["lenient", "strict"])
    args = ap.parse_args()
    os.chdir(ROOT)
    sys.path[:0] = [str(HERE), str(ROOT / "research" / "attacks_core"), str(ROOT / "research" / "blackbox")]
    import gates as gates_mod
    import paraphraser as para_mod
    import samples as samples_mod
    import sentences as sentences_mod
    import sta_core

    out = Path(args.out)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    flagged = {x["sample_id"]: x["z0"] for x in json.load(open(args.screening)) if x["flagged"]}
    by_id = {s.sample_id: s for s in samples_mod.load_samples(args.files)}
    todo = [sid for sid in flagged if (args.only is None or sid in args.only) and sid in by_id]
    tok = sta_core.load_tokenizer()
    score_text = lambda t: sta_core.score_text(tok, t)["z"]
    para = para_mod.Paraphraser()
    minilm = gates_mod.MiniLM()
    gate = gates_mod.Gates(gates_mod.PRESETS[args.preset], gates_mod.KnownWords(gates_mod.vocabulary_from_c4(), gates_mod.load_wordnet()), minilm.similarity, para.perplexity)

    rows = []
    for n, sid in enumerate(todo, 1):
        path = out / "samples" / (sid.replace(":", "__") + ".json")
        if path.exists():
            res = json.load(open(path))["nobox"]
        else:
            seg = sentences_mod.split_span(by_id[sid].span)
            t0 = time.time()
            res = attack_span_nobox(seg, para, gate, score_text, n_cands=args.n_cands, seed=args.seed)
            res["z0"], res["seconds"] = flagged[sid], round(time.time() - t0, 1)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"sample_id": sid, "nobox": res}, indent=1, default=str))
            os.replace(tmp, path)
        rows.append((sid, res))
        print("[%d/%d] %s z0 %.2f -> %.2f | %s | %d of %d sentences rewritten"
              % (n, len(todo), sid, res["z0"], res["z_final"], res["status"], res["sentences_rewritten"], res["eligible"]))

    broken = sum(1 for _, r in rows if r["status"] == "broken")
    p, lo, hi = wilson(broken, len(rows))
    doc = ["# No-box baseline (paraphrase-and-check-once, zero detector feedback)\n",
           "Same spans, paraphraser and quality gates as the adaptive attack; the only difference is the detector gives no feedback during "
           "rewriting (query budget = 1, used only to measure the final z). Compare the broken rate here against the adaptive attack's on the same spans: "
           "a large gap is evidence that detector guidance specifically helps on this watermark.\n",
           "- spans attacked: %d | broken (z <= 2.0): %d (%.0f%% [%.0f%%, %.0f%%])" % (len(rows), broken, p, lo, hi), ""]
    doc += ["| span | z0 | z_final | status | sentences rewritten |", "|---|---|---|---|---|"]
    for sid, r in rows:
        doc.append("| %s | %.2f | %.2f | %s | %d of %d |" % (sid, r["z0"], r["z_final"], r["status"], r["sentences_rewritten"], r["eligible"]))
    (out / "NOBOX.md").write_text("\n".join(doc) + "\n")
    import csv
    with open(out / "NOBOX.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample", "z0", "z_final", "status", "sentences_rewritten", "eligible"])
        for sid, r in rows:
            w.writerow([sid, "%.4f" % r["z0"], "%.4f" % r["z_final"], r["status"], r["sentences_rewritten"], r["eligible"]])
    print("\n".join(doc[2:4]))
    print("written:", out / "NOBOX.md")


if __name__ == "__main__":
    main()
