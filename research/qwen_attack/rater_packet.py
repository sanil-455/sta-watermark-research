"""
A blind rating packet for the final edits (item 4: independent human evaluation).

  python research/qwen_attack/rater_packet.py [--n200 60] [--n500 40] [--controls 5] [--seed 0]     (CPU only; run final_all.py first)

Writes results/rating/:
  rater_sheet.csv         pair_id, original, rewrite, rating, note   (give each rater their own copy, saved as rating_<name>.csv)
  RATING_INSTRUCTIONS.md  the rubric, with invented examples
  answer_key.csv          PRIVATE: what each pair is (final edit of which span, or a control), and the reading's verdict
The sheet mixes a random sample of the final edits of each length with hidden controls (known meaning changes from READING.md, and identical pairs).
"""

import argparse
import csv
import html
import json
import os
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BAD = ("meaning changed", "garbled")

INSTRUCTIONS = """# Rating instructions

You will see pairs of sentences: an ORIGINAL sentence and a REWRITE of it. Judge only whether the rewrite says the same thing as the original. Do not judge style or quality of writing.
Put ONE letter in the `rating` column for every pair (optionally a short remark in `note`):

| letter | meaning | use it when |
|---|---|---|
| F | faithful | the same facts, names, numbers, quotes, negations, and who-did-what-to-whom; only the wording differs |
| M | minor drift | a small shift in tone, emphasis or word choice, or a tiny detail added or dropped, but no fact, name, number, quote, negation or relation is changed |
| C | meaning changed | a fact, name, number, quote, negation, relation or attribution is changed, added or dropped, or the sentence now says something different |
| U | unreadable | the rewrite is so garbled or ungrammatical that you cannot tell what it says |

Rules:
- Some originals are themselves odd, typo-ridden or truncated. Judge the rewrite against the original AS WRITTEN: if the rewrite invents a claim that is not in the original, it is C.
- Decide each pair on its own. Do not try to guess which pairs came from a system or how the others rated them. Please do not discuss your ratings with other raters until all sheets are returned.
- If you are unsure between F and M, choose M. If you are unsure between M and C, ask yourself: would a reader of the rewrite believe a different fact than a reader of the original? If yes, C.
- Do not look anything up and do not use a translation or AI tool.

Examples (invented, not in the sheet):

| original | rewrite | rating | why |
|---|---|---|---|
| The council approved the budget on Monday after a long debate. | After a lengthy debate, the council passed the budget on Monday. | F | same facts |
| She said the clinic would open in March. | She mentioned that the clinic should open around March. | M | softer certainty, same claim |
| He sold 40 cars in 2019. | He sold 14 cars in 2019. | C | the number changed |
| The mayor did not attend the meeting. | The mayor attended the meeting. | C | the negation was lost |
| The mayor did not attend the meeting. | Meeting attended mayor the of not did. | U | cannot be read |

Return your sheet saved as `rating_<your initials>.csv`.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n200", type=int, default=60)
    ap.add_argument("--n500", type=int, default=40)
    ap.add_argument("--controls", type=int, default=5, help="known-bad controls and identical controls, each")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.chdir(ROOT)
    sys.path[:0] = [str(HERE), str(ROOT / "research" / "attacks_core"), str(ROOT / "research" / "blackbox")]
    import samples
    import sentences
    from repair import flagged_from_reading
    from tally_reading import classify
    rng = random.Random(args.seed)
    final, bad_pool = {"200": [], "500": []}, []
    for d in sorted(Path("results/qwen_attack").glob("run_*")):
        run = json.load(open(d / "run.json"))
        if run["args"].get("only"):
            continue
        key = "500" if any("len500" in f for f in run["args"].get("files", [])) else "200"
        final[key] = []
        files = [f for f in run["args"]["files"] if os.path.exists(f)]
        spans = {s.sample_id: s.span for s in samples.load_samples(files)}
        verdicts = flagged_from_reading((d / "READING.md").read_text(), classify) if (d / "READING.md").exists() else {}
        samp = {}
        for p in (d / "samples").glob("*.json"):
            s = json.load(open(p))
            if "attack" in s:
                samp[s["sample_id"]] = s
        repairs = {}
        for p in (d / "repair").glob("*.json") if (d / "repair").exists() else []:
            st = json.load(open(p))
            repairs[st["sample_id"]] = st
        for row in csv.DictReader(open(d / "PER_EDIT_FINAL.csv")):
            sid, idx, origin = row["sample"], int(row["segment"]), row["origin_of_rewrite"]
            rec = samp[sid]["attack"]["rewrites"].get(str(idx))
            if origin.startswith("attack"):
                orig, new = rec["original"], rec["rewrite"]
            else:
                orig = {g.index: g.text for g in sentences.split_span(spans[sid]).segments}[idx]
                new = {int(k): v for k, v in repairs[sid]["repl"].items()}[idx]
            mine = verdicts.get(sid, {}).get(idx) if origin != "approved by the user" else "approved by the user"
            final[key].append({"kind": "final", "group": key, "sample": sid, "segment": idx, "origin": origin, "original": orig, "rewrite": new, "mine": mine or ""})
        for sid, v in verdicts.items():
            for idx, verdict in v.items():
                rec = samp.get(sid, {}).get("attack", {}).get("rewrites", {}).get(str(idx))
                if verdict in BAD and rec:
                    bad_pool.append({"kind": "control_known_meaning_change", "group": key, "sample": sid, "segment": idx, "origin": "reverted in the repair", "original": rec["original"], "rewrite": rec["rewrite"], "mine": verdict})
    items = rng.sample(final["200"], min(args.n200, len(final["200"]))) + rng.sample(final["500"], min(args.n500, len(final["500"])))
    items += rng.sample(bad_pool, min(args.controls, len(bad_pool)))
    for e in rng.sample(final["200"] + final["500"], min(args.controls, len(final["200"]) + len(final["500"]))):
        items.append({"kind": "control_identical", "group": e["group"], "sample": e["sample"], "segment": e["segment"], "origin": "identical pair", "original": e["original"], "rewrite": e["original"], "mine": "faithful (identical)"})
    rng.shuffle(items)
    out = ROOT / "results" / "rating"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "rater_sheet.csv", "w", newline="") as f, open(out / "answer_key.csv", "w", newline="") as g:
        w, k = csv.writer(f), csv.writer(g)
        w.writerow(["pair_id", "original", "rewrite", "rating", "note"])
        k.writerow(["pair_id", "kind", "group", "sample", "segment", "origin", "reading_verdict"])
        for i, it in enumerate(items, 1):
            pid = "P%03d" % i
            w.writerow([pid, html.unescape(it["original"]).strip(), html.unescape(it["rewrite"]).strip(), "", ""])
            k.writerow([pid, it["kind"], it["group"], it["sample"], it["segment"], it["origin"], it["mine"]])
    (out / "RATING_INSTRUCTIONS.md").write_text(INSTRUCTIONS)
    kinds = {}
    for it in items:
        kinds[it["kind"], it["group"]] = kinds.get((it["kind"], it["group"]), 0) + 1
    print("%d pairs written to %s: %s" % (len(items), out / "rater_sheet.csv", dict(sorted(kinds.items()))))
    print("final edits available: 200-token %d, 500-token %d; known-bad controls available: %d" % (len(final["200"]), len(final["500"]), len(bad_pool)))


if __name__ == "__main__":
    main()
