"""
Interactive repair of spans that are not clean.

  python research/qwen_attack/repair.py results/qwen_attack/run_...
  python research/qwen_attack/repair.py results/qwen_attack/run_... --spans samples_len200:43 samples_len200:74

Starts from the rewrites that the reading did not mark "meaning changed" (or garbled),
puts the ORIGINAL sentence back where it did, then proposes new rewrites for those
sentences and, if the span is still detected, for other sentences. Each proposal is shown
next to its original and you answer:
    y  accept          n  reject, show the next proposal
    r  new batch       s  skip this sentence          q  stop this span
A span is done when its z <= 2.0, and then every rewrite in it was either kept from the
reading or accepted by you. Every decision is saved at once: stop with Ctrl+C and run the
same command to continue. Needs the GPU (Qwen); run it in the foreground.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

Z_TARGET = 2.0
BAD = ("meaning changed", "garbled")


def flagged_from_reading(text, classify):
    """sample id -> {segment: verdict}, from a filled READING.md."""
    out, sample, idx = {}, None, None
    for line in text.split("\n"):
        m = re.match(r"^## (\S+): z ", line)
        if m:
            sample = m.group(1)
        m = re.match(r"^\*\*\[(\d+)\] rewrite:\*\*", line)
        if m:
            idx = int(m.group(1))
        m = re.match(r"^Verdict:[ \t]*(.*)$", line)
        if m and sample is not None:
            out.setdefault(sample, {})[idx] = classify(m.group(1))
    return out


def new_state(seg, kept, old_bad):
    return {"repl": dict(kept), "provenance": {i: "kept from the reading" for i in kept}, "old_bad": dict(old_bad),
            "rejected": {}, "skipped": [], "decisions": [], "calls": 0, "status": None}


def repair_span(seg, state, paraphraser, gates, oracle, ask, say, save, n_cands=60, max_rounds=3, base_seed=10000):
    """Runs until z <= 2.0, the sentences run out, or the user stops. Returns the updated state."""
    by_index = {g.index: g for g in seg.segments}
    state["old_bad"] = {int(k): v for k, v in state["old_bad"].items()}
    state["repl"] = {int(k): v for k, v in state["repl"].items()}
    state["rejected"] = {int(k): v for k, v in state["rejected"].items()}
    state["provenance"] = {int(k): v for k, v in state["provenance"].items()}
    eligible = sorted((g for g in seg.eligible() if g.index not in state["repl"] and g.index not in state["old_bad"]),
                      key=lambda g: (-len(g.text), g.index))
    todo = sorted(state["old_bad"]) + [g.index for g in eligible]
    z = oracle.query(seg.reassemble(state["repl"]))
    state["z_start_of_repair"] = z
    if z <= Z_TARGET:
        say("z is already %.2f after putting the original sentences back: the span is broken without any new rewrite." % z)
    for idx in todo:
        if z <= Z_TARGET or idx in state["skipped"] or idx in state["repl"]:       # a sentence already approved is not asked again on resume
            continue
        g = by_index[idx]
        pool, rounds, shown = [], 0, 0
        while True:
            if not pool:
                if rounds >= max_rounds:
                    say("no more usable proposals for this sentence; skipping it.")
                    state["skipped"].append(idx)
                    save(state)
                    break
                rounds += 1
                state["calls"] += 1
                cands, _ = paraphraser.rewrite(g.text, n=n_cands, seed=base_seed + 101 * idx + 7 * state["calls"])
                texts, _ = gates.run(g.text, cands)
                banned = set(state["rejected"].get(idx, [])) | {state["old_bad"].get(idx)}
                pool = sorted((oracle.query(seg.reassemble({**state["repl"], idx: t})), t) for t in texts if t not in banned)
                pool = [(zt, t) for zt, t in pool if zt < z]
                if not pool:
                    say("  (batch %d: nothing usable that lowers z)" % rounds)
                    continue
            zt, text = pool.pop(0)
            shown += 1
            sim = gates.similarity(g.text, [text])[0]
            say("\n=== span %s | z now %.2f, target %.2f | sentence %d | proposal %d" % (seg_id(state), z, Z_TARGET, idx, shown))
            if idx in state["old_bad"]:
                say("    (the reading marked the earlier rewrite of this sentence as changing the meaning)")
            say("ORIGINAL: " + g.text)
            say("PROPOSAL: " + text)
            say("meaning similarity %.2f | z would become %.2f (drop %.2f)" % (sim, zt, z - zt))
            while True:
                a = ask("Is this correct?  [y] accept  [n] reject  [r] new batch  [s] skip sentence  [q] stop span > ").strip().lower()
                if a in ("y", "n", "r", "s", "q"):
                    break
                say("please answer y, n, r, s or q")
            state["decisions"].append({"segment": idx, "answer": a, "proposal": text, "z_before": z, "z_after": zt, "similarity": sim})
            if a == "y":
                state["repl"][idx], state["provenance"][idx], z = text, "approved by the user", zt
                save(state)
                break
            if a == "n":
                state["rejected"].setdefault(idx, []).append(text)
            elif a == "r":
                state["rejected"].setdefault(idx, []).append(text)
                pool = []
            elif a == "s":
                state["skipped"].append(idx)
                save(state)
                break
            elif a == "q":
                state["status"], state["z_final"] = "stopped by the user", z
                save(state)
                return state
            save(state)
    state["z_final"] = z
    state["status"] = "broken" if z <= Z_TARGET else "not broken: no accepted proposal got z under 2.0"
    state["final_span"] = seg.reassemble(state["repl"])
    save(state)
    return state


def seg_id(state):
    return state.get("sample_id", "?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rundir")
    ap.add_argument("--spans", nargs="*", default=None)
    ap.add_argument("--preset", default="strict", choices=["strict", "lenient"])
    ap.add_argument("--n-cands", type=int, default=60)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    os.chdir(root)
    sys.path[:0] = [str(Path(__file__).resolve().parent), str(root / "research" / "attacks_core"), str(root / "research" / "blackbox")]
    import attack, gates, paraphraser, samples, sentences, sta_core
    from oracle import DetectorOracle
    import importlib.util
    spec = importlib.util.spec_from_file_location("qwen_run_attack", Path(__file__).resolve().parent / "run_attack.py")    # by file path: attacks_core has another run_attack.py
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    prompt_snippets, write_json = runner.prompt_snippets, runner.write_json
    from tally_reading import classify

    rd = Path(args.rundir).resolve()
    run = json.load(open(rd / "run.json"))
    verdicts = flagged_from_reading((rd / "READING.md").read_text(), classify)
    todo = [s for s, v in verdicts.items() if any(x in BAD for x in v.values())]
    if args.spans:
        todo = [s for s in todo if s in args.spans]
    todo.sort()
    print("%d spans to repair: %s" % (len(todo), ", ".join(todo)))
    files = [f for f in run["args"]["files"] if os.path.exists(f)]
    by_id = {s.sample_id: s for s in samples.load_samples(files)}
    snippets = prompt_snippets(files)
    tok = sta_core.load_tokenizer()
    para = paraphraser.Paraphraser()
    minilm = gates.MiniLM()
    gate = gates.Gates(gates.PRESETS[args.preset], gates.KnownWords(gates.vocabulary_from_c4(), gates.load_wordnet()), minilm.similarity, para.perplexity)
    summary = []
    for sid in todo:
        path = rd / "repair" / (sid.replace(":", "__") + ".json")
        sample = json.load(open(rd / "samples" / (sid.replace(":", "__") + ".json")))
        seg = sentences.split_span(by_id[sid].span)
        rewrites = sample["attack"]["rewrites"]
        bad = {int(i) for i, v in verdicts[sid].items() if v in BAD}
        if path.exists():
            state = json.load(open(path))
            if state.get("status") not in (None, "stopped by the user"):
                print("%s already done: %s" % (sid, state["status"]))
                summary.append(state)
                continue
        else:
            state = new_state(seg, {int(i): r["rewrite"] for i, r in rewrites.items() if int(i) not in bad}, {i: rewrites[str(i)]["rewrite"] for i in bad})
        state["sample_id"], state["z0"] = sid, sample["attack"]["z0"]
        state["status"] = None
        oracle = attack.SpanOracle(DetectorOracle(tok, "score"), snippets.get(sid, ""))
        t0 = time.time()

        save = lambda st, path=path: write_json(path, st)

        state = repair_span(seg, state, para, gate, oracle, input, print, save, n_cands=args.n_cands)
        state["queries"], state["seconds"] = oracle.asked, round(time.time() - t0, 1)
        save(state)
        summary.append(state)
        print("\n%s: %s (z %.2f -> %.2f)" % (sid, state["status"], state["z0"], state["z_final"]))
        if state["status"] == "stopped by the user":
            break
    (rd / "repair").mkdir(parents=True, exist_ok=True)
    lines = ["# Repair summary (%s gates)\n" % args.preset, "| span | z0 | z after putting originals back | z final | result | approved | skipped | rejected |", "|---|---|---|---|---|---|---|---|"]
    for s in summary:
        lines.append("| %s | %.2f | %.2f | %.2f | %s | %d | %d | %d |" % (
            s["sample_id"], s["z0"], s.get("z_start_of_repair", float("nan")), s.get("z_final", float("nan")), s["status"],
            sum(1 for v in s["provenance"].values() if v.startswith("approved")), len(s["skipped"]), sum(len(v) for v in s["rejected"].values())))
    (rd / "repair" / "REPAIR_SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
