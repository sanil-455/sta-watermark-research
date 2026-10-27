"""
Automatic audit of the rewrites in a run folder, instead of reading all of
them by eye. It is NOT a human reading: its reliability is measured by
comparing it with a small sample you read yourself.

  python research/qwen_attack/audit.py run    results/qwen_attack/run_...     audit every rewrite of every broken span
  python research/qwen_attack/audit.py sample results/qwen_attack/run_...     write CALIBRATION.md: N rewrites for you to judge (verdicts hidden)
  python research/qwen_attack/audit.py score  results/qwen_attack/run_...     compare your verdicts with the automatic ones

Each rewrite is judged in both directions by a natural-language-inference
model (original entails rewrite, rewrite entails original), plus rules:
numbers, quotations, negation count, capitalised names, unknown words and
glued punctuation. Verdicts: faithful, minor drift, meaning changed,
garbled. A span is clean only if all its rewrites are faithful or minor
drift. The thresholds below are defaults that the calibration should check.
"""

import argparse
import html
import json
import os
import random
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NLI_MODEL = "FacebookAI/roberta-large-mnli"
FAITHFUL, MINOR, CHANGED, GARBLED = "faithful", "minor drift", "meaning changed", "garbled"
ACCEPTABLE = (FAITHFUL, MINOR)


@dataclass
class Thresholds:
    faithful: float = 0.80       # both entailment probabilities at least this: faithful
    minor: float = 0.40          # both at least this: minor drift; below: meaning changed
    contradiction: float = 0.50  # either contradiction probability at least this: meaning changed


def label_indices(id2label):
    idx = {}
    for i, name in id2label.items():
        n = str(name).lower()
        key = "entailment" if "entail" in n else "contradiction" if "contra" in n else "neutral" if "neutral" in n else None
        if key:
            idx[key] = int(i)
    if set(idx) != {"entailment", "neutral", "contradiction"}:
        raise ValueError("cannot map the model's labels: %s" % id2label)
    return idx


class NLI:
    def __init__(self, name=NLI_MODEL, device="cpu"):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch, self.name, self.device = torch, name, device
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).to(device).eval()
        self.idx = label_indices(self.model.config.id2label)
        self.parameters = sum(p.numel() for p in self.model.parameters())

    def probs(self, pairs, batch=16):
        out = []
        for i in range(0, len(pairs), batch):
            chunk = pairs[i:i + batch]
            enc = self.tok([a for a, _ in chunk], [b for _, b in chunk], padding=True, truncation=True,
                           max_length=256, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                p = self.torch.softmax(self.model(**enc).logits, dim=-1).cpu().tolist()
            out += [{k: row[j] for k, j in self.idx.items()} for row in p]
        return out


def rule_issues(original, rewrite, known, gates):
    issues, nr = [], gates.norm(rewrite)
    if gates.numbers(original) != gates.numbers(rewrite):
        issues.append("numbers changed")
    if any(q not in nr for q in gates.quotes(original)):
        issues.append("quotation changed")
    if len(gates.NEGATION.findall(gates.norm(original))) != len(gates.NEGATION.findall(nr)):
        issues.append("negation count changed")
    if not gates.names(original) <= set(gates.NAME.findall(nr)):
        issues.append("name changed")
    seen = {w.lower() for w in gates.WORD.findall(gates.norm(original))}
    if [w for w in gates.WORD.findall(nr) if w.lower() not in seen and w not in known]:
        issues.append("unknown word")
    if Counter(gates.GLUE.findall(rewrite)) - Counter(gates.GLUE.findall(original)):
        issues.append("glued punctuation")
    return issues


def judge(issues, fwd, bwd, th):
    """(verdict, reasons) from the rule issues and the two NLI directions."""
    reasons = list(issues)
    if "unknown word" in issues or "glued punctuation" in issues:
        return GARBLED, reasons
    lowest = min(fwd["entailment"], bwd["entailment"])
    contra = max(fwd["contradiction"], bwd["contradiction"])
    if contra >= th.contradiction:
        reasons.append("contradiction %.2f" % contra)
    if lowest < th.minor:
        reasons.append("entailment %.2f" % lowest)
    if any(i in issues for i in ("numbers changed", "quotation changed", "negation count changed", "name changed")) \
            or contra >= th.contradiction or lowest < th.minor:
        return CHANGED, reasons
    return (FAITHFUL if lowest >= th.faithful else MINOR), reasons


def audit_run(rundir, probs, known, gates, th):
    spans = []
    for path in sorted((Path(rundir) / "samples").glob("*.json")):
        s = json.load(open(path))
        a = s.get("attack")
        if not a or a["status"] != "broken":
            continue
        items = sorted(a["rewrites"].items(), key=lambda kv: int(kv[0]))
        pairs = [(html.unescape(v["original"]), html.unescape(v["rewrite"])) for _, v in items]
        fwd, bwd = probs(pairs), probs([(b, a_) for a_, b in pairs])
        rewrites = []
        for (idx, v), f, b in zip(items, fwd, bwd):
            verdict, reasons = judge(rule_issues(v["original"], v["rewrite"], known, gates), f, b, th)
            rewrites.append({"segment": int(idx), "original": v["original"], "rewrite": v["rewrite"], "verdict": verdict,
                             "reasons": reasons, "original_entails_rewrite": f["entailment"], "rewrite_entails_original": b["entailment"],
                             "contradiction": max(f["contradiction"], b["contradiction"])})
        spans.append({"sample_id": s["sample_id"], "z0": a["z0"], "z_final": a["z_final"], "rewrites": rewrites,
                      "status": "clean" if all(r["verdict"] in ACCEPTABLE for r in rewrites) else "not clean"})
    return spans


def summary_md(spans, th, nli_name, nli_params, agreement=None):
    allr = [r for s in spans for r in s["rewrites"]]
    n_clean = sum(1 for s in spans if s["status"] == "clean")
    out = ["# Automatic audit of the rewrites\n",
           "**This is an automatic judgement, not a human reading.** Each rewrite was judged in both directions by `%s` (%s parameters) and by rule checks. "
           "Thresholds: faithful if both entailment probabilities >= %.2f; minor drift if both >= %.2f; meaning changed below that, "
           "if either contradiction probability >= %.2f, or if a number, quotation, negation count or capitalised name changed; garbled if a new unknown word or glued punctuation appears.\n"
           % (nli_name, "{:,}".format(nli_params) if nli_params else "n/a", th.faithful, th.minor, th.contradiction)]
    if agreement:
        out.append("**Measured against your own verdicts on %d rewrites:** exact agreement %.0f%%; agreement on acceptable (faithful or minor drift) versus not %.0f%%; Cohen's kappa on that binary split %.2f.\n"
                   % (agreement["n"], 100 * agreement["exact"], 100 * agreement["binary"], agreement["kappa"]))
    else:
        out.append("**Not yet calibrated against human verdicts** (run `audit.py sample` and `audit.py score`).\n")
    out.append("broken spans audited: %d | clean: %d | not clean: %d | rewrites: %d (%s)\n" % (
        len(spans), n_clean, len(spans) - n_clean, len(allr),
        ", ".join("%s %d" % (v, sum(1 for r in allr if r["verdict"] == v)) for v in (FAITHFUL, MINOR, CHANGED, GARBLED))))
    reasons = Counter(x.split(" ")[0] + (" " + x.split(" ")[1] if x.split(" ")[0] in ("numbers", "quotation", "negation", "unknown", "glued", "name") and len(x.split(" ")) > 1 else "")
                      for r in allr if r["verdict"] in (CHANGED, GARBLED) for x in r["reasons"])
    out.append("reasons behind meaning changed / garbled: %s\n" % dict(reasons.most_common()))
    out.append("| span | z0 -> z | rewrites | verdicts | result |\n|---|---|---|---|---|")
    for s in spans:
        out.append("| %s | %.2f -> %.2f | %d | %s | **%s** |" % (s["sample_id"], s["z0"], s["z_final"], len(s["rewrites"]), ", ".join(r["verdict"] for r in s["rewrites"]), s["status"]))
    out.append("\n## Every rewrite that was not judged faithful\n")
    for s in spans:
        for r in s["rewrites"]:
            if r["verdict"] != FAITHFUL:
                out += ["**%s [%d] %s** (%s; entailment %.2f / %.2f)\n" % (s["sample_id"], r["segment"], r["verdict"].upper(), "; ".join(r["reasons"]) or "no rule fired",
                                                                         r["original_entails_rewrite"], r["rewrite_entails_original"]),
                        "- original: " + r["original"], "- rewrite: " + r["rewrite"] + "\n"]
    return "\n".join(out) + "\n"


def sample_sheet(spans, n, seed):
    flat = [(s["sample_id"], r) for s in spans for r in s["rewrites"]]
    random.Random(seed).shuffle(flat)
    chosen = flat[:n]
    lines = ["# Calibration sheet: judge these %d rewrites yourself\n" % len(chosen),
             "Write faithful, minor drift, meaning changed or garbled after each Verdict (spoken-English bar: only unreadable text is garbled). The automatic verdicts are hidden.\n"]
    for k, (sid, r) in enumerate(chosen, 1):
        lines += ["### %d | %s | segment %d\n" % (k, sid, r["segment"]), "**original:** " + r["original"] + "\n", "**rewrite:** " + r["rewrite"] + "\n", "Verdict: ______\n"]
    return "\n".join(lines) + "\n"


def score_sheet(text, spans):
    from tally_reading import classify
    auto = {(s["sample_id"], r["segment"]): r["verdict"] for s in spans for r in s["rewrites"]}
    rows = []
    for block in re.split(r"\n### ", "\n" + text)[1:]:
        m = re.match(r"\d+ \| (\S+) \| segment (\d+)", block)
        v = re.search(r"^Verdict:[ \t]*(.*)$", block, re.M)
        if not m or not v:
            continue
        human = classify(v.group(1))
        if human in ("unread", "unrecognised"):
            continue
        key = (m.group(1), int(m.group(2)))
        rows.append((key, human, auto[key]))
    n = len(rows)
    if n == 0:
        raise ValueError("no judged rewrites found in the sheet")
    exact = sum(h == a for _, h, a in rows) / n
    hb = [h in ACCEPTABLE for _, h, _ in rows]
    ab = [a in ACCEPTABLE for _, _, a in rows]
    binary = sum(x == y for x, y in zip(hb, ab)) / n
    pe = sum(((sum(hb) / n) if c else 1 - sum(hb) / n) * ((sum(ab) / n) if c else 1 - sum(ab) / n) for c in (True, False))
    kappa = (binary - pe) / (1 - pe) if pe < 1 else 1.0
    confusion = Counter((h, a) for _, h, a in rows)
    return {"n": n, "exact": exact, "binary": binary, "kappa": kappa, "confusion": dict(confusion),
            "disagreements": [(k, h, a) for k, h, a in rows if h != a]}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("rundir")
    r.add_argument("--device", default="cpu")
    r.add_argument("--faithful", type=float, default=0.80)
    r.add_argument("--minor", type=float, default=0.40)
    r.add_argument("--contradiction", type=float, default=0.50)
    for name in ("sample", "score"):
        p = sub.add_parser(name)
        p.add_argument("rundir")
        if name == "sample":
            p.add_argument("--n", type=int, default=20)
            p.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    rd = Path(args.rundir).resolve()
    os.chdir(ROOT)
    sys.path[:0] = [str(HERE)]
    if args.cmd == "run":
        import gates
        th = Thresholds(args.faithful, args.minor, args.contradiction)
        nli = NLI(device=args.device)
        known = gates.KnownWords(gates.vocabulary_from_c4(), gates.load_wordnet())
        spans = audit_run(rd, nli.probs, known, gates, th)
        data = {"model": nli.name, "parameters": nli.parameters, "thresholds": asdict(th), "spans": spans}
        (rd / "audit.json").write_text(json.dumps(data, indent=1))
        (rd / "AUDIT.md").write_text(summary_md(spans, th, nli.name, nli.parameters))
        print((rd / "AUDIT.md").read_text()[:2500])
        print("\nwritten: %s and %s" % (rd / "AUDIT.md", rd / "audit.json"))
        return
    data = json.loads((rd / "audit.json").read_text())
    if args.cmd == "sample":
        (rd / "CALIBRATION.md").write_text(sample_sheet(data["spans"], args.n, args.seed))
        print("written %s: judge those rewrites, save, then run:  python research/qwen_attack/audit.py score %s" % (rd / "CALIBRATION.md", rd))
        return
    res = score_sheet((rd / "CALIBRATION.md").read_text(), data["spans"])
    th = Thresholds(**data["thresholds"])
    (rd / "AUDIT.md").write_text(summary_md(data["spans"], th, data["model"], data["parameters"], res))
    print("judged by you: %d | exact agreement %.0f%% | acceptable-vs-not agreement %.0f%% | kappa %.2f" % (res["n"], 100 * res["exact"], 100 * res["binary"], res["kappa"]))
    print("confusion (you, automatic): %s" % res["confusion"])
    for key, h, a in res["disagreements"]:
        print("  disagree on %s segment %d: you said %s, automatic said %s" % (key[0], key[1], h, a))
    print("AUDIT.md updated with the measured agreement")


if __name__ == "__main__":
    main()
