"""
Tally the verdicts you filled in READING.md.

  python research/qwen_attack/tally_reading.py results/qwen_attack/run_.../READING.md

Write one of these after each "Verdict:": faithful, minor drift, meaning
changed, garbled. A span is CLEAN only if every one of its rewrites is
faithful or minor drift. Writes TALLY.md next to the sheet.
"""

import re
import sys
from pathlib import Path

GOOD = {"faithful": "faithful", "minor": "minor drift"}
BAD = {"changed": "meaning changed", "garbled": "garbled"}


def classify(text):
    t = text.strip().lower()
    if not t or set(t) <= {"_"}:
        return "unread"
    for key, name in list(BAD.items()) + list(GOOD.items()):          # a bad word wins over a good one
        if key in t:
            return name
    return "unrecognised"


def tally(text):
    spans = []
    for block in re.split(r"\n## ", "\n" + text)[1:]:
        m = re.match(r"(\S+): z ([\d.-]+) -> ([\d.-]+)", block)
        if not m:
            continue
        verdicts = [classify(v) for v in re.findall(r"^Verdict:[ \t]*(.*)$", block, re.M)]
        if any(v in ("unread", "unrecognised") for v in verdicts):
            status = "unread" if any(v == "unread" for v in verdicts) else "unrecognised"
        elif all(v in GOOD.values() for v in verdicts):
            status = "clean"
        else:
            status = "not clean"
        spans.append({"id": m.group(1), "z0": float(m.group(2)), "z": float(m.group(3)), "verdicts": verdicts, "status": status})
    return spans


def summary(spans):
    counts = {}
    for s in spans:
        counts[s["status"]] = counts.get(s["status"], 0) + 1
    all_v = [v for s in spans for v in s["verdicts"]]
    lines = ["# Reading tally\n", "broken spans on the sheet: %d | clean: %d | not clean: %d | still unread: %d | unrecognised verdict: %d\n" % (
        len(spans), counts.get("clean", 0), counts.get("not clean", 0), counts.get("unread", 0), counts.get("unrecognised", 0))]
    lines.append("rewrites by verdict: " + ", ".join("%s %d" % (k, all_v.count(k)) for k in ("faithful", "minor drift", "meaning changed", "garbled", "unread", "unrecognised")) + "\n")
    lines.append("| span | z0 -> z | rewrites | verdicts | result |\n|---|---|---|---|---|")
    for s in spans:
        lines.append("| %s | %.2f -> %.2f | %d | %s | **%s** |" % (s["id"], s["z0"], s["z"], len(s["verdicts"]), ", ".join(s["verdicts"]), s["status"]))
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    path = Path(sys.argv[1])
    out = summary(tally(path.read_text()))
    (path.parent / "TALLY.md").write_text(out)
    print(out)
