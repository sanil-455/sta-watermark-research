"""
Why does the fresh z0 sometimes differ from the saved z?

Run:  python research/qwen_attack/check_tokens.py

For every sample made by the generation script, three scores are compared:
  saved   z over the generated token ids, counting the pair that straddles
          the prompt boundary
  ideal   z over [start-of-text] + the generated ids: what a detector would
          score if the decoded text turned back into exactly the same tokens
  z0      z of the span text re-tokenised: what the detector really scores

  saved -> ideal  is the first-pair effect (at most one pair)
  ideal -> z0     is the re-tokenisation effect: the text became different tokens

Every difference between the generated tokens and the re-tokenised ones is
labelled with one of these causes, and anything else is reported as
UNEXPECTED so it cannot hide:
  start-of-text space prefix   the span begins with a token written without a
                               leading space (for example "." attached to the
                               prompt's last word); tokenised alone, the
                               tokeniser adds a space in front of it
  trailing whitespace dropped  the span ends with a newline or space token
  same text, different token split   the model sampled a split of a word that
                               the tokeniser does not choose by itself

The paper's own detector also scores text, so z0 is the right baseline; this
only explains where the difference to the saved number comes from.
Loads no language model: only the Llama tokenizer and the detector hash.
"""

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FILES = ["results/raw/samples_len200.json", "results/raw/samples_len500.json"]

WHITESPACE = {"▁", "<0x0A>", "<0x0D>", "<0x09>", "<0x20>"}
BYTE_TEXT = {"<0x0A>": "\n", "<0x0D>": "\r", "<0x09>": "\t", "<0x20>": " "}
START_PREFIX = "start-of-text space prefix"
TRAILING = "trailing whitespace dropped"
SPLIT = "same text, different token split"
UNEXPECTED = "UNEXPECTED: the text differs"


def text_of(pieces):
    """The string a list of token pieces spells."""
    return "".join(BYTE_TEXT.get(p, p.replace("▁", " ")) for p in pieces)


def trim(pieces):
    pieces = list(pieces)
    while pieces and pieces[-1] in WHITESPACE:
        pieces.pop()
    return pieces


def classify(ideal_p, retok_p):
    """Causes of the difference between two piece lists that both start with <s>."""
    causes, a, b = [], ideal_p[1:], retok_p[1:]
    if a and b and not a[0].startswith("▁") and b[0] == "▁" + a[0]:
        causes.append(START_PREFIX)
        a, b = a[1:], b[1:]
    if a == b:
        return causes
    ta, tb = text_of(a), text_of(b)
    if ta.rstrip() != tb.rstrip():
        return causes + [UNEXPECTED]
    if ta != tb:
        causes.append(TRAILING)
    if trim(a) != trim(b):
        causes.append(SPLIT)
    return causes


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b)) if len(a) != len(b) else None


def diagnose(rows_by_file, master, encode, z_of, bos, pieces):
    """Returns (summary lines, detail lines, list of changed samples)."""
    import samples
    done = {s["prompt_id"]: s for s in master["done"]}
    n, changed = 0, []
    for tag, rows in rows_by_file.items():
        for row in rows:
            s = done[row["prompt_id"]]
            p, k = s["prompt_tokens"], row["n_generated"]
            ideal = [bos] + s["full_ids"][p:p + k]
            retok = encode(samples.span_of(row))
            n += 1
            if retok == ideal:
                continue
            ip, rp = pieces(ideal), pieces(retok)
            causes = classify(ip, rp)
            z_ideal, z0 = z_of(ideal), z_of(retok)
            shown = []
            for lo_a, lo_b in ((0, 0), (2, 2)) if START_PREFIX in causes else ((0, 0),):
                i = first_diff(ideal[lo_a:], retok[lo_b:])
                if i is not None:
                    at = lo_a + i
                    shown.append((at, ip[max(0, at - 2):at + 3], rp[max(0, at - 2):at + 3]))
            changed.append({
                "id": "%s:%s" % (tag, row["prompt_id"]), "t_ideal": len(ideal) - 1,
                "t_retok": len(retok) - 1, "causes": causes, "shown": shown,
                "first_pair": z_ideal - row["watermarked_z"], "retok": z0 - z_ideal,
                "total": z0 - row["watermarked_z"]})
    counts = {}
    for c in changed:
        for cause in c["causes"]:
            counts[cause] = counts.get(cause, 0) + 1
    summary = ["samples checked: %d | re-tokenise to exactly the generated ids: %d | differ: %d"
               % (n, n - len(changed), len(changed))]
    for cause, num in sorted(counts.items(), key=lambda kv: -kv[1]):
        summary.append("  %-36s %d samples" % (cause, num))
    unexplained = sum(1 for c in changed if not c["causes"] or UNEXPECTED in c["causes"])
    summary.append("unexplained differences: %d" % unexplained)
    if changed:
        worst = max(changed, key=lambda c: abs(c["retok"]))
        summary.append("largest re-tokenisation effect on z: %+.3f (%s)" % (worst["retok"], worst["id"]))
    detail = []
    for c in changed:
        lines = ["%-20s tokens %3d -> %3d | %s | saved->ideal %+.3f | ideal->z0 %+.3f | total %+.3f"
                 % (c["id"], c["t_ideal"], c["t_retok"], "; ".join(c["causes"]) or "NO CAUSE FOUND",
                    c["first_pair"], c["retok"], c["total"])]
        for at, g, r in c["shown"]:
            lines.append("      at %3d  generated: %s\n              re-tokenised: %s" % (at, g, r))
        detail.append("\n".join(lines))
    return summary, detail, changed


def main():
    os.chdir(ROOT)
    sys.path[:0] = [str(HERE), str(ROOT / "research" / "attacks_core")]
    import sta_core
    tok = sta_core.load_tokenizer()
    master = json.load(open("results/raw/gen_master.json"))
    rows_by_file = {Path(p).stem: json.load(open(p)) for p in FILES if os.path.exists(p)}
    summary, detail, _ = diagnose(
        rows_by_file, master,
        encode=lambda text: sta_core.text_to_ids(tok, text),
        z_of=lambda ids: sta_core.sta_stats(ids)["z"],
        bos=tok.bos_token_id, pieces=lambda ids: tok.convert_ids_to_tokens(ids))
    print("\n".join(summary))
    print("\n".join(detail))


if __name__ == "__main__":
    main()
