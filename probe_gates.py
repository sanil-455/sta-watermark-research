"""
Find which gate blocks the most sentences.

The sweep shows documents that run 4-5 sentences drop z by
40-64%, while those running 0-1 barely move. Prompt 20 ran zero
sentences, prompt 2 and 25 ran one each.

So the limit is sentences being skipped, not candidate quality.
This prints, for every sentence of a document, how many
candidates survive each gate in turn, so the blocking one is
visible rather than guessed.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "research" / "blackbox"))
sys.path.insert(0, str(ROOT / "research" / "attacks_core"))

from sta_core import load_tokenizer
from beam import Semantic
from paraphrase import Paraphraser
from para_attack import (split_sentences, clean, semantic_filter,
                         fluency_filter)
from content_checks import passes

# the three that ran fewest sentences
PROMPTS = [20, 2, 25]


def main():
    tok = load_tokenizer()
    para = Paraphraser()
    sem = Semantic()

    base = {}
    for f in ["results/raw/baseline_safe.json",
              "results/raw/baseline_extra.json"]:
        for r in json.load(open(f)):
            base[r["prompt_id"]] = r

    for pid in PROMPTS:
        r = base[pid]
        pc = len(r["prompt"])
        sents = [s for s in split_sentences(r["watermarked_text"])
                 if s[0] >= pc]

        print("=" * 70)
        print("prompt %d: %d sentences in the generated span"
              % (pid, len(sents)))
        print("=" * 70)

        for i, (a, b) in enumerate(sents):
            orig = r["watermarked_text"][a:b]

            if '"' in orig or "\u201c" in orig:
                print("  %d  QUOTATION, skipped: %s" % (i, orig[:70]))
                continue

            raw = para.rewrite(orig, n=40)
            c1 = clean(raw, orig)
            c2 = semantic_filter(sem, orig, c1, 0.60)
            c3 = [c for c in c2 if passes(orig, c)]
            c4 = fluency_filter(para, c3)

            print("  %d  raw %2d -> clean %2d -> sem %2d -> "
                  "content %2d -> fluency %2d   %s"
                  % (i, len(raw), len(c1), len(c2), len(c3),
                     len(c4), orig[:55]))

            # show what the blocking gate rejected
            if c1 and not c2:
                print("        semantic killed: %s" % c1[0][:85])
            elif c2 and not c3:
                print("        content killed: %s" % c2[0][:85])
        print()


if __name__ == "__main__":
    main()
