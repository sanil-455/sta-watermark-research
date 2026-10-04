"""
Find the loosest semantic threshold that still preserves meaning.

0.88 starved the attack: four of seven sentences produced no
candidates. 0.80 recovered some but still only reached 4.5 from
6.5. The question is how far it can relax before rewrites stop
being faithful.

Runs the attack at several thresholds on one document and prints
every rewrite, so the boundary is judged by reading rather than
by a number.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "research" / "blackbox"))
sys.path.insert(0, str(ROOT / "research" / "attacks_core"))

from sta_core import load_tokenizer
from beam import Semantic
from oracle import DetectorOracle
from span_attack import gen_span_z
from paraphrase import Paraphraser
from para_attack import attack

THRESHOLDS = [0.30,0.45, 0.60]
PROMPT_ID = 1


def main():
    tok = load_tokenizer()
    para = Paraphraser()
    sem = Semantic()

    base = {}
    for f in ["results/raw/baseline_safe.json",
              "results/raw/baseline_extra.json"]:
        for r in json.load(open(f)):
            base[r["prompt_id"]] = r

    r = base[PROMPT_ID]
    pc = len(r["prompt"])
    sz = lambda t: gen_span_z(tok, t, r["prompt_tokens"])
    z0 = sz(r["watermarked_text"])

    print("prompt %d, start span z = %.4f\n" % (PROMPT_ID, z0))

    for th in THRESHOLDS:
        o = DetectorOracle(tok, mode="score")
        a, z1, text = attack(o, para, r["watermarked_text"], pc, sz,
                             semantic=sem, n_cands=60,
                             sem_threshold=th,
                             log=lambda *x: None)

        print("=" * 66)
        print("threshold %.2f -> span z %.4f  (%.1f%% drop)  %s"
              % (th, z1, 100 * (z0 - z1) / z0,
                 "BROKE" if text else ""))
        print("=" * 66)
        for x in a:
            print()
            print("WAS:", x["original"][:150])
            print("NOW:", x["rewritten"][:150])
        print()


if __name__ == "__main__":
    main()
