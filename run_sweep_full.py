"""
Paraphrase attack sweep, unattended.

Settings are fixed rather than tuned per document: 0.92 starved
the candidate pool, 0.85 let meaning drift, 0.88 was best.

Results are written after every document, so an interruption
loses nothing already completed.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "research" / "blackbox"))
sys.path.insert(0, str(ROOT / "research" / "attacks_core"))

from sta_core import load_tokenizer
from fluency import Fluency
from beam import Semantic
from oracle import DetectorOracle
from span_attack import gen_span_z
from para_attack import attack

BASELINES = [
    "results/raw/baseline_safe.json",
    "results/raw/baseline_extra.json",
    "results/raw/baseline_more.json",
]
OUT = "results/raw/para_sweep_full.json"

SEM_THRESHOLD = 0.88
N_CANDS = 60
MAX_SENTS = 14


def main():
    log = lambda m: print(m, flush=True)

    tok = load_tokenizer()
    flu = Fluency(tok)
    sem = Semantic()

    rows = []
    for f in BASELINES:
        if Path(f).exists():
            rows.extend(json.load(open(f)))

    # only documents the detector flags; an undetected document
    # has nothing to remove
    rows = [r for r in rows if r["watermarked_z"] > 2.0]
    log("attacking %d detected documents" % len(rows))
    log("%4s %8s %8s %6s %8s %7s"
        % ("id", "z0", "z1", "sents", "queries", "evaded"))

    results = []
    for row in rows:
        oracle = DetectorOracle(tok, mode="score")
        sz = lambda t, n=row["prompt_tokens"]: gen_span_z(tok, t, n)
        z0 = sz(row["watermarked_text"])

        try:
            applied, z1, text = attack(
                oracle, tok, flu.model, row["watermarked_text"],
                len(row["prompt"]), n_cands=N_CANDS,
                max_sents=MAX_SENTS, span_z=sz, semantic=sem,
                sem_threshold=SEM_THRESHOLD, fluency=flu,
                log=lambda *a: None,
            )
        except Exception as e:
            log("%4d FAILED %s" % (row["prompt_id"], e))
            continue

        results.append({
            "prompt_id": row["prompt_id"],
            "z0": z0, "z1": z1,
            "sentences": len(applied),
            "queries": oracle.queries,
            "broke": text is not None,
            "rewrites": applied,
            "text": text,
        })

        log("%4d %8.3f %8.3f %6d %8d %7s"
            % (row["prompt_id"], z0, z1, len(applied),
               oracle.queries, text is not None))

        json.dump(results, open(OUT, "w"), indent=2)

    broke = [r for r in results if r["broke"]]
    log("")
    log("evaded: %d of %d" % (len(broke), len(results)))
    if broke:
        log("mean sentences: %.1f | mean queries: %.0f"
            % (sum(r["sentences"] for r in broke) / len(broke),
               sum(r["queries"] for r in broke) / len(broke)))


if __name__ == "__main__":
    main()
