"""
Black-box attack, end to end.

Pipeline: propose candidates -> contextual-fit filter (no
queries) -> rank by detector queries -> greedy with quality
gates.

The fit filter runs before ranking so the query budget is spent
only on candidates that could survive.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "attacks_core"))

from sta_core import load_tokenizer, text_to_ids
from context_fit import ContextFit
from fit_filter import filter_by_fit
from fluency import Fluency
from oracle import DetectorOracle
from candidates_bb import propose_all
from rank import rank_candidates
from greedy_filtered import greedy_filtered

BASELINE = Path("results/raw/baseline_safe.json")

def attack(prompt_id, fit=7.0, mode="score", baseline=BASELINE):
    tok = load_tokenizer()
    row = [r for r in json.load(Path(baseline).open())
           if r["prompt_id"] == prompt_id][0]
    ids = text_to_ids(tok, row["watermarked_text"])

    pool = propose_all(ids, tok)
    print(f"pool: {len(pool)} candidates")

    fluency = Fluency(tok)
    fit_scorer = ContextFit(tok, fluency.model, fluency.device)
    pool = filter_by_fit(pool, ids, fit_scorer, max_drop=fit)

    oracle = DetectorOracle(tok, mode=mode)
    _, ranked = rank_candidates(oracle, tok, ids, pool)

    print()
    edits, z, text = greedy_filtered(
        oracle, tok, ids, ranked, fluency=fluency
    )

    print()
    print(f"queries {oracle.queries} | edits {len(edits)} | z {z:.6f}")
    return edits, z, text, oracle.queries


if __name__ == "__main__":
    pid = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    _, _, text, _ = attack(pid)
    if text:
        print()
        print(text)
