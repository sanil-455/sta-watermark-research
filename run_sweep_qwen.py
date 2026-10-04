"""
Paraphrase sweep using Qwen2.5-7B-Instruct.

Semantic threshold 0.60: probing showed quality holds down to
0.30 while z barely improves below 0.60, so the content check
is carrying fidelity and a tighter semantic gate only starves
candidates.

Quotations are skipped, since paraphrasing a quote alters what
someone said.

Results are written after every document, so an interruption
loses nothing already completed.
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "research" / "blackbox"))
sys.path.insert(0, str(ROOT / "research" / "attacks_core"))

from sta_core import load_tokenizer
from beam import Semantic
from oracle import DetectorOracle
from span_attack import gen_span_z
from paraphrase import Paraphraser
from para_attack import attack, edit_metrics

BASELINES = ["results/raw/baseline_safe.json",
             "results/raw/baseline_extra.json"]
OUT = "results/raw/para_sweep_qwen.json"


def main():
    log = lambda m: print(m, flush=True)

    tok = load_tokenizer()
    para = Paraphraser()
    sem = Semantic()

    rows = []
    for f in BASELINES:
        if Path(f).exists():
            rows.extend(json.load(open(f)))
    rows = [r for r in rows if r["watermarked_z"] > 2.0]

    log("attacking %d detected documents" % len(rows))
    log("%4s %8s %8s %7s %6s %8s %7s"
        % ("id", "z0", "z1", "drop%", "sents", "queries", "broke"))
    t_start = time.time()
    total = len(rows)
    results = []
    for r in rows:
        pc = len(r["prompt"])
        o = DetectorOracle(tok, mode="score")
        sz = lambda t, n=r["prompt_tokens"]: gen_span_z(tok, t, n)
        z0 = sz(r["watermarked_text"])

        try:
            a, z1, text = attack(o, para, r["watermarked_text"], pc, sz,
                                 semantic=sem, n_cands=60,
                                 sem_threshold=0.60,
                                 log=lambda *x: None)
        except Exception as e:
            log("%4d FAILED %s" % (r["prompt_id"], e))
            continue

        final = text if text else r["watermarked_text"]
        m = edit_metrics(tok, r["watermarked_text"], final, pc, len(a))

        results.append({
            "prompt_id": r["prompt_id"], "z0": z0, "z1": z1,
            "drop_pct": 100.0 * (z0 - z1) / z0,
            "sentences": len(a), "queries": o.queries,
            "broke": text is not None, "metrics": m,
            "rewrites": a, "text": text,
        })

        done = len(results)
        elapsed = time.time() - t_start
        per_doc = elapsed / done
        remaining = per_doc * (total - done)

        log("%4d %8.3f %8.3f %6.1f%% %6d %8d %7s  | %d/%d  %.0fm elapsed  ~%.0fm left"
            % (r["prompt_id"], z0, z1, 100.0 * (z0 - z1) / z0,
               len(a), o.queries, text is not None,
               done, total, elapsed / 60, remaining / 60))

        json.dump(results, open(OUT, "w"), indent=2)

    broke = [x for x in results if x["broke"]]
    log("")
    log("evaded: %d of %d" % (len(broke), len(results)))
    log("mean z drop: %.1f%%"
        % (sum(x["drop_pct"] for x in results) / len(results)))


if __name__ == "__main__":
    main()
