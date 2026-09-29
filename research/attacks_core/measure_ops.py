"""
Measure every single-edit operation: predicted vs actual.

Predicted drop is local (two adjacent pairs). Actual effect
comes from decode/re-encode, which includes re-tokenisation
ripple. The gap between them is the evidence that the
per-token locality assumption does not hold.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from sta_core import load_tokenizer, text_to_ids, sta_stats
from candidates import scan_all
from edit_ops import edits_to_text

OUT = Path("results/raw/op_measurements")


def main(prompt_id):
    tok = load_tokenizer()
    rows = json.load(open("results/raw/baseline_safe.json"))
    row = [r for r in rows
           if int(r["prompt_id"]) == prompt_id][0]

    ids = text_to_ids(tok, row["watermarked_text"])
    base = sta_stats(ids)

    pool_by_kind = scan_all(ids, tok, min_drop=-2)
    pool = (pool_by_kind["replace"]
            + pool_by_kind["delete"])

    print(f"prompt {prompt_id} | z={base['z']:.6f} "
          f"| pool={len(pool)}")

    out = []
    for n, e in enumerate(pool, 1):
        text = edits_to_text(tok, ids, [e])
        st = sta_stats(text_to_ids(tok, text))

        out.append({
            "type": e["type"],
            "pos": e["pos"],
            "label": e["label"],
            "pos_tag": e.get("pos_tag"),
            "predicted_drop": e["drop"],
            "actual_green_change":
                st["green_count"] - base["green_count"],
            "actual_pair_change":
                st["pair_count"] - base["pair_count"],
            "actual_delta_z": st["z"] - base["z"],
            "ripple":
                e["drop"] + (st["green_count"]
                             - base["green_count"]),
        })

        if n % 50 == 0:
            print(f"  {n}/{len(pool)}")

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"prompt_{prompt_id}.json"
    path.write_text(json.dumps({
        "prompt_id": prompt_id,
        "baseline_z": base["z"],
        "baseline_green": base["green_count"],
        "baseline_pairs": base["pair_count"],
        "operations": out,
    }, indent=2))

    exact = sum(1 for o in out if o["ripple"] == 0)
    print(f"\nlocal prediction exact: {exact}/{len(out)} "
          f"({100*exact/len(out):.1f}%)")
    print(f"saved: {path}")


if __name__ == "__main__":
    main(int(sys.argv[1]))
