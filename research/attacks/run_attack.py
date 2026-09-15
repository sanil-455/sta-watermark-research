"""
Entry point: beam search against a chosen baseline sample.

    python research/attacks/run_attack.py 2        # Sample 3
    python research/attacks/run_attack.py 1 --depth 10
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from sta_core import load_tokenizer, text_to_ids, sta_stats
from candidates import scan_all
from beam import search

BASELINE = Path("results/raw/baseline_safe.json")
OUT_DIR = Path("results/raw/beam_attack")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt_id", type=int)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--beam", type=int, default=200)
    ap.add_argument("--rank-width", type=int, default=1200)
    ap.add_argument("--sem", type=float, default=0.95)
    ap.add_argument("--breaks", type=int, default=5)
    args = ap.parse_args()

    tok = load_tokenizer()
    rows = json.load(BASELINE.open())

    match = [r for r in rows
             if int(r["prompt_id"]) == args.prompt_id]
    if len(match) != 1:
        raise SystemExit(
            f"prompt_id={args.prompt_id} not unique"
        )

    row = match[0]
    text = row["watermarked_text"]
    stored = float(row["watermarked_z"])

    ids = text_to_ids(tok, text)
    base = sta_stats(ids)

    if abs(base["z"] - stored) > 1e-9:
        raise SystemExit(
            f"detector drift: {base['z']} vs {stored}"
        )

    if not base["detected"]:
        raise SystemExit(
            f"baseline z={base['z']:.4f} is not detected; "
            f"a break would be undefined"
        )

    print(f"prompt {args.prompt_id} | tokens={len(ids)} "
          f"| z={base['z']:.6f} (verified)")

    pool_by_kind = scan_all(ids, tok)
    pool = pool_by_kind["replace"] + pool_by_kind["delete"]

    print(f"pool: {len(pool_by_kind['replace'])} replace + "
          f"{len(pool_by_kind['delete'])} delete")

    if not pool:
        raise SystemExit("empty candidate pool")

    breaks, best, history = search(
        tok, ids, pool,
        max_depth=args.depth,
        beam=args.beam,
        rank_width=args.rank_width,
        sem_threshold=args.sem,
        max_breaks=args.breaks,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"prompt_{args.prompt_id}.json"

    out.write_text(json.dumps({
        "prompt_id": args.prompt_id,
        "baseline_z": base["z"],
        "baseline_green": base["green_count"],
        "baseline_pairs": base["pair_count"],
        "threshold": 2.0,
        "pool_size": len(pool),
        "settings": vars(args),
        "best_z": best[0],
        "delta_z": best[0] - base["z"],
        "breaks_found": len(breaks),
        "breaks": breaks,
        "history": history,
    }, indent=2))

    print()
    print(f"baseline z : {base['z']:.6f}")
    print(f"best z     : {best[0]:.6f}")
    print(f"delta z    : {best[0] - base['z']:+.6f}")
    print(f"breaks     : {len(breaks)}")
    print(f"saved      : {out}")

    if breaks:
        print("\nbreaks:")
        for b in breaks[:5]:
            print(f"  z={b['z']:.4f} sim={b['sim']:.4f} "
                  f"d={b['depth']} | {b['edits']}")


if __name__ == "__main__":
    main()
