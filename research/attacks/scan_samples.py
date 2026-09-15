import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from sta_core import load_tokenizer, text_to_ids, sta_stats
from candidates import scan_all


BASELINE = "results/raw/baseline_safe.json"

TARGETS = [
    (1, "Sample 2"),
    (2, "Sample 3"),
]


def main():
    tok = load_tokenizer()
    rows = json.load(open(BASELINE))

    for pid, name in TARGETS:
        matches = [r for r in rows if r["prompt_id"] == pid]

        if len(matches) != 1:
            raise RuntimeError(
                f"could not uniquely find prompt_id={pid}"
            )

        row = matches[0]
        ids = text_to_ids(tok, row["watermarked_text"])
        st = sta_stats(ids)

        print("=" * 66)
        print(
            f"{name} | tokens={len(ids)} "
            f"z={st['z']:.4f} "
            f"green={st['green_count']}/{st['pair_count']}"
        )
        print(f"needs delta z <= {2.0 - st['z']:.4f}")
        print("=" * 66)

        pool = scan_all(ids, tok)

        for kind in ("replace", "delete"):
            lst = pool[kind]
            d2 = sum(1 for e in lst if e["drop"] >= 2)

            print(
                f"\n{kind}: {len(lst)} candidates "
                f"({d2} with drop=2)"
            )

            for e in lst[:10]:
                print(
                    f"   drop={e['drop']} "
                    f"pos={e['pos']:4d} {e['label']}"
                )

        total = sum(len(v) for v in pool.values())
        print(f"\nTOTAL POOL: {total}")
        print()


if __name__ == "__main__":
    main()
