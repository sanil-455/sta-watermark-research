"""
Run the black-box attack across every flagged sample and
collect edits, queries and efficiency into one table.

Only documents the detector already flags are attacked. An
unflagged document has nothing to remove, so attacking it
would not be meaningful.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "attacks_core"))
sys.path.insert(0, str(HERE.parent / "math"))

from run_bb import attack
from cost_law import eta_observed, z_score
from sta_core import Z_THRESHOLD


def sweep(files, fit=7.0, out="results/raw/bb_sweep.json"):
    """Attack every flagged sample across the given baselines."""
    results = []

    for path in files:
        for row in json.load(Path(path).open()):
            pid, z0 = row["prompt_id"], row["watermarked_z"]
            if z0 <= Z_THRESHOLD:
                print(f"skip {pid}: z={z0:.3f} not flagged")
                continue

            print(f"\n=== prompt {pid} (z={z0:.3f}) ===")
            try:
                edits, z1, text, queries = attack(
                    pid, fit=fit, baseline=path
                )
            except Exception as e:
                print(f"  failed: {e}")
                continue

            results.append({
                "prompt_id": pid,
                "source": str(path),
                "z_before": z0,
                "z_after": z1,
                "broke": z1 <= Z_THRESHOLD,
                "edits": len(edits),
                "queries": queries,
                "labels": [e["label"] for e in edits],
                "text": text,
            })

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(results, indent=2))
    return results


def summarise(results):
    broke = [r for r in results if r["broke"]]
    print("\n" + "=" * 58)
    print(f"{len(broke)} of {len(results)} broke")
    print("=" * 58)
    print(f"{'id':>4s} {'z0':>7s} {'z1':>7s} {'edits':>6s} {'queries':>8s} {'broke':>6s}")
    for r in results:
        print(f"{r['prompt_id']:4d} {r['z_before']:7.3f} {r['z_after']:7.3f} "
              f"{r['edits']:6d} {r['queries']:8d} {str(r['broke']):>6s}")

    if broke:
        e = sum(r["edits"] for r in broke) / len(broke)
        q = sum(r["queries"] for r in broke) / len(broke)
        print(f"\nmean edits {e:.1f} | mean queries {q:.1f}")


if __name__ == "__main__":
    files = ["results/raw/baseline_safe.json",
             "results/raw/baseline_extra.json"]
    summarise(sweep(files))
