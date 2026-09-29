import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from sta_core import load_tokenizer, score_text

BASELINE = Path("results/raw/baseline_safe.json")

tok = load_tokenizer()
rows = json.load(BASELINE.open())

worst = 0.0

for r in rows:
    got = score_text(tok, r["watermarked_text"])["z"]
    exp = float(r["watermarked_z"])
    err = abs(got - exp)
    worst = max(worst, err)

    print(
        f"prompt {r['prompt_id']}: "
        f"{got:.12f} vs {exp:.12f} | err {err:.2e}"
    )

print()
print(f"worst error: {worst:.2e}")

if worst > 1e-9:
    raise SystemExit(
        "FAILED - detector does not reproduce baseline"
    )

print("PASSED")
