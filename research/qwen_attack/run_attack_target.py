"""
Item 5: push the attack's target below the paper's default z <= 2.0.

A thin wrapper around run_attack.py. attack.py's stopping condition (z <= Z_THRESHOLD) is a
module-level constant, not a CLI flag, so this patches attack.Z_THRESHOLD before run_attack's
main() runs; Python resolves that name against the attack module's namespace at call time (not
at def time), so the patch takes effect for every attack_span call made afterwards, with zero
changes to attack.py or run_attack.py themselves. All other run_attack.py flags (--files, --only,
--preset, --n-cands, --seed, --resume, ...) pass through unchanged.

  python research/qwen_attack/run_attack_target.py --target 1.5 --files ... --only ... --resume ...
"""

import argparse
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--target", type=float, default=2.0, help="z threshold the attack stops at (paper's default: 2.0)")
    args, rest = ap.parse_known_args()

    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))   # must match run_attack.py's own sys.path setup, so "import attack" below
    import attack                   # and run_attack.py's later "import attack" share the SAME cached module object
    attack.Z_THRESHOLD = args.target

    import run_attack
    sys.argv = [sys.argv[0]] + rest
    run_attack.main()


if __name__ == "__main__":
    main()
