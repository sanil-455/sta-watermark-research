"""
Validate lag_covariance against sequences whose correlation
structure we already know, before trusting it on real data.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from covariance import lag_covariance


def independent_sequence(n, p=0.5, seed=0):
    """Each entry drawn fresh. No relationship between neighbours."""
    rng = random.Random(seed)
    return [1 if rng.random() < p else 0 for _ in range(n)]


def correlated_sequence(n, copy_prob=0.8, seed=0):
    """Each entry copies the previous one with probability copy_prob."""
    rng = random.Random(seed)
    seq = [1]
    for _ in range(n - 1):
        if rng.random() < copy_prob:
            seq.append(seq[-1])
        else:
            seq.append(1 - seq[-1])
    return seq


def anticorrelated_sequence(n, flip_prob=0.8, seed=0):
    """Each entry flips the previous one with probability flip_prob."""
    rng = random.Random(seed)
    seq = [1]
    for _ in range(n - 1):
        if rng.random() < flip_prob:
            seq.append(1 - seq[-1])
        else:
            seq.append(seq[-1])
    return seq


if __name__ == "__main__":
    N = 20000

    indep = independent_sequence(N)
    corr = correlated_sequence(N)
    anti = anticorrelated_sequence(N)

    c_indep = lag_covariance(indep, 1)
    c_corr = lag_covariance(corr, 1)
    c_anti = lag_covariance(anti, 1)

    print(f"independent   : {c_indep:+.6f}   expect near 0")
    print(f"correlated    : {c_corr:+.6f}   expect clearly > 0")
    print(f"anticorrelated: {c_anti:+.6f}   expect clearly < 0")
    print()

    ok = (abs(c_indep) < 0.01 and c_corr > 0.05 and c_anti < -0.05)
    print("VALIDATION PASSED" if ok else "VALIDATION FAILED")
