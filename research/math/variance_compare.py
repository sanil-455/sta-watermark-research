"""
Compare the variance the paper assumes against the variance the
data actually shows.

Appendix C.3 assumes independence, giving Var = T*p*(1-p).
If adjacent verdicts are correlated, the true variance carries
additional covariance terms that the proof drops.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from covariance import mean, lag_covariance, correlation_profile


def compare_variance(seq):
    """
    Both variance estimates, side by side.

    paper_variance assumes every pair is independent.
    corrected_variance adds the lag-1 covariance terms, which
    is the dominant correction when dependence is local.

    ratio > 1 means the paper's figure is too small, so its
    bound on the type II error is more optimistic than the
    data supports.
    """
    T = len(seq)
    p = mean(seq)

    paper_variance = T * p * (1 - p)

    cov1 = lag_covariance(seq, 1)
    correction = 2 * (T - 1) * cov1 if cov1 is not None else 0.0

    corrected = paper_variance + correction

    return {
        "T": T,
        "p": p,
        "paper_variance": paper_variance,
        "lag1_covariance": cov1,
        "correction": correction,
        "corrected_variance": corrected,
        "ratio": corrected / paper_variance if paper_variance > 1e-12 else None,
    }


def report(seq, label=""):
    """Print one comparison in readable form."""
    r = compare_variance(seq)

    print(f"--- {label} ---")
    print(f"  pairs (T)            : {r['T']}")
    print(f"  green fraction (p)   : {r['p']:.4f}")
    print(f"  lag-1 covariance     : {r['lag1_covariance']:+.6f}")
    print()
    print(f"  variance, paper      : {r['paper_variance']:10.3f}")
    print(f"  covariance correction: {r['correction']:+10.3f}")
    print(f"  variance, corrected  : {r['corrected_variance']:10.3f}")
    print(f"  ratio corrected/paper: {r['ratio']:.4f}")
    print()
    print("  correlation by lag:")
    for lag, c in correlation_profile(seq).items():
        print(f"    lag {lag:2d}: {c:+.6f}")
    print()
