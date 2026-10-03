"""
Testing the independence assumption in Theorem 3 of the STA-1 paper.

Appendix C.3 states, without proof:
    "Because these random variables are independent of each
     other, the variance of their sum equals the sum of their
     variances."

This file builds up, piece by piece, the measurement needed to
test that claim on real watermarked text.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "attacks_core"))

from sta_core import load_tokenizer, text_to_ids, is_green_pair, GAMMA


def green_sequence(ids):
    """
    Turn a list of token ids into the raw 0/1 verdict for every
    adjacent pair.

    12 tokens -> 11 pairs, because the last token has nothing
    after it to pair with. This is the same T = tokens - 1
    relationship the detector itself uses.
    """
    return [
        int(is_green_pair(ids[i], ids[i + 1]))
        for i in range(len(ids) - 1)
    ]

def mean(values):
    """
    Average of a list of numbers.

    For a list of 0s and 1s this equals the fraction that are 1,
    since the zeros add nothing to the sum. That fraction is our
    estimate of p, the true probability a pair comes out green.

    Both the covariance formula and the paper's variance formula
    T*p*(1-p) need this value, so it is computed first.
    """
   # part of bernoulli distribution
    if not values: # edge case of value 0 so that not it gets divided by 0
        return None
    return sum(values) / len(values)

def lag_covariance(seq, lag=1):
    """
    Covariance between each verdict and the one `lag` positions
    later.

    Cov(X,Y) = average of (X - Xbar)(Y - Ybar)

    The product is positive when both values sit on the same
    side of their averages and negative when they sit on
    opposite sides, so the average of those products is
    positive for quantities that move together and near zero
    for unrelated ones.

    lag=1 compares each pair's verdict with its immediate
    neighbour, which is the case Theorem 3 assumes independent.
    """
   # no.of comparison pairs we will get
    n = len(seq) - lag
    if n <= 1:
        return None

    x = seq[:-lag]
    y = seq[lag:]

    mx = mean(x)
    my = mean(y)

    products = [(a - mx) * (b - my) for a, b in zip(x, y)]
    return mean(products)

def variance(values):
    """
    Average squared distance from the mean.

    Squaring stops positive and negative deviations from
    cancelling each other out, so this measures spread
    regardless of direction.
    """
    if not values:
        return None
    m = mean(values)
    return mean([(v - m) ** 2 for v in values])


def lag_correlation(seq, lag=1):
    """
    Covariance rescaled to the range [-1, +1].

    Raw covariance has no natural scale, so a value of 0.15
    tells you little on its own. Dividing by the two standard
    deviations removes the scale, making the result directly
    comparable across documents of different lengths and
    different green fractions.

      +1 : perfectly locked together
       0 : no linear relationship
      -1 : perfectly opposite
    """
    cov = lag_covariance(seq, lag)
    if cov is None:
        return None

    x = seq[:-lag]
    y = seq[lag:]

    spread = (variance(x) * variance(y)) ** 0.5

    if spread < 1e-12:
        return None

    return cov / spread

def correlation_profile(seq, lags=(1, 2, 3, 5, 10, 20)):
    """
    Correlation measured at several distances.

    A single lag-1 number cannot distinguish local dependence
    from something systematic. Adjacent pairs share a token, so
    the structural argument predicts a value that is clearly
    nonzero at lag 1 and decays toward zero at larger lags.

    If instead the correlation stays flat across every lag, the
    shared-token explanation does not account for it and the
    real cause has to be found before drawing any conclusion.
    """
    return {
        lag: lag_correlation(seq, lag)
        for lag in lags
        if len(seq) - lag > 1
    }
