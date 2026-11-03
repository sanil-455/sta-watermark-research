# Query budget: how much does the attack rely on detector access?

Reconstructed from the saved per-step logs of the lenient attack (no new attack run). A query budget of N means: what z would the attacker be stuck at if the detector refused all queries past the Nth (the first query is always the z0 check). Lengths are never pooled.

## 200-token spans (46 flagged, attacked spans)

| query budget | broken (z<=2.0) | break rate (95%% interval) |
|---|---|---|
| 5 | 1 of 46 | 2% [0%, 11%] |
| 10 | 1 of 46 | 2% [0%, 11%] |
| 20 | 2 of 46 | 4% [1%, 15%] |
| 50 | 4 of 46 | 9% [3%, 20%] |
| 100 | 17 of 46 | 37% [25%, 51%] |
| 200 | 34 of 46 | 74% [60%, 84%] |
| 500 | 35 of 46 | 76% [62%, 86%] |
| unlimited (actual) | 35 of 46 | 76% [62%, 86%] |

- of the spans the attack actually broke, queries needed: min 2, median 101, max 252
  - 50% of eventually-broken spans needed 101 queries or fewer
  - 75% of eventually-broken spans needed 147 queries or fewer
  - 90% of eventually-broken spans needed 175 queries or fewer

## 500-token spans (21 flagged, attacked spans)

| query budget | broken (z<=2.0) | break rate (95%% interval) |
|---|---|---|
| 5 | 0 of 21 | 0% [0%, 15%] |
| 10 | 0 of 21 | 0% [0%, 15%] |
| 20 | 0 of 21 | 0% [0%, 15%] |
| 50 | 0 of 21 | 0% [0%, 15%] |
| 100 | 0 of 21 | 0% [0%, 15%] |
| 200 | 1 of 21 | 5% [1%, 23%] |
| 500 | 15 of 21 | 71% [50%, 86%] |
| unlimited (actual) | 18 of 21 | 86% [65%, 95%] |

- of the spans the attack actually broke, queries needed: min 107, median 308, max 857
  - 50% of eventually-broken spans needed 304 queries or fewer
  - 75% of eventually-broken spans needed 393 queries or fewer
  - 90% of eventually-broken spans needed 577 queries or fewer

