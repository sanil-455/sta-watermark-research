# Bound check: does the necessary condition predict which spans can be broken?

The bound (gamma = 0.5, T tokens, z0 the starting z): rewriting a share r of the pairs at chance level gives z = (1 - r) * z0, so z <= 2 needs
r >= 1 - 2/z0. Only the eligible sentences (character share e) may change; when e is smaller than that, the new pairs must be green LESS often
than chance by delta = ((1 - e) * z0 - 2) / (2 * e * sqrt(T)). The attack stores it as delta_needed (None: more than 0.5 or no eligible text,
i.e. impossible). This table shows how often each class of span really broke.

## 200-token spans (batch 1 + batch 2) (253 attacked spans, 195 broken)

| class of span, by the bound | spans | broken | break rate (95% interval) |
|---|---|---|---|
| chance-level rewriting is enough (delta <= 0) | 169 | 167 | 99% [96%, 100%] |
| needs selection gain in (0, 0.05] | 36 | 26 | 72% [56%, 84%] |
| needs selection gain in (0.05, 0.10] | 16 | 2 | 12% [3%, 36%] |
| needs selection gain in (0.10, 0.20] | 13 | 0 | 0% [0%, 23%] |
| needs selection gain above 0.20 | 9 | 0 | 0% [0%, 30%] |
| impossible by the bound (gain above 0.5 or no eligible text) | 10 | 0 | 0% [-0%, 28%] |

- a lower delta_needed predicts a break: AUC 0.969 (1.0 = every broken span needed less gain than every unbroken one, 0.5 = no information)
- the rule "predicted to break when delta_needed <= t" is right for 95% of spans at its best threshold t = 0.047, and for 88% at t = 0 (chance-level rewriting only)
- largest delta_needed among spans that broke: 0.081; smallest among spans that did not (finite ones): 0.000
- among spans that needed a gain above chance: the attack's measured gain reached it for 58 of 72 (broken: 28 of 58)
- the attack's measured selection gain (delta_achieved) over all spans, mean / median / max: 0.151 / 0.147 / 0.362

## 500-token spans (21 attacked spans, 18 broken)

| class of span, by the bound | spans | broken | break rate (95% interval) |
|---|---|---|---|
| chance-level rewriting is enough (delta <= 0) | 11 | 11 | 100% [74%, 100%] |
| needs selection gain in (0, 0.05] | 6 | 6 | 100% [61%, 100%] |
| needs selection gain in (0.05, 0.10] | 1 | 1 | 100% [21%, 100%] |
| needs selection gain in (0.10, 0.20] | 1 | 0 | 0% [0%, 79%] |
| needs selection gain above 0.20 | 0 | 0 | n/a |
| impossible by the bound (gain above 0.5 or no eligible text) | 2 | 0 | 0% [0%, 66%] |

- a lower delta_needed predicts a break: AUC 1.000 (1.0 = every broken span needed less gain than every unbroken one, 0.5 = no information)
- the rule "predicted to break when delta_needed <= t" is right for 100% of spans at its best threshold t = 0.051, and for 67% at t = 0 (chance-level rewriting only)
- largest delta_needed among spans that broke: 0.051; smallest among spans that did not (finite ones): 0.105
- among spans that needed a gain above chance: the attack's measured gain reached it for 8 of 8 (broken: 7 of 8)
- the attack's measured selection gain (delta_achieved) over all spans, mean / median / max: 0.143 / 0.131 / 0.250

