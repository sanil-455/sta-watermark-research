# Project status

## Verified results

- Detector reproduces the official implementation to zero error.
- White-box attack (uses the key), scored on the FULL document (prompt + generation):
  prompt 1 z 2.816 -> 1.997 in 9 edits; prompt 2 z 3.050 -> 1.963 in 11 edits.
- Black-box attack (detector queries only, no key) matches both: 9 edits / 55 queries
  and 11 edits / 62 queries. Oracle ranking recovered exactly the drop=2 candidates.
- Sweep over 9 detected documents, quality audit by reading the text:
  5 of 9 break with clean text (prompts 1, 2, 20, 24, 25) after blocking 6 bad candidates.
  fit=10.0 breaks 6 but readmits known wrong-sense errors, so it is not reportable.
- Cost law E = [(p - gamma)T - z*sqrt(gamma(1-gamma)T)] / eta, with eta = 1.27 and 1.14.
  Cross-prediction error -1.1 and +1.0 edits. Query model error +2 and +1.
- Independence assumption in Theorem 3 holds empirically (pooled lag-1 correlation 0.020,
  noise level). That critique FAILED and is not a finding.

## Critical limitation (found late, must be stated in any write-up)

Every break above is against FULL-document scoring. On the generated span alone
(what the paper evaluates, T = 200) all five broken documents are STILL detected:
z_gen after attack = 5.74, 4.81, 2.97, 4.24, 3.82. The attack defeats a detector that
includes prompt text. It does NOT defeat STA-1 as the paper evaluates it.

## Corrections to earlier claims

- The "40% false-negative rate" is WRONG. Prompts 3 and 4 score 3.68 and 3.39 on their
  generated spans. They looked undetected only because prompt text diluted the score.
- research/attacks_core/README.md says "three of the five documents were never detected".
  The correct count is two of five, and it is a dilution artifact. Fix before publishing.
- The attack is white-box in its first form. The black-box version removes that limit.

## Span-restricted feasibility (measured)

Substitution alone cannot break the generated span for 5 of 9 documents (ceiling below
the greens that must be removed). Insertions are not supply-limited: 25.5% of filler
tokens give two red pairs (theory 25%), 198 of 199 positions have a usable one.
Single-adverb insertion breaks z (2.816 -> 1.34 with 15) but the text is saturated with
hedges. Clause insertion is the proposed fix: about 66% of random 5-token clauses help.

## Rule

Before building on any measurement, check that it targets the signal the paper evaluates.

## Next steps

1. Clause insertion restricted to the generated span, with quality gates, oracle-ranked.
2. Rerun on the 9 documents, report success rate, edits, queries, and z_gen.
3. Generate more samples if time allows (n=50 is about 2-3 hours unattended GPU).
4. Fix the README, then write up.
