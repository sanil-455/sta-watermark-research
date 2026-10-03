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

## Span-restricted attack (the paper's evaluation protocol)

Attacks the generated span only, with fit filtering and no clause
repetition. Substitutions plus comma-anchored clause insertions.

  id     z0      z1   edits  queries  broke
  25   3.818   2.210      9       32  no
  20   3.253   1.833      7       17  YES
   0   3.960   2.796      5       12  no
  22   4.667   1.745     16       76  YES
  24   5.091   3.522      8       36  no
   2   5.515   3.252     12       75  no
  23   5.374   3.717      7       28  no
   1   6.505   4.371     12       34  no
  27   6.364   4.888      7       21  no

2 of 9. Every failure is pool exhaustion, not a quality rejection.
Success tracks pool size relative to greens needed, not baseline z
alone: prompt 22 broke from 4.667 because it had 43 substitutions,
while prompt 25 stalled from 3.818 with only 11.

Clause insertion is ~2.6x more effective per operation than
substitution, because each added token raises gamma*T by 0.5.
Measured: 25.5% of filler tokens give two red pairs (theory 25%).

Known unfixed: clause placement uses "preceded by a comma", which
also matches commas inside lists, producing "with sharp, bulbous,
as the company noted, bright orange lights". Fixing this needs the
dependency parse and would cut slots further.

## Two threat models, two results

Full-document scoring (detector sees prompt + generation):
  5 of 9 break with clean text, ~50 queries each.
Generated-span scoring (the paper's protocol):
  2 of 9 break.

Only the second speaks to STA-1 as published. The first is a real
deployment scenario but a different claim.

## Session 2: expansion attack

Multi-token substitution ("said" -> "went on to say") destroys the
two pairs around the trigger AND raises gamma*T by 0.5(k-1), so the
numerator falls by 2 + 0.5(k-1). At k=5 that is -4.0, twice a plain
substitution. Most efficient operation found.

Numerator change per operation, best case:
  single substitution   -2.0
  deletion              -1.5  (also shrinks the denominator)
  adjacent swap         -3.0  (untested)
  insert 5 tokens       -3.5
  substitute 1->5       -4.0

research/blackbox/expansions.py: 123 triggers, 248 phrases.
Coverage 9-17 hits per 200-token span, clears the ~8 needed.

Span sweep WITH expansions (results/raw/span_sweep_exp.json):
  id     z0      z1   edits  queries  broke
  25   3.818   1.816      9       17  YES
  20   3.253   1.888     10       12  YES
   0   3.960   1.925     10       14  YES
  22   4.667   1.972     20       81  YES
  24   5.091   3.283     16       74  no
   2   5.515   1.957     19       77  YES
  23   5.374   2.133     20      105  no
   1   6.505   3.486     18       72  no
  27   6.364   2.772     18       36  no

5 of 9 on z. QUALITY AUDIT NOT DONE except prompt 0, which FAILED.

## Why prompt 0 failed, and the fix

The table assumed every expansion is interchangeable with its
trigger in any context. False for function words:

  "isn't the no more than one"     only->no more than
  "in love along with Fiona"       with->along with
  "loves, admires, as well as"     and->as well as, inside a list
  "she stood a chance to be"       might->stood a chance to

Expansions that DID read correctly are content verbs and
clause-initial connectives, where the grammatical role is fixed:
reported->went on to report, said->went on to say,
now->as things stand, although->in spite of the fact that.

NEXT: prune every trigger whose grammatical role varies -- and,
with, only, just, about, all, both, some, each, set, put, and all
modals. Keep reporting verbs and clause connectives. Coverage will
roughly halve, so the rate may fall below 5 of 9. Then rerun and
audit every breaking text by reading it.

## Final span-restricted result (pruned expansion table)

Pruning removed triggers whose grammatical role varies. Coverage
fell from 9-17 hits per span to 1-5, because the high-frequency
function words (and, with, only, about, modals) were carrying
nearly all the coverage and are exactly the unsafe ones.

  id     z0      z1   edits  queries  broke
  25   3.818   1.987      8       33  YES
  20   3.253   1.925      9       16  YES
   0   3.960   2.119      8       16  no
  22   4.667   1.933     16       68  YES
  24   5.091   3.598      9       48  no
   2   5.515   2.463     16       72  no
  23   5.374   2.472     16       82  no
   1   6.505   4.075     15       38  no
  27   6.364   4.220      9       23  no

3 of 9 on z. Texts in results/raw/final_audit.txt.

Unpruned table gave 5 of 9 but prompt 0's text failed on
"in love along with Fiona" and "stood a chance to be coming".

## Summary of everything established

VERIFIED
- Detector reproduces the official implementation exactly.
- Black-box oracle recovers what the key provides: same edits,
  same final z, ~55 queries. The key is not the security boundary.
- Cost law E = [(p-g)T - z*sqrt(g(1-g)T)]/eta predicts edits within
  one, fitted on two documents (eta 1.27 and 1.14).
- Operation efficiency: substitution -2.0, deletion -1.5,
  insert-5 -3.5, substitute-1-to-5 -4.0 on the numerator.
- Prompt inclusion suppresses detection: z_gen 3.4-6.5 against
  z_all 0.6-3.4 on the same documents.
- Theorem 3's independence assumption holds empirically
  (lag-1 correlation 0.020). Our critique of it failed.

NOT ESTABLISHED
- Any claim to break STA-1 generally. 3 of 9 under its own
  protocol, and the full-document result attacks a signal a
  deployed detector would not see.
- Binary-oracle resistance. Never tested, only asserted.
- Statistical confidence. n=9.

LIMIT
A 200-token span contains too few grammatically safe edits.
Prompt 1 needs 32 greens removed; after filtering it has roughly
5 expansion sites and 7 substitutions. Arithmetic, not tuning.
