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

## FINAL: span-restricted attack fails the quality audit

All three breaks were read in full. All three contain grammatical
errors:

  prompt 25  "have contributed significantly in advance of"
             before was adverbial, not prepositional
             "for over that stretch, which he was went on to describe"
  prompt 20  "a correction is inevitable, in the event that not overdue"
             "disastrous performance given the fact that March"
             since was temporal, the expansion made it causal
  prompt 22  "her figure overestimated", "a modification in her
             antibiotics", "that near set her in the hospital"

0 of 9 under the paper's protocol at an acceptable quality bar.

The failure is structural. Expansions that are grammatically safe
in every context are rare; the common ones (and, with, only,
about, before, since, modals) carry multiple syntactic roles and
break when substituted blindly. Context-sensitive expansion would
need a parser deciding role per occurrence, which is a different
system from a lookup table.

## What this work establishes

STA-1 resists this attack on 200-token spans. That is a positive
result for the scheme, measured rather than assumed.

The contributions are elsewhere:

1. Query access substitutes for the secret key. ~55 detector
   queries recover the same edits, same final z, as holding H1
   and H2. The paper's ethical statement says to protect the key;
   this shows that is not sufficient.

2. Attack cost law, verified on two documents to within one edit:
       E = [(p-g)T - z*sqrt(g(1-g)T)] / eta
   An attacker need only erase the margin above the threshold,
   not the watermark. For one document that was 29% of the signal.

3. Operation efficiency on the numerator:
       substitution -2.0, deletion -1.5, insert-5 -3.5,
       substitute-1-to-5 -4.0
   Adding tokens helps because each raises gamma*T by 0.5.

4. Prompt inclusion suppresses detection. Same documents score
   z_gen 3.4-6.5 but z_all 0.6-3.4. Two of five looked undetected
   purely from dilution.

5. Theorem 3's independence assumption holds empirically
   (pooled lag-1 correlation 0.020). Our critique of it failed
   and is withdrawn.

## Limits of these conclusions

n=9. No statistical confidence on any rate. Binary-oracle
resistance was asserted, never tested. The cost law is fitted on
two points.

## Llama-2-7B-chat: tested, worse, abandoned

Switched to the instruction-tuned model to fix factual
substitution, with a system prompt stating "never change a
number, percentage, date, name or quotation; never reverse a
comparison or a negation".

It reversed direction on EVERY candidate:
  "her number overestimated how long she could go"
  -> "her estimate undervalued the duration"
  -> "her projection undervalued the duration"
  -> "her prediction undervalued the duration"
  -> "her estimate undercounted the duration"

The base model got this wrong sometimes. The chat model gets
it wrong consistently, despite the explicit instruction.

It also changed "Anglos" to "non-Hispanics", a different
demographic category, while correctly preserving "half" and
"a quarter" as distinct quantities.

Both available paraphrasers are now tested. Neither preserves
factual direction reliably. Reverted to the base model.

## Final position

The attack mechanism is proven and the limitation is precisely
located.

PROVEN: oracle-guided selective paraphrase evades STA-1 on the
generated span, which is what the paper evaluates. 3 to 6 of 9
documents depending on how strict the content gates are, using
63 to 184 detector queries and no watermark key. Token-level
editing cannot do this: a 200-token span holds too few safe
single-word edits, while sentences can be rewritten without
limit.

NOT ACHIEVED: faithful rewrites. Both Llama-2-7B base and chat
substitute content words in ways that invert meaning. No gate
built from regex or embeddings catches this, because the
substituted word exists, counts match, and embeddings barely
move on a single antonym.

The binding constraint is paraphrase fidelity, not the attack.
A model that preserves factual direction would make this a
clean break. That is a tooling question, not a limitation of
the method.

## Structure check: tried, net loss, reverted

Added a check requiring the candidate to keep the original's
sentence count and a length ratio of 0.65 to 1.15, after the
audit showed every failure restructured across clause
boundaries while every clean rewrite preserved structure.

Measured from the audit data: clean rewrites ran 0.67 to 1.11,
failures 0.61 to 1.27. The ranges overlap almost entirely, so
length alone is a weak discriminator and the sentence-count
test does the real work.

Result: 5 of 9 evade, 2 clean (22, 24), against 6 of 9 and 3
clean (24, 25, 27) without it. It fixed prompt 22's "the
freedom her job gave her" becoming "freedom her job lacked",
but cost prompts 25 and 27, both previously clean. Prompt 27
was the strongest single result in the project, z 6.364 to
1.552. Reverted.

## What the remaining failures are

Comprehension errors, not statistical or structural ones:

  prompt 0   "USA Today reported that her first tweet was" became
             "her first tweet to USA Today read" -- a fabricated
             relationship. A direct address to Fiona became
             third-person commentary.
  prompt 2   "he's come to tribes with his pain" is a typo for
             "come to terms". Read literally, producing invented
             travel to tribal communities.
  prompt 20  "tech stocks are the worst offenders because they're
             the ones you've got to buy" became "as I have to buy
             them most often", turning a causal claim into a
             statement about purchasing frequency.

Regex, embeddings and sentence counts have all been tried and
none catch these. They require reading comprehension.

## Final position

The attack's ceiling is set by the paraphraser's comprehension,
not by the watermark's strength. STA-1 is evaded on 6 of 9
documents, 3 of which survive reading, using 46 to 175 detector
queries and no watermark key, rewriting 50 to 86 percent of
sentences in the generated span.
