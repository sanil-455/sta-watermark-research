# Trial 7 — When the filters are not enough

## What we tried

Sample 2 had been broken cleanly. The same pipeline was
pointed at Sample 3, a book review of 574 words with a
starting score of 3.05, the highest of any document we tested.

## What happened first

It broke, at 1.92, but every one of the seven results
contained the same problem.

> "So, with Gibberish he's **went** from pornography to an
> exploration of what incest and trauma"

That is a grammatical error, not a matter of taste. After
"he's" English requires the past participle gone. Went is the
simple past and does not belong there.

The check that inflects replacements to match the original had
worked correctly. It was given the wrong information. The
original word moved, sitting after "he's", is a past
participle, but the tagger labelled it simple past. The
inflector did exactly what it was told and produced the wrong
form. The grammar checker then let it through.

## Tightening did not fix it

The obvious response was to tighten the contextual threshold
until the bad candidate fell out. It scored 6.11, so a cutoff
of 6.0 removes it.

The trouble is what else goes at the same time. Sample 3 needs
thirteen green pairs removed, and only four positions in the
whole document remove two at once:

    matters -> things       fits well
    companion -> familiar   fits well, means the wrong thing
    away -> off             clumsy but correct
    moved -> went           grammatically wrong

A cutoff of 6.0 removes both the broken candidate and one of
the three good ones. With the remaining two the search reached
2.05 and stopped. Close, but above the threshold.

Widening the search did not help either. At the wider setting
the run ended with no new states to explore, which means the
candidates were exhausted rather than the quality budget. More
searching cannot help when there is nothing left to search.

## What actually worked

Three candidates were blocked by name, after a person read the
output and identified them:

    moved -> went       "he's went from pornography"
    audio -> sound      "the companion sound-CD"
    reaches -> makes    "as he makes two critical milestones"

Each is a different kind of failure. The first is a tagging
error. The second breaks a hyphenated compound rather than
replacing a word. The third breaks a fixed phrase: you reach
or hit a milestone, you do not make one.

With those three blocked and the threshold left loose enough
to keep the merely awkward candidates, the attack reached
1.96 with no grammatical errors at all.

## Result

| | |
|---|---|
| Score before | 3.05, flagged |
| Score after | 1.96, not flagged |
| Words changed | 11 |
| Proportion of document | 1.9% |
| Meaning preserved | 99.84% |
| Grammatical errors | 0 |

The eleven changes:

    want    -> desire        really -> truly
    very    -> really        earnest -> solemn
    move    -> go            spot -> place
    matters -> things        begin -> start
    stories -> tales         reaches -> hits
    away    -> off

Two of these are clumsy. "Truly a really rich memoir" stacks
two intensifiers, and "moved off from the spectacle" is
stiffer than the original. Neither is wrong, and the standard
we set was grammatical correctness rather than elegance.

One oddity in the text is not ours. The phrase "he's come to
tribes with his pain" appears in the original watermarked
document, where it should read "come to terms". The model
generated it that way.

## What this trial shows

Every automated check we built passed all three of the blocked
candidates.

Meaning similarity passed them, because one word in five
hundred barely moves the score. The grammar checker passed
them, including the one that was genuinely ungrammatical. The
fluency measure passed them, because a single odd word
disappears into the average of a long document. The contextual
fitness score, which had caught so much in trial 5, rated
"companion to familiar" as a *good* fit.

A person reading the output caught all three in under a
minute.

The honest conclusion is that the automated pipeline narrows
the field but does not close it. It took a pool of 159 raw
candidates down to 20 sensible ones, which made the search
possible. It did not produce publishable text on its own.
