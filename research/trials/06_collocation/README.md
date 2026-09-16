# Trial 6 — A broken phrase

## What we tried

With contextual fitness scoring in place, the attack produced
its cleanest text yet. Eight of nine changes read correctly.

One did not.

## What went wrong

> "We have been **doing** some good progress with the regional
> leadership team"

You make progress. You do not do progress.

The phrase is fixed. Both words are ordinary, both are common,
and the combination is wrong. A reader notices instantly even
though nothing is ungrammatical in any formal sense.

## Why it happened

The contextual check scores the replaced word in place. Doing
scored well, because "we have been doing" is a perfectly
normal thing to write.

A later version of the check also scored the word immediately
after the change. That did not help either. The word after is
some, and "doing some" is also perfectly normal.

The break only appears at progress, three words downstream:

    position     word
    -1           been
    0            making    replaced with doing
    +1           some      still fine
    +2           good      still fine
    +3           progress  this is where it breaks

Looking one word ahead was not enough. The partner in a fixed
phrase can sit several words away, and there is no reliable
limit on how far.

## What we did instead

Widening the window would catch this particular case. It would
not catch a phrase whose partner sits six or eight words out,
and we do not know how wide is wide enough.

Rather than guess, we used a property of the search itself.

The attack does not produce one result. It produces fifty
different combinations of changes, all of which drive the
score below the threshold. Thirty-five of those fifty never
touch the word making at all.

An attacker needs one working result. Picking a clean one from
a list of fifty costs nothing.

This is worth stating plainly, because it changes what a
defence has to achieve. The defender must stop every variant.
The attacker only needs one to survive. Observing that most
candidate edits look wrong is not a defence when the search
generates dozens of alternatives.

## Where this leaves things

The collocation problem is worked around, not solved. A proper
fix would score the phrase rather than the word, and that
remains open.

The result that came out of this trial is the one reported in
the main README. Nine changes, no grammatical errors, no
misused words, no broken phrases, and a detection score that
falls from 2.82 to 2.00.
