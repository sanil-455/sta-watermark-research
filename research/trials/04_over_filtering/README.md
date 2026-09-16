# Trial 4 — A fix that cut too deep

## What we tried

To stop plural words being replaced by singular ones, the
simplest approach is to refuse to touch any word that is not
already in its dictionary form. WordNet returns dictionary
forms, so if the original is also a dictionary form, they
match automatically.

## What went wrong

Nothing grammatically. The sentences were correct. There were
just not enough of them left to work with.

The candidate pool collapsed:

    Sample 2    159 candidates  ->  29
    Sample 3    115 candidates  ->  29

More importantly, the useful candidates went with them. Only
words that kill two green pairs at once are worth much, and
those dropped from 34 to 7 in one sample, and from 26 to 3 in
the other.

With three useful candidates and a target needing thirteen
pairs removed, the attack could not succeed. Not because the
text would read badly, but because there was nothing left to
change.

## Why it happened

Most words in ordinary writing are inflected. Verbs carry
tense, nouns carry number. Refusing to touch inflected words
means refusing to touch most of the document.

We removed three bad candidates and about seventy percent of
the good ones along with them.

## What the next attempt changed

The right move is not to avoid inflected words. It is to
inflect the replacement to match.

If the original is plural, make the replacement plural. If the
original is past tense, put the replacement in past tense. A
small library called lemminflect does this conversion.

So matters can now be offered issues rather than issue, and
reaches can be offered hits rather than hit.

The pool recovered:

    Sample 2    29 candidates  ->  43
    Sample 3    29 candidates  ->  37

And the grammatical errors from trial 3 stayed gone. This is
the version that produced the first result worth reading.
