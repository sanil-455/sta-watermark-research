# Trial 3 — Singular and plural mixed up

## What we tried

Candidates now came from the first and most common meaning of
each word, with the part of speech matched. No fragments, no
obscure senses, no proper nouns.

## What went wrong

The remaining errors were grammatical.

> "These are also **matter**, however, that move him closer"

> "Caveney **detail** his formative years"

> "Previous **estimate** had targeted separation costs"

> "as he **hit** two critical milestones"

Each sentence has a plural subject and a singular word, or a
present tense clashing with a past one. A reader notices
immediately.

More from the same run:

    matters   ->  matter      plural became singular
    details   ->  detail      plural became singular
    tribes    ->  tribe       plural became singular
    reaches   ->  hit         present became base form
    presented ->  present     past became present

## Why it happened

WordNet stores dictionary headwords. Look up matters and it
gives you back matter, because matter is the form that
appears in a dictionary.

So the suggested synonym was not a different word at all. It
was the same word with its ending stripped off.

This slipped through every check we had. The part of speech
matched, because matter and matters are both nouns. The
meaning matched, because they mean the same thing. Only the
grammatical form was wrong, and nothing was looking at that.

## What the next attempt changed

Trial 4 detected the grammatical form of the original word
and required the replacement to be in the same form.

That turned out to need more care than expected, and the
first version of the fix caused its own problem. That is
trial 4.
