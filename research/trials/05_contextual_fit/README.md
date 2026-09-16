# Trial 5 — A synonym that does not fit

## What we tried

By this point every candidate was a whole word, from the most
common meaning, with the right part of speech and the right
grammatical form. Three separate checks ran on the output:
meaning similarity, a grammar checker, and a fluency score
from the language model itself.

The attack broke the watermark. The text still read wrong.

## What went wrong

> "Earlier in the summer, it **denoted** its purpose to invest
> $780 million"

> "which we expect to result in lower working capital **asks**"

> "You have probably **applied** its products"

> "presented on the **familiar** audio-CD"

None of these are grammatical errors. Every sentence parses.
Every word is the right part of speech and the right number.
They are simply not what anyone would write.

Denote means to signify. A company does not denote an
investment, it announces one. An ask is a request; working
capital needs are not asks. You use a product, you do not
apply it. A familiar, as a noun, is a spirit that attends a
witch.

## Why it happened

Every one of these is a defensible synonym in isolation.
Denote and announce overlap in some contexts. Ask and need
overlap in others. A dictionary is right to list them
together.

The problem is that a dictionary describes words in the
abstract. It cannot know that this particular sentence, about
this particular company, needs one sense and not the other.

None of the three checks caught it either. Meaning similarity
compares whole documents, and one word in eight hundred barely
moves the score. The grammar checker found nothing wrong
because nothing was grammatically wrong. The fluency score
looked at the document as a whole, where a single odd word
disappears into the average.

## What the next attempt changed

Instead of asking a dictionary what a word means, ask the
language model what belongs here.

For each candidate, measure how likely the model thinks that
word is at that exact position, given everything around it.
Compare against the original word. If the replacement is far
less expected, reject it.

The scores confirmed the intuition straight away:

    maintain -> keep        -1.5   model prefers the swap
    additional -> extra     -1.6   model prefers the swap
    benefit -> profit        0.8   about equally good
    intent -> purpose        6.4   noticeably worse
    estimates -> ideas       7.4   clearly wrong
    needs -> asks           12.2   badly wrong
    source -> beginning     18.4   nonsense

Everything the reader had flagged appeared near the bottom.
Everything that read well appeared near the top, and two
candidates scored better than the words they replaced.

Setting the cutoff at 6.0 removed every failure listed above.
The pool fell from 46 candidates to 20, and the attack still
worked. A smaller set of well-fitting candidates outperformed
a larger noisy one, because bad candidates crowd out good ones
during the search.
