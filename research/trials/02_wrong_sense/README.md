# Trial 2 — The right word with the wrong meaning

## What we tried

With word fragments blocked and deletions restricted to a
short list of optional words, the attack now only touched
whole words. Synonyms came from WordNet, a dictionary that
lists words with similar meanings.

## What went wrong

The text filled up with words that are real synonyms and
completely wrong here.

> "the start-stop vehicle **mart** opportunity"

> "expand its battery manufacturing **content**"

> "management announced its **purpose** to invest"

A mart is a shop. The sentence is about a market sector.
Manufacturing capacity is not manufacturing content. A
company announces an intent, not a purpose.

More from the same run:

    market   ->  mart        wrong kind of market
    capacity ->  content     unrelated meaning
    yield    ->  fruit       farming sense of yield
    was      ->  Washington  proper noun for a verb

## Why it happened

WordNet lists every meaning a word can carry. The word market
has a sense meaning a physical shop, and mart is a synonym of
that sense. It is simply not the sense used in a financial
report.

We were asking for synonyms of the word. We should have been
asking for synonyms of the word as used in this sentence.

## What the next attempt changed

WordNet orders meanings by how common they are. The first
entry is the everyday sense; later entries get progressively
more obscure. Mart came from the third sense of market.
Content came from the fourth sense of capacity.

Trial 3 restricted candidates to the first sense only.

It also required the part of speech to match. Asking for
synonyms of was as a verb no longer returns Washington,
because that comes from a noun sense.

This cut the substitution pool for one sample from 262
candidates down to 159, and removed every example listed
above.
