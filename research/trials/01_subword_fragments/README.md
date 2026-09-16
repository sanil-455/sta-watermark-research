# Trial 1 — Deleting pieces of words

## What we tried

Scan every position in the document. For each one, work out
whether deleting that word or swapping it for a synonym would
turn green pairs red. Keep anything that helps.

The only filter was a check that the text looked like a word:
letters only, at least two or three characters.

## What went wrong

The attack deleted parts of words.

> "he grew up in **Carr**, a small town"

The original said Carrington. The attack removed the ending
and left the rest standing.

Other examples from the same run:

    del(ington)    Washington  ->  Wash
    del(ance)      acceptance  ->  accept
    del(mem)       remember    ->  re...ber
    del(SE)        NYSE        ->  NY

## Why it happened

Language models do not store words the way people do. Llama
keeps common words whole but splits rarer ones into pieces.
Washington is stored as two chunks, roughly Wash and ington.

Our filter asked: does this chunk consist of letters? Yes,
ington is all letters. Is it long enough? Yes. So it passed.

The filter was asking the wrong question. It should have
asked whether the chunk starts a word, not whether it looks
like one.

## What the next attempt changed

Llama marks the beginning of a word with a special character
that does not survive being printed. Checking the raw stored
form rather than the printed text separates the ending from
the whole word immediately.

Trial 2 added that check, plus two more:

Skip anything tagged as a name, place, or organisation.
Deleting England is grammatical and still wrong.

Only delete words from a short list of genuinely optional
ones, such as very, really and also. Deleting a noun changes
what the sentence says.

This cut the deletion pool from 310 candidates to 3, which
turned out to matter later.
