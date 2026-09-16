# Breaking the STA-1 Watermark with Contextual Substitution

This documents an attack on STA-1, the text watermarking
scheme from *Watermarking Large Language Models: An Unbiased
and Low-risk Method* (ACL 2025).

The short version: changing about one word in a hundred is
enough to make watermarked text look unwatermarked, and the
result still reads like ordinary English.

Six attempts failed before one worked. Each failure is
documented separately in [`../trials/`](../trials/), because
the progression of what broke is probably more useful than the
final number.

---

## How STA-1 works

The watermark is a statistical fingerprint, not a hidden
message. Nothing is embedded in the text itself.

Take any two neighbouring words. Convert them to the numbers
the model uses internally, combine those with a secret
formula, and use the result to seed a random number generator.
Draw one number between zero and one. If it lands below one
half, call that pair green.

Do this for every adjacent pair. In ordinary writing about
half come out green, because the process is a coin flip. In
watermarked text the model was nudged toward choices that
produce green pairs, so the count runs high.

The detector measures how far above half you are, in standard
deviations. That number is the z-score. Above 2.0 the text is
flagged. Below 2.0 it is not.

Three details matter for the attack.

The secret is two numbers. Anyone holding them can compute the
verdict for any pair. There is no cryptography, only secrecy.

Each pair is scored on its own. Changing one word affects only
the two pairs that word sits between.

Detection is a threshold on a sum. It does not matter which
pairs are green, only how many.

---

## Verifying the detector first

Before attacking anything we rebuilt the detector and checked
it against the original on five watermarked documents. All
five matched to zero difference. Not close enough, but
bit-for-bit identical.

This matters because every number downstream depends on it.

    python research/attacks/verify_sta_core.py

One wrinkle worth recording: the paper's published equation
and the paper's released code disagree. The equation counts
words; the code counts pairs of words, which is one fewer. We
follow the code, because the code produced the reference
numbers.

---

## The result

Sample 2, a Johnson Controls earnings article, 788 words.

| | |
|---|---|
| Score before | 2.82, flagged as watermarked |
| Score after | 2.00, not flagged |
| Words changed | 9 |
| Proportion of document | 1.1% |
| Meaning preserved | 99.98% |
| New grammar errors | 0 |

The nine changes:

    reports       ->  describes
    provides      ->  supplies
    additional    ->  extra
    maintain      ->  keep
    report        ->  describe
    benefit       ->  profit
    particularly  ->  especially
    probably      ->  likely
    also          ->  (deleted)

Every one is a word a copy editor might suggest. The text was
read in full and contains no grammatical errors, no misused
words, and no broken phrases.

For comparison, the original paper reports that rewriting text
entirely with GPT-3.5 reduces detection accuracy to 0.63. Nine
words achieve complete evasion.

---

## What this means

**Small pools beat large noisy ones.** The final run used 18
candidate words, down from 159 before filtering, and found
more working attacks at greater depth than the unfiltered
version. Bad candidates crowd out good ones in a beam search.

**Meaning similarity is not a quality check.** At one point
1,150 edits passed a 0.95 similarity threshold and only 400
survived a basic grammar check. Sentence embeddings measure
what a text is about. They are close to blind to whether it is
written correctly.

**Fifty attacks, one needs to work.** The defender must stop
every variant. The attacker picks the best one.

---

## An open question about the mathematics

The paper's proof of detection reliability assumes each green
verdict is independent of the others.

They are not. Pair 50 covers words 50 and 51. Pair 51 covers
words 51 and 52. They share a word. When one changes, both
change.

The proof adds up the individual variances and stops. The
terms describing how neighbouring pairs move together are
absent. This means the threshold of 2.0 does not carry the
statistical meaning the paper assigns to it.

The size of this gap has not yet been quantified. That is the
next piece of work.

---

## Running it

    python research/attacks/verify_sta_core.py
    python research/attacks/scan_samples.py
    python research/attacks/run_attack.py 1 --depth 14 --breaks 50 --fit 7.0

Needs nltk with WordNet, spacy with en_core_web_sm,
lemminflect, language-tool-python which needs Java, and a
local copy of Llama-2-7B.

| File | What it does |
|---|---|
| sta_core.py | The detector, verified against the original |
| edit_ops.py | Applying changes and converting back to text |
| candidates.py | Finding words worth changing |
| context_fit.py | Rejecting words that do not fit |
| fluency.py | Checking the text still reads naturally |
| beam.py | The search |
| run_attack.py | Entry point |

Every change is applied to the text, written out, and read
back before scoring, because that is what a real detector
does. It receives writing, not internal representations.

---

## Still to do

Sample 3 has not been run with the final filters. It is a
harder target: thirteen pairs to flip rather than twelve, from
a smaller pool. It may not break under these constraints,
which would itself be worth reporting.

The mathematics above needs working out properly.

The collocation problem from trial 6 deserves a real solution
rather than the workaround used here.
