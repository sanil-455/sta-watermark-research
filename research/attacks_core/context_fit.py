"""
Contextual fit scoring for substitution candidates.

WordNet guarantees a candidate MEANS roughly the same thing.
It does not guarantee the candidate FITS. All of these are
valid dictionary synonyms that read wrong in place:

    announced -> denoted    "it denoted its purpose"
    used      -> applied    "you've probably applied its products"
    needs     -> asks       "lower working capital asks"
    intent    -> purpose    "announced its purpose to invest"
    companion -> familiar   "on the familiar audio-CD"

This scores each candidate by its conditional log-probability
at the edit position, given the full surrounding context, and
keeps those within `max_drop` nats of the original word.

# nat- unit we get after taking natural log to base e

Filtering happens once, before the search, so the cost is one
forward pass per candidate rather than per depth.
"""

import torch

WINDOW = 48  # context tokens either side to decide a given word's relevance

class ContextFit:
    def __init__(self, tokenizer, model, device):
        self.tok = tokenizer
        self.model = model
        self.device = device

    @torch.no_grad()
    def _logprob_at(self, ids, index):
        """
        log P(ids[index] | ids[:index]) under the model,: ie. prob of the tok at cur idx given the tokens before it
        using a local window to keep the pass cheap.
        """
        lo = max(0, index - WINDOW) # 48 tokens before index 
        hi = min(len(ids), index + WINDOW) # 48 tokens after index
        window = ids[lo:hi]
        target = index - lo

        # if the word is close enough to the beginning of the doc
        if target < 1:
            return None
        # converting normal py no.s to tensors
        x = torch.tensor(
            [window], device=self.device
        )
        # forward pass of NN that gibves a list of numbers 1 for each word
        logits = self.model(x).logits[0]
        logprobs = torch.log_softmax(
            logits[target - 1].float(), dim=-1
        )
        return float(logprobs[window[target]])

    def score(self, base_ids, pos, new_id):
        """
        Log-probability drop from substituting new_id at pos.

        Returns a positive number: how many nats less likely
        the replacement is than the original in this context.
        Smaller is better. Negative means the replacement is
        actually more expected than the original.
        """
        # how likely the model thought the orginal word was at this position in un edited text
        original = self._logprob_at(base_ids, pos)
        # if measurement fails return +inf as then its guaranteed this candidate gets rejected by any threshold
        if original is None:
            return float("inf")
        # make a copy of the token list and swap in the candidate word: ie. what if we make the edit version, just to measure
        swapped = list(base_ids)
        swapped[pos] = new_id

        # now measure how likely the model thinks the new word os
        replacement = self._logprob_at(swapped, pos)
        if replacement is None:
            return float("inf")

        total = original - replacement

        #  how much less likely does the
        # FOLLOWING word become? Catches broken fixed phrases
        # where the replacement is plausible on its own:
        #  making -> doing   "doing some good progress"
        # "doing" scores fine alone, but "progress" collocates
        # with "make", not "do".

        # as given the case above, this only checks 1 word ahead
        # if in case the phrase is longer it FAILS
        if pos + 1 < len(base_ids):
            after_orig = self._logprob_at(base_ids, pos + 1)
            after_swap = self._logprob_at(swapped, pos + 1)
            if after_orig is not None and after_swap is not None:
                total += after_orig - after_swap

        return total

# Candidates the contextual scorer rates well but that are
# wrong in sense rather than merely clumsy. The scorer works
# on local likelihood, so a word can look plausible to the
# model and still mean the wrong thing:
#
#   companion -> familiar   "on the familiar audio-CD"
#     scores 2.67, a good fit, but a familiar is a witch's
#     attendant spirit. The companion CD is a companion.
BLOCKED = {
    # A familiar is a witch's attendant spirit. The companion
    # CD is a companion. Scores 2.67, a good local fit, and
    # still the wrong word.
    "companion->familiar",

    # "he's went from pornography to an exploration" needs the
    # past participle, gone. spaCy tags the original moved as
    # simple past, so the inflection matched the wrong form
    # and LanguageTool did not catch it.
    "moved->went",

    # "the companion sound-CD". The compound is audio-CD; the
    # replacement breaks a hyphenated term rather than a word.
    "audio->sound",

    # "as he makes two critical milestones". You reach or hit
    # a milestone. Make does not collocate with it. The same
    # position offers reaches->hits, which is correct.
    "reaches->makes",
    
    # "doing some good progress". You make progress, not do it.
    # The fit scorer looks one token ahead, so it cannot see that
    # the word it breaks sits three positions later.
    "making->doing",

    # Collocation breaks. The fit scorer looks one token ahead,
    # so it cannot see the word further on that these destroy:
    #   make a plan, put a hold on, cast a ballot, have a margin
    "make->do",
    "putting->setting",
    "cast->throw",
    "have->hold",

    # "aid you charter the boat". Help takes a bare infinitive,
    # aid does not -- it needs "aid you in chartering".
    "help->aid",

    # "a cave construction". Construction is the act of building,
    # not the thing built.
    "structure->construction",
}


def filter_pool(pool, base_ids, fit, max_drop=4.0, log=print):
    """
    Keep candidates whose contextual fit is within max_drop.

    Deletions pass through unscored: there is no replacement
    token to evaluate, and the DROPPABLE allow-list already
    constrains them.

    max_drop is in nats. 4.0 means the replacement may be up
    to e^4 ~ 55x less likely than the original word in this
    position. Loose enough to admit real synonyms, tight
    enough to reject words that do not belong.
    """
    kept, dropped = [], []

    for e in pool:
        if e["type"] != "replace":
            kept.append(e)
            continue

        if e["label"] in BLOCKED:
            # setting it to inf so that all those rejected can ahve a common number
            e["fit_drop"] = float("inf")
            dropped.append(e)
            continue
        # eevrything that passes store the result in edit dict as a new key
        drop = fit.score(base_ids, e["pos"], e["new_id"])
        e["fit_drop"] = drop
        # if the score is at or below threshold,keep it,else reject
        if drop <= max_drop:
            kept.append(e)
        else:
            dropped.append(e)
    # sort everything else by predicted drop, use fit_drop as tiebreaker for everything else
    kept.sort(key=lambda e: (-e["drop"], e.get("fit_drop", 0)))

    log(f"contextual fit: {len(kept)} kept, "
        f"{len(dropped)} rejected (max_drop={max_drop})")

    if dropped:
        worst = sorted(
            dropped, key=lambda e: -e["fit_drop"]
        )[:5]
        log("  worst rejected: " + ", ".join(
            f"{e['label']} ({e['fit_drop']:.1f})"
            for e in worst
        ))

    return kept
# totoal= original-replacemnt and not other way as log probs are always negative and this means
# positive number makes replacemnt look worse while it is not
# deletion path skips the scoring entirely as there are no words to compare against
