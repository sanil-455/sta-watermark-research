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

Filtering happens once, before the search, so the cost is one
forward pass per candidate rather than per depth.
"""

import torch

WINDOW = 48  # context tokens either side


class ContextFit:
    def __init__(self, tokenizer, model, device):
        self.tok = tokenizer
        self.model = model
        self.device = device

    @torch.no_grad()
    def _logprob_at(self, ids, index):
        """
        log P(ids[index] | ids[:index]) under the model,
        using a local window to keep the pass cheap.
        """
        lo = max(0, index - WINDOW)
        hi = min(len(ids), index + WINDOW)
        window = ids[lo:hi]
        target = index - lo

        if target < 1:
            return None

        x = torch.tensor(
            [window], device=self.device
        )
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
        original = self._logprob_at(base_ids, pos)
        if original is None:
            return float("inf")

        swapped = list(base_ids)
        swapped[pos] = new_id

        replacement = self._logprob_at(swapped, pos)
        if replacement is None:
            return float("inf")

        total = original - replacement

        # Collocation term: how much less likely does the
        # FOLLOWING word become? Catches broken fixed phrases
        # where the replacement is plausible on its own:
        #   making -> doing   "doing some good progress"
        # "doing" scores fine alone, but "progress" collocates
        # with "make", not "do".
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
            e["fit_drop"] = float("inf")
            dropped.append(e)
            continue

        drop = fit.score(base_ids, e["pos"], e["new_id"])
        e["fit_drop"] = drop

        if drop <= max_drop:
            kept.append(e)
        else:
            dropped.append(e)

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
