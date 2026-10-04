"""
Oracle-guided selective paraphrase.

Replacing a sentence destroys every pair that sentence
participates in. For a 20-token sentence at 73% green that is
15.3 greens. The replacement creates the same number of pairs,
green at chance, so about 10.5 return. Generating many
candidates and keeping the one the detector scores lowest pulls
that to roughly 4, for a net of about -11 per sentence.

Three sentences then suffice where 16 single substitutions would
have been needed, and unlike synonyms, paraphrases never run out.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "attacks_core"))

from sta_core import text_to_ids
from paraphrase import paraphrase
from beam import Semantic
from content_checks import passes as content_passes

# Discard model output that echoes the prompt or drifts into
# commentary. Llama-2-7B base is not instruction tuned, so a
# fraction of samples come back as list markers or meta text.
JUNK = re.compile(
    r"^(#|\(|[a-z]\.|-\s|I need|Original|Rewritten|Example|Note)",
    re.IGNORECASE,
)

# Things a paraphrase must carry across unchanged. A rewrite may
# change how something is said, never what is said.
#
# This exists because span similarity does not catch factual
# drift: a rewrite scoring 0.91 turned "Europe, Middle East and
# Africa produced 27%" into "representing the remaining 40%".
# Embeddings measure topical overlap, not accuracy.
FACT_PATTERNS = [
    r"\d+(?:[.,]\d+)?%",          # percentages
    r"\$\s?[\d.,]+",              # currency
    r"\b\d{4}\b",                 # years
    r"\b\d+(?:[.,]\d+)?\b",       # any number
    r'"[^"]{10,}"',               # direct quotations
]


def extract_facts(text):
    """
    Every factual token a paraphrase must preserve.

    Returned as a multiset so a candidate that drops one of two
    occurrences of the same figure is rejected.
    """
    out = []
    for pat in FACT_PATTERNS:
        out.extend(re.findall(pat, text))
    return sorted(out)


def facts_preserved(original, candidate):
    """
    True when the candidate carries every fact from the original.

    Extra facts are also rejected: a rewrite that invents a
    figure is as wrong as one that drops it.
    """
    return extract_facts(original) == extract_facts(candidate)

def clean(cands, original):
    """Keep rewrites that are plausible sentences."""
    out = []
    for c in cands:
        c = c.strip().strip('"').strip()
        if JUNK.match(c):
            continue
        if len(c) < 15:
            continue
        if c.lower() == original.strip().lower():
            continue
        # a rewrite far longer or shorter than the original has
        # usually drifted rather than paraphrased
        if not 0.5 <= len(c) / max(1, len(original)) <= 2.0:
            continue
        out.append(c)
    return out

def semantic_filter(semantic, original, cands, threshold=0.75):
    """
    Drop paraphrases that changed the meaning.

    Compares each candidate against the original SENTENCE, not
    the whole document. Document-level similarity is useless
    here: one rewritten sentence out of thirty barely moves it,
    which is why an inverted meaning slipped through earlier
    ("this isn't rocket science" became "rocket science is hard
    to do").

    Threshold 0.75 is deliberately permissive. The bar is that
    the sentence still says the same thing, not that it says it
    the same way.
    """
    if not cands:
        return []

    sims = semantic.similarity(original, cands)
    keep = [(c, s) for c, s in zip(cands, sims) if s >= threshold]
    keep.sort(key=lambda x: -x[1])
    return [c for c, _ in keep]

def fluency_filter(fluency, cands, factor=1.3):
    """
    Drop rewrites whose word order is wrong.

    The semantic filter cannot catch these. Embeddings are
    largely order-insensitive, so a scrambled sentence shares
    its content words with the original and scores high:
    "This isn't rocket science is important manifestation"
    passed at 0.85.

    Perplexity measures how expected each word is given what
    came before, so word order is precisely what it sees.

    The cutoff is relative to the median of this sentence's own
    candidates rather than absolute, because perplexity varies
    hugely between sentences: one carrying a long quotation
    scores nothing like a plain declarative.
    """
    if len(cands) <= 2:
        return cands

    scored = [(fluency.perplexity(c), c) for c in cands]
    med = sorted(p for p, _ in scored)[len(scored) // 2]
    return [c for p, c in scored if p <= med * factor]
def split_sentences(text):
    """
    Break text into sentences with their character offsets.

    Splitting on punctuation followed by a space and a capital.
    Crude, but it does not need to be perfect: a bad split just
    produces a worse paraphrase, which the oracle then rejects.
    """
    spans, start = [], 0
    for m in re.finditer(r"(?<=[.!?])\s+(?=[A-Z])", text):
        spans.append((start, m.start()))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))
    return [(a, b) for a, b in spans if b - a > 30]


def attack(oracle, tokenizer, model, full_text, n_prompt_chars,
           n_cands=40, max_sents=6, target=2.0, span_z=None,
           semantic=None, sem_threshold=0.75,log=print,fluency=None):
    """
    Paraphrase sentences one at a time, best candidate first.

    Only sentences beginning after n_prompt_chars are touched, so
    the attack stays inside the generated span.

    Each candidate costs one detector query. The best is kept and
    the text updated before moving to the next sentence.
    """
    text = full_text
    sents = [s for s in split_sentences(text) if s[0] >= n_prompt_chars]
    log("sentences in generated span: %d" % len(sents))

    applied = []

    for idx in range(min(max_sents, len(sents))):
        # offsets shift as we rewrite, so recompute each round
        sents = [s for s in split_sentences(text)
                 if s[0] >= n_prompt_chars]
        if idx >= len(sents):
            break

        a, b = sents[idx]
        original = text[a:b]

        raw = paraphrase(model, tokenizer, original, n=n_cands)
        cands = clean(raw, original)
        if semantic is not None:
            before = len(cands)
            cands = semantic_filter(semantic, original, cands,
                                    sem_threshold)
            log("  sentence %d: %d cands, %d pass semantics"
                % (idx, before, len(cands)))
        before = len(cands)
        cands = [c for c in cands if content_passes(original, c)]

        if before != len(cands):
            log("  sentence %d: %d pass content" % (idx, len(cands)))
        if fluency is not None:
            before = len(cands)
            cands = fluency_filter(fluency, cands)
            if before != len(cands):
                log("  sentence %d: %d pass fluency"
                    % (idx, len(cands)))
        if not cands:
            log("  sentence %d: no usable paraphrase" % idx)
            continue

        best = None
        for c in cands:
            trial = text[:a] + c + text[b:]
            z = oracle.query(trial)
            if best is None or z < best[0]:
                best = (z, c, trial)

        z, chosen, trial = best
        sz = span_z(trial) if span_z else None

        text = trial
        applied.append({"original": original, "rewritten": chosen})

        log("  sentence %d: %d cands, doc z=%.4f%s"
            % (idx, len(cands), z,
               "" if sz is None else ", span z=%.4f" % sz))

        if sz is not None and sz <= target:
            log("SPAN BROKEN after %d sentences, %d queries"
                % (len(applied), oracle.queries))
            return applied, sz, text

    final = span_z(text) if span_z else None
    log("stopped: span z=%s after %d sentences"
        % ("n/a" if final is None else "%.4f" % final, len(applied)))
    return applied, final, None

def edit_metrics(tokenizer, original, attacked, n_prompt_chars,
                 n_sentences_changed):
    """
    How much of the text changed, for comparison against published
    attacks.

    Three units, because they say different things:

      sentences   the natural unit for paraphrase, and what
                  DIPPER and GPT-3.5 attacks are measured in
      tokens      comparable to the copy-paste attack, which the
                  STA paper reports at 25% replacement
      characters  independent of tokenisation, so it survives a
                  change of model

    All are measured over the GENERATED SPAN only. Including the
    prompt would understate the edit rate, since the prompt is
    never touched.
    """
    import difflib

    orig_span = original[n_prompt_chars:]
    att_span = attacked[n_prompt_chars:]

    orig_tok = tokenizer.encode(orig_span, add_special_tokens=False)
    att_tok = tokenizer.encode(att_span, add_special_tokens=False)

    # character-level difference, counting both sides of each edit
    sm = difflib.SequenceMatcher(None, orig_span, att_span)
    changed_chars = sum(
        max(i2 - i1, j2 - j1)
        for tag, i1, i2, j1, j2 in sm.get_opcodes()
        if tag != "equal"
    )

    total_sents = len(split_sentences(orig_span))

    return {
        "sentences_changed": n_sentences_changed,
        "sentences_total": total_sents,
        "sentence_pct": (100.0 * n_sentences_changed / total_sents
                         if total_sents else 0.0),
        "tokens_before": len(orig_tok),
        "tokens_after": len(att_tok),
        "token_delta_pct": (100.0 * (len(att_tok) - len(orig_tok))
                            / max(1, len(orig_tok))),
        "chars_changed_pct": (100.0 * changed_chars
                              / max(1, len(orig_span))),
    }
