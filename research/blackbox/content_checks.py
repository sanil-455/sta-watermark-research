"""
Content checks for paraphrase candidates.

Added after reading the first sweep, where three rewrites
evaded detection but corrupted meaning in ways no existing
gate could see.
"""

import re

# Quotes at ANY length. The earlier pattern required 10+
# characters and so missed "Hello Fiona!" becoming
# "Hello, Twitter!".
FACT_PATTERNS = [
    r"\d+(?:[.,]\d+)?%",
    r"\$\s?[\d.,]+",
    r"\b\d{4}\b",
    r"\b\d+(?:[.,]\d+)?\b",
    r'"[^"]+"',
    "\u201c[^\u201d]+\u201d",
]

# Dropping one of these inverts the claim while moving an
# embedding hardly at all.
NEGATIONS = {
    "not", "no", "never", "none", "neither", "nor", "cannot",
    "without", "nothing", "nobody", "nowhere", "hardly",
    "scarcely", "barely", "n't",
}

# These carry a relationship. "without a change in X" is not
# "without X".
RELATIONAL = {
    "in", "of", "to", "from", "with", "without", "within",
    "between", "among", "through", "across", "against",
    "under", "over", "before", "after", "during", "since",
}


def extract_facts(text):
    """Every factual token, as a sorted multiset."""
    out = []
    for pat in FACT_PATTERNS:
        out.extend(re.findall(pat, text))
    return sorted(out)


def count_words(text, vocab):
    """
    How often each word from vocab appears.

    Contractions are handled separately: "won't" tokenises as
    one word, so the n't suffix is counted on its own.
    """
    counts = {}
    for w in re.findall(r"[a-z']+", text.lower()):
        if w in vocab:
            counts[w] = counts.get(w, 0) + 1
        if w.endswith("n't"):
            counts["n't"] = counts.get("n't", 0) + 1
    return counts


def passes(original, candidate):
    """
    True when the candidate preserves the original's content.

    facts      exact match both ways, so a rewrite that invents
               a figure is as wrong as one that drops it
    negation   exact count match, since dropping one inverts
               the claim
    relations  prepositions may be added but not dropped; a
               rewrite is allowed to be more explicit
    """
    if extract_facts(original) != extract_facts(candidate):
        return False

    if count_words(original, NEGATIONS) != count_words(candidate, NEGATIONS):
        return False

    o_rel = count_words(original, RELATIONAL)
    c_rel = count_words(candidate, RELATIONAL)
    for word, n in o_rel.items():
        if c_rel.get(word, 0) < n:
            return False

    return True
