"""
Oracle-guided selective paraphrase.

Replacing a sentence destroys every pair that sentence takes
part in. For a 20-token sentence at 73% green that is about 15
greens, against 2 for a single word substitution. The rewrite
creates the same number of pairs, green at chance, so roughly
half come back. Generating many candidates and keeping whichever
the detector scores lowest pulls that down further.

This works where token-level editing could not: a 200-token span
holds too few grammatically safe single-word edits, while
sentences can be rewritten without limit.

Paraphrases come from Qwen2.5-7B-Instruct. Llama-2-7B base
reversed factual direction sometimes and the chat model reversed
it on every candidate; Qwen preserved it 6 of 6 on the same test.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "attacks_core"))

from content_checks import passes as content_passes

JUNK = re.compile(
    r"^(#|\(|[a-z]\.|-\s|I need|Original|Rewritten|Example|Note|"
    r"Here|Sure|Certainly|The following)",
    re.IGNORECASE,
)


def split_sentences(text):
    """
    Sentences with their character offsets.

    Splits on punctuation followed by whitespace then a capital
    or an opening quote. Crude, but a bad split only produces a
    worse candidate, which the gates then reject.
    """
    spans, start = [], 0
    for m in re.finditer(r"(?<=[.!?])\s+(?=[A-Z\"\u201c])", text):
        spans.append((start, m.start()))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))
    return [(a, b) for a, b in spans if b - a > 30]


def clean(cands, original):
    """Keep rewrites that are plausible sentences."""
    out = []
    for c in cands:
        c = c.strip().strip('"').strip()
        if JUNK.match(c) or len(c) < 15:
            continue
        if c.lower() == original.strip().lower():
            continue
        # a rewrite far longer or shorter has usually drifted
        if not 0.5 <= len(c) / max(1, len(original)) <= 2.0:
            continue
        out.append(c)
    return out

def structure_preserved(original, candidate):
    """
    Reject rewrites that restructure across sentence boundaries.

    Every failure in the audit restructured; every clean rewrite
    kept the sentence arrangement. Prompt 2 split one sentence
    into three and reinterpreted a typo as a new fact. Prompt 0
    merged a quoted tweet into third-person commentary and lost
    a hashtag.

    The semantic filter cannot catch this, because embeddings
    are largely order-insensitive, so all three passed at 0.60.

    Two tests:

      sentence count must match, since splitting or merging is
      where reinterpretation happens

      length within 0.75 to 1.35, tighter than the 0.5 to 2.0
      used by clean(), because compressing a long sentence into
      a short one is summarising rather than paraphrasing
    """
    if len(split_sentences(original)) != len(split_sentences(candidate)):
        return False

    ratio = len(candidate) / max(1, len(original))
    return 0.65 <= ratio <= 1.15

def semantic_filter(semantic, original, cands, threshold):
    """
    Drop rewrites that changed the meaning.

    Compares against the original SENTENCE, not the document.
    Document-level similarity is useless here: MiniLM truncates
    at 512 tokens, so on a long document it only ever embeds the
    prompt and returns 1.0000 regardless of what changed.
    """
    if not cands:
        return []
    sims = semantic.similarity(original, cands)
    keep = [(c, s) for c, s in zip(cands, sims) if s >= threshold]
    keep.sort(key=lambda x: -x[1])
    return [c for c, _ in keep]


def fluency_filter(para, cands, factor=1.3):
    """
    Drop rewrites whose word order is wrong.

    Embeddings are largely order-insensitive, so a scrambled
    sentence keeps its content words and scores high on
    similarity. Perplexity sees order directly.

    The cutoff is relative to the median of this sentence's own
    candidates, because absolute perplexity varies hugely
    between sentences and between models.
    """
    if len(cands) <= 5:
        return cands
    scored = [(para.perplexity(c), c) for c in cands]
    med = sorted(p for p, _ in scored)[len(scored) // 2]
    return [c for p, c in scored if p <= med * factor]


def attack(oracle, para, full_text, n_prompt_chars, span_z,
           semantic=None, n_cands=60, max_sents=14, target=2.0,
           sem_threshold=0.88, log=print):
    """
    Rewrite sentences until the generated span falls below target.

    Sentence offsets are recomputed every round because a rewrite
    changes the text length and shifts everything after it.

    Only sentences starting at or after n_prompt_chars are
    touched, so the attack stays inside the generated span.
    """
    text = full_text
    applied = []

    for idx in range(max_sents):
        sents = [s for s in split_sentences(text)
                 if s[0] >= n_prompt_chars]
        if idx >= len(sents):
            break

        a, b = sents[idx]
        original = text[a:b]
	
        # A real quotation cannot be faithfully paraphrased:
        # rewriting the words inside changes what someone said.
        # But a quoted TITLE is just a name -- "Mad Money",
        # "Gibberish" -- and skipping those cost six of nine
        # sentences across the three worst documents.
        #
        # Treat a quote as speech only when it runs past a few
        # words, which titles rarely do.
        quoted = re.findall(r'"([^"]+)"|\u201c([^\u201d]+)\u201d',
                            original)
        speech = any(len((a or b).split()) > 4 for a, b in quoted)
        if speech:
            log("  sentence %d: contains speech, skipped" % idx)
            continue
        cands = clean(para.rewrite(original, n=n_cands), original)
        if semantic is not None:
            cands = semantic_filter(semantic, original, cands,
                                    sem_threshold)
        cands = [c for c in cands if content_passes(original, c)]
        cands = fluency_filter(para, cands)
        cands = [c for c in cands if content_passes(original, c)]
        # Structure check tried and removed. It fixed prompt 22's
        # "job gave her freedom" inversion but cost prompts 25 and
        # 27, both clean without it: 6 evading / 3 clean became
        # 5 evading / 2 clean.
        # cands = [c for c in cands if structure_preserved(original, c)]
        cands = fluency_filter(para, cands)
        if not cands:
            log("  sentence %d: nothing survives the gates" % idx)
            continue

        best = None
        for c in cands:
            trial = text[:a] + c + text[b:]
            z = oracle.query(trial)
            if best is None or z < best[0]:
                best = (z, c, trial)

        _, chosen, trial = best
        text = trial
        applied.append({"original": original, "rewritten": chosen})

        sz = span_z(text)
        log("  sentence %d: %d cands, span z=%.4f" % (idx, len(cands), sz))

        if sz <= target:
            log("SPAN BROKEN: %d sentences, %d queries"
                % (len(applied), oracle.queries))
            return applied, sz, text

    sz = span_z(text)
    log("stopped at span z=%.4f after %d sentences" % (sz, len(applied)))
    return applied, sz, None

def edit_metrics(tokenizer, original, attacked, n_prompt_chars,
                 n_sentences):
    """
    How much of the generated span changed, in three units.

    Sentences is the natural unit for paraphrase and matches how
    DIPPER and GPT-3.5 attacks are reported. Tokens is comparable
    to the copy-paste attack, which the paper reports at 25%
    replacement. Characters is independent of tokenisation.

    Measured over the generated span only. Including the prompt
    would divide by a much larger denominator and understate the
    edit rate, since the prompt is never touched.
    """
    import difflib

    o = original[n_prompt_chars:]
    a = attacked[n_prompt_chars:]

    ot = tokenizer.encode(o, add_special_tokens=False)
    at = tokenizer.encode(a, add_special_tokens=False)

    sm = difflib.SequenceMatcher(None, o, a)
    # for a replacement, count the larger of removed or added,
    # so a rewrite of equal length is not understated
    changed = sum(max(i2 - i1, j2 - j1)
                  for tag, i1, i2, j1, j2 in sm.get_opcodes()
                  if tag != "equal")

    total = len(split_sentences(o))

    return {
        "sentences_changed": n_sentences,
        "sentences_total": total,
        "sentence_pct": 100.0 * n_sentences / max(1, total),
        "tokens_before": len(ot),
        "tokens_after": len(at),
        "token_delta_pct": 100.0 * (len(at) - len(ot)) / max(1, len(ot)),
        "chars_changed_pct": 100.0 * changed / max(1, len(o)),
    }
