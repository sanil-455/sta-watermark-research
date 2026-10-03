"""
Span-restricted black-box attack.

Everything before this scored the whole document, prompt
included. The paper evaluates the generated span alone, so no
edit here is placed below prompt_tokens.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "attacks_core"))

from sta_core import text_to_ids, is_green_pair, GAMMA
from candidates import build_pos_map, is_word_start
from edit_ops import make_edit
from candidates_bb import propose_substitutions, propose_deletions
from edit_ops import make_edit, edits_to_text, edits_conflict
from context_fit import ContextFit, BLOCKED
from expansions import EXPANSIONS

def propose_expansions(tokenizer, ids, start):
    """
    Replace a single token with a longer phrase meaning the same.

    Destroys the two pairs around the trigger AND raises gamma*T
    by 0.5(k-1) for a k-token replacement, so the numerator falls
    by 2 + 0.5(k-1). At k=5 that is -4.0, against -2.0 for a
    single-word substitution.

    No spacing rule: expansions land on different words across
    the span, so they do not cluster the way asides do.
    """
    out = []

    for i in range(start, len(ids) - 1):
        if not is_word_start(tokenizer, ids[i]):
            continue

        word = tokenizer.decode([ids[i]]).strip()
        key = word.lower()
        if key not in EXPANSIONS:
            continue

        for phrase in EXPANSIONS[key]:
            enc = tokenizer.encode(" " + phrase,
                                   add_special_tokens=False)
            if len(enc) < 2:
                continue
            out.append(
                make_edit(i, "replace", enc,
                          "%s->%s" % (word, phrase))
            )

    return out

def filter_span_pool(pool, ids, fit, max_drop=8.0, log=print):
    """
    Drop substitutions whose replacement does not fit context.

    Only substitutions are scored. Deletions have no replacement
    token to compare against, and clause insertions are multi
    token, which the scorer does not handle: their quality
    problem is placement, not word choice.
    """
    kept, cut = [], 0

    for e in pool:
        if e["type"] != "replace":
            kept.append(e)
            continue

        if e["label"] in BLOCKED:
            cut += 1
            continue

        nid = e["new_id"]
        if isinstance(nid, (list, tuple)):
            kept.append(e)
            continue

        score = fit.score(ids, e["pos"], nid)
        if score <= max_drop:
            e["fit_drop"] = score
            kept.append(e)
        else:
            cut += 1

    subs = sum(1 for e in kept if e["type"] == "replace")
    ins = sum(1 for e in kept if e["type"] == "insert")
    log("fit filter: %d kept (%d subs, %d ins), %d cut"
        % (len(kept), subs, ins, cut))

    kept.sort(key=lambda e: e.get("fit_drop", -99))
    return kept

def attack_span(oracle, tokenizer, ids, pool, n_prompt,
                target=2.0, max_edits=40, log=print):
    """
    Greedy attack on the generated span.

    The oracle returns whole-document z, which is what a real
    detector would report. Success is judged on span z, which
    is what the paper evaluates. The two move together, so
    ranking on document z still selects useful edits.

    Candidates are tried in pool order. The first that lowers
    document z is taken, then the span is re-measured.
    """
    base_text = tokenizer.decode(ids, skip_special_tokens=True)
    doc_z = oracle.query(base_text)
    span_z = gen_span_z(tokenizer, base_text, n_prompt)

    log("start: doc z=%.4f  span z=%.4f" % (doc_z, span_z))

    applied = []
    for step in range(max_edits):
        best = None

        for cand in pool:
            if any(edits_conflict(cand, a) for a in applied):
                continue

            trial = applied + [cand]
            text = edits_to_text(tokenizer, ids, trial)
            z = oracle.query(text)

            if z < doc_z:
                best = (cand, z, text)
                break

        if best is None:
            log("no candidate lowers z")
            break

        cand, doc_z, text = best
        applied.append(cand)
        if cand["type"] == "insert":
            pool = [
                e for e in pool
                if not (e["type"] == "insert"
                        and e["label"] == cand["label"])
            ]
        span_z = gen_span_z(tokenizer, text, n_prompt)

        log("  %2d: doc=%.4f span=%.4f  %s"
            % (len(applied), doc_z, span_z, cand["label"]))

        if span_z <= target:
            log("SPAN BROKEN: %d edits, %d queries"
                % (len(applied), oracle.queries))
            return applied, span_z, text

    log("stopped: span z=%.4f after %d edits" % (span_z, len(applied)))
    return applied, span_z, None

def build_pool(tokenizer, ids, n_prompt):
    """
    Every candidate, restricted to the generated span.

    Substitutions and deletions are filtered by position so no
    edit lands in the prompt. Clause insertions are added on top
    because substitution alone cannot reach the target on most
    documents: measured ceilings sit at or below the greens that
    must be removed.

    Returns the pool and a breakdown, since the counts explain
    why a document does or does not break.
    """
    pos_map = build_pos_map(tokenizer, ids)
    start = max(0, n_prompt - 1)

    subs = [
        e for e in propose_substitutions(ids, tokenizer, pos_map)
        if e["pos"] >= start
    ]
    dels = [
        e for e in propose_deletions(ids, tokenizer, pos_map)
        if e["pos"] >= start
    ]
    ins, n_slots = propose_clauses(tokenizer, ids, pos_map, start)
    exps = propose_expansions(tokenizer, ids, start)
    stats = {
        "substitutions": len(subs),
        "deletions": len(dels),
        "clause_slots": n_slots,
        "clause_candidates": len(ins),
    }
    stats = {
        "substitutions": len(subs),
        "deletions": len(dels),
        "clause_slots": n_slots,
        "clause_candidates": len(ins),
        "expansions": len(exps),
    }
    return subs + dels + exps + ins, stats

# Mid-sentence asides. Each follows a comma and ends with one,
# so no capitalisation is needed and the sentence resumes
# cleanly. Drawn from ordinary reporting hedges rather than
# filler words, which saturate the text when repeated.
CLAUSES = [
    "according to the filing,",
    "as the company noted,",
    "the report said,",
    "based on those figures,",
    "in the same period,",
    "by that measure,",
    "the statement added,",
    "over that stretch,",
    "as management put it,",
    "per the disclosure,",
    "the document shows,",
    "on that reading,",
]


def propose_clauses(tokenizer, ids, pos_map, start, spacing=15):
    """
    Every clause as a candidate at every usable slot.

    Slots are thinned so no two insertions land within `spacing`
    tokens. Clustered asides read as padding however natural each
    one is on its own.

    Returns the candidates and the number of slots kept, since
    that count is the binding constraint on how far insertion
    alone can go.
    """
    slots = clause_slots(tokenizer, ids, pos_map, start)

    thinned, last = [], -999
    for i in slots:
        if i - last >= spacing:
            thinned.append(i)
            last = i

    out = []
    for i in thinned:
        for c in CLAUSES:
            enc = tokenizer.encode(c, add_special_tokens=False)
            out.append(
                make_edit(i, "insert", enc, "ins(%s)" % c.rstrip(","))
            )

    return out, len(thinned)

def clause_slots(tokenizer, ids, pos_map, start):
    """
    Positions where a comma-led aside can go.

    Three conditions, all necessary:

      word start      so we never split a word into pieces.
                      Llama stores 'rebalanced' as re|bal|anced
                      and inserting at the middle piece would
                      produce nonsense.

      parse agrees    the spaCy token at this index must match
                      the actual token. A third of positions in
                      the span fail this, and the grammar
                      information there cannot be trusted.

      after a comma   so the clause needs no capitalisation and
                      the sentence reads continuously.
    """
    out = []
    for i in range(start + 1, len(ids) - 1):
        if not is_word_start(tokenizer, ids[i]):
            continue

        tok = pos_map.get(i)
        prev = pos_map.get(i - 1)
        if tok is None or prev is None:
            continue

        if tokenizer.decode([ids[i]]).strip().lower() != tok.text.lower():
            continue

        # A clause boundary, where an aside can sit without
        # needing a capital. Two kinds:
        #
        #   after a comma, but NOT one inside a list. spaCy
        #   marks list items with conj/appos dependencies, so
        #   a comma whose following token has one of those is
        #   skipped: that is what produced "with sharp,
        #   bulbous, as the company noted, bright orange".
        #
        #   before a subordinating conjunction, where an aside
        #   reads naturally and no comma is required.
        SUBORD = ("which", "while", "although", "because",
                  "since", "whereas", "though")

        if prev.text in (",", ";"):
            if tok.dep_ not in ("conj", "appos", "amod"):
                out.append(i)
        elif tok.text.lower() in SUBORD:
            out.append(i)

    return out

def gen_span_z(tokenizer, text, n_prompt):
    """
    z over the generated pairs only.

    A pair belongs to the generated span when its SECOND token
    was produced by the model, since that is the token the
    accept/resample step acted on. The first generated token is
    at index n_prompt, so the span starts at pair n_prompt - 1.

    Uses is_green_pair, which needs the key. This is measurement
    only: it tells us whether an attack worked. The attack
    itself never calls it.
    """
    ids = text_to_ids(tokenizer, text)
    start = max(0, n_prompt - 1)

    pairs = [
        int(is_green_pair(ids[i], ids[i + 1]))
        for i in range(start, len(ids) - 1)
    ]

    T = len(pairs)
    if T < 2:
        return 0.0

    G = sum(pairs)
    return (G - GAMMA * T) / math.sqrt(GAMMA * (1 - GAMMA) * T)

