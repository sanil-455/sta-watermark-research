"""
Greedy black-box attack with quality gates.

The gates need no key and make no detector queries. They run on
public models the attacker already has, so filtering costs
compute but not query budget. That separation matters: rate
limiting a detector does not stop an attacker from producing
clean text, it only limits how fast they can find which edits
to make.

Gates, cheapest first, matching beam.py's ordering:
    semantic   MiniLM cosine against the original
    grammar    LanguageTool, no new errors allowed
    perplexity Llama ratio against the original
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "attacks_core"))

from edit_ops import edits_conflict, edits_to_text
from beam import Semantic, Grammar
from fluency import Fluency


def greedy_filtered(oracle, tokenizer, base_ids, ranked,
                    sem_threshold=0.95, max_grammar_errors=0,
                    max_ppl_ratio=1.08, max_edits=40,
                    fluency=None, log=print):
    """
    Same greedy loop, but a candidate must pass all three gates
    before it is accepted.

    Query cost is unchanged from the unfiltered version: gates
    reject candidates without ever consulting the detector.
    """
    base_text = tokenizer.decode(base_ids, skip_special_tokens=True)

    semantic = Semantic()
    grammar = Grammar()
    if fluency is None:
        fluency = Fluency(tokenizer)

    base_errors = grammar.set_reference(base_text)
    base_ppl = fluency.set_reference(base_text)
    log(f"baseline grammar errors = {base_errors}")
    log(f"baseline perplexity     = {base_ppl:.3f}")

    applied = []
    current_z = oracle.query(base_text)
    log(f"start z = {current_z:.6f}")

    rejected = {"conflict": 0, "no_help": 0,
                "semantic": 0, "grammar": 0, "ppl": 0}

    for step in range(max_edits):
        best = None

        for edit, _, _ in ranked:
            if any(edits_conflict(edit, a) for a in applied):
                rejected["conflict"] += 1
                continue

            trial = applied + [edit]
            text = edits_to_text(tokenizer, base_ids, trial)

            z = oracle.query(text)
            if z >= current_z:
                rejected["no_help"] += 1
                continue

            sim = semantic.similarity(base_text, [text])[0]
            if sim < sem_threshold:
                rejected["semantic"] += 1
                continue

            if grammar.extra_errors(text) > max_grammar_errors:
                rejected["grammar"] += 1
                continue

            ratio = fluency.ratio(text)
            if ratio > max_ppl_ratio:
                rejected["ppl"] += 1
                continue

            best = (edit, z, text, sim, ratio)
            break

        if best is None:
            log("no candidate passes all gates")
            break

        edit, z, text, sim, ratio = best
        applied.append(edit)
        current_z = z

        log(f"  edit {len(applied):2d}: z={z:.6f} sim={sim:.4f} "
            f"ppl={ratio:.3f}  {edit['label']}")

        if oracle.is_broken(text):
            log(f"BROKEN: {len(applied)} edits, "
                f"{oracle.queries} queries")
            log(f"rejections: {rejected}")
            return applied, current_z, text

    log(f"stopped at z={current_z:.6f}, {len(applied)} edits")
    log(f"rejections: {rejected}")
    return applied, current_z, None
