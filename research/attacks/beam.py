"""
Beam search over edit sets.

At each depth every surviving state is extended by one edit.
States are ranked by true z (measured after decode/re-encode),
then the best are semantically gated and carried forward.

Semantic scoring is the expensive step, so it runs only on
states that already rank well by z.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from sta_core import sta_stats, text_to_ids, Z_THRESHOLD
from edit_ops import (
    edits_conflict, edits_to_text, edit_signature, describe,
)

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Grammar:
    """
    Counts LanguageTool errors relative to the original.

    Embedding similarity measures topical overlap and is
    blind to grammar: "capable to execute" scored 0.997.
    This catches what the embedding cannot.
    """

    def __init__(self):
        import language_tool_python
        self.tool = language_tool_python.LanguageTool("en-US")
        self.base = None

    def set_reference(self, text):
        self.base = len(self.tool.check(text))
        return self.base

    def extra_errors(self, text):
        return len(self.tool.check(text)) - self.base


class Semantic:
    def __init__(self, device=None):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.tok = AutoTokenizer.from_pretrained(EMBED_MODEL)
        self.model = AutoModel.from_pretrained(
            EMBED_MODEL
        ).to(self.device).eval()

    def embed(self, texts):
        enc = self.tok(
            texts, padding=True, truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            out = self.model(**enc)[0]

        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return F.normalize(pooled, p=2, dim=1)

    def similarity(self, reference, texts, batch=32):
        ref = self.embed([reference])
        scores = []

        for i in range(0, len(texts), batch):
            chunk = self.embed(texts[i:i + batch])
            scores.extend(
                torch.mm(chunk, ref.T).squeeze(1).tolist()
            )

        return scores


def _extend(state, pool):
    """All valid one-edit extensions of a state."""
    for cand in pool:
        if any(edits_conflict(cand, e) for e in state):
            continue
        yield state + [cand]


def search(
    tokenizer,
    base_ids,
    pool,
    max_depth=8,
    beam=200,
    rank_width=1200,
    sem_threshold=0.95,
    max_grammar_errors=0,
    max_breaks=5,
    log=print,
):
    """
    Returns (breaks, best_state, history).

    `rank_width` controls how many top-z states get semantic
    scoring each depth. Wider costs more but recovers states
    that rank well on z yet need the gate to confirm.
    """
    semantic = Semantic()
    grammar = Grammar()
    reference = tokenizer.decode(
        base_ids, skip_special_tokens=True
    )
    base_errors = grammar.set_reference(reference)
    log(f"baseline grammar errors: {base_errors}")

    base = sta_stats(base_ids)
    log(f"baseline z={base['z']:.6f} "
        f"green={base['green_count']}/{base['pair_count']}")
    log(f"pool={len(pool)} beam={beam} depth={max_depth}")

    states = [[]]
    seen = {()}
    breaks = []
    best = (base["z"], [])
    history = []

    for depth in range(1, max_depth + 1):
        scored = []

        for state in states:
            for nxt in _extend(state, pool):
                sig = edit_signature(nxt)
                if sig in seen:
                    continue
                seen.add(sig)

                text = edits_to_text(tokenizer, base_ids, nxt)
                st = sta_stats(text_to_ids(tokenizer, text))
                scored.append((st["z"], nxt, text, st))

        if not scored:
            log(f"depth {depth}: no new states")
            break

        scored.sort(key=lambda x: x[0])
        head = scored[:rank_width]

        sims = semantic.similarity(
            reference, [h[2] for h in head]
        )

        sem_ok = [
            (z, edits, text, st, sim)
            for (z, edits, text, st), sim in zip(head, sims)
            if sim >= sem_threshold
        ]

        # Grammar runs only on states that already passed
        # the semantic gate, keeping the volume low.
        passed = []
        for z, edits, text, st, sim in sem_ok:
            if grammar.extra_errors(text) <= max_grammar_errors:
                passed.append((z, edits, text, st, sim))
            if len(passed) >= beam * 2:
                break

        if not passed:
            log(f"depth {depth}: {len(scored)} states, "
                f"{len(sem_ok)} passed semantic, "
                f"none passed grammar")
            break

        for z, edits, text, st, sim in passed:
            if z <= Z_THRESHOLD:
                breaks.append({
                    "z": z, "sim": sim, "depth": depth,
                    "edits": describe(edits), "text": text,
                    "green": st["green_count"],
                    "pairs": st["pair_count"],
                })

        if passed[0][0] < best[0]:
            best = (passed[0][0], passed[0][1])

        history.append({
            "depth": depth,
            "generated": len(scored),
            "passed": len(passed),
            "best_z": passed[0][0],
            "best_sim": passed[0][4],
            "breaks": len(breaks),
        })

        log(f"depth {depth}: {len(scored):6d} states | "
            f"{len(sem_ok):4d} sem | {len(passed):4d} gram | "
            f"best z={passed[0][0]:.6f} "
            f"(sim {passed[0][4]:.4f}) | "
            f"breaks={len(breaks)}")

        if len(breaks) >= max_breaks:
            log(f"stopping: {len(breaks)} breaks found")
            break

        states = [p[1] for p in passed[:beam]]

    return breaks, best, history
