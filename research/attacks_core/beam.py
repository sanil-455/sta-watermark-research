"""
Beam search over edit sets.

each of the below gates runs only in the specific order when it has survived the gate before it
this is because:
z rank essentially free,
semantic similarity one small NN pass
grammar checking takes no NN but takes time per doc
Perplexity req running entire 7B model
    z-rank      cheap, cached      -> rank_width
    semantic    one batched pass   -> sem_threshold
    grammar     LanguageTool       -> max_grammar_errors
    perplexity  Llama forward pass -> max_ppl_ratio

Cost ordering matters: perplexity is a full 7B forward pass,
so it must only see candidates that survived the cheap checks.
"""
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from sta_core import sta_stats, text_to_ids, Z_THRESHOLD
from fluency import Fluency
from edit_ops import (
    edits_conflict, edits_to_text, edit_signature, describe,
)
# model that takes a whole sentence or doc and produces a list of numbers (embeddings) vectors that captures its overall meaning
# this is the model behind sim score
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# its a rule based grammar and spell checker as it works from a large collection of hand-written linguistic rules
# not a NN
class Grammar:
    """
    LanguageTool errors relative to the original.
    Embedding similarity is close to blind to grammar:
    "capable to execute" scored 0.997. This catches some of
    what the embedding cannot, though not word sense.
    """
    def __init__(self):
        import language_tool_python
        self.tool = language_tool_python.LanguageTool("en-US")
        self.base = None
    # this records how many errors the original unedited doc contain,
    # then measures the errors in edited doc
    def set_reference(self, text):
        # self.tool.check(text) returns a list of every grammar issue LanguageTool finds; len(...) just counts them
        self.base = len(self.tool.check(text))
        return self.base

    # the above fn checks and this reports the number of extra errors between un edited and edited doc
    def extra_errors(self, text):
        return len(self.tool.check(text)) - self.base
# loads the minilm model and its tokenizer which is diffrent from LLama
class Semantic:
    def __init__(self, device=None):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.tok = AutoTokenizer.from_pretrained(EMBED_MODEL)
        self.model = AutoModel.from_pretrained(
            EMBED_MODEL
        ).to(self.device).eval() #.eval() means we are only using the model and not training it
    # kept padding true as tokenizing words of differnt length
    def embed(self, texts):
        enc = self.tok(
            texts, padding=True, truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            # models raw output vector with a whole vector for every single token in every text
            out = self.model(**enc)[0]
        # masking the padded tokens so taht not tokenization affects them
        mask = enc["attention_mask"].unsqueeze(-1).float()
        # multiply every token's vector by either 1 (keep it) or 0,then add up all the token vectors together
        # divide by the real (non-padding) tokens there actually were, giving a true average.
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        # L2 normalization, rescaling each document's vector so its overall length is exactly 1, without changing its direction
        # this is so as each vector has length exactly 1,explaining how similar 2 vectors are by dot product
        # thus no need to divide each time by each vector's length
        return F.normalize(pooled, p=2, dim=1)

    def similarity(self, reference, texts, batch=32):
        # refernce is the original text
        # embedding the whole ref doc at once
        # then process the list of candidate texts at batch of 32 so as to avoid a memory crash
        ref = self.embed([reference])
        scores = []

        for i in range(0, len(texts), batch):
            chunk = self.embed(texts[i:i + batch])
            # matrix multiplication between the batch of candidate vectors and the reference vector (transposed).
            # this directly gives the similarity score between each candidate and reference
            scores.extend(
                torch.mm(chunk, ref.T).squeeze(1).tolist()
            )
        return scores
# a state here means a specific combination of edits tried
# This fn tries adding one more edit, from the whole candidate pool, onto an existing state
def _extend(state, pool):
    """All valid one-edit extensions of a state."""
    for cand in pool:
        if any(edits_conflict(cand, e) for e in state):
            continue
        # yield and not return makes _extend a generator, producing one new extended state at a time, on demand, rather than building the entire list
        # prevents memory crash
        yield state + [cand]

def search(
    tokenizer,
    base_ids,
    pool,
    max_depth=8,# max number of edits at most to try at once
    beam=200, # how many states to carry forward and preevnt an exhaustive search
              # here what happpens is we keep top 200 edits in all possible 1 edit combinations
              # similarly top 200 in 2 all possible 2 edit combinations,3 edit all possible top 200 and so on
              # instead of doing all possible combinations brute force as this would be highly expensive and exhaustive(800 billion almost for 10 edits)
    rank_width=1200,
    sem_threshold=0.95,
    max_grammar_errors=0,
    max_ppl_ratio=1.05,
    max_breaks=50,
    fluency=None, # instead of loading LLaMa model each time it accepts teh existing and loads only when none there
    log=print,
):
    """
    Returns (breaks, best_state, history).

    Each element of `passed` is
        (z, edits, text, stats, similarity, ppl_ratio)

    Breaks are deduplicated by edit signature: many states
    share a core edit set and differ by one substitution,
    producing identical z. Only structurally distinct edit
    sets are recorded.
    """
    # builds the semantica nd grammar check objects afresh
    semantic = Semantic()
    grammar = Grammar()
    # Decode the original token IDs back into real text
    reference = tokenizer.decode(
        base_ids, skip_special_tokens=True
    )
    # Record the original document's grammar-error count, perplexity, and z-score
    base_errors = grammar.set_reference(reference)
    log(f"baseline grammar errors: {base_errors}")

    # perplexity is the degree of certainity (eg:for the next word generation of LLm)
    # low perplexity means high certainity
    # if high perplexity that means the word we used (for replacement) does not match well with the context or in the sentence ka sense
    if fluency is None:
        fluency = Fluency(tokenizer)
    base_ppl = fluency.set_reference(reference)
    log(f"baseline perplexity: {base_ppl:.3f}")

    base = sta_stats(base_ids)
    log(f"baseline z={base['z']:.6f} "
        f"green={base['green_count']}/{base['pair_count']}")
    log(f"pool={len(pool)} beam={beam} depth={max_depth} "
        f"ppl_max={max_ppl_ratio}")

    states = [[]] # search begins with an empty list of editsie. no chanegs made yet
    seen = {()}
    breaks = []
    # deduplicating sets,prevents the same underlying combination of edits (reached via two different search paths) from being recorded twice as separate breaks
    break_sigs = set()
    best = (base["z"], [])
    history = []

    for depth in range(1, max_depth + 1):
        scored = []

        for state in states:
            # for every sriving state try every valid one edit combination 
            for nxt in _extend(state, pool):
                sig = edit_signature(nxt)
                # skip anything already tried before
                if sig in seen:
                    continue
                seen.add(sig)
                # for everything genuinely new build the real edited text and retokenize them
                text = edits_to_text(tokenizer, base_ids, nxt)
                # and score them
                st = sta_stats(text_to_ids(tokenizer, text))
                scored.append((st["z"], nxt, text, st))
        # if every possible possible combination is conflicting or explored, stop it
        if not scored:
            log(f"depth {depth}: no new states")
            break
        # Gate 1: z score
        # sort every newly scored z score,lowest first
        scored.sort(key=lambda x: x[0])
        # keep only the top 1200 by deafult (rank width)
        head = scored[:rank_width]

        # Gate 2:sematic similarity
        # Batch-embed all 1200 surviving candidate texts at once,(it internally chunks this into manageble batches
        # and keeps the ones whose similarity matches
        sims = semantic.similarity(
            reference, [h[2] for h in head]
        )
        sem_ok = [
            (z, edits, text, st, sim)
            for (z, edits, text, st), sim in zip(head, sims)
            if sim >= sem_threshold
        ]
        
        # Gate 3: grammar check
        # check each candidate 1 at a time as Lnaguage tool offres no features of batch efficiency
        # it doesnt check every candidate through the gate, stops when it chcked beam*2 tokens
        # tradeoff to priortize speed over exhaustiveness
        gram_ok = []
        for row in sem_ok:
            if grammar.extra_errors(row[2]) <= max_grammar_errors:
                gram_ok.append(row)
            if len(gram_ok) >= beam * 2:
                break

        # most expensive check 
        # like upper fn this also stops early when it has chcked (beam) no.of candidates
        passed = []
        for z, edits, text, st, sim in gram_ok:
            ratio = fluency.ratio(text)
            if ratio <= max_ppl_ratio:
                passed.append(
                    (z, edits, text, st, sim, ratio)
                )
            if len(passed) >= beam:
                break
        # if nothing survived all 4 gates search sttops
        # this happened quite a few times when ppl died to 0
        if not passed:
            log(f"depth {depth}: {len(scored)} states | "
                f"{len(sem_ok)} sem | {len(gram_ok)} gram | "
                f"0 ppl")
            break

        # recording actual breaks
        # like z threshold break
        for z, edits, text, st, sim, ratio in passed:
            if len(breaks) >= max_breaks:
                break
            if z > Z_THRESHOLD:
                continue

            sig = edit_signature(edits)
            if sig in break_sigs:
                continue
            break_sigs.add(sig)

            breaks.append({
                "z": z,
                "sim": sim,
                "ppl_ratio": ratio,
                "depth": depth,
                "edits": describe(edits),
                "text": text,
                "green": st["green_count"],
                "pairs": st["pair_count"],
            })
        # tracking the best logging and continuing
        if passed[0][0] < best[0]:
            best = (passed[0][0], passed[0][1])

        history.append({
            "depth": depth,
            "generated": len(scored),
            "sem_ok": len(sem_ok),
            "gram_ok": len(gram_ok),
            "passed": len(passed),
            "best_z": passed[0][0],
            "best_sim": passed[0][4],
            "best_ppl_ratio": passed[0][5],
            "breaks": len(breaks),
        })

        log(f"depth {depth}: {len(scored):6d} states | "
            f"{len(sem_ok):4d} sem | {len(gram_ok):4d} gram | "
            f"{len(passed):4d} ppl | "
            f"best z={passed[0][0]:.6f} "
            f"(sim {passed[0][4]:.4f} "
            f"ppl {passed[0][5]:.3f}) | "
            f"breaks={len(breaks)}")

        # if enough breaks have been collected stop early :limit 50 breaks
        if len(breaks) >= max_breaks:
            log(f"stopping: {len(breaks)} breaks found")
            break
        states = [p[1] for p in passed[:beam]]
        
    return breaks, best, history
    # we stop grammar,perplexity check as measure of speed tradeoff as they take the longest time
