"""
Check Module 3 on real sentences.

Run:  python research/qwen_attack/check_paraphraser.py

Loads Qwen on the GPU (stop anything else that uses it), prints the
model facts the final report needs, then rewrites the longest eligible
sentence of three real spans, 20 candidates each, and shows what came
back. Nothing is filtered and nothing is saved.
"""

import os
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FILES = ["results/raw/baseline_safe.json", "results/raw/baseline_extra.json",
         "results/raw/samples_len200.json", "results/raw/samples_len500.json"]
PICK = ("baseline_safe:1", "samples_len200:44", "samples_len500:34")
N = 20


def pick_sentences(sample_list, ids):
    """The longest eligible segment of each named span: (sample id, index, text)."""
    import sentences
    by_id = {s.sample_id: s for s in sample_list}
    items = []
    for sid in ids:
        if sid in by_id:
            eligible = sentences.split_span(by_id[sid].span).eligible()
            if eligible:
                g = max(eligible, key=lambda x: len(x.text))
                items.append((sid, g.index, g.text))
    return items


def show(d, indent=2):
    lines = []
    for k, v in d.items():
        if isinstance(v, dict):
            lines.append(" " * indent + "%s:" % k)
            lines += show(v, indent + 2)
        else:
            lines.append(" " * indent + "%-38s %s" % (k, v))
    return lines


def run_check(p, items, n=N, seed=1, memory_fn=None):
    out = ["MODEL FACTS"] + show(p.describe())
    for sid, idx, sentence in items:
        cands, rec = p.rewrite(sentence, n=n, seed=seed)
        texts = [c.text for c in cands]
        toks = [c.n_tokens for c in cands] or [0]
        out += ["", "=== %s, segment %d: %d characters, %d tokens" % (sid, idx, len(sentence), rec["sentence_tokens"]),
                "ORIGINAL: " + sentence,
                "original perplexity %.1f" % p.perplexity(sentence),
                "requested %d | returned %d | batches %d | %.1f s (%.2f s per candidate) | token limit %d | hit the limit %d | out-of-memory events %d"
                % (rec["requested"], rec["returned"], rec["batches"], rec["seconds"],
                   rec["seconds"] / max(1, rec["returned"]), rec["max_new_tokens"],
                   rec["hit_limit"], len(rec["oom_events"])),
                "empty %d | more than one line %d | identical to the original %d | distinct %d of %d | tokens min/median/max %d/%d/%d"
                % (sum(1 for t in texts if not t), sum(1 for t in texts if "\n" in t),
                   sum(1 for t in texts if t.lower() == sentence.strip().lower()),
                   len(set(texts)), len(texts), min(toks), statistics.median(toks), max(toks))]
        for i, c in enumerate(cands[:6], 1):
            ppl = p.perplexity(c.text) if c.text else float("nan")
            out.append("%2d. [ppl %5.1f%s] %s" % (i, ppl, ", HIT LIMIT" if c.hit_limit else "", c.text.replace("\n", " / ")))
    if memory_fn:
        out += ["", memory_fn()]
    return out


def main():
    os.chdir(ROOT)
    sys.path[:0] = [str(HERE)]
    import samples
    import paraphraser
    files = [f for f in FILES if os.path.exists(f)]
    items = pick_sentences(samples.load_samples(files), PICK)
    p = paraphraser.Paraphraser()
    import torch
    memory = lambda: "peak GPU memory: allocated %.2f GiB, reserved %.2f GiB" % (
        torch.cuda.max_memory_allocated() / 2 ** 30, torch.cuda.max_memory_reserved() / 2 ** 30)
    print("\n".join(run_check(p, items, memory_fn=memory)))


if __name__ == "__main__":
    main()
