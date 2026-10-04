"""
Paraphrase candidate generation.

Replacing a span of L tokens destroys all L+1 pairs that span
participates in. At 73% green that is 15.3 greens for L=20,
against 2 for a single substitution.

The replacement creates L+1 new pairs, green at chance, so about
10.5 come back. Generating many candidates and keeping the one
the detector likes least pulls that down to roughly 4.

Unlike synonyms or expansion triggers, paraphrases are unlimited.
That is the point: every other operation ran out of candidates.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "attacks_core"))


@torch.no_grad()
def paraphrase(model, tokenizer, text, n=20,
               temperature=1.0, max_new=80, batch=3):
    """
    Generate n rewrites of one span.

    Sampling, not greedy decoding: greedy gives the same output
    every time, and variety is the whole point.

    Generated in batches of `batch` because each sequence needs
    its own attention cache. Asking for 40 at once allocates 40
    caches alongside the 13GB model and exhausts a 16GB card.
    """
    # Base Llama is not instruction tuned, so a bare instruction
    # produces prompt echo and meta commentary. Few-shot examples
    # establish the pattern far more reliably: a base model
    # continues a pattern rather than following a command.
    #
    # The three examples show different transformations --
    # reordering, voice change, fronting -- so the model does not
    # latch onto one trick.
    # Llama-2-chat expects [INST] markers. Plain text would be
    # read as the model's own prior output rather than an
    # instruction, discarding the instruction tuning.
    #
    # The system block states the fidelity requirements, since
    # they apply to every rewrite. The failures that motivated
    # this switch were all substitutions a base model made
    # freely: "overestimated" became "underestimated", "half"
    # became "a little less than half".
    system = (
        "You rewrite sentences using different words while "
        "keeping the meaning exactly. Never change a number, "
        "percentage, date, name or quotation. Never reverse a "
        "comparison or a negation. Reply with only the "
        "rewritten sentence."
    )
    prompt = (
        "[INST] <<SYS>>\n%s\n<</SYS>>\n\n"
        "Rewrite this sentence:\n%s [/INST]"
        % (system, text.strip())
    )

    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = enc["input_ids"].shape[1]

    seen, results = set(), []
    remaining = n

    while remaining > 0:
        k = min(batch, remaining)
        remaining -= k

        try:
            out = model.generate(
                **enc,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                max_new_tokens=max_new,
                num_return_sequences=k,
                pad_token_id=tokenizer.eos_token_id,
            )
        except torch.cuda.OutOfMemoryError:
            # memory fragments across sentences, so a batch can
            # fail partway through a run. Return what we have
            # rather than losing the whole attack.
            torch.cuda.empty_cache()
            break

        for row in out:
            gen = tokenizer.decode(row[prompt_len:],
                                   skip_special_tokens=True)
            line = gen.strip().split("\n")[0].strip()
            if len(line) < 10 or line in seen:
                continue
            seen.add(line)
            results.append(line)

        del out
        torch.cuda.empty_cache()

    return results

