"""
Screen instruction-tuned models for paraphrase fidelity.

Both Llama-2-7B base and chat reversed direction on "her number
overestimated how long she could go", producing "undervalued"
and "undercounted". The chat model did it on every candidate
despite being told never to reverse a comparison.

This tests whether any available model preserves factual
direction. Each is loaded in 4-bit, tested, then unloaded
before the next.
"""

import gc

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

# 4-bit stores each weight in 4 bits rather than 16, so a 13B
# model needs about 7GB instead of 26. nf4 spaces its 16 levels
# to match how weights are actually distributed, which loses
# less than evenly spaced levels would. Compute is still fp16:
# weights unpack to 16-bit for each matrix multiply.
QUANT = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)


def load(repo):
    """Load one model in 4-bit. Returns (tokenizer, model)."""
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(
        repo,
        quantization_config=QUANT,
        device_map="auto",
    ).eval()
    return tok, model

def unload(*names_and_globals):
    """
    Free the card before loading the next model.

    Deleting the parameter inside a function does not help: the
    caller still holds a reference, so Python never frees the
    object. The caller must drop its own reference first, which
    is why this takes no model argument -- it only does the
    collection and cache release.

    All three steps are needed. gc.collect() makes Python
    actually free the object, and empty_cache() returns the
    memory to the driver rather than leaving it reserved in
    PyTorch's pool where the next load cannot use it.
    """
    gc.collect()
    torch.cuda.empty_cache()

SYSTEM = (
    "You rewrite sentences using different words while keeping "
    "the meaning exactly. Never change a number, percentage, "
    "date, name or quotation. Never reverse a comparison or a "
    "negation. Reply with only the rewritten sentence."
)


def build_prompt(tokenizer, sentence):
    """
    Format the instruction using the model's own chat template.

    Llama-2, Mistral and Qwen each use different markers.
    apply_chat_template reads the right format from the
    tokenizer config, so one code path serves all three and the
    comparison stays fair.
    """
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",
         "content": "Rewrite this sentence:\n" + sentence},
    ]
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # some Mistral versions reject a system role, so fold
        # the instruction into the user turn instead
        merged = [{
            "role": "user",
            "content": SYSTEM + "\n\nRewrite this sentence:\n" + sentence,
        }]
        return tokenizer.apply_chat_template(
            merged, tokenize=False, add_generation_prompt=True
        )


@torch.no_grad()
def generate(model, tokenizer, sentence, n=6, batch=3):
    """
    Produce n rewrites.

    Batched at 3 because each sequence carries its own attention
    cache. Asking for all of them at once is what caused the
    out-of-memory crashes earlier in this project.
    """
    prompt = build_prompt(tokenizer, sentence)
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    plen = enc["input_ids"].shape[1]

    out = []
    while len(out) < n:
        k = min(batch, n - len(out))
        try:
            gen = model.generate(
                **enc,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                max_new_tokens=80,
                num_return_sequences=k,
                pad_token_id=tokenizer.eos_token_id,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            break

        for row in gen:
            text = tokenizer.decode(row[plen:],
                                    skip_special_tokens=True)
            line = text.strip().split("\n")[0].strip()
            if len(line) > 10:
                out.append(line)

        del gen
        torch.cuda.empty_cache()

    return out[:n]
