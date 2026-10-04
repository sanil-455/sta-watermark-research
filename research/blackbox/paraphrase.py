"""
Paraphrase generation using Qwen2.5-7B-Instruct.

Llama-2-7B base reversed factual direction sometimes; the chat
model reversed it on every candidate, turning "overestimated"
into "undervalued" despite being told not to. Qwen preserved it
6 of 6, and kept "half" and "a quarter" distinct where the
chat model collapsed "Anglos" into "non-Hispanics".

Loaded in 4-bit at about 5.5GB, which leaves room for the
detector's Llama tokenizer and the KV cache on a 16GB card.
"""

import gc

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

REPO = "Qwen/Qwen2.5-7B-Instruct"

QUANT = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

SYSTEM = (
    "You rewrite sentences using different words while keeping "
    "the meaning exactly. Never change a number, percentage, "
    "date, name or quotation. Never reverse a comparison or a "
    "negation. Reply with only the rewritten sentence."
)


class Paraphraser:
    """
    Holds the model once, since loading costs minutes.

    Also exposes perplexity, so one model serves both the
    rewriting and the fluency gate. Two 7B models would not fit
    alongside the detector on a 16GB card.
    """

    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained(REPO)
        self.model = AutoModelForCausalLM.from_pretrained(
            REPO, quantization_config=QUANT, device_map="auto",
        ).eval()

    def _prompt(self, sentence):
        msgs = [
            {"role": "system", "content": SYSTEM},
            {"role": "user",
             "content": "Rewrite this sentence:\n" + sentence},
        ]
        return self.tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )

    @torch.no_grad()
    def rewrite(self, sentence, n=20, batch=4):
        """
        Produce up to n distinct rewrites.

        Batched because each sequence carries its own attention
        cache; asking for all at once caused out-of-memory
        crashes earlier in this project.
        """
        enc = self.tok(self._prompt(sentence),
                       return_tensors="pt").to(self.model.device)
        plen = enc["input_ids"].shape[1]

        seen, out = set(), []
        remaining = n

        while remaining > 0:
            k = min(batch, remaining)
            remaining -= k
            try:
                gen = self.model.generate(
                    **enc, do_sample=True, temperature=1.5,
                    top_p=0.98, max_new_tokens=100,
                    num_return_sequences=k,
                    pad_token_id=self.tok.eos_token_id,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                break

            for row in gen:
                line = self.tok.decode(
                    row[plen:], skip_special_tokens=True
                ).strip().split("\n")[0].strip()
                if len(line) > 10 and line not in seen:
                    seen.add(line)
                    out.append(line)

            del gen
            torch.cuda.empty_cache()

        return out

    @torch.no_grad()
    def perplexity(self, text):
        """
        Perplexity under Qwen, used only as a relative measure.

        The fluency gate compares candidates against the median
        of that sentence's own candidates, so the absolute scale
        does not matter and need not match Llama's.
        """
        ids = self.tok(text, return_tensors="pt",
                       truncation=True, max_length=2048
                       ).input_ids.to(self.model.device)
        if ids.shape[1] < 2:
            return float("inf")
        loss = self.model(ids, labels=ids).loss
        return float(torch.exp(loss))
