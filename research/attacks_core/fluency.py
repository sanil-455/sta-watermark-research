"""
Perplexity gate using the generating model.

Embedding similarity measures topical overlap and is close to
blind to grammar and word sense: "capable to execute" scored
0.997, "These are also matter" passed both cosine and
LanguageTool. Perplexity under Llama-2 asks a different
question -- does this read like text the model would produce
-- which catches agreement errors, wrong senses and
awkwardness in one number.
"""

import torch
from transformers import AutoModelForCausalLM

MODEL_PATH = "hf_models/Llama-2-7b-hf"


class Fluency:
    def __init__(self, tokenizer, device=None):
        self.tok = tokenizer
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            device_map=self.device,
        ).eval()
        self.base = None

    @torch.no_grad()
    def perplexity(self, text):
        ids = self.tok(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).input_ids.to(self.device)

        if ids.shape[1] < 2:
            return float("inf")

        loss = self.model(ids, labels=ids).loss
        return float(torch.exp(loss))

    def set_reference(self, text):
        self.base = self.perplexity(text)
        return self.base

    def ratio(self, text):
        """
        Perplexity relative to the original.

        A ratio of 1.10 means the edited text is 10% less
        predictable than the original. Absolute perplexity
        varies by document, so the ratio is the portable
        quantity.
        """
        return self.perplexity(text) / self.base
