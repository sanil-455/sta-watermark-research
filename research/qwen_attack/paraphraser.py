"""
Module 3 of the Qwen paraphrase attack: the paraphraser.

Produces candidate rewrites of ONE sentence with Qwen2.5-7B-Instruct
loaded in 4-bit. Every output comes back exactly as generated, with its
length and a flag; throwing away junk, duplicates and bad rewrites is
Module 4's job, and Module 4 counts what it throws away.

Kept from the earlier attack: nf4 with double quantisation and float16
compute, the same system prompt, temperature 1.5, top_p 0.98, batches
of 4. Changed:
  - the fixed 100-token limit is gone. A rewrite may run to
    length_factor x the sentence plus extra_tokens, and a rewrite that
    hits that limit is flagged, never silently cut
  - top_k and repetition_penalty come from the model's own generation
    config (the earlier code used them without saying so); they are read
    at load time and recorded
  - every batch has its own seed, so a call can be repeated exactly
  - an out-of-memory error halves the batch and is recorded; it is never
    swallowed
The model is also used for perplexity, so one 7B model does both jobs.
"""

import time
from dataclasses import asdict, dataclass

REPO = "Qwen/Qwen2.5-7B-Instruct"
SYSTEM = (
    "You rewrite sentences using different words while keeping "
    "the meaning exactly. Never change a number, percentage, "
    "date, name or quotation. Never reverse a comparison or a "
    "negation. Reply with only the rewritten sentence."
)
USER_PREFIX = "Rewrite this sentence:\n"


@dataclass
class Settings:
    temperature: float = 1.5
    top_p: float = 0.98
    top_k: int = None                  # None: read the model's own value at load
    repetition_penalty: float = None   # None: read the model's own value at load
    batch: int = 4
    length_factor: float = 2.0         # a rewrite may be this many times the sentence length
    extra_tokens: int = 32             # plus this many tokens


@dataclass
class Candidate:
    text: str          # the output as generated, outer whitespace removed
    n_tokens: int      # tokens generated before the stop token
    hit_limit: bool    # stopped by the length limit, not by the model
    batch: int         # which batch it came from
    seed: int          # the seed of that batch


def length_limit(sentence_tokens, settings):
    return int(settings.length_factor * sentence_tokens) + settings.extra_tokens


def generate_candidates(sentence, n, backend, settings, seed=0):
    """Returns (candidates, record). The record is what the report needs."""
    started = time.time()
    prompt = backend.encode_prompt(sentence)
    limit = length_limit(backend.count_tokens(sentence), settings)
    stops = set(backend.stop_ids)
    out, events = [], []
    remaining, size, batch = n, settings.batch, 0
    while remaining > 0:
        k = min(size, remaining)
        batch_seed = seed + 1000 * batch
        rows, oom = None, False
        try:
            rows = backend.generate(prompt, k, limit, batch_seed, settings)
        except backend.oom_exc:
            oom = True
        if oom:
            backend.free()                  # only after the except block has ended
            events.append({"batch": batch, "size": k})
            if k == 1:
                raise RuntimeError("out of memory with a single sequence: %s" % events)
            size = max(1, k // 2)
            continue
        for row in rows:
            cut = next((i for i, t in enumerate(row) if t in stops), None)
            body = row if cut is None else row[:cut]
            out.append(Candidate(backend.decode(body).strip(), len(body),
                                 cut is None and len(row) >= limit, batch, batch_seed))
        remaining -= k
        batch += 1
    record = {
        "sentence_tokens": backend.count_tokens(sentence), "max_new_tokens": limit,
        "requested": n, "returned": len(out), "batches": batch, "seed": seed,
        "hit_limit": sum(c.hit_limit for c in out), "oom_events": events,
        "seconds": round(time.time() - started, 2),
    }
    return out, record


def qwen2_parameters(cfg):
    """Exact parameter count of a Qwen2-style model, from its config."""
    h, inter, layers = cfg["hidden_size"], cfg["intermediate_size"], cfg["num_hidden_layers"]
    heads, kv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    kv_dim = kv * (h // heads)
    attn = (h * h + h) + 2 * (h * kv_dim + kv_dim) + h * h      # q, k, v carry a bias; o does not
    per_layer = attn + 3 * h * inter + 2 * h                      # MLP gate, up, down; two norms
    head = 0 if cfg.get("tie_word_embeddings") else cfg["vocab_size"] * h
    return cfg["vocab_size"] * h + layers * per_layer + h + head


class QwenBackend:
    """The real model behind generate_candidates."""

    def __init__(self, tok, model, torch):
        self.tok, self.model, self.torch = tok, model, torch
        eos = model.generation_config.eos_token_id
        eos = list(eos) if isinstance(eos, (list, tuple)) else [eos]
        self.stop_ids = sorted(set(eos + [tok.eos_token_id]))
        self.oom_exc = torch.cuda.OutOfMemoryError

    def encode_prompt(self, sentence):
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": USER_PREFIX + sentence}]
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self.tok(text, return_tensors="pt").to(self.model.device)

    def count_tokens(self, sentence):
        return len(self.tok(sentence, add_special_tokens=False)["input_ids"])

    def generate(self, enc, k, limit, seed, settings):
        self.torch.manual_seed(seed)
        with self.torch.no_grad():
            gen = self.model.generate(
                **enc, do_sample=True, temperature=settings.temperature,
                top_p=settings.top_p, top_k=settings.top_k,
                repetition_penalty=settings.repetition_penalty,
                max_new_tokens=limit, num_return_sequences=k,
                pad_token_id=self.tok.eos_token_id)
        plen = enc["input_ids"].shape[1]
        rows = [r[plen:].tolist() for r in gen]
        del gen
        return rows

    def decode(self, ids):
        return self.tok.decode(ids, skip_special_tokens=True)

    def free(self):
        import gc
        gc.collect()
        self.torch.cuda.empty_cache()


class Paraphraser:
    def __init__(self, settings=None, repo=REPO):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        started = time.time()
        self.torch, self.repo = torch, repo
        self.settings = settings or Settings()
        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
        self.tok = AutoTokenizer.from_pretrained(repo)
        self.model = AutoModelForCausalLM.from_pretrained(
            repo, quantization_config=quant, device_map="auto").eval()
        self.load_seconds = round(time.time() - started, 1)
        gen_cfg = self.model.generation_config
        if self.settings.top_k is None:
            self.settings.top_k = gen_cfg.top_k
        if self.settings.repetition_penalty is None:
            self.settings.repetition_penalty = gen_cfg.repetition_penalty
        self.backend = QwenBackend(self.tok, self.model, torch)

    def rewrite(self, sentence, n=20, seed=0):
        return generate_candidates(sentence, n, self.backend, self.settings, seed)

    def perplexity(self, text):
        """Perplexity under Qwen; only ever used as a relative measure."""
        with self.torch.no_grad():
            ids = self.tok(text, return_tensors="pt", truncation=True,
                           max_length=2048).input_ids.to(self.model.device)
            if ids.shape[1] < 2:
                return float("inf")
            return float(self.torch.exp(self.model(ids, labels=ids).loss))

    def describe(self):
        """Everything about the model that the final report needs."""
        from importlib import metadata
        torch, m = self.torch, self.model
        cfg = m.config.to_dict()
        dtypes = {}
        for p in m.parameters():
            dtypes[str(p.dtype)] = dtypes.get(str(p.dtype), 0) + p.numel()
        quant = getattr(m.config, "quantization_config", None)
        derived, reported = qwen2_parameters(cfg), m.num_parameters()
        versions = {}
        for name in ("torch", "transformers", "bitsandbytes", "accelerate"):
            try:
                versions[name] = metadata.version(name)
            except Exception:
                versions[name] = "unknown"
        return {
            "repo": self.repo, "revision": getattr(m.config, "_commit_hash", None),
            "architectures": cfg.get("architectures"),
            "parameters_exact": derived, "parameters_reported_by_transformers": reported,
            "parameters_agree": derived == reported,
            "stored_elements_by_dtype": dtypes,
            "hidden_size": cfg.get("hidden_size"), "layers": cfg.get("num_hidden_layers"),
            "attention_heads": cfg.get("num_attention_heads"),
            "kv_heads": cfg.get("num_key_value_heads"),
            "intermediate_size": cfg.get("intermediate_size"),
            "vocab_size": cfg.get("vocab_size"),
            "context_length": cfg.get("max_position_embeddings"),
            "quantization": quant.to_dict() if hasattr(quant, "to_dict") else quant,
            "memory_footprint_gib": round(m.get_memory_footprint() / 2 ** 30, 2),
            "gpu_allocated_gib": round(torch.cuda.memory_allocated() / 2 ** 30, 2),
            "gpu_reserved_gib": round(torch.cuda.memory_reserved() / 2 ** 30, 2),
            "load_seconds": self.load_seconds, "settings": asdict(self.settings),
            "stop_token_ids": self.backend.stop_ids, "versions": versions,
        }
