"""
Module 4 of the Qwen paraphrase attack: the gates.

Candidate rewrites of one sentence pass through five gates, in this
order. Every gate counts what it rejects and why, so the report can say
where candidates are lost. Every gate compares against the ORIGINAL
sentence, never against an earlier rewrite, so repeated rewriting cannot
drift away from the meaning.

  clean      empty, more than one line, hit the length limit, identical to
             the original, under 10 characters, CJK text, an answer that
             starts with a preamble ("Here is..."), exact duplicates
  spelling   a word known neither to the C4 vocabulary nor to WordNet nor to
             the original sentence (glued words such as "tousually"), or a
             glued pair (",x" or "aB") that the original does not contain
  content    the numbers, the quoted strings, the negations and (optionally)
             the capitalised names of the original are all kept
  semantic   MiniLM similarity to the original at or above a threshold
  fluency    perplexity at most `fluency` x the median of the survivors
             (skipped when fewer than 3 candidates are left)

None of this proves the meaning is kept. Quality is judged by reading.
"""

import json
import re
import statistics
from collections import Counter
from dataclasses import dataclass

CJK = re.compile("[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")
PREAMBLE = re.compile(r"^(here(?:'s| is| are)|sure|certainly|of course|rewritten|rewrite|revised|paraphrase)\b", re.I)
NUMBER = re.compile(r"\d[\d,]*\.?\d*")
QUOTED = re.compile(r'"([^"]+)"')
NEGATION = re.compile(r"\b(?:not|no|never|none|nobody|nothing|nowhere|neither|nor|without|cannot)\b|n't\b", re.I)
GLUE = re.compile(r"[a-z][A-Z]|[,;:][A-Za-z]")
WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
NAME = re.compile(r"\b[A-Z][a-zA-Z]+\b")
ENTITY = re.compile(r"&(?:[a-zA-Z]+|#\d+);")

CLEAN_REASONS = ("empty", "several lines", "hit length limit", "identical to original",
                 "too short", "CJK text", "preamble", "duplicate")


def norm(text):
    """Straight quotes and apostrophes, HTML entities removed: for comparing."""
    return ENTITY.sub(" ", text.replace("\u201c", '"').replace("\u201d", '"')
                      .replace("\u2019", "'").replace("\u2018", "'"))


def numbers(text):
    return sorted(n.replace(",", "").rstrip(".") for n in NUMBER.findall(norm(text)))


def quotes(text):
    return [q.strip() for q in QUOTED.findall(norm(text))]


def names(text):
    t = norm(text)
    start = re.search(r"[A-Za-z]", t)                  # the first word is capitalised anyway: "What's", "Start-up"
    skip = start.start() if start else -1
    return {m.group() for m in NAME.finditer(t) if m.start() != skip}


@dataclass
class GateConfig:
    semantic: float = 0.75
    fluency: float = 1.8
    min_for_fluency: int = 3
    spelling: bool = True
    max_unknown: int = 0
    check_numbers: bool = True
    check_quotes: bool = True
    check_negations: bool = True
    check_names: bool = False


PRESETS = {
    "lenient": GateConfig(),                                        # round 1: spoken-English quality, meaning kept at sentence level
    "strict": GateConfig(semantic=0.88, fluency=1.3, check_names=True),
}


class Vocabulary:
    """Words seen at least `min_count` times in a body of ordinary text."""

    def __init__(self, texts, min_count=2):
        counts = {}
        for t in texts:
            for w in WORD.findall(t.replace("\u2019", "'").lower()):
                counts[w] = counts.get(w, 0) + 1
        self.words = {w for w, c in counts.items() if c >= min_count}

    def __contains__(self, word):
        return word.replace("\u2019", "'").lower() in self.words

    def __len__(self):
        return len(self.words)


def vocabulary_from_c4(path="data/C4/sampled_dataset_train.json"):
    return Vocabulary(json.load(open(path)))


def load_wordnet():
    """WordNet through nltk, or None when the corpus is not installed."""
    try:
        from nltk.corpus import wordnet
        wordnet.morphy("test")
        return wordnet
    except Exception:
        return None


class KnownWords:
    """A word is known when the C4 vocabulary has it (possessive 's removed) or WordNet does."""

    def __init__(self, vocab, wordnet=None):
        self.vocab, self.wn = vocab, wordnet

    @property
    def has_wordnet(self):
        return self.wn is not None

    def __contains__(self, word):
        w = word.replace("\u2019", "'").lower()
        if w.endswith("'s"):
            w = w[:-2]
        return w in self.vocab or (self.wn is not None and bool(self.wn.morphy(w)))


class MiniLM:
    """Sentence similarity with all-MiniLM-L6-v2 (the model the earlier attack used)."""

    def __init__(self, device=None):
        import torch
        from transformers import AutoModel, AutoTokenizer
        name = "sentence-transformers/all-MiniLM-L6-v2"
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(self.device).eval()

    def embed(self, texts):
        torch = self.torch
        enc = self.tok(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**enc)[0]
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return torch.nn.functional.normalize(pooled, p=2, dim=1)

    def similarity(self, reference, texts, batch=32):
        ref, scores = self.embed([reference]), []
        for i in range(0, len(texts), batch):
            scores.extend(self.torch.mm(self.embed(texts[i:i + batch]), ref.T).squeeze(1).tolist())
        return scores


class Gates:
    def __init__(self, config, vocab, similarity, perplexity):
        self.cfg, self.vocab, self.similarity, self.perplexity = config, vocab, similarity, perplexity

    def clean(self, original, cands, rejected):
        out, seen = [], set()
        for c in cands:
            t = c.text
            reason = ("empty" if not t else "several lines" if "\n" in t
                      else "hit length limit" if c.hit_limit
                      else "identical to original" if t.lower() == original.strip().lower()
                      else "too short" if len(t) < 10
                      else "CJK text" if CJK.search(t) and not CJK.search(original)
                      else "preamble" if PREAMBLE.match(t)
                      else "duplicate" if t in seen else None)
            if reason:
                rejected[reason] = rejected.get(reason, 0) + 1
            else:
                seen.add(t)
                out.append(t)
        return out

    def spelling(self, original, texts, rejected):
        known = {w.lower() for w in WORD.findall(norm(original))}
        glue_allowed = Counter(GLUE.findall(original))
        out = []
        for t in texts:
            unknown = [w for w in WORD.findall(norm(t)) if w.lower() not in known and w not in self.vocab]
            if len(unknown) > self.cfg.max_unknown:
                rejected["unknown word"] = rejected.get("unknown word", 0) + 1
            elif Counter(GLUE.findall(t)) - glue_allowed:
                rejected["glued punctuation"] = rejected.get("glued punctuation", 0) + 1
            else:
                out.append(t)
        return out

    def content(self, original, texts, rejected):
        cfg, out = self.cfg, []
        nums, qs, neg, nm = numbers(original), quotes(original), len(NEGATION.findall(norm(original))), names(original)
        for t in texts:
            nt = norm(t)
            if cfg.check_numbers and numbers(t) != nums:
                reason = "numbers changed"
            elif cfg.check_quotes and any(q not in nt for q in qs):
                reason = "quotation changed"
            elif cfg.check_negations and len(NEGATION.findall(nt)) != neg:
                reason = "negation count changed"
            elif cfg.check_names and not nm <= set(NAME.findall(nt)):
                reason = "name changed"
            else:
                out.append(t)
                continue
            rejected[reason] = rejected.get(reason, 0) + 1
        return out

    def run(self, original, cands):
        """Returns (surviving texts, funnel). The funnel is what the report aggregates."""
        rejected, funnel = {}, {"generated": len(cands)}
        texts = self.clean(original, cands, rejected)
        funnel["after_clean"] = len(texts)
        if self.cfg.spelling:
            texts = self.spelling(original, texts, rejected)
        funnel["after_spelling"] = len(texts)
        texts = self.content(original, texts, rejected)
        funnel["after_content"] = len(texts)
        if texts:
            sims = self.similarity(original, texts)
            keep = [t for t, s in zip(texts, sims) if s >= self.cfg.semantic]
            rejected["meaning too different"] = len(texts) - len(keep)
            texts = keep
        funnel["after_semantic"] = len(texts)
        if len(texts) >= self.cfg.min_for_fluency:
            ppl = [self.perplexity(t) for t in texts]
            limit = self.cfg.fluency * statistics.median(ppl)
            keep = [t for t, p in zip(texts, ppl) if p <= limit]
            rejected["perplexity too high"] = len(texts) - len(keep)
            texts, funnel["fluency_skipped"] = keep, False
        else:
            funnel["fluency_skipped"] = True
        funnel["after_fluency"] = len(texts)
        funnel["rejected"] = {k: v for k, v in rejected.items() if v}
        return texts, funnel
