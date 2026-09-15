import re
from difflib import SequenceMatcher

import numpy as np
from sentence_transformers import SentenceTransformer


SEMANTIC_THRESHOLD = 0.90
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class SemanticChecker:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME, device="cpu")

    def _embed(self, texts):
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def _cosine(self, a, b):
        return float(np.dot(a, b))

    def _chunks(self, text, words_per_chunk=180):
        words = text.split()

        if not words:
            return [""]

        return [
            " ".join(words[i:i + words_per_chunk])
            for i in range(0, len(words), words_per_chunk)
        ]

    def global_similarity(self, original, attacked):
        original_chunks = self._chunks(original)
        attacked_chunks = self._chunks(attacked)

        original_embedding = self._embed(original_chunks).mean(axis=0)
        attacked_embedding = self._embed(attacked_chunks).mean(axis=0)

        original_embedding /= np.linalg.norm(original_embedding)
        attacked_embedding /= np.linalg.norm(attacked_embedding)

        return self._cosine(original_embedding, attacked_embedding)

    def _changed_region(self, original, attacked, window=500):
        matcher = SequenceMatcher(None, original, attacked)
        opcodes = matcher.get_opcodes()

        changed = [
            opcode for opcode in opcodes
            if opcode[0] != "equal"
        ]

        if not changed:
            return original, attacked

        first = changed[0]

        original_start = max(0, first[1] - window)
        original_end = min(len(original), first[2] + window)

        attacked_start = max(0, first[3] - window)
        attacked_end = min(len(attacked), first[4] + window)

        return (
            original[original_start:original_end],
            attacked[attacked_start:attacked_end],
        )

    def local_similarity(self, original, attacked):
        original_region, attacked_region = self._changed_region(
            original,
            attacked,
        )

        embeddings = self._embed(
            [original_region, attacked_region]
        )

        return self._cosine(embeddings[0], embeddings[1])

    def check(self, original, attacked):
        global_score = self.global_similarity(original, attacked)
        local_score = self.local_similarity(original, attacked)

        accepted = (
            global_score >= SEMANTIC_THRESHOLD
            and local_score >= SEMANTIC_THRESHOLD
        )

        return {
            "global_similarity": global_score,
            "local_similarity": local_score,
            "threshold": SEMANTIC_THRESHOLD,
            "accepted": accepted,
        }


def main():
    checker = SemanticChecker()

    original = (
        "The company has experienced strong growth over the past year, "
        "although demand has recently slowed."
    )

    attacked = (
        "The company has experienced strong growth over the past year, "
        "although demand has recently slowed slightly."
    )

    result = checker.check(original, attacked)

    print("Semantic checker test")
    print(f"Global similarity: {result['global_similarity']:.6f}")
    print(f"Local similarity:  {result['local_similarity']:.6f}")
    print(f"Threshold:         {result['threshold']:.2f}")
    print(f"Accepted:          {result['accepted']}")


if __name__ == "__main__":
    main()
