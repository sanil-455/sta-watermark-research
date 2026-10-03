"""
A black-box detector oracle.

The attacker does not know the watermark key. The only thing it
can do is hand text to a detector and read back a verdict. This
module is written so the key never appears in it: no H1, no H2,
no is_green_pair, no notion of which specific pairs are green.

Every call is counted. Query count is the attack's real cost
under this threat model, the way edit count is its cost under
the white-box one.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "attacks_core"))

from sta_core import score_text, Z_THRESHOLD


class DetectorOracle:
    """
    Wraps the detector and counts how often it is consulted.

    mode "score"  : returns the z-score, modelling a research
                    API that reports confidence
    mode "binary" : returns True/False only, modelling a
                    deployed service that answers nothing more
                    than "is this watermarked"
    """

    def __init__(self, tokenizer, mode="score"):
        if mode not in ("score", "binary"):
            raise ValueError("mode must be 'score' or 'binary'")
        self.tok = tokenizer
        self.mode = mode
        self.queries = 0
        self._cache = {}

    def query(self, text):
        """
        One detector call.

        Identical text is served from cache and not counted
        twice, since an attacker would not pay to ask the same
        question again.
        """
        if text in self._cache:
            return self._cache[text]

        self.queries += 1
        stats = score_text(self.tok, text)

        answer = (
            stats["z"] if self.mode == "score"
            else stats["z"] > Z_THRESHOLD
        )
        self._cache[text] = answer
        return answer

    def is_broken(self, text):
        """Has this text escaped detection? Costs one query."""
        answer = self.query(text)
        if self.mode == "binary":
            return not answer
        return answer <= Z_THRESHOLD

    def reset_count(self):
        """Zero the counter without clearing the cache."""
        self.queries = 0
