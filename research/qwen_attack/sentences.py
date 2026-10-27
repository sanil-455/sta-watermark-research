"""
Module 2 of the Qwen paraphrase attack: sentences and the skip rules.

The span is cut into segments. Each segment is one sentence-like piece
of text with its exact character offsets; the whitespace between
segments is kept untouched in the gaps. Nothing is dropped: a segment
that must not be rewritten stays in the list with the reason, so the
report can say exactly how much of the text was attackable and why the
rest was not.

A segment is eligible for rewriting unless one of these rules applies
(checked in this order, the first that applies is recorded):
  starts mid-sentence       the first segment does not start like a sentence
                            (the span continues the prompt's last sentence)
  does not start like a sentence   any later segment, e.g. a bullet or a
                            lowercase line
  trailing fragment         the last segment has no final punctuation (the
                            token limit cut it off)
  no final punctuation      a segment inside the span without it (a heading)
  quotation                 inside or part of a quotation, or a quotation
                            of more than QUOTE_WORDS words ("Mad Money" is a
                            title and may be rewritten around; speech is not)
  URL, hashtag or handle    rewriting these would corrupt them
  too short                 under MIN_CHARS characters

Sentence boundaries are a heuristic: a full stop, ! or ? (plus closing
quotes) followed by whitespace and a capital letter or opening quote, and
every line break. A full stop after an abbreviation or an initial is not
a boundary, so "U.S. Rep. Beto O'Rourke" stays in one piece. The price is
that a sentence ending in an abbreviation merges with the next one.
"""

import re
from dataclasses import dataclass

MIN_CHARS = 30         # shorter pieces are recorded but not rewritten
QUOTE_WORDS = 4        # quotations longer than this are never rewritten

SKIP_MID = "starts mid-sentence"
SKIP_START = "does not start like a sentence"
SKIP_TRAILING = "trailing fragment"
SKIP_PUNCT = "no final punctuation"
SKIP_QUOTE = "quotation"
SKIP_URL = "URL, hashtag or handle"
SKIP_SHORT = "too short"

ABBREVIATIONS = {
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.", "mt.", "ft.",
    "rep.", "sen.", "gov.", "gen.", "col.", "lt.", "capt.", "sgt.", "cmdr.",
    "adm.", "rev.", "hon.", "pres.", "supt.", "inc.", "ltd.", "co.", "corp.",
    "bros.", "vs.", "v.", "no.", "nos.", "fig.", "vol.", "approx.", "dept.",
    "est.", "jan.", "feb.", "mar.", "apr.", "jun.", "jul.", "aug.", "sep.",
    "sept.", "oct.", "nov.", "dec.",
}

BOUNDARY = re.compile(r"""([.!?]["”’')\]]*)([ \t]+)(?=["“‘(\[]*[A-Z])""")
STARTS_LIKE_SENTENCE = re.compile(r"""["“‘(\[]*[A-Z]""")
ENDS_LIKE_SENTENCE = re.compile(r"""[.!?]["”’')\]]*$""")
URL_LIKE = re.compile(r"""https?://|www\.|\b[\w-]+\.(?:com|org|net|gov|edu|io)\b|(?<!\w)[#@]\w+""")


def is_abbreviation(token):
    """True when a token ending in a full stop is an abbreviation or an initial."""
    core = token.lstrip("\"“‘'([")
    if core.lower() in ABBREVIATIONS:
        return True
    if re.fullmatch(r"[A-Z]\.", core):                      # an initial: J.
        return True
    return bool(re.fullmatch(r"(?:[A-Za-z]\.){2,}", core))  # U.S.  a.m.  e.g.


OPEN_BEFORE = set("([{\u2014\u2013-:")   # characters that may directly precede an opening quote


def quote_marks(text):
    """(index, "open" or "close") for every quotation mark in text.

    A curly mark says what it is. A straight mark is an opening one when it
    follows a space, the start of the text or an opening bracket and is
    followed by a non-space; otherwise it closes. A mark after a digit is an
    inch mark (5") and is ignored.
    """
    marks = []
    for i, ch in enumerate(text):
        if ch == "\u201c":
            marks.append((i, "open"))
        elif ch == "\u201d":
            marks.append((i, "close"))
        elif ch == '"':
            prev = text[i - 1] if i > 0 else ""
            nxt = text[i + 1] if i + 1 < len(text) else ""
            if prev.isdigit():
                continue
            opening = ((prev == "" or prev.isspace() or prev in OPEN_BEFORE)
                       and nxt != "" and not nxt.isspace())
            marks.append((i, "open" if opening else "close"))
    return marks


def paragraph_starts_in_quote(paragraph):
    """True when the first quotation mark of a paragraph closes a quote.

    The quote was opened before the paragraph began, for example in the
    prompt, so everything up to that mark is speech.
    """
    marks = quote_marks(paragraph)
    return bool(marks) and marks[0][1] == "close"


def quote_profile(text, in_quote):
    """(starts inside a quote, ends inside a quote, longest quoted run in words)."""
    kinds = dict(quote_marks(text))
    starts, longest, run = in_quote, 0, []
    for i, ch in enumerate(text):
        kind = kinds.get(i)
        if kind == "open":
            in_quote, run = True, []
        elif kind == "close":
            if in_quote:
                longest = max(longest, len("".join(run).split()))
            in_quote, run = False, []
        elif in_quote:
            run.append(ch)
    if in_quote:
        longest = max(longest, len("".join(run).split()))
    return starts, in_quote, longest


@dataclass
class Segment:
    index: int
    start: int                 # offset of the first character in the span
    end: int                   # offset just after the last character
    text: str                  # span[start:end]
    skip: str = None           # None when it may be rewritten, else the reason

    @property
    def eligible(self):
        return self.skip is None


class Segmentation:
    def __init__(self, span, segments):
        self.span, self.segments = span, segments

    def eligible(self):
        return [s for s in self.segments if s.eligible]

    def reassemble(self, replacements=None):
        """The span with some segments replaced. Gaps stay exactly as they were.

        Only eligible segments may be replaced: a quotation or a fragment can
        never be rewritten by mistake.
        """
        replacements = replacements or {}
        by_index = {s.index: s for s in self.segments}
        for i in replacements:
            if i not in by_index:
                raise IndexError("no segment %s" % i)
            if not by_index[i].eligible:
                raise ValueError("segment %d is skipped (%s) and must not be rewritten"
                                 % (i, by_index[i].skip))
        out, pos = [], 0
        for seg in self.segments:
            out.append(self.span[pos:seg.start])
            out.append(replacements.get(seg.index, seg.text))
            pos = seg.end
        out.append(self.span[pos:])
        return "".join(out)

    def summary(self):
        reasons = {}
        for s in self.segments:
            if s.skip:
                reasons[s.skip] = reasons.get(s.skip, 0) + 1
        elig = self.eligible()
        eligible_chars = sum(len(s.text) for s in elig)
        first = self.segments[0] if self.segments else None
        last = self.segments[-1] if self.segments else None
        return {
            "chars": len(self.span), "segments": len(self.segments),
            "eligible": len(elig), "skipped": len(self.segments) - len(elig),
            "skip_reasons": reasons, "eligible_chars": eligible_chars,
            "eligible_char_share": eligible_chars / len(self.span) if self.span else 0.0,
            "leading_fragment": first.text if first and first.skip == SKIP_MID else None,
            "trailing_fragment": last.text if last and last.skip == SKIP_TRAILING else None,
        }


def raw_pieces(span):
    """(start, end) of every sentence-like piece, whitespace excluded."""
    pieces = []
    for line in re.finditer(r"[^\n]+", span):
        base, text, start = line.start(), line.group(), 0
        for m in BOUNDARY.finditer(text):
            period = m.start(1)
            if text[period] == ".":
                tokens = text[:period + 1].split()
                if tokens and is_abbreviation(tokens[-1]):
                    continue
            pieces.append((base + start, base + m.end(1)))
            start = m.end()
        pieces.append((base + start, base + len(text)))
    cleaned = []
    for s, e in pieces:
        chunk = span[s:e]
        left = len(chunk) - len(chunk.lstrip())
        right = len(chunk.rstrip())
        if right > left:
            cleaned.append((s + left, s + right))
    return cleaned


def split_span(span, min_chars=MIN_CHARS, quote_words=QUOTE_WORDS):
    cleaned = raw_pieces(span)
    segments, in_quote, prev_end = [], False, 0
    for i, (s, e) in enumerate(cleaned):
        text = span[s:e]
        if i == 0 or "\n" in span[prev_end:s]:   # a new paragraph: its first quotation mark
            par_end = span.find("\n", s)           # says whether it begins inside a quote
            in_quote = paragraph_starts_in_quote(span[s:par_end if par_end != -1 else len(span)])
        starts_in, in_quote, longest = quote_profile(text, in_quote)
        prev_end = e
        if not STARTS_LIKE_SENTENCE.match(text):
            skip = SKIP_MID if i == 0 else SKIP_START
        elif not ENDS_LIKE_SENTENCE.search(text):
            skip = SKIP_TRAILING if i == len(cleaned) - 1 else SKIP_PUNCT
        elif starts_in or in_quote or longest > quote_words:
            skip = SKIP_QUOTE
        elif URL_LIKE.search(text):
            skip = SKIP_URL
        elif len(text) < min_chars:
            skip = SKIP_SHORT
        else:
            skip = None
        segments.append(Segment(i, s, e, text, skip))
    return Segmentation(span, segments)


def settings():
    """The values that shape the result, for the run record."""
    return {"min_chars": MIN_CHARS, "quote_words": QUOTE_WORDS,
            "abbreviations": len(ABBREVIATIONS)}
