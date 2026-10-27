"""
Module 1 of the Qwen paraphrase attack: what we attack.

A sample is the text the watermarked model generated, and nothing else
(no prompt). Its baseline score z0 is what the detector says about
exactly that text. Success later means getting that score to 2.0 or
below while the text still reads properly.

Three file layouts exist and all of them are handled:
  prompt-free files   "prompt" is empty, saved z = detector on the span text
  first-version files "prompt" holds the prompt joined to the text; saved z
                      is over the generated token ids (true boundary pair)
  baseline files      prompt joined to the text; saved z covers the WHOLE
                      document, so it is never compared with anything
"""

import json
from dataclasses import dataclass
from pathlib import Path

Z_THRESHOLD = 2.0   # the paper's detection threshold; sta_core uses the same value

# how far a saved z may differ from the fresh z0 before it counts as a mismatch
#   detector: the very same function, so it must match exactly
#   ids:      at most TWO pairs may differ. In the saved number the first pair is
#             (last prompt token, first generated token); in the fresh one it is
#             (start-of-text token, first generated token). If the span begins with
#             a token written without a leading space, tokenising it alone adds one,
#             and that also changes the second pair. Every later pair is identical.
#             Anything beyond two pairs is flagged and must be explained by
#             check_tokens.py.
PAIRS_ALLOWED = 2


@dataclass
class Sample:
    sample_id: str            # unique name: "<file>:<C4 entry>"
    source: str               # the file it was loaded from
    prompt_id: int            # which C4 entry it was generated from
    span: str                 # the generated text, and nothing else
    layout: str               # "span" or "prompt+text"
    stored_z: float = None    # z saved by the generation script, if comparable
    stored_z_kind: str = None  # "detector", "ids", or None when not comparable
    n_generated: int = None   # tokens the model generated, if the file says
    prompt_tokens: int = None  # tokens of the discarded prompt (provenance only)
    green_fraction: float = None
    z0: float = None          # what the detector says about span; set by screen()


def span_of(row):
    """The generated text of one saved row, whichever file layout it came from.

    Nothing is stripped. An earlier version called strip(), which dropped a
    trailing newline token and changed one sample's token count.
    """
    text = row["watermarked_text"]
    prompt = row.get("prompt", "")
    if prompt:                                   # prompt and text joined
        if not text.startswith(prompt):
            raise ValueError("row %s: the text does not start with its prompt, "
                             "so the generated part cannot be located"
                             % row.get("prompt_id"))
        text = text[len(prompt):]
        if text.startswith(" "):                 # the one space that joined them
            text = text[1:]
    return text                                  # exactly what was generated


def load_samples(paths):
    """Read every row of every file into Sample records. Nothing is dropped."""
    samples, seen = [], set()
    for path in paths:
        tag = Path(path).stem
        rows = json.load(open(path))
        if not isinstance(rows, list):
            raise ValueError("%s is not a list of samples" % path)
        for row in rows:
            sid = "%s:%s" % (tag, row["prompt_id"])
            if sid in seen:
                raise ValueError("duplicate sample %s" % sid)
            seen.add(sid)
            if "source_prompt" in row:           # prompt-free files
                kind, ptoks = "detector", row.get("source_prompt_tokens")
            elif "n_generated" in row:           # first-version generation files
                kind, ptoks = "ids", row.get("prompt_tokens")
            else:                                # baseline files
                kind, ptoks = None, row.get("prompt_tokens")
            span = span_of(row)
            if not span.strip():
                raise ValueError("sample %s has an empty span" % sid)
            samples.append(Sample(
                sample_id=sid, source=str(path), prompt_id=row["prompt_id"],
                span=span, layout="prompt+text" if row.get("prompt") else "span",
                stored_z=row["watermarked_z"] if kind else None,
                stored_z_kind=kind, n_generated=row.get("n_generated"),
                prompt_tokens=ptoks, green_fraction=row.get("green_fraction")))
    return samples


def screen(samples, oracle):
    """Ask the detector about every span. Returns (flagged, not_flagged)."""
    if getattr(oracle, "mode", "score") != "score":
        raise ValueError("screening needs z-scores, not yes/no answers")
    flagged, not_flagged = [], []
    for s in samples:
        s.z0 = oracle.query(s.span)
        (flagged if s.z0 > Z_THRESHOLD else not_flagged).append(s)
    return flagged, not_flagged


def compared(samples):
    """Samples that have both a comparable saved z and a fresh z0."""
    return [s for s in samples if s.stored_z_kind and s.z0 is not None]


def allowed_difference(sample):
    """How far this sample's saved z and fresh z0 may differ, in z units."""
    if sample.stored_z_kind == "detector":
        return 1e-6
    n = sample.n_generated or 200                  # pairs scored; one pair is 1/(0.5*sqrt(n))
    return PAIRS_ALLOWED / (0.5 * n ** 0.5) + 1e-6


def mismatches(samples):
    """Compared samples whose saved z and fresh z0 differ by more than allowed."""
    return [s for s in compared(samples)
            if abs(s.stored_z - s.z0) > allowed_difference(s)]


def check_prefixes(samples):
    """A prompt that appears at two lengths: the shorter span must start the longer.

    Returns (number of pairs checked, list of pairs that fail).
    """
    by_prompt = {}
    for s in samples:
        by_prompt.setdefault(s.prompt_id, []).append(s)
    pairs, bad = 0, []
    for group in by_prompt.values():
        group = sorted(group, key=lambda s: len(s.span))
        for shorter, longer in zip(group, group[1:]):
            pairs += 1
            if not longer.span.startswith(shorter.span):
                bad.append((shorter.sample_id, longer.sample_id))
    return pairs, bad


def describe(samples):
    """One line per sample, for the log and the report."""
    lines = ["%-24s %-12s %6s %6s %8s %-8s %8s %7s"
             % ("sample", "layout", "chars", "tokens", "saved z", "kind", "z0", "diff")]
    for s in samples:
        saved = "-" if s.stored_z is None else "%.3f" % s.stored_z
        z0 = "-" if s.z0 is None else "%.3f" % s.z0
        diff = ("%+.3f" % (s.z0 - s.stored_z)
                if s.stored_z is not None and s.z0 is not None else "-")
        toks = "-" if s.n_generated is None else str(s.n_generated)
        lines.append("%-24s %-12s %6d %6s %8s %-8s %8s %7s"
                     % (s.sample_id, s.layout, len(s.span), toks, saved,
                        s.stored_z_kind or "-", z0, diff))
    return "\n".join(lines)
