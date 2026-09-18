"""
Vulnerability scanning with grammatical safety filters.

Only whole, meaning-preserving edits are proposed:
  - no subword fragments
  - no proper nouns, numbers, or entities
  - deletions restricted to genuinely droppable words
  - substitutions matched on part of speech

Predicted drops are LOCAL and ignore re-tokenisation ripple;
they rank candidates cheaply. Truth comes from the detector.
"""

import spacy
from lemminflect import getInflection
from nltk.corpus import wordnet as wn

from sta_core import is_green_pair
from edit_ops import make_edit

NLP = spacy.load("en_core_web_sm")

# Words safe to delete without breaking meaning.
DROPPABLE = {
    "very", "quite", "really", "rather", "fairly", "pretty",
    "just", "simply", "merely", "actually", "basically",
    "truly", "totally", "completely", "absolutely", "so",
    "somewhat", "slightly", "extremely", "highly", "indeed",
    "certainly", "clearly", "obviously", "perhaps", "maybe",
    "also", "even", "still", "already", "however", "though",
}

# POS tags whose members may be deleted if droppable.
DELETABLE_POS = {"ADV", "ADJ", "INTJ", "PART"}

# POS tags eligible for substitution.
REPLACEABLE_POS = {"NOUN", "VERB", "ADJ", "ADV"}

# Fine-grained tags that are already base forms.
#
# WordNet lemma names are always base forms, so an inflected
# source word can only receive a mismatched replacement:
#   matters (NNS) -> matter (NN)   "These are also matter"
#   reaches (VBZ) -> hit  (VB)     "as he hit two milestones"
# Restricting substitution to base-form sources removes this
# whole failure class.
#
# The check is on the SOURCE only. Tagging a candidate in
# isolation is unreliable -- spaCy reads bare "solemn" as a
# verb and bare "place" as a noun -- so comparing source tag
# to isolated candidate tag would reject good edits.
BASE_FORM_TAGS = {"NN", "VB", "VBP", "JJ", "RB"}

# WordNet POS codes keyed by spaCy tag.
WN_POS = {
    "NOUN": wn.NOUN,
    "VERB": wn.VERB,
    "ADJ": wn.ADJ,
    "ADV": wn.ADV,
}

_SYN_CACHE = {}


def is_word_start(tokenizer, tok_id):
    """Llama marks word-initial tokens with U+2581."""
    tok = tokenizer.convert_ids_to_tokens([tok_id])[0]
    return tok.startswith("\u2581")


def build_pos_map(tokenizer, ids):
    """
    Map each token index to the spaCy token covering it.

    Decoding each token and walking the text keeps the two
    tokenisations aligned by character offset.
    """
    text = tokenizer.decode(ids, skip_special_tokens=True)
    doc = NLP(text)

    spans = [(t.idx, t.idx + len(t.text), t) for t in doc]
    pos_map = {}
    cursor = 0

    for i, tok_id in enumerate(ids):
        piece = tokenizer.decode([tok_id])
        if not piece:
            continue

        start = text.find(piece.strip(), cursor) \
            if piece.strip() else -1
        if start < 0:
            continue

        end = start + len(piece.strip())
        cursor = start

        for s, e, t in spans:
            if s <= start and end <= e:
                pos_map[i] = t
                break

    return pos_map


def synonyms(word, pos_tag, limit=6):
    """Same-POS WordNet lemmas, single words only."""
    key = (word.lower(), pos_tag)

    if key in _SYN_CACHE:
        return _SYN_CACHE[key]

    wn_pos = WN_POS.get(pos_tag)
    out = []

    if wn_pos:
        # Only the first synset. WordNet orders senses by
        # frequency, so later synsets give rare readings
        # (market->mart, capacity->content) that are valid
        # dictionary entries but wrong in context.
        for syn in wn.synsets(word.lower(), pos=wn_pos)[:1]:
            for lemma in syn.lemmas():
                name = lemma.name()
                if "_" in name or name.lower() == word.lower():
                    continue
                if not name.isalpha() or name[0].isupper():
                    continue
                if name not in out:
                    out.append(name)

    out = out[:limit]
    _SYN_CACHE[key] = out
    return out


def _green(ids, i):
    return int(is_green_pair(ids[i], ids[i + 1]))


def scan_substitutions(ids, tokenizer, pos_map, min_drop=1):
    found = []

    for i in range(1, len(ids) - 1):
        if not is_word_start(tokenizer, ids[i]):
            continue

        tok = pos_map.get(i)
        if tok is None or tok.pos_ not in REPLACEABLE_POS:
            continue
        if tok.ent_type_ or tok.text[0].isupper():
            continue

        word = tokenizer.decode([ids[i]]).strip()
        if not word.isalpha() or len(word) < 3:
            continue
        if word.lower() != tok.text.lower():
            continue

        before = _green(ids, i - 1) + _green(ids, i)

        for syn in synonyms(word, tok.pos_):
            # WordNet lemmas are base forms. Inflect to match
            # the source tag, otherwise an inflected source
            # receives a mismatched replacement:
            #   matters (NNS) -> matter  "These are also matter"
            #   reaches (VBZ) -> hit     "as he hit two ..."
            if tok.tag_ not in BASE_FORM_TAGS:
                forms = getInflection(syn, tag=tok.tag_)
                if not forms:
                    continue
                syn = forms[0]

            enc = tokenizer.encode(
                " " + syn, add_special_tokens=False
            )
            if len(enc) != 1 or enc[0] == ids[i]:
                continue

            new_id = enc[0]
            after = int(is_green_pair(ids[i - 1], new_id)) \
                + int(is_green_pair(new_id, ids[i + 1]))

            drop = before - after
            if drop >= min_drop:
                found.append({
                    **make_edit(i, "replace", new_id,
                                f"{word}->{syn}"),
                    "drop": drop,
                    "pos_tag": tok.pos_,
                })

    return found


def scan_deletions(ids, tokenizer, pos_map, min_drop=1):
    found = []

    for i in range(1, len(ids) - 1):
        if not is_word_start(tokenizer, ids[i]):
            continue

        # The next token must also start a word, otherwise
        # this token is the head of a multi-token word.
        if i + 1 < len(ids) and \
                not is_word_start(tokenizer, ids[i + 1]):
            continue

        tok = pos_map.get(i)
        if tok is None or tok.ent_type_:
            continue

        word = tokenizer.decode([ids[i]]).strip()
        if not word.isalpha() or len(word) < 2:
            continue
        if word.lower() != tok.text.lower():
            continue

        droppable = (
            word.lower() in DROPPABLE
            and tok.pos_ in DELETABLE_POS
        )
        if not droppable:
            continue

        before = _green(ids, i - 1) + _green(ids, i)
        after = int(is_green_pair(ids[i - 1], ids[i + 1]))

        drop = before - after
        if drop >= min_drop:
            found.append({
                **make_edit(i, "delete", None, f"del({word})"),
                "drop": drop,
                "pos_tag": tok.pos_,
            })

    return found


def scan_all(ids, tokenizer, top_k=None, min_drop=1):
    pos_map = build_pos_map(tokenizer, ids)

    out = {
        "replace": scan_substitutions(
            ids, tokenizer, pos_map, min_drop),
        "delete": scan_deletions(
            ids, tokenizer, pos_map, min_drop),
    }

    for key in out:
        out[key].sort(key=lambda e: (-e["drop"], e["pos"]))
        if top_k:
            out[key] = out[key][:top_k]

    return out
