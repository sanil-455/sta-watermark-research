"""
Candidate generation without the watermark key.

Everything linguistic in candidates.py works without H1 or H2:
word boundaries, part of speech, named entities, WordNet senses,
morphological inflection. Only the green-pair arithmetic needs
the key, and that is removed here.

The result is a pool of valid substitutions with no prediction
of which ones help. Ranking them is the oracle's job.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "attacks_core"))
# library that changes a word's grammatical form.
# we need it as WordNet — the dictionary we use for synonyms — always returns words in their base form
from lemminflect import getInflection

# NLP is the grammar analyser. DROPPABLE is a list of words safe to delete.
# DELETABLE_POS and REPLACEABLE_POS are lists of word types.
# BASE_FORM_TAGS marks which grammatical forms are base forms. 
# is_word_start checks whether a token begins a word.
#build_pos_map works out each word's grammatical role. synonyms looks words up in the dictionary.
from candidates import (
    NLP, DROPPABLE, DELETABLE_POS, REPLACEABLE_POS,
    BASE_FORM_TAGS, is_word_start, build_pos_map, synonyms,
)
from edit_ops import make_edit

def propose_substitutions(ids, tokenizer, pos_map):
    """
    Every grammatically valid single-token substitution.

    Identical to scan_substitutions except that no green-pair
    arithmetic happens and no `drop` is recorded, because a
    key-free attacker cannot compute either.
    """
    found = []

    for i in range(1, len(ids) - 1):
        # ie. if its not a complete english word and just a token adn hence is_word_start is false we continue
        if not is_word_start(tokenizer, ids[i]):
            continue
        # checking for the part of speech that i belongs to
        tok = pos_map.get(i)
        if tok is None or tok.pos_ not in REPLACEABLE_POS:
            continue
        # ig analyser recognised it as a named entity,a proper noun or so continue
        if tok.ent_type_ or tok.text[0].isupper():
            continue
        # Turn the number back into text
        word = tokenizer.decode([ids[i]]).strip()
        # rejects punctuation,very short words that usually dont have useful meaning and numbers
        if not word.isalpha() or len(word) < 3:
            continue
        # tokeniser and analyser should say the same thing
        if word.lower() != tok.text.lower():
            continue
        # check for the synonyms but within the restricted parts of speech
        for syn in synonyms(word, tok.pos_):
            if tok.tag_ not in BASE_FORM_TAGS:
                forms = getInflection(syn, tag=tok.tag_)
                # if inflection failed skip the word
                if not forms:
                    continue
                syn = forms[0]
            # convert synonym to numebers , leading space matters
            enc = tokenizer.encode(" " + syn, add_special_tokens=False)
            # we need a single token substitution else it will change the length of every token after them
            if len(enc) != 1 or enc[0] == ids[i]:
                continue
            # recored the change we done(ie. the substitution) and compare it with the white box attack
            found.append(make_edit(i, "replace", enc[0], f"{word}->{syn}"))
    return found
# here in blackbox attack the no.of deletions is more than whitebox as:
# because white-box discards anything it can see won't help, while black-box has to keep everything.

# the explanation of below is just like that of replacemnt case
def propose_deletions(ids, tokenizer, pos_map):
    """Every safe single-token deletion, again unscored."""
    found = []

    for i in range(1, len(ids) - 1):
        if not is_word_start(tokenizer, ids[i]):
            continue
        if i + 1 < len(ids) and not is_word_start(tokenizer, ids[i + 1]):
            continue

        tok = pos_map.get(i)
        if tok is None or tok.ent_type_:
            continue

        word = tokenizer.decode([ids[i]]).strip()
        if not word.isalpha() or len(word) < 2:
            continue
        if word.lower() != tok.text.lower():
            continue

        if word.lower() not in DROPPABLE or tok.pos_ not in DELETABLE_POS:
            continue

        found.append(make_edit(i, "delete", None, f"del({word})"))

    return found

def propose_all(ids, tokenizer):
    """Build the pos map once, then both candidate kinds."""
    pos_map = build_pos_map(tokenizer, ids)
    return (
        propose_substitutions(ids, tokenizer, pos_map)
        + propose_deletions(ids, tokenizer, pos_map)
    )
