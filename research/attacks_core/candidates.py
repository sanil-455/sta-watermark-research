"""
it looks at the whole document and produces a list of candidate edits which if changed would break grammar or maening
Only whole, meaning-preserving edits are proposed:
  - no subword fragments
  - no proper nouns, numbers, or entities
  - deletions restricted to genuinely droppable words
  - substitutions matched on part of speech

Predicted drops are LOCAL and ignore re-tokenisation ripple;
they rank candidates cheaply. Truth comes from the detector.
"""
# spacy: library that takes raw texts and adds grammatical understanding to it
import spacy
# solves small problems like given a base word and a target grammatical shape,produces the correct inferred word
from lemminflect import getInflection
# dictionary that gruups words that mean the same thing in a particular sense
# it alwasy gives the base word
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

# Fine-grained tags tells that soem of teh words are already base forms.
# WordNet lemma names are always base forms, so an inflected helps
# source word can only receive a mismatched replacement:
#   matters (NNS) -> matter (NN)   "These are also matter"
#   reaches (VBZ) -> hit  (VB)     "as he hit two milestones"
# Restricting substitution to base-form sources removes this
# whole failure class.

# The check is on the SOURCE only. Tagging a candidate in
# isolation is unreliable -- spaCy reads bare "solemn" as a
# verb and bare "place" as a noun -- so comparing source tag
# to isolated candidate tag would reject good edits.
BASE_FORM_TAGS = {"NN", "VB", "VBP", "JJ", "RB"}
# JJ-adjective
# RB- Adverb

# WordNet POS codes keyed by spaCy tag.
# so that rest of the codes can work easily on wordnet mentioning spacy,s tag
WN_POS = {
    "NOUN": wn.NOUN,
    "VERB": wn.VERB,
    "ADJ": wn.ADJ,
    "ADV": wn.ADV,
}
# caches words that have the same meaning in the same context
_SYN_CACHE = {}

def is_word_start(tokenizer, tok_id):
    # Llama marks word-initial tokens with U+2581, 
    # to check if the token starts with a new space a small block character is used '_'
    # this is added onto the main token
    tok = tokenizer.convert_ids_to_tokens([tok_id])[0]
    return tok.startswith("\u2581")

def build_pos_map(tokenizer, ids):
    """
    Map each token index to the spaCy token covering it.
    Decoding each token and walking the text keeps the two
    tokenisations aligned by character offset.
    """
    # first converting token ids to text and then run the same text through spacy to produce doc which is its own analysed version
    # of sentence each carrying grammar tags
    text = tokenizer.decode(ids, skip_special_tokens=True)
    doc = NLP(text)

    # for every spacy token t see where it is in char string from t.idx to t.idx + len(t.text)
    # essentially a mapping which states which chars belong to which word
    spans = [(t.idx, t.idx + len(t.text), t) for t in doc]
    pos_map = {}
    cursor = 0

  # loop through llama's tokens 1 at a time ,deocde this and if it gives nothing skip it
    for i, tok_id in enumerate(ids):
        piece = tokenizer.decode([tok_id])
        if not piece:
            continue
  # find where exactly the llama's text sit in the full decoded string,searching from the position we last found
  # and not from the start every time
      # also else text,find would find the first occurence every time
        start = text.find(piece.strip(), cursor) \
            if piece.strip() else -1
        if start < 0:
            continue

        end = start + len(piece.strip())
        cursor = start
      # check if every llams's character token range fall entirely inside spacy word char range (s to e)
      # if yes we found the word that owns the llama token and store it in pos_map under lalm token index i and thus break
      # entirely inside instead of exactly matches as single spacy word may correspond to multiple llama tokens
        for s, e, t in spans:
            if s <= start and end <= e:
                pos_map[i] = t
                break
    # pos map is a dict connecting Llama token index to the spaCy word/grammar-tag object that covers it.
    return pos_map


def synonyms(word, pos_tag, limit=6):
    key = (word.lower(), pos_tag)
    # if we've already looked up synonyms for this exact (word, pos) combination, return the saved answer instead of asking WordNet again.
    if key in _SYN_CACHE:
        return _SYN_CACHE[key]
    
    wn_pos = WN_POS.get(pos_tag)
    out = []
    # Translate spaCy's POS tag into WordNet's own code using the table from earlier. wn.synsets(word, pos=...)
  # asks WordNet: "give me every synset — every meaning-group — that this word belongs to, restricted to this part of speech.
    if wn_pos:
        # Only the first synset. WordNet orders senses by
        # frequency, so later synsets give rare readings
        # (market->mart, capacity->content) that are valid
        # dictionary entries but wrong in context.
      # loops through all the found words in the synset dict that are synonyms
        for syn in wn.synsets(word.lower(), pos=wn_pos)[:1]:
            for lemma in syn.lemmas():
                name = lemma.name()
                if "_" in name or name.lower() == word.lower():# accepts multiple words connected by _
                    continue
                if not name.isalpha() or name[0].isupper(): # skip ones with caps as 1st letter as it denotes proper noun
                    continue
                if name not in out:
                    out.append(name)
  # cap the limit at 6 (default) candidates and then save it to cache and return
    out = out[:limit]
    _SYN_CACHE[key] = out
    return out
# converts the fns's green pair at i,i+1 from boolean to 1/0 so that it can be added as convenience later
def _green(ids, i):
    return int(is_green_pair(ids[i], ids[i + 1]))

def scan_substitutions(ids, tokenizer, pos_map, min_drop=1):
    found = []
# scans pairs from 1 to n-1 as they are pairs so makes no sense to start from 0
    for i in range(1, len(ids) - 1):
      # skip anything that is not the start of the word. this allows only words with emaning to be choosen rather than subfragments
        if not is_word_start(tokenizer, ids[i]):
            continue
  # look up at the grammar behind the coosen word using pos_map
  # If there's no match, or the part of speech isn't one of the allowed replaceable categories, skip it
      #Also skip if spacy detected this as part of a named entity
        tok = pos_map.get(i)
        if tok is None or tok.pos_ not in REPLACEABLE_POS:
            continue
        if tok.ent_type_ or tok.text[0].isupper():
            continue
  # skip any words that is <3 char long or not purely alphabetic
        word = tokenizer.decode([ids[i]]).strip()
        if not word.isalpha() or len(word) < 3:
            continue
      # also we check if the decoded Llama token actually match the text of the spacy word pos_map
      # it belongs to If they don't match, something has gone wrong in the alignment from build_pos_map else we skip it
        if word.lower() != tok.text.lower():
            continue
    # count how many of the 2 pairs are actually green
        before = _green(ids, i - 1) + _green(ids, i)
  # Now try every candidate synonym WordNet offered for this word
        for syn in synonyms(word, tok.pos_):
            # WordNet lemmas are base forms. Inflect to match
            # the source tag, otherwise an inflected source
            # receives a mismatched replacement:
            #   matters (NNS) -> matter  "These are also matter"
            #   reaches (VBZ) -> hit     "as he hit two ..."
          # if words are not in the base form call lemminflect to inflect upon wordnet to check upon the synonyms
            if tok.tag_ not in BASE_FORM_TAGS:
                forms = getInflection(syn, tag=tok.tag_)
              # If lemminflect can't produce any valid inflected form for this word (forms comes back empty)
              # skip this candidate entirely
                if not forms:
                    continue
              # Otherwise, take the first inflected form it offers and use that as the actual candidate word 
              # overwriting the original base-form syn variable.
                syn = forms[0]
      # encode the original word, check 2 things: if the word substitution is that of a single word
      # else leave as the planfor 1 word to 1 word changes
            enc = tokenizer.encode(
                " " + syn, add_special_tokens=False
            )
            if len(enc) != 1 or enc[0] == ids[i]:
                continue
    # now we check if the substituion were made what changes would be there in red green pair list
            new_id = enc[0]
            after = int(is_green_pair(ids[i - 1], new_id)) \
                + int(is_green_pair(new_id, ids[i + 1]))
    # if we find more >=1 we store it in a dict labeling the words with to be replaced words
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
  # for substituion we check if the first char indicates the start only for 1 word, but for deletion,
        # The next token must also start a word, otherwise
        # this token is the head of a multi-token word.
        if i + 1 < len(ids) and \
                not is_word_start(tokenizer, ids[i + 1]):
            continue
# same grammar lookup and entity cehck as before
        tok = pos_map.get(i)
        if tok is None or tok.ent_type_:
            continue
  # Same alphabetic/alignment checks as substitution though here the minimum length is 2 instead of 3
        word = tokenizer.decode([ids[i]]).strip()
        if not word.isalpha() or len(word) < 2:
            continue
        if word.lower() != tok.text.lower():
            continue
  # checks if the word to be deleted is in both droppable (manual) whitelist
      #its part of speech has to be one of the DELETABLE_POS categories.
        droppable = (
            word.lower() in DROPPABLE
            and tok.pos_ in DELETABLE_POS
        )
        if not droppable:
            continue
# same as substitution except that now i-1 pairs with i+1 as ith word is deleted
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

# build the grammar map once and let build_pos_map run spacy run over thw whole texrt once and then 
# we can reuse its results for both deletion and substituions
def scan_all(ids, tokenizer, top_k=None, min_drop=1):
    pos_map = build_pos_map(tokenizer, ids)

# Sort each list,replace and delete by drop, highest first (the'-' negates it so ascending sort behaves like descending),
# and use e["pos"] as a tiebreaker for anything with equal drop values, keeping results in a stable, predictable order
# If top_k was given, trim each list down to just that many entries. Return a dictionary with two keys, "replace" and delete
# each holding a sorted list of candidate edits,this is the exact structure the beam search consumes as its starting pool.
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
