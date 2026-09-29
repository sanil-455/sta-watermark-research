"""
An Edit is a dict:
    {"pos": int, "type": "replace"|"delete"|"insert",
     "new_id": int|None, "label": str}
"""
# pos is the original pos in sequence
# removal is done right to left so as not to disturb the pos of other words and change their score 
from sta_core import text_to_ids

def make_edit(pos, op, new_id=None, label=""):
    # op- what kind of operation to be performed
    if op not in ("replace", "delete", "insert"):
        raise ValueError(f"unknown op: {op!r}")

    # this basically means if no new id id provided for operations such as replace and insert raise error
    # as default :new id id 0
    if op in ("replace", "insert") and new_id is None:
        raise ValueError(f"{op} needs new_id")
    
    return {
        "pos": pos,
        "type": op,
        "new_id": new_id,
        "label": label,
    }


def edits_conflict(a, b):
 #if two edits (eg: insert,replace) are simultaneously applied at the same position it acuses a conflict
    return a["pos"] == b["pos"]


def apply_edits(token_ids, edits):
    """
    Apply edits to a token-ID list.
    Insertion places the new token BEFORE `pos`, matching the
    spaCy-based combiners used earlier in this project.
    """
    out = list(token_ids)

    for e in sorted(edits, key=lambda x: x["pos"], reverse=True):
    # sort the edits by position,highest first by looking at the pos value
        pos = e["pos"]

        if pos < 0 or pos >= len(token_ids):
            raise IndexError(
                f"pos {pos} outside sequence of length "
                f"{len(token_ids)}"
            )
            # replacement
        if e["type"] == "replace":
            out[pos] = e["new_id"]
            # deletion
        elif e["type"] == "delete":
            del out[pos]
        else:
            # insertion
            out.insert(pos, e["new_id"])

    return out


def edits_to_text(tokenizer, token_ids, edits):
    """
    Apply edits and decode to text.
    Decoding is mandatory: the detector receives text and
    re-tokenizes, so an edit's true effect includes any
    re-segmentation it causes downstream.
    """
    return tokenizer.decode(
        apply_edits(token_ids, edits),
        #calls apply edits : converts numbers back to actual strings taht are readable
        skip_special_tokens=True,
    )


def roundtrip_ids(tokenizer, token_ids, edits):
    """Edited ids -> text -> ids, as the detector would see."""
    return text_to_ids(
        tokenizer,
        edits_to_text(tokenizer, token_ids, edits),
    )


def edit_signature(edits):
    return tuple(sorted(
        (e["pos"], e["type"], e["new_id"]) for e in edits
    ))


def describe(edits):
    return " + ".join(
        e["label"] or f"{e['type']}@{e['pos']}"
        for e in sorted(edits, key=lambda x: x["pos"])
    )
