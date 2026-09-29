"""
Token-level edit representation and application.

An Edit is a dict:
    {"pos": int, "type": "replace"|"delete"|"insert",
     "new_id": int|None, "label": str}

`pos` always refers to the ORIGINAL token sequence. A set of
edits is applied right-to-left so earlier positions stay valid.
"""

from sta_core import text_to_ids


def make_edit(pos, op, new_id=None, label=""):
    if op not in ("replace", "delete", "insert"):
        raise ValueError(f"unknown op: {op!r}")

    if op in ("replace", "insert") and new_id is None:
        raise ValueError(f"{op} needs new_id")

    return {
        "pos": pos,
        "type": op,
        "new_id": new_id,
        "label": label,
    }


def edits_conflict(a, b):
    """
    Two edits conflict if they touch the same slot.

    Insert-before-pos and replace-at-pos are distinct
    operations, but allowing both at one position makes the
    result order-dependent, so they are treated as a conflict.
    """
    return a["pos"] == b["pos"]


def apply_edits(token_ids, edits):
    """
    Apply edits to a token-ID list.

    Insertion places the new token BEFORE `pos`, matching the
    spaCy-based combiners used earlier in this project.
    """
    out = list(token_ids)

    for e in sorted(edits, key=lambda x: x["pos"], reverse=True):
        pos = e["pos"]

        if pos < 0 or pos >= len(token_ids):
            raise IndexError(
                f"pos {pos} outside sequence of length "
                f"{len(token_ids)}"
            )

        if e["type"] == "replace":
            out[pos] = e["new_id"]
        elif e["type"] == "delete":
            del out[pos]
        else:
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
        skip_special_tokens=True,
    )


def roundtrip_ids(tokenizer, token_ids, edits):
    """Edited ids -> text -> ids, as the detector would see."""
    return text_to_ids(
        tokenizer,
        edits_to_text(tokenizer, token_ids, edits),
    )


def edit_signature(edits):
    """Order-independent key identifying a set of edits."""
    return tuple(sorted(
        (e["pos"], e["type"], e["new_id"]) for e in edits
    ))


def describe(edits):
    return " + ".join(
        e["label"] or f"{e['type']}@{e['pos']}"
        for e in sorted(edits, key=lambda x: x["pos"])
    )
