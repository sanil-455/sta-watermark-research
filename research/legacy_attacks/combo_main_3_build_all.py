import json
from pathlib import Path

import spacy


BASELINE_PATH = Path(
    "results/raw/baseline_safe.json"
)

A2_PATH = Path(
    "results/raw/attack2_candidates_by_sample/"
    "sample_1.json"
)

OUT_PATH = Path(
    "results/raw/attack12_combined/"
    "sample_1_all_combinations.jsonl"
)

SUMMARY_PATH = Path(
    "results/raw/attack12_combined/"
    "sample_1_all_combinations_summary.json"
)

PROMPT_ID = 0
SAMPLE_NUMBER = 1

EXPECTED_BASELINE_Z = 3.363765319856875

A1_OPERATIONS = [
    ("might", "could"),
    ("might", "can"),
]

NLP = None
A1_POSITION = None


def load_json(path):
    with path.open() as f:
        return json.load(f)


def replace_word(text, old, new):
    """
    Replace the single occurrence of `old` with `new`.

    Fails loudly if the word is absent or ambiguous.
    """

    doc = NLP(text)

    matches = [
        t for t in doc
        if t.text == old
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one '{old}', "
            f"found {len(matches)}"
        )

    t = matches[0]

    return (
        text[:t.idx]
        + new
        + text[t.idx + len(t.text):]
    )


def apply_candidate(doc, candidate, expect_word=None):
    """
    Apply one A2 insertion or deletion to `doc`.

    `expect_word` (when given) is the token text that must
    be present at `position`. This guards against applying
    a baseline-derived position to a document whose token
    at that position has changed.
    """

    position = candidate["position"]

    if not isinstance(position, int):
        raise ValueError(
            f"position is not an int: {position!r}"
        )

    if position < 0 or position >= len(doc):
        raise ValueError(
            f"position {position} outside token "
            f"sequence of length {len(doc)}"
        )

    token = doc[position]

    if expect_word is not None and token.text != expect_word:
        raise ValueError(
            f"anchor mismatch at position {position}: "
            f"expected {expect_word!r}, got {token.text!r}"
        )

    attack_type = candidate["attack_type"]

    if attack_type == "deletion":

        if token.text != candidate["original_word"]:
            raise ValueError(
                f"token mismatch: expected "
                f"{candidate['original_word']!r}, "
                f"got {token.text!r}"
            )

        start = token.idx
        end = token.idx + len(token.text)

        return doc.text[:start] + doc.text[end:]

    if attack_type == "insertion":

        word = candidate.get("replacement")

        if not word:
            raise ValueError(
                "insertion candidate has no replacement"
            )

        if position == 0:
            return word + " " + doc.text

        start = token.idx

        return (
            doc.text[:start]
            + word
            + " "
            + doc.text[start:]
        )

    raise ValueError(
        f"unknown attack type: {attack_type!r}"
    )


def collides_with_a1(candidate):
    """
    An A2 operation collides with A1 when it targets the
    token A1 already rewrote.

    A deletion at A1_POSITION would remove or mismatch the
    A1 replacement. An insertion at A1_POSITION is anchored
    on a token whose identity A1 changed.
    """

    return candidate["position"] == A1_POSITION


def resolve_a1_position(baseline_doc):
    """
    Locate the A1 anchor word in spaCy token space.

    Positions carried over from the Llama tokenizer are NOT
    valid here. spaCy tokenizes differently, so the anchor
    must be found in this token space rather than assumed.
    """

    a1_source_words = {
        old for old, _ in A1_OPERATIONS
    }

    if len(a1_source_words) != 1:
        raise RuntimeError(
            "All A1 operations must share one source "
            f"word; found {sorted(a1_source_words)}"
        )

    a1_source = a1_source_words.pop()

    matches = [
        t.i for t in baseline_doc
        if t.text == a1_source
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {a1_source!r} in "
            f"baseline, found {len(matches)} at "
            f"positions {matches}"
        )

    return a1_source, matches[0]


def main():

    global NLP, A1_POSITION

    NLP = spacy.blank("en")

    # --------------------------------------------------
    # Load and verify baseline
    # --------------------------------------------------

    baseline_rows = load_json(BASELINE_PATH)

    matches = [
        r for r in baseline_rows
        if int(r["prompt_id"]) == PROMPT_ID
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Could not uniquely identify "
            f"prompt_id={PROMPT_ID}; "
            f"found {len(matches)} rows"
        )

    baseline_row = matches[0]

    baseline = baseline_row["watermarked_text"]

    stored_z = float(
        baseline_row["watermarked_z"]
    )

    if abs(stored_z - EXPECTED_BASELINE_Z) > 1e-9:
        raise RuntimeError(
            f"Baseline z mismatch: expected "
            f"{EXPECTED_BASELINE_Z}, got {stored_z}"
        )

    candidate_data = load_json(A2_PATH)

    if int(candidate_data.get("prompt_id", PROMPT_ID)) != PROMPT_ID:
        raise RuntimeError(
            "A2 candidate file prompt_id mismatch"
        )

    candidates = candidate_data["candidates"]

    print("========== DIAGNOSTIC ==========")
    print("Baseline characters:", len(baseline))
    print("Baseline stored z:", stored_z)
    print("A2 candidates:", len(candidates))

    baseline_doc = NLP(baseline)

    print(
        "Baseline spaCy tokens:",
        len(baseline_doc)
    )

    # --------------------------------------------------
    # Resolve A1 position in spaCy token space
    # --------------------------------------------------

    a1_source, A1_POSITION = resolve_a1_position(
        baseline_doc
    )

    print(
        f"A1 anchor {a1_source!r} resolved to spaCy "
        f"token {A1_POSITION}"
    )

    # --------------------------------------------------
    # Build and verify A1 variants
    # --------------------------------------------------

    a1_texts = []

    for old, new in A1_OPERATIONS:

        attacked = replace_word(
            baseline,
            old,
            new,
        )

        attacked_doc = NLP(attacked)

        if len(attacked_doc) != len(baseline_doc):
            raise RuntimeError(
                f"A1 {old}->{new} changed token count "
                f"({len(baseline_doc)} -> "
                f"{len(attacked_doc)})"
            )

        if attacked_doc[A1_POSITION].text != new:
            raise RuntimeError(
                f"A1 {old}->{new} did not land at "
                f"token {A1_POSITION}: got "
                f"{attacked_doc[A1_POSITION].text!r}"
            )

        a1_texts.append(
            {
                "old": old,
                "new": new,
                "text": attacked,
                "doc": attacked_doc,
            }
        )

        print(
            f"A1 CHECK OK: {old}->{new} | "
            f"tokens={len(attacked_doc)}"
        )

    # --------------------------------------------------
    # Validate every A2 candidate against its saved text
    # --------------------------------------------------

    valid_a2 = []
    invalid_a2 = []
    collided_a2 = []

    for i, candidate in enumerate(candidates):

        try:

            constructed = apply_candidate(
                baseline_doc,
                candidate,
            )

            saved = candidate.get("attacked_text")

            if not saved:
                raise ValueError(
                    "candidate has no attacked_text "
                    "to validate against"
                )

            if constructed != saved:
                raise ValueError(
                    "constructed text != saved "
                    "attacked_text"
                )

            if collides_with_a1(candidate):

                collided_a2.append(
                    {
                        "candidate_index": i,
                        "position":
                            candidate["position"],
                        "attack_type":
                            candidate["attack_type"],
                        "reason":
                            f"targets A1 position "
                            f"{A1_POSITION}",
                    }
                )

                continue

            valid_a2.append(
                {
                    "candidate_index": i,
                    **candidate,
                }
            )

        except Exception as exc:

            invalid_a2.append(
                {
                    "candidate_index": i,
                    "position":
                        candidate.get("position"),
                    "attack_type":
                        candidate.get("attack_type"),
                    "error_type":
                        type(exc).__name__,
                    "reason": str(exc),
                }
            )

    print()
    print(
        "A2 exact validation:",
        len(valid_a2),
        "valid /",
        len(invalid_a2),
        "invalid /",
        len(collided_a2),
        "collided with A1",
    )

    expected_total = len(valid_a2) * len(a1_texts)

    print("Potential combinations:", expected_total)

    if len(valid_a2) == 0:
        raise RuntimeError(
            "Zero valid A2 candidates. Stopping. "
            "Check whether candidate positions were "
            "generated under a different tokenizer."
        )

    # --------------------------------------------------
    # Build every combination
    # --------------------------------------------------

    OUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if OUT_PATH.exists():
        OUT_PATH.unlink()

    saved_count = 0
    failed = []

    with OUT_PATH.open("w") as out:

        for a1 in a1_texts:

            for a2 in valid_a2:

                try:

                    # The anchor token at this position is
                    # unchanged by A1, because A1 rewrote
                    # only A1_POSITION and collisions were
                    # already filtered out.
                    anchor = baseline_doc[
                        a2["position"]
                    ].text

                    combined = apply_candidate(
                        a1["doc"],
                        a2,
                        expect_word=anchor,
                    )

                    if combined == baseline:
                        raise RuntimeError(
                            "Combined text equals baseline"
                        )

                    if combined == a1["text"]:
                        raise RuntimeError(
                            "A2 edit had no effect"
                        )

                    combined_doc = NLP(combined)

                    # Verify the A1 edit survived at its own
                    # position, not merely somewhere in the
                    # text.
                    a1_offset = 0

                    if a2["attack_type"] == "insertion":
                        if a2["position"] <= A1_POSITION:
                            a1_offset = 1

                    elif a2["attack_type"] == "deletion":
                        if a2["position"] < A1_POSITION:
                            a1_offset = -1

                    a1_index = A1_POSITION + a1_offset

                    if a1_index < 0 or a1_index >= len(combined_doc):
                        raise RuntimeError(
                            f"A1 position {a1_index} "
                            f"outside combined sequence "
                            f"of length {len(combined_doc)}"
                        )

                    if combined_doc[a1_index].text != a1["new"]:
                        raise RuntimeError(
                            f"A1 replacement not found at "
                            f"token {a1_index}: got "
                            f"{combined_doc[a1_index].text!r}, "
                            f"expected {a1['new']!r}"
                        )

                    row = {
                        "sample_number": SAMPLE_NUMBER,
                        "prompt_id": PROMPT_ID,
                        "a1_original_word": a1["old"],
                        "a1_replacement": a1["new"],
                        "a1_position": A1_POSITION,
                        "a1_position_in_combined":
                            a1_index,
                        "a2_candidate_index":
                            a2["candidate_index"],
                        "a2_attack_type":
                            a2["attack_type"],
                        "a2_original_word":
                            a2.get("original_word", ""),
                        "a2_position":
                            a2["position"],
                        "a2_replacement":
                            a2.get("replacement", ""),
                        "combined_text": combined,
                        "baseline_char_count":
                            len(baseline),
                        "combined_char_count":
                            len(combined),
                        "char_count_change":
                            len(combined) - len(baseline),
                        "baseline_word_count":
                            len(baseline.split()),
                        "combined_word_count":
                            len(combined.split()),
                        "word_count_change":
                            len(combined.split())
                            - len(baseline.split()),
                        "baseline_spacy_tokens":
                            len(baseline_doc),
                        "combined_spacy_tokens":
                            len(combined_doc),
                    }

                    out.write(
                        json.dumps(row) + "\n"
                    )
                    out.flush()

                    saved_count += 1

                    if saved_count % 50 == 0:
                        print(
                            f"[✓ SAVED] {saved_count}/"
                            f"{expected_total}"
                        )

                except Exception as exc:

                    # One bad combination must not stop
                    # the run.
                    failed.append(
                        {
                            "a1_original_word": a1["old"],
                            "a1_replacement": a1["new"],
                            "a2_candidate_index":
                                a2["candidate_index"],
                            "a2_attack_type":
                                a2["attack_type"],
                            "a2_position":
                                a2["position"],
                            "error_type":
                                type(exc).__name__,
                            "error": str(exc),
                        }
                    )

                    print(
                        f"[! FAILED] a1={a1['old']}->"
                        f"{a1['new']} a2_idx="
                        f"{a2['candidate_index']} | "
                        f"{type(exc).__name__}: {exc}"
                    )

                    continue

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    summary = {
        "sample_number": SAMPLE_NUMBER,
        "prompt_id": PROMPT_ID,
        "baseline_z": stored_z,
        "baseline_spacy_tokens": len(baseline_doc),
        "a1_anchor_word": a1_source,
        "a1_position": A1_POSITION,
        "a1_operations": len(a1_texts),
        "a2_total": len(candidates),
        "a2_valid": len(valid_a2),
        "a2_invalid": len(invalid_a2),
        "a2_collided_with_a1": len(collided_a2),
        "expected_combinations": expected_total,
        "combinations_saved": saved_count,
        "combinations_failed": len(failed),
        "invalid_a2_records": invalid_a2,
        "collided_a2_records": collided_a2,
        "failed_records": failed,
        "output": str(OUT_PATH),
    }

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with SUMMARY_PATH.open("w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("========== COMPLETE ==========")
    print("A1 position (spaCy):", A1_POSITION)
    print("A2 valid:", len(valid_a2))
    print("A2 invalid:", len(invalid_a2))
    print("A2 collided with A1:", len(collided_a2))
    print("Combinations saved:", saved_count)
    print("Combinations failed:", len(failed))
    print(f"[✓ SAVED] {OUT_PATH}")
    print(f"[✓ SAVED] {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
