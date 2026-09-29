import json
from difflib import SequenceMatcher
from pathlib import Path

import spacy


BASELINE_PATH = Path(
    "results/raw/baseline_safe.json"
)

A1_DIR = Path(
    "results/raw/manual_attacks/sample2_replacements"
)

A2_PATH = Path(
    "results/raw/attack2_candidates_by_sample/sample_2.json"
)

OUT_DIR = Path(
    "results/raw/attack12_combined_sample2"
)

OUT_PATH = OUT_DIR / "sample_2_all_combinations.jsonl"
SUMMARY_PATH = OUT_DIR / "sample_2_all_combinations_summary.json"

PROMPT_ID = 1
SAMPLE_NUMBER = 2

EXPECTED_BASELINE_Z = 2.8160458729081004
Z_THRESHOLD = 2.0

EXPECTED_A1_COUNT = 13
EXPECTED_A2_COUNT = 1531

NLP = None


def load_json(path):
    with path.open() as f:
        return json.load(f)


def recover_single_edit(original, attacked):
    """
    Recover the single character-level edit that produced
    `attacked` from `original`.

    This is used instead of searching for the replaced word
    by string match, because a word such as 'growth' can
    occur several times in the baseline and a string search
    cannot tell which occurrence the original attack edited.
    The saved attacked_text resolves that unambiguously.
    """

    opcodes = SequenceMatcher(
        None,
        original,
        attacked,
        autojunk=False,
    ).get_opcodes()

    changes = [
        op for op in opcodes
        if op[0] != "equal"
    ]

    if len(changes) != 1:
        raise ValueError(
            f"Expected exactly one localized edit, "
            f"found {len(changes)}"
        )

    tag, i1, i2, j1, j2 = changes[0]

    if tag != "replace":
        raise ValueError(
            f"Expected a substitution, got {tag!r}"
        )

    rebuilt = (
        original[:i1]
        + attacked[j1:j2]
        + original[i2:]
    )

    if rebuilt != attacked:
        raise ValueError(
            "Recovered edit does not reproduce the "
            "saved attacked text"
        )

    return {
        "start": i1,
        "end": i2,
        "old_text": original[i1:i2],
        "new_text": attacked[j1:j2],
    }


def token_index_for_char_span(doc, start, end):
    """
    Return the index of the single spaCy token covering
    [start, end).

    Raises if the span does not align to exactly one token,
    because every downstream position calculation assumes
    A1 replaces exactly one baseline token.
    """

    covered = [
        t.i
        for t in doc
        if t.idx < end and (t.idx + len(t.text)) > start
    ]

    if len(covered) != 1:
        raise ValueError(
            f"Character span [{start}, {end}) covers "
            f"{len(covered)} spaCy tokens, expected 1"
        )

    index = covered[0]
    token = doc[index]

    if token.idx != start or (
        token.idx + len(token.text)
    ) != end:
        raise ValueError(
            f"Character span [{start}, {end}) does not "
            f"align to token {index} "
            f"[{token.idx}, "
            f"{token.idx + len(token.text)})"
        )

    return index


def apply_a2(doc, candidate, position, expect_word=None):
    """
    Apply one A2 insertion or deletion to `doc`.

    `position` is the position in the CURRENT spaCy
    document, not necessarily the original baseline
    position.

    `expect_word` guards against applying a baseline-derived
    position to the wrong token after A1 has changed the
    token count.
    """

    if not isinstance(position, int):
        raise ValueError(
            f"position is not an int: {position!r}"
        )

    if position < 0 or position >= len(doc):
        raise ValueError(
            f"position {position} outside token sequence "
            f"of length {len(doc)}"
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


def map_a2_position_after_a1(
    baseline_position,
    a1_position,
    a1_token_delta,
):
    """
    Map a baseline spaCy token position into the A1-modified
    spaCy token sequence.

    A1 replaces exactly one baseline token, possibly with
    several tokens. Positions before A1 are unchanged.
    Positions after A1 shift by the token-count delta A1
    introduced.

    A2 at the A1 position itself is a collision and is
    handled by the caller.
    """

    if baseline_position < a1_position:
        return baseline_position

    if baseline_position > a1_position:
        return baseline_position + a1_token_delta

    raise ValueError(
        "A2 position equals A1 position; this pair must "
        "be treated as a collision"
    )


def a1_position_after_a2(a1_position, attack_type, a2_position):
    """
    Where the A1 token lands after the A2 edit is applied.

    An insertion at or before the A1 position shifts it
    right by one. A deletion strictly before it shifts it
    left by one. An edit after it leaves it unchanged.

    Insertion uses <= because the inserted word is placed
    immediately before the anchor token, so an insertion at
    exactly the A1 position still pushes A1 right.

    Both arguments must already be in the A1-modified
    coordinate system.
    """

    if attack_type == "insertion":
        if a2_position <= a1_position:
            return a1_position + 1

    elif attack_type == "deletion":
        if a2_position < a1_position:
            return a1_position - 1

    return a1_position


def main():
    global NLP

    # Same tokenizer used for word-level attack
    # construction. No statistical components required.
    NLP = spacy.blank("en")

    # --------------------------------------------------
    # Baseline
    # --------------------------------------------------

    baseline_data = load_json(BASELINE_PATH)

    matches = [
        x for x in baseline_data
        if int(x["prompt_id"]) == PROMPT_ID
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Could not uniquely identify "
            f"prompt_id={PROMPT_ID}; "
            f"found {len(matches)} rows"
        )

    sample = matches[0]

    baseline = sample["watermarked_text"]

    stored_z = float(sample["watermarked_z"])

    print("=" * 90)
    print("SAMPLE 2 - A1 x A2 COMBINATION BUILDER")
    print("=" * 90)

    print("Sample:", SAMPLE_NUMBER)
    print("Prompt ID:", PROMPT_ID)
    print("Expected baseline z:", EXPECTED_BASELINE_Z)
    print("Stored baseline z:", stored_z)

    if abs(stored_z - EXPECTED_BASELINE_Z) > 1e-6:
        raise RuntimeError(
            f"Baseline z mismatch: expected "
            f"{EXPECTED_BASELINE_Z}, got {stored_z}"
        )

    if stored_z <= Z_THRESHOLD:
        raise RuntimeError(
            f"Sample 2 baseline z={stored_z} is not above "
            f"the detection threshold {Z_THRESHOLD}. "
            f"A watermark break would be undefined."
        )

    baseline_doc = NLP(baseline)

    print("Baseline characters:", len(baseline))
    print("Baseline spaCy tokens:", len(baseline_doc))

    # --------------------------------------------------
    # Load and deduplicate the verified A1 operations
    # --------------------------------------------------

    a1_candidates = {}
    a1_skipped = []

    for path in sorted(A1_DIR.glob("*.json")):

        try:
            data = load_json(path)

        except Exception as exc:
            a1_skipped.append({
                "file": str(path),
                "reason": f"unreadable: {exc}",
            })
            continue

        records = (
            data if isinstance(data, list) else [data]
        )

        for r in records:

            if not isinstance(r, dict):
                continue

            if r.get("sample") != SAMPLE_NUMBER:
                continue

            # Only completed attacks are valid A1 sources.
            # Error records carry no attacked_text.
            if r.get("status") != "completed":
                a1_skipped.append({
                    "file": str(path),
                    "position": r.get("position"),
                    "reason":
                        f"status={r.get('status')!r}",
                })
                continue

            if (
                r.get("position") is None
                or r.get("original_word") is None
                or r.get("replacement") is None
                or r.get("attacked_text") is None
            ):
                a1_skipped.append({
                    "file": str(path),
                    "position": r.get("position"),
                    "reason": "missing required field",
                })
                continue

            key = (
                r["position"],
                r["original_word"],
                r["replacement"],
            )

            a1_candidates[key] = {
                "position": r["position"],
                "old": r["original_word"],
                "new": r["replacement"],
                "attacked_text": r["attacked_text"],
                "source_file": str(path),
            }

    a1 = sorted(
        a1_candidates.values(),
        key=lambda x: (
            x["position"],
            x["old"],
            x["new"],
        ),
    )

    print()
    print("A1 unique candidates:", len(a1))
    print("A1 records skipped:", len(a1_skipped))

    if len(a1) != EXPECTED_A1_COUNT:
        raise RuntimeError(
            f"Expected exactly {EXPECTED_A1_COUNT} unique "
            f"A1 candidates, found {len(a1)}"
        )

    # --------------------------------------------------
    # Validate A1 candidates and resolve spaCy positions
    #
    # The stored `position` is Llama-tokenizer provenance
    # and is not valid in spaCy space. The spaCy position
    # is recovered from the saved attacked_text by diff,
    # which is unambiguous even when the replaced word
    # occurs several times in the baseline.
    #
    # A1 is allowed to change the spaCy token count, so
    # that multi-word replacements such as
    # 'releases' -> 'eases off' are supported.
    # --------------------------------------------------

    for i, candidate in enumerate(a1, 1):

        edit = recover_single_edit(
            baseline,
            candidate["attacked_text"],
        )

        if edit["old_text"] != candidate["old"]:
            raise RuntimeError(
                f"A1 recovered old text "
                f"{edit['old_text']!r} does not match "
                f"recorded original_word "
                f"{candidate['old']!r}"
            )

        if edit["new_text"] != candidate["new"]:
            raise RuntimeError(
                f"A1 recovered new text "
                f"{edit['new_text']!r} does not match "
                f"recorded replacement "
                f"{candidate['new']!r}"
            )

        spacy_position = token_index_for_char_span(
            baseline_doc,
            edit["start"],
            edit["end"],
        )

        a1_doc = NLP(candidate["attacked_text"])

        a1_token_delta = len(a1_doc) - len(baseline_doc)

        if spacy_position >= len(a1_doc):
            raise RuntimeError(
                f"A1 position {spacy_position} outside "
                f"A1 document of length {len(a1_doc)}"
            )

        # The replacement may span several spaCy tokens.
        # Verify the whole replacement sits at the old
        # position, not merely its first token.
        replacement_doc = NLP(candidate["new"])

        replacement_length = len(replacement_doc)

        if replacement_length == 0:
            raise RuntimeError(
                f"A1 replacement produced zero spaCy "
                f"tokens: {candidate['new']!r}"
            )

        if replacement_length - 1 != a1_token_delta:
            raise RuntimeError(
                f"A1 token delta {a1_token_delta:+d} is "
                f"inconsistent with a replacement of "
                f"{replacement_length} tokens"
            )

        end_position = spacy_position + replacement_length

        if end_position > len(a1_doc):
            raise RuntimeError(
                f"A1 replacement span exceeds A1 "
                f"document length"
            )

        found_span = [
            a1_doc[j].text
            for j in range(spacy_position, end_position)
        ]

        expected_span = [
            t.text for t in replacement_doc
        ]

        if found_span != expected_span:
            raise RuntimeError(
                f"A1 {candidate['old']}->"
                f"{candidate['new']} did not land at "
                f"spaCy tokens "
                f"[{spacy_position}, {end_position}): "
                f"got {found_span}, expected "
                f"{expected_span}"
            )

        candidate["spacy_position"] = spacy_position
        candidate["a1_token_delta"] = a1_token_delta
        candidate["a1_spacy_token_count"] = len(a1_doc)
        candidate["replacement_token_count"] = (
            replacement_length
        )
        candidate["char_start"] = edit["start"]
        candidate["char_end"] = edit["end"]
        candidate["doc"] = a1_doc

        print(
            f"A1 {i:02d}: "
            f"source_pos={candidate['position']} "
            f"spacy_pos={spacy_position} "
            f"chars=[{edit['start']},{edit['end']}) | "
            f"{candidate['old']} -> "
            f"{candidate['new']} | "
            f"spaCy tokens "
            f"{len(baseline_doc)} -> {len(a1_doc)} "
            f"(delta {a1_token_delta:+d})"
        )

    # --------------------------------------------------
    # Load A2
    # --------------------------------------------------

    a2_data = load_json(A2_PATH)

    if int(a2_data["sample_number"]) != SAMPLE_NUMBER:
        raise RuntimeError("A2 sample number mismatch")

    if int(a2_data["prompt_id"]) != PROMPT_ID:
        raise RuntimeError("A2 prompt ID mismatch")

    if abs(
        float(a2_data["baseline_z"])
        - EXPECTED_BASELINE_Z
    ) > 1e-6:
        raise RuntimeError("A2 baseline z mismatch")

    all_a2 = a2_data["candidates"]

    if len(all_a2) != EXPECTED_A2_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_A2_COUNT} A2 candidates, "
            f"found {len(all_a2)}"
        )

    print()
    print("A2 candidates:", len(all_a2))

    # --------------------------------------------------
    # Validate A2 candidates by exact reconstruction
    # --------------------------------------------------

    valid_a2 = []
    invalid_a2 = []

    for index, candidate in enumerate(all_a2):

        try:

            position = candidate["position"]

            if (
                not isinstance(position, int)
                or position < 0
                or position >= len(baseline_doc)
            ):
                raise ValueError(
                    f"invalid position {position!r} for "
                    f"doc of length {len(baseline_doc)}"
                )

            saved = candidate.get("attacked_text")

            if not saved:
                raise ValueError(
                    "candidate has no attacked_text"
                )

            anchor = baseline_doc[position].text

            reconstructed = apply_a2(
                baseline_doc,
                candidate,
                position,
                expect_word=anchor,
            )

            if reconstructed != saved:
                raise ValueError(
                    "saved attacked_text does not match "
                    "reconstruction"
                )

            valid_a2.append(candidate)

        except Exception as exc:

            invalid_a2.append({
                "candidate_index":
                    candidate.get(
                        "candidate_index", index
                    ),
                "position": candidate.get("position"),
                "attack_type":
                    candidate.get("attack_type"),
                "error_type": type(exc).__name__,
                "error": str(exc),
            })

    print("Valid A2:", len(valid_a2))
    print("Invalid A2:", len(invalid_a2))

    if len(valid_a2) == 0:
        raise RuntimeError(
            "Zero valid A2 candidates. Stopping. Check "
            "whether candidate positions were generated "
            "under a different tokenizer."
        )

    # --------------------------------------------------
    # Expected combination count
    #
    # A collision is per-pair, not global. An A2 candidate
    # at the same spaCy position as one A1 operation is
    # still usable with all the others.
    # --------------------------------------------------

    expected_total = sum(
        1
        for a1_candidate in a1
        for c in valid_a2
        if c["position"]
        != a1_candidate["spacy_position"]
    )

    total_collisions = (
        len(a1) * len(valid_a2)
    ) - expected_total

    print()
    print(
        "Per-pair collisions excluded:",
        total_collisions,
    )
    print("Expected combinations:", expected_total)

    # --------------------------------------------------
    # Build every combination
    # --------------------------------------------------

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if OUT_PATH.exists():
        OUT_PATH.unlink()

    saved_count = 0
    skipped = []
    failed = []

    with OUT_PATH.open("w") as out:

        for a1_candidate in a1:

            a1_spacy_position = a1_candidate[
                "spacy_position"
            ]

            a1_doc = a1_candidate["doc"]

            a1_token_delta = a1_candidate[
                "a1_token_delta"
            ]

            replacement_length = a1_candidate[
                "replacement_token_count"
            ]

            for a2 in valid_a2:

                a2_baseline_position = a2["position"]

                # Per-pair collision: this A2 targets the
                # baseline token this A1 rewrote.
                if (
                    a2_baseline_position
                    == a1_spacy_position
                ):
                    skipped.append({
                        "a1_old": a1_candidate["old"],
                        "a1_new": a1_candidate["new"],
                        "a1_spacy_position":
                            a1_spacy_position,
                        "a2_candidate_index":
                            a2.get("candidate_index"),
                        "a2_position":
                            a2_baseline_position,
                        "reason":
                            "A2 targets the A1 token",
                    })
                    continue

                try:

                    # A2 positions originate in the
                    # baseline document. If A1 changed the
                    # token count, positions after A1 must
                    # be shifted before A2 is applied.
                    a2_position_in_a1 = (
                        map_a2_position_after_a1(
                            a2_baseline_position,
                            a1_spacy_position,
                            a1_token_delta,
                        )
                    )

                    anchor = baseline_doc[
                        a2_baseline_position
                    ].text

                    if (
                        a2_position_in_a1 < 0
                        or a2_position_in_a1
                        >= len(a1_doc)
                    ):
                        raise RuntimeError(
                            f"Mapped A2 position "
                            f"{a2_position_in_a1} outside "
                            f"A1 document length "
                            f"{len(a1_doc)}"
                        )

                    mapped_anchor = a1_doc[
                        a2_position_in_a1
                    ].text

                    if mapped_anchor != anchor:
                        raise RuntimeError(
                            f"A2 anchor mapping mismatch: "
                            f"baseline position "
                            f"{a2_baseline_position} has "
                            f"{anchor!r}, mapped A1 "
                            f"position "
                            f"{a2_position_in_a1} has "
                            f"{mapped_anchor!r}"
                        )

                    combined = apply_a2(
                        a1_doc,
                        a2,
                        a2_position_in_a1,
                        expect_word=anchor,
                    )

                    if combined == baseline:
                        raise RuntimeError(
                            "Combined text equals baseline"
                        )

                    if combined == a1_candidate[
                        "attacked_text"
                    ]:
                        raise RuntimeError(
                            "A2 edit had no effect"
                        )

                    combined_doc = NLP(combined)

                    # Verify A1 survived, accounting for
                    # the shift A2 introduces. Both inputs
                    # are in A1-modified coordinates.
                    expected_a1_position = (
                        a1_position_after_a2(
                            a1_spacy_position,
                            a2["attack_type"],
                            a2_position_in_a1,
                        )
                    )

                    end_position = (
                        expected_a1_position
                        + replacement_length
                    )

                    if (
                        expected_a1_position < 0
                        or end_position
                        > len(combined_doc)
                    ):
                        raise RuntimeError(
                            f"A1 span "
                            f"[{expected_a1_position}, "
                            f"{end_position}) outside "
                            f"combined sequence of length "
                            f"{len(combined_doc)}"
                        )

                    found_span = [
                        combined_doc[j].text
                        for j in range(
                            expected_a1_position,
                            end_position,
                        )
                    ]

                    expected_span = [
                        t.text
                        for t in NLP(
                            a1_candidate["new"]
                        )
                    ]

                    if found_span != expected_span:
                        raise RuntimeError(
                            f"A1 replacement not at "
                            f"tokens "
                            f"[{expected_a1_position}, "
                            f"{end_position}): got "
                            f"{found_span}, expected "
                            f"{expected_span}"
                        )

                    row = {
                        "sample_number":
                            SAMPLE_NUMBER,
                        "prompt_id":
                            PROMPT_ID,

                        "a1_original_word":
                            a1_candidate["old"],
                        "a1_replacement":
                            a1_candidate["new"],
                        "a1_position":
                            a1_candidate["position"],
                        "a1_spacy_position":
                            a1_spacy_position,
                        "a1_token_delta":
                            a1_token_delta,
                        "a1_position_in_combined":
                            expected_a1_position,

                        "a2_candidate_index":
                            a2.get("candidate_index"),
                        "a2_attack_type":
                            a2["attack_type"],
                        "a2_original_word":
                            a2.get("original_word", ""),
                        "a2_position":
                            a2_baseline_position,
                        "a2_position_in_a1":
                            a2_position_in_a1,
                        "a2_replacement":
                            a2.get("replacement", ""),

                        "combined_text": combined,

                        "baseline_char_count":
                            len(baseline),
                        "combined_char_count":
                            len(combined),
                        "char_count_change":
                            len(combined)
                            - len(baseline),

                        "baseline_word_count":
                            len(baseline.split()),
                        "combined_word_count":
                            len(combined.split()),
                        "word_count_change":
                            len(combined.split())
                            - len(baseline.split()),

                        "baseline_spacy_tokens":
                            len(baseline_doc),
                        "a1_spacy_tokens":
                            len(a1_doc),
                        "combined_spacy_tokens":
                            len(combined_doc),
                    }

                    out.write(
                        json.dumps(row) + "\n"
                    )
                    out.flush()

                    saved_count += 1

                    if saved_count % 100 == 0:
                        print(
                            f"[SAVED] {saved_count}/"
                            f"{expected_total}"
                        )

                except Exception as exc:

                    failed.append({
                        "a1_old": a1_candidate["old"],
                        "a1_new": a1_candidate["new"],
                        "a1_spacy_position":
                            a1_spacy_position,
                        "a2_candidate_index":
                            a2.get("candidate_index"),
                        "a2_position":
                            a2_baseline_position,
                        "a2_attack_type":
                            a2.get("attack_type"),
                        "error_type":
                            type(exc).__name__,
                        "error": str(exc),
                    })

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    summary = {
        "sample_number": SAMPLE_NUMBER,
        "prompt_id": PROMPT_ID,
        "baseline_z": stored_z,
        "baseline_char_count": len(baseline),
        "baseline_spacy_tokens": len(baseline_doc),
        "z_threshold": Z_THRESHOLD,

        "a1_candidate_count": len(a1),
        "a1_records_skipped": len(a1_skipped),

        "a2_candidate_count": len(all_a2),
        "a2_valid_count": len(valid_a2),
        "a2_invalid_count": len(invalid_a2),

        "pairwise_collisions_excluded": len(skipped),
        "expected_combination_count": expected_total,
        "saved_combination_count": saved_count,
        "failed_combination_count": len(failed),

        "a1_candidates": [
            {
                k: v
                for k, v in c.items()
                if k not in ("doc", "attacked_text")
            }
            for c in a1
        ],

        "a1_skipped_records": a1_skipped,
        "invalid_a2": invalid_a2,
        "collision_records": skipped,
        "failed_combinations": failed,
    }

    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2)
    )

    print()
    print("=" * 90)
    print("FINAL")
    print("=" * 90)
    print("A1 candidates:", len(a1))
    print("A1 skipped records:", len(a1_skipped))
    print("A2 total:", len(all_a2))
    print("A2 valid:", len(valid_a2))
    print("A2 invalid:", len(invalid_a2))
    print("Pairwise collisions:", len(skipped))
    print("Expected:", expected_total)
    print("Saved:", saved_count)
    print("Failed:", len(failed))

    print()
    print(f"[SAVED] {OUT_PATH}")
    print(f"[SAVED] {SUMMARY_PATH}")

    if saved_count != expected_total or failed:
        raise RuntimeError(
            f"Combination build incomplete: saved "
            f"{saved_count}, expected {expected_total}, "
            f"failed {len(failed)}. See "
            f"failed_combinations in the summary."
        )


if __name__ == "__main__":
    main()

