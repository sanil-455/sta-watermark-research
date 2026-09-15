import json
import difflib
from pathlib import Path


BASELINE_PATH = Path(
    "results/raw/baseline_safe.json"
)

A1_PATH = Path(
    "results/raw/manual_attacks/"
    "sample1_pos345_might_to_could_corrected.json"
)

A2_PATH = Path(
    "results/raw/attack2_candidates_by_sample/"
    "sample_1.json"
)


def load_json(path):
    with path.open() as f:
        return json.load(f)


def get_one_diff(old, new):
    matcher = difflib.SequenceMatcher(
        None,
        old,
        new,
    )

    changes = [
        op for op in matcher.get_opcodes()
        if op[0] != "equal"
    ]

    if len(changes) != 1:
        raise RuntimeError(
            f"Expected one edit, found {changes}"
        )

    return changes[0]


def apply_translated_edit(
    baseline,
    a1_text,
    a2_text,
):
    # Find the exact A2 character edit
    # relative to the original baseline.
    a2_op = get_one_diff(
        baseline,
        a2_text,
    )

    tag, a2_start, a2_end, _, _ = a2_op

    # Find the exact A1 character edit.
    a1_op = get_one_diff(
        baseline,
        a1_text,
    )

    a1_tag, a1_start, a1_end, a1_new_start, a1_new_end = (
        a1_op
    )

    # The two edits must not overlap.
    if (
        a2_start < a1_end
        and a2_end > a1_start
    ):
        raise RuntimeError(
            "A1 and A2 edits overlap"
        )

    # A1 changes the character length of the text.
    a1_old_len = a1_end - a1_start
    a1_new_len = a1_new_end - a1_new_start
    shift = a1_new_len - a1_old_len

    if a2_start >= a1_end:
        start = a2_start + shift
        end = a2_end + shift
    else:
        start = a2_start
        end = a2_end

    if tag == "insert":
        replacement = a2_text[
            a2_start:a2_end
        ]
    else:
        replacement = a2_text[
            a2_start:a2_end
        ]

    # The SequenceMatcher opcode needs the
    # actual replacement text from a2_text.
    if tag == "replace":
        replacement = a2_text[
            a2_start:
            a2_start + (
                len(a2_text) - len(baseline)
                + (a2_end - a2_start)
            )
        ]

    elif tag == "insert":
        replacement = a2_text[
            a2_start:
            a2_start + (
                len(a2_text) - len(baseline)
            )
        ]

    elif tag == "delete":
        replacement = ""

    else:
        raise RuntimeError(
            f"Unsupported edit type: {tag}"
        )

    combined = (
        a1_text[:start]
        + replacement
        + a1_text[end:]
    )

    return combined, a2_op


def main():
    baseline_data = load_json(
        BASELINE_PATH
    )

    baseline_row = next(
        x for x in baseline_data
        if x["prompt_id"] == 0
    )

    baseline = baseline_row[
        "watermarked_text"
    ]

    a1 = load_json(A1_PATH)
    a1_text = a1["attacked_text"]

    a2_data = load_json(A2_PATH)
    a2 = a2_data["candidates"][0]
    a2_text = a2["attacked_text"]

    print("A1:")
    print(
        a1["old_word"],
        "->",
        a1["new_word"]
    )

    print("A2:")
    print(
        a2["attack_type"],
        a2["original_word"],
        "->",
        a2["replacement"]
    )

    combined, a2_op = apply_translated_edit(
        baseline,
        a1_text,
        a2_text,
    )

    print()
    print("A2 character diff:")
    print(a2_op)

    print()
    print("Baseline length:", len(baseline))
    print("A1 length:", len(a1_text))
    print("A2 length:", len(a2_text))
    print("Combined length:", len(combined))

    print()
    print("Combined beginning:")
    print(repr(combined[:120]))

    print()
    print("Combined ending:")
    print(repr(combined[-180:]))

    print()
    print("A1 word still present:")
    print("could" in combined)

    print("A1 old word absent:")
    print("might" not in combined)

    print("A2 deleted word absent at beginning:")
    print(
        not combined.startswith("The ")
    )

    print()
    print("✓ ONE COMBINATION CONSTRUCTED")


if __name__ == "__main__":
    main()
