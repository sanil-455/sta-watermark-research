import json
from pathlib import Path
from difflib import SequenceMatcher


BASELINE_PATH = Path(
    "results/raw/baseline_safe.json"
)

A1_DIR = Path(
    "results/raw/manual_attacks"
)

A1_FILES = [
    "sample1_pos345_might_to_could_corrected.json",
    "sample1_pos345_might_to_can.json",
    "sample1_pos345_might_to_may.json",
]

PROMPT_ID = 0


def load_json(path):
    with path.open("r") as f:
        return json.load(f)


def get_baseline():
    rows = load_json(BASELINE_PATH)

    matches = [
        row
        for row in rows
        if int(row["prompt_id"]) == PROMPT_ID
    ]

    if len(matches) != 1:
        raise RuntimeError(
            "Could not uniquely find Sample 1."
        )

    return matches[0]["watermarked_text"]


def inspect_edit(original, attacked):
    matcher = SequenceMatcher(
        None,
        original,
        attacked,
        autojunk=False,
    )

    changes = [
        op
        for op in matcher.get_opcodes()
        if op[0] != "equal"
    ]

    print(
        f"Number of changed regions: "
        f"{len(changes)}"
    )

    for number, op in enumerate(
        changes,
        start=1,
    ):
        tag, i1, i2, j1, j2 = op

        old_text = original[i1:i2]
        new_text = attacked[j1:j2]

        print()
        print(
            f"Change {number}:"
        )
        print(
            f"  type       = {tag}"
        )
        print(
            f"  original span = [{i1}, {i2}]"
        )
        print(
            f"  attacked span = [{j1}, {j2}]"
        )
        print(
            f"  old text   = {old_text!r}"
        )
        print(
            f"  new text   = {new_text!r}"
        )

        left = max(
            0,
            i1 - 35,
        )

        right = min(
            len(original),
            i2 + 35,
        )

        print(
            f"  original context = "
            f"{original[left:right]!r}"
        )

    rebuilt = original

    for op in reversed(changes):
        tag, i1, i2, j1, j2 = op

        new_text = attacked[j1:j2]

        rebuilt = (
            rebuilt[:i1]
            + new_text
            + rebuilt[i2:]
        )

    print()
    print(
        f"Reconstruction matches saved attack: "
        f"{rebuilt == attacked}"
    )

    return changes


def main():
    baseline = get_baseline()

    print(
        "Sample 1 baseline loaded."
    )

    print(
        f"Baseline length: {len(baseline)}"
    )

    for filename in A1_FILES:
        print()
        print("=" * 60)
        print(
            f"Checking: {filename}"
        )
        print("=" * 60)

        path = A1_DIR / filename

        data = load_json(path)

        original = data[
            "original_text"
        ]

        attacked = data[
            "attacked_text"
        ]

        if original != baseline:
            raise RuntimeError(
                "Original text does not match "
                "Sample 1 baseline."
            )

        print(
            f"Saved old_word = "
            f"{data['old_word']!r}"
        )

        print(
            f"Saved new_word = "
            f"{data['new_word']!r}"
        )

        print(
            f"Saved original z = "
            f"{data['original_z']}"
        )

        print(
            f"Saved attacked z = "
            f"{data['attacked_z']}"
        )

        print(
            f"Saved delta_z = "
            f"{data['delta_z']}"
        )

        calculated_delta = (
            float(data["attacked_z"])
            - float(data["original_z"])
        )

        print(
            f"Calculated attacked-original = "
            f"{calculated_delta}"
        )

        inspect_edit(
            original,
            attacked,
        )

    print()
    print(
        "A1 edit inspection complete."
    )


if __name__ == "__main__":
    main()
