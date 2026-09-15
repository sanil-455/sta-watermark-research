import json
from difflib import SequenceMatcher
from pathlib import Path


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


def get_change(original, attacked):
    changes = [
        op
        for op in SequenceMatcher(
            None,
            original,
            attacked,
            autojunk=False,
        ).get_opcodes()
        if op[0] != "equal"
    ]

    if len(changes) != 1:
        return None

    tag, i1, i2, j1, j2 = changes[0]

    return {
        "tag": tag,
        "start": i1,
        "end": i2,
        "old_text": original[i1:i2],
        "new_text": attacked[j1:j2],
    }


def main():
    baseline = get_baseline()

    valid = []
    invalid = []

    print("Sample 1 A1 validation")
    print("prompt_id = 0")
    print()

    for filename in A1_FILES:
        data = load_json(
            A1_DIR / filename
        )

        original = data["original_text"]
        attacked = data["attacked_text"]

        change = get_change(
            original,
            attacked,
        )

        print(filename)

        if original != baseline:
            print("  INVALID: original text mismatch")
            invalid.append(filename)
            continue

        if change is None:
            print("  INVALID: not one localized edit")
            invalid.append(filename)
            continue

        print(
            f"  diff type  = {change['tag']}"
        )
        print(
            f"  old text   = {change['old_text']!r}"
        )
        print(
            f"  new text   = {change['new_text']!r}"
        )

        whole_word_match = (
            change["old_text"]
            == data["old_word"]
            and
            change["new_text"]
            == data["new_word"]
        )

        correct_type = (
            change["tag"] == "replace"
        )

        if whole_word_match and correct_type:
            print("  RESULT     = VALID")
            valid.append(
                {
                    "file": filename,
                    "old_word": data["old_word"],
                    "new_word": data["new_word"],
                    "start": change["start"],
                    "end": change["end"],
                    "attacked_z": data["attacked_z"],
                    "delta_z_calculated": (
                        float(data["attacked_z"])
                        - float(data["original_z"])
                    ),
                }
            )
        else:
            print("  RESULT     = INVALID")
            print(
                f"  whole-word match = "
                f"{whole_word_match}"
            )
            print(
                f"  replacement type = "
                f"{correct_type}"
            )

            invalid.append(filename)

        print()

    output = {
        "sample_number": 1,
        "prompt_id": 0,
        "valid_count": len(valid),
        "invalid_count": len(invalid),
        "valid": valid,
        "invalid": invalid,
    }

    output_path = Path(
        "results/raw/attack12_combined/"
        "sample_1_a1_validation.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("w") as f:
        json.dump(
            output,
            f,
            indent=2,
        )

    print(
        f"[✓ SAVED] {output_path}"
    )

    print()
    print(
        f"Valid A1 operations: {len(valid)}"
    )
    print(
        f"Invalid A1 operations: {len(invalid)}"
    )


if __name__ == "__main__":
    main()
