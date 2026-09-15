import json
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

EXPECTED_PROMPT_ID = 0
EXPECTED_BASELINE_Z = 3.363765319856875


def load_json(path):
    with path.open("r") as f:
        return json.load(f)


def load_baseline():
    rows = load_json(BASELINE_PATH)

    matches = [
        row
        for row in rows
        if int(row["prompt_id"])
        == EXPECTED_PROMPT_ID
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one baseline row "
            f"for prompt_id={EXPECTED_PROMPT_ID}, "
            f"found {len(matches)}"
        )

    row = matches[0]

    stored_z = float(
        row["watermarked_z"]
    )

    if abs(
        stored_z - EXPECTED_BASELINE_Z
    ) > 1e-6:
        raise RuntimeError(
            f"Unexpected baseline z: {stored_z}"
        )

    print(
        "Baseline verified"
    )

    print(
        f"prompt_id = {row['prompt_id']}"
    )

    print(
        f"baseline z = {stored_z:.12f}"
    )

    print(
        f"text length = "
        f"{len(row['watermarked_text'])}"
    )

    return row


def inspect_a1_file(
    path,
    baseline_text,
):
    data = load_json(path)

    print()
    print(
        f"FILE: {path.name}"
    )

    print(
        f"sample = {data.get('sample')}"
    )

    print(
        f"position = {data.get('position')}"
    )

    print(
        f"old_word = "
        f"{data.get('old_word')!r}"
    )

    print(
        f"new_word = "
        f"{data.get('new_word')!r}"
    )

    original_text = data.get(
        "original_text"
    )

    attacked_text = data.get(
        "attacked_text"
    )

    print(
        f"original text matches baseline = "
        f"{original_text == baseline_text}"
    )

    if original_text != baseline_text:
        print(
            "WARNING: original_text differs "
            "from baseline"
        )

        return data

    if not attacked_text:
        raise RuntimeError(
            "attacked_text is missing"
        )

    print(
        f"attacked text length = "
        f"{len(attacked_text)}"
    )

    print(
        f"original z = "
        f"{data.get('original_z')}"
    )

    print(
        f"attacked z = "
        f"{data.get('attacked_z')}"
    )

    print(
        f"delta z = "
        f"{data.get('delta_z')}"
    )

    return data


def main():
    baseline = load_baseline()

    baseline_text = baseline[
        "watermarked_text"
    ]

    print()
    print(
        "Checking A1 files..."
    )

    loaded = []

    for filename in A1_FILES:
        path = A1_DIR / filename

        if not path.exists():
            raise RuntimeError(
                f"Missing file: {path}"
            )

        data = inspect_a1_file(
            path,
            baseline_text,
        )

        loaded.append(data)

    print()
    print(
        f"Successfully inspected "
        f"{len(loaded)} A1 files."
    )


if __name__ == "__main__":
    main()
