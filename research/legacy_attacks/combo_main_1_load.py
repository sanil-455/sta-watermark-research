import json
from pathlib import Path


BASELINE_PATH = Path(
    "results/raw/baseline_safe.json"
)

A2_PATH = Path(
    "results/raw/attack2_candidates_by_sample/"
    "sample_1.json"
)

A1_FILES = [
    Path(
        "results/raw/manual_attacks/"
        "sample1_pos345_might_to_could_corrected.json"
    ),
    Path(
        "results/raw/manual_attacks/"
        "sample1_pos345_might_to_can.json"
    ),
]


def load_json(path):
    with path.open() as f:
        return json.load(f)


def main():

    baseline_data = load_json(
        BASELINE_PATH
    )

    baseline = None

    for row in baseline_data:
        if row["prompt_id"] == 0:
            baseline = row
            break

    if baseline is None:
        raise RuntimeError(
            "Sample 1 baseline not found"
        )

    print("BASELINE")
    print("prompt_id =", baseline["prompt_id"])
    print("z =", baseline["watermarked_z"])
    print(
        "text length =",
        len(baseline["watermarked_text"])
    )

    a1 = []

    for path in A1_FILES:
        row = load_json(path)

        if row["original_text"] != (
            baseline["watermarked_text"]
        ):
            raise RuntimeError(
                f"A1 text mismatch: {path}"
            )

        a1.append(row)

    print()
    print("A1")
    print("count =", len(a1))

    for row in a1:
        print(
            row["old_word"],
            "->",
            row["new_word"]
        )

    a2_data = load_json(A2_PATH)
    a2 = a2_data["candidates"]

    print()
    print("A2")
    print("count =", len(a2))

    print()
    print("FIRST A2 OBJECT")
    print(
        json.dumps(
            a2[0],
            indent=2
        )
    )

    print()
    print("LAST A2 OBJECT")
    print(
        json.dumps(
            a2[-1],
            indent=2
        )
    )

    print()
    print("✓ LOAD CHECK PASSED")


if __name__ == "__main__":
    main()
