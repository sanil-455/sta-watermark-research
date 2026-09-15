import json
import difflib
from pathlib import Path


BASELINE_PATH = Path(
    "results/raw/baseline_safe.json"
)

A2_PATH = Path(
    "results/raw/attack2_candidates_by_sample/"
    "sample_1.json"
)


def main():
    with BASELINE_PATH.open() as f:
        rows = json.load(f)

    baseline = next(
        x["watermarked_text"]
        for x in rows
        if x["prompt_id"] == 0
    )

    with A2_PATH.open() as f:
        data = json.load(f)

    candidate = data["candidates"][85]

    attacked = candidate["attacked_text"]

    print("CANDIDATE 85")
    print(
        json.dumps(
            {
                k: candidate.get(k)
                for k in [
                    "sample_number",
                    "prompt_id",
                    "attack_type",
                    "original_word",
                    "replacement",
                    "position",
                ]
            },
            indent=2,
        )
    )

    print()
    print("LENGTHS")
    print("baseline:", len(baseline))
    print("attacked:", len(attacked))
    print("difference:", len(attacked) - len(baseline))

    print()
    print("DIFF")

    matcher = difflib.SequenceMatcher(
        None,
        baseline,
        attacked,
    )

    for op in matcher.get_opcodes():
        if op[0] != "equal":
            tag, i1, i2, j1, j2 = op

            print()
            print(tag)
            print(
                "baseline:",
                repr(baseline[i1:i2])
            )
            print(
                "attacked :",
                repr(attacked[j1:j2])
            )

    print()
    print("✓ INSPECTION COMPLETE")


if __name__ == "__main__":
    main()
