import json
import os

WORKSHEET_DIR = "results/raw/manual_sta_worksheets"

TARGET_POSITION = 275
TARGET_WORD = "plant"

found = []

for sample in range(1, 6):

    path = os.path.join(
        WORKSHEET_DIR,
        f"sample_{sample}.json",
    )

    if not os.path.exists(path):
        print(
            "[MISSING]",
            path,
        )
        continue

    with open(path) as f:
        data = json.load(f)

    for candidate in data.get("candidates", []):

        position = candidate.get("position")
        word = candidate.get("old_token")

        if (
            position == TARGET_POSITION
            or word.strip().lower() == TARGET_WORD
        ):
            found.append(
                {
                    "sample": sample,
                    "position": position,
                    "target": word,
                    "red_red_opportunities": len(
                        candidate.get(
                            "red_replacements",
                            [],
                        )
                    ),
                    "context": candidate.get(
                        "context",
                        "",
                    ),
                }
            )


print()
print("=" * 90)
print("SEARCH RESULT")
print("=" * 90)

if not found:
    print(
        "NO MATCH FOUND for position 275 or word 'plant'."
    )
else:
    for i, result in enumerate(found, 1):

        print()
        print(
            f"MATCH {i}"
        )

        print(
            "Sample:",
            result["sample"],
        )

        print(
            "Position:",
            result["position"],
        )

        print(
            "Target:",
            repr(result["target"]),
        )

        print(
            "RED-RED opportunities:",
            result["red_red_opportunities"],
        )

        print(
            "Context:"
        )

        print(
            result["context"]
        )

print()
print("=" * 90)
print("NO ATTACK PERFORMED")
print("=" * 90)
