import json
import os
import re


INPUT = (
    "results/raw/manual_attacks/"
    "sample1_vulnerability_shortlist.json"
)

OUTPUT = (
    "results/raw/manual_attacks/"
    "sample1_top50_meaningful_review.json"
)


with open(INPUT) as f:
    data = json.load(f)

positions = data["positions"]

review = []

for rank, row in enumerate(positions[:50], 1):

    target = row["target"]

    # Remove tokenizer whitespace markers / surrounding whitespace.
    cleaned = target.strip()

    # A deliberately conservative definition of a
    # meaningful ordinary English word candidate.
    meaningful = bool(
        re.fullmatch(r"[A-Za-z]{2,}", cleaned)
    )

    review.append(
        {
            "rank": rank,
            "position": row["position"],
            "target": target,
            "cleaned_target": cleaned,
            "token_id": row["token_id"],
            "red_red_opportunities": (
                row["red_red_opportunities"]
            ),
            "meaningful_word_candidate": meaningful,
            "context": row["context"],
        }
    )


result = {
    "sample": data["sample"],
    "source": INPUT,
    "coverage": "top_50",
    "total_positions_reviewed": len(review),
    "meaningful_candidates": sum(
        r["meaningful_word_candidate"]
        for r in review
    ),
    "positions": review,
}


os.makedirs(
    os.path.dirname(OUTPUT),
    exist_ok=True,
)

with open(OUTPUT, "w") as f:
    json.dump(
        result,
        f,
        indent=2,
    )


print()
print("=" * 90)
print("TOP 50 STA-VULNERABLE POSITIONS")
print("=" * 90)

print(
    "Meaningful-looking candidates:",
    result["meaningful_candidates"],
    "/",
    len(review),
)

print()

for row in review:

    marker = (
        "MEANINGFUL"
        if row["meaningful_word_candidate"]
        else "SKIP"
    )

    print("=" * 90)

    print(
        f"#{row['rank']} | "
        f"position={row['position']} | "
        f"target={row['target']!r} | "
        f"{marker}"
    )

    print(
        "RED-RED opportunities:",
        row["red_red_opportunities"],
    )

    print("Context:")
    print(row["context"])

print()
print("=" * 90)
print("[✓ SAVED]", OUTPUT)
print("=" * 90)
