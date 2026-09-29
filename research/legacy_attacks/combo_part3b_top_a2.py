import json
from pathlib import Path


RESULT_PATH = Path(
    "results/raw/attack2_results_by_sample/sample_1.jsonl"
)


def main():
    results = []

    with RESULT_PATH.open("r") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)

                if (
                    row.get("status") == "ok"
                    and row.get("semantic_pass") is True
                    and row.get("attacked_z") is not None
                ):
                    results.append(row)

    # delta_z must be calculated from the actual z values.
    results.sort(
        key=lambda x:
        x["attacked_z"] - x["original_z"]
    )

    top10 = results[:10]

    print("Valid A2 results:", len(results))
    print()
    print("TOP 10 A2 ATTACKS")
    print("-" * 60)

    for rank, row in enumerate(top10, 1):
        delta = (
            row["attacked_z"]
            - row["original_z"]
        )

        print(
            f"{rank}. "
            f"candidate_index={row['candidate_index']} | "
            f"type={row['attack_type']} | "
            f"word={row['original_word']} | "
            f"replacement={row['replacement']} | "
            f"position={row['position']}"
        )

        print(
            f"   z={row['attacked_z']:.9f} | "
            f"delta_z={delta:.9f}"
        )

        print(
            f"   semantic="
            f"{row['global_similarity']:.4f}/"
            f"{row['local_similarity']:.4f}"
        )

        print(
            f"   tokens="
            f"{row['original_token_count']} -> "
            f"{row['attacked_token_count']}"
        )

        print()

    output = {
        "sample_number": 1,
        "prompt_id": 0,
        "valid_a2_count": len(results),
        "top10": top10,
    }

    out_path = Path(
        "results/raw/attack12_combined/"
        "sample_1_top10_a2.json"
    )

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with out_path.open("w") as f:
        json.dump(
            output,
            f,
            indent=2,
        )

    print(
        f"[✓ SAVED] {out_path}"
    )


if __name__ == "__main__":
    main()
