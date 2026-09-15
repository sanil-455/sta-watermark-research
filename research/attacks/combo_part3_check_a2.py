import json
from pathlib import Path


RESULT_PATH = Path(
    "results/raw/attack2_results_by_sample/sample_1.jsonl"
)

CANDIDATE_PATH = Path(
    "results/raw/attack2_candidates_by_sample/sample_1.json"
)


def load_json(path):
    with path.open("r") as f:
        return json.load(f)


def main():
    candidate_data = load_json(
        CANDIDATE_PATH
    )

    candidates = candidate_data["candidates"]

    print(
        f"Candidate count = {len(candidates)}"
    )

    results = []

    with RESULT_PATH.open("r") as f:
        for line in f:
            if line.strip():
                results.append(
                    json.loads(line)
                )

    print(
        f"Result records = {len(results)}"
    )

    indices = [
        int(row["candidate_index"])
        for row in results
        if row.get("candidate_index") is not None
    ]

    print(
        f"Minimum candidate_index = "
        f"{min(indices)}"
    )

    print(
        f"Maximum candidate_index = "
        f"{max(indices)}"
    )

    candidate_count = len(candidates)

    if (
        min(indices) == 0
        and max(indices) == candidate_count - 1
    ):
        indexing = "zero_based"

    elif (
        min(indices) == 1
        and max(indices) == candidate_count
    ):
        indexing = "one_based"

    else:
        raise RuntimeError(
            "Could not determine candidate indexing."
        )

    print(
        f"Detected indexing = {indexing}"
    )

    latest = {}

    for row in results:
        if row.get("candidate_index") is None:
            continue

        latest[
            int(row["candidate_index"])
        ] = row

    valid = []

    for row in latest.values():

        if row.get("status") != "ok":
            continue

        if row.get("semantic_pass") is not True:
            continue

        if row.get("attacked_z") is None:
            continue

        index = int(
            row["candidate_index"]
        )

        if indexing == "zero_based":
            list_index = index
        else:
            list_index = index - 1

        if not (
            0 <= list_index < candidate_count
        ):
            raise RuntimeError(
                f"Cannot map candidate {index}"
            )

        candidate = candidates[
            list_index
        ]

        if not isinstance(
            candidate,
            dict,
        ):
            raise RuntimeError(
                f"Candidate {index} is not a dictionary."
            )

        valid.append(
            {
                "candidate_index": index,
                "list_index": list_index,
                "attack_type":
                    candidate.get("attack_type"),
                "original_word":
                    candidate.get(
                        "original_word"
                    ),
                "replacement":
                    candidate.get(
                        "replacement"
                    ),
                "attacked_z":
                    float(
                        row["attacked_z"]
                    ),
                "delta_z":
                    float(
                        row["delta_z"]
                    ),
            }
        )

    valid.sort(
        key=lambda x: x["delta_z"]
    )

    print()
    print(
        f"Valid semantic-passing A2 attacks = "
        f"{len(valid)}"
    )

    print()
    print(
        "Top 10 A2 candidates:"
    )

    for rank, row in enumerate(
        valid[:10],
        start=1,
    ):
        print(
            f"{rank:2d}. "
            f"candidate={row['candidate_index']} "
            f"list_index={row['list_index']} "
            f"| {row['attack_type']} "
            f"| {row['original_word']} "
            f"-> {row['replacement']} "
            f"| z={row['attacked_z']:.6f} "
            f"| dz={row['delta_z']:.6f}"
        )

    if len(valid) < 10:
        raise RuntimeError(
            "Fewer than 10 valid A2 candidates."
        )

    output = {
        "sample_number": 1,
        "prompt_id": 0,
        "candidate_count": candidate_count,
        "result_count": len(results),
        "indexing": indexing,
        "valid_count": len(valid),
        "top_10": valid[:10],
    }

    output_path = Path(
        "results/raw/attack12_combined/"
        "sample_1_a2_validation.json"
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

    print()
    print(
        f"[✓ SAVED] {output_path}"
    )


if __name__ == "__main__":
    main()
