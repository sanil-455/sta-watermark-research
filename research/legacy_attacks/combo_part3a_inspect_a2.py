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

    print("CANDIDATE FILE")
    print(
        "sample_number:",
        candidate_data.get("sample_number"),
    )
    print(
        "prompt_id:",
        candidate_data.get("prompt_id"),
    )
    print(
        "candidate count:",
        len(candidates),
    )

    print()
    print("FIRST CANDIDATE")
    print(
        json.dumps(
            candidates[0],
            indent=2,
        )
    )

    print()
    print("SECOND CANDIDATE")
    print(
        json.dumps(
            candidates[1],
            indent=2,
        )
    )

    print()
    print("LAST CANDIDATE")
    print(
        json.dumps(
            candidates[-1],
            indent=2,
        )
    )

    results = []

    with RESULT_PATH.open("r") as f:
        for line in f:
            if line.strip():
                results.append(
                    json.loads(line)
                )

    print()
    print("RESULT FILE")
    print(
        "result count:",
        len(results),
    )

    print()
    print("FIRST 5 RESULTS")

    for i, row in enumerate(
        results[:5]
    ):
        print()
        print(
            f"RESULT {i + 1}"
        )
        print(
            json.dumps(
                row,
                indent=2,
            )
        )

    print()
    print("LAST 5 RESULTS")

    for i, row in enumerate(
        results[-5:]
    ):
        print()
        print(
            f"RESULT {len(results) - 4 + i}"
        )
        print(
            json.dumps(
                row,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()

