import json
from pathlib import Path

import spacy


BASELINE_PATH = Path(
    "results/raw/baseline_safe.json"
)

A2_PATH = Path(
    "results/raw/attack2_candidates_by_sample/"
    "sample_1.json"
)

OUT_PATH = Path(
    "results/raw/attack12_combined/"
    "sample_1_a2_validation_0_49.json"
)


START = 0
END = 50


def load_json(path):
    with path.open() as f:
        return json.load(f)


def main():

    rows = load_json(BASELINE_PATH)

    baseline = next(
        row["watermarked_text"]
        for row in rows
        if row["prompt_id"] == 0
    )

    data = load_json(A2_PATH)
    candidates = data["candidates"]

    nlp = spacy.blank("en")

    doc = nlp(baseline)

    results = []

    for index in range(START, END):

        candidate = candidates[index]

        position = candidate["position"]
        word = candidate["original_word"]

        result = {
            "candidate_index": index,
            "attack_type": candidate["attack_type"],
            "original_word": word,
            "replacement": candidate["replacement"],
            "position": position,
            "valid": False,
            "reason": None,
        }

        if position < 0 or position >= len(doc):
            result["reason"] = "invalid_token_position"
            results.append(result)
            print(f"{index}: INVALID position")
            continue

        token = doc[position]

        if (
            candidate["attack_type"]
            == "deletion"
        ):
            if token.text != word:
                result["reason"] = (
                    "token_does_not_match_original_word"
                )
                results.append(result)
                print(
                    f"{index}: INVALID token mismatch"
                )
                continue

        elif (
            candidate["attack_type"]
            == "insertion"
        ):
            if word is not None:
                result["reason"] = (
                    "insertion_has_original_word"
                )
                results.append(result)
                print(
                    f"{index}: INVALID insertion metadata"
                )
                continue

        else:
            result["reason"] = "unknown_attack_type"
            results.append(result)
            print(f"{index}: INVALID attack type")
            continue

        result["token_text"] = token.text
        result["token_start"] = token.idx
        result["token_end"] = (
            token.idx + len(token.text)
        )

        result["candidate_text_length"] = len(
            candidate["attacked_text"]
        )

        result["baseline_text_length"] = len(
            baseline
        )

        result["valid"] = True
        result["reason"] = "metadata_valid"

        results.append(result)

        print(
            f"{index}: VALID metadata | "
            f"{candidate['attack_type']} | "
            f"{word or candidate['replacement']} | "
            f"token={position}"
        )

    OUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUT_PATH.open("w") as f:
        json.dump(
            {
                "sample_number": 1,
                "prompt_id": 0,
                "range_start": START,
                "range_end": END - 1,
                "results": results,
            },
            f,
            indent=2,
        )

    valid = sum(
        x["valid"]
        for x in results
    )

    print()
    print("DONE")
    print("Checked:", len(results))
    print("Metadata valid:", valid)
    print("Metadata invalid:", len(results) - valid)
    print(f"[✓ SAVED] {OUT_PATH}")


if __name__ == "__main__":
    main()
