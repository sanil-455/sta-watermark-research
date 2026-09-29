import json
from difflib import SequenceMatcher
from pathlib import Path


BASELINE_PATH = Path(
    "results/raw/baseline_safe.json"
)

# Research sample -> actual prompt_id.
SAMPLE_TO_PROMPT_ID = {
    1: 0,
    2: 1,
    3: 2,
}

# These are the files actually produced so far.
RESULT_FILES = {
    1: Path(
        "results/raw/attack2_results_by_sample/"
        "sample_1.jsonl"
    ),
    2: Path(
        "results/raw/attack2_results/"
        "sample_1.jsonl"
    ),
    3: Path(
        "results/raw/attack2_results/"
        "sample_2.jsonl"
    ),
}

CANDIDATE_FILES = {
    1: Path(
        "results/raw/attack2_candidates_by_sample/"
        "sample_1.json"
    ),
    2: Path(
        "results/raw/attack2_candidates/"
        "sample_1.json"
    ),
    3: Path(
        "results/raw/attack2_candidates/"
        "sample_2.json"
    ),
}


def load_json(path):
    with path.open() as f:
        return json.load(f)


def load_results(path):
    results = []

    with path.open() as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    return results


def get_changed_region(
    original,
    attacked,
    radius=350,
):
    matcher = SequenceMatcher(
        None,
        original,
        attacked,
    )

    changes = [
        op
        for op in matcher.get_opcodes()
        if op[0] != "equal"
    ]

    if not changes:
        return original, attacked

    op = changes[0]

    original_start = max(
        0,
        op[1] - radius,
    )

    original_end = min(
        len(original),
        op[2] + radius,
    )

    attacked_start = max(
        0,
        op[3] - radius,
    )

    attacked_end = min(
        len(attacked),
        op[4] + radius,
    )

    return (
        original[original_start:original_end],
        attacked[attacked_start:attacked_end],
    )


def main():
    baselines = load_json(BASELINE_PATH)

    baseline_by_id = {
        int(x["prompt_id"]): x
        for x in baselines
    }

    output_path = Path(
        "results/raw/attack2_manual_audit.txt"
    )

    all_output = []

    for sample_number in (1, 2, 3):
        prompt_id = SAMPLE_TO_PROMPT_ID[
            sample_number
        ]

        baseline = baseline_by_id[prompt_id]

        result_path = RESULT_FILES[
            sample_number
        ]

        candidate_path = CANDIDATE_FILES[
            sample_number
        ]

        results = load_results(result_path)
        candidate_data = load_json(
            candidate_path
        )

        candidates = candidate_data[
            "candidates"
        ]

        # Only candidates that passed the fixed
        # semantic filter and have an STA result.
        valid = [
            r
            for r in results
            if r.get("status") == "ok"
            and r.get("semantic_pass") is True
            and r.get("attacked_z") is not None
        ]

        valid.sort(
            key=lambda r: r["delta_z"]
        )

        # Keep the strongest candidates, but avoid
        # showing many edits with exactly the same z.
        selected = []
        seen_z = set()

        for result in valid:
            z_key = round(
                result["attacked_z"],
                8,
            )

            if z_key in seen_z:
                continue

            seen_z.add(z_key)
            selected.append(result)

            if len(selected) >= 15:
                break

        all_output.append(
            "\n" + "=" * 90
        )
        all_output.append(
            f"RESEARCH SAMPLE {sample_number} "
            f"(prompt_id={prompt_id})"
        )
        all_output.append(
            f"Baseline z = "
            f"{baseline['watermarked_z']:.6f}"
        )
        all_output.append(
            f"Candidates passing semantic filter = "
            f"{len(valid)}"
        )
        all_output.append("=" * 90)

        original_text = (
            baseline["watermarked_text"]
        )

        for rank, result in enumerate(
            selected,
            1,
        ):
            candidate_index = result[
                "candidate_index"
            ]

            candidate = candidates[
                candidate_index
            ]

            attacked_text = candidate[
                "attacked_text"
            ]

            original_region, attacked_region = (
                get_changed_region(
                    original_text,
                    attacked_text,
                )
            )

            if result["attack_type"] == "deletion":
                edit = (
                    f"DELETE "
                    f"'{result['original_word']}'"
                )
            else:
                edit = (
                    f"INSERT "
                    f"'{result['replacement']}'"
                )

            all_output.append("")
            all_output.append(
                f"--- Candidate {rank} ---"
            )
            all_output.append(
                f"Edit: {edit}"
            )
            all_output.append(
                f"Token position: "
                f"{result['position']}"
            )
            all_output.append(
                f"Global similarity: "
                f"{result['global_similarity']:.6f}"
            )
            all_output.append(
                f"Local similarity: "
                f"{result['local_similarity']:.6f}"
            )
            all_output.append(
                f"Original z: "
                f"{result['original_z']:.6f}"
            )
            all_output.append(
                f"Attacked z: "
                f"{result['attacked_z']:.6f}"
            )
            all_output.append(
                f"Delta z: "
                f"{result['delta_z']:.6f}"
            )

            all_output.append(
                "\nORIGINAL CONTEXT:"
            )
            all_output.append(
                original_region.strip()
            )

            all_output.append(
                "\nATTACKED CONTEXT:"
            )
            all_output.append(
                attacked_region.strip()
            )

    with output_path.open("w") as f:
        f.write(
            "\n".join(all_output)
        )

    print(
        f"[✓ SAVED] Manual audit -> "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()
