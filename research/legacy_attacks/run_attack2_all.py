from pathlib import Path

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

import evaluate_attack2_by_sample as evaluator


BASELINE_PATH = Path(
    "results/raw/baseline_safe.json"
)


def load_json(path):
    with path.open() as f:
        import json
        return json.load(f)


def main():
    print("Loading semantic model...")
    model = SentenceTransformer(
        evaluator.MODEL_NAME,
        device="cpu",
    )

    print("Loading Llama tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "hf_models/Llama-2-7b-hf",
        local_files_only=True,
    )

    baselines = load_json(
        BASELINE_PATH
    )

    baseline_by_id = {
        int(x["prompt_id"]): x
        for x in baselines
    }

    for sample_number in (1, 2, 3):
        expected_id = evaluator.SAMPLE_TO_PROMPT_ID[
            sample_number
        ]

        sample = evaluator.find_baseline(
            sample_number,
            baselines,
        )

        actual_id = int(
            sample["prompt_id"]
        )

        if actual_id != expected_id:
            raise RuntimeError(
                f"STOP: Research Sample "
                f"{sample_number} expected "
                f"prompt_id={expected_id}, "
                f"got {actual_id}."
            )

        print(
            "\n" + "=" * 70
        )
        print(
            f"STARTING RESEARCH SAMPLE "
            f"{sample_number} "
            f"(prompt_id={actual_id})"
        )
        print(
            f"Baseline z = "
            f"{sample['watermarked_z']}"
        )
        print(
            "=" * 70
        )

        sample_with_number = dict(sample)
        sample_with_number[
            "sample_number"
        ] = sample_number

        evaluator.process_sample(
            sample_with_number,
            model,
            tokenizer,
        )

        print(
            f"[✓ SAVED] Research Sample "
            f"{sample_number} complete"
        )

    print(
        "\n[✓ SAVED] Attack 2 evaluation "
        "complete for Samples 1, 2, 3."
    )


if __name__ == "__main__":
    main()

