import json
import math
import os
import torch
from transformers import AutoTokenizer


MODEL_PATH = "hf_models/Llama-2-7b-hf"
DATA_PATH = "results/raw/baseline_safe.json"
OUT_DIR = "results/raw/manual_sta_worksheets"

HASH_KEY1 = 15485863
HASH_KEY2 = 17624813
GAMMA = 0.5

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required.")

rng = torch.Generator(device="cuda")


def is_green(a, b):
    seed = (
        HASH_KEY1 * int(a)
        + HASH_KEY2 * int(b)
    )

    rng.manual_seed(seed)

    return (
        torch.rand(
            1,
            device="cuda",
            generator=rng
        ).item()
        < GAMMA
    )


def token_text(token_id):
    return tokenizer.decode(
        [token_id],
        clean_up_tokenization_spaces=False
    )


def main():

    os.makedirs(OUT_DIR, exist_ok=True)

    with open(DATA_PATH, "r") as f:
        data = json.load(f)

    for sample_index, sample in enumerate(data):

        print()
        print("=" * 70)
        print(
            f"SAMPLE {sample_index + 1}/{len(data)}"
        )
        print("=" * 70)

        wm = tokenizer(
            sample["watermarked_text"],
            add_special_tokens=True,
            truncation=True,
            max_length=2048
        )["input_ids"]

        uwm = tokenizer(
            sample["unwatermarked_text"],
            add_special_tokens=True,
            truncation=True,
            max_length=2048
        )["input_ids"]

        donor_tokens = sorted(
            set(int(x) for x in uwm)
            - set(int(x) for x in tokenizer.all_special_ids)
        )

        candidates = []

        for pos in range(1, len(wm) - 1):

            old_left = is_green(
                wm[pos - 1],
                wm[pos]
            )

            old_right = is_green(
                wm[pos],
                wm[pos + 1]
            )

            if not (old_left and old_right):
                continue

            red_replacements = []

            for new_token in donor_tokens:

                if new_token == wm[pos]:
                    continue

                new_left = is_green(
                    wm[pos - 1],
                    new_token
                )

                new_right = is_green(
                    new_token,
                    wm[pos + 1]
                )

                if not new_left and not new_right:
                    red_replacements.append({
                        "token_id": new_token,
                        "token": token_text(new_token)
                    })

            if red_replacements:

                candidates.append({
                    "position": pos,

                    "old_token_id":
                        int(wm[pos]),

                    "old_token":
                        token_text(wm[pos]),

                    "previous_token_id":
                        int(wm[pos - 1]),

                    "previous_token":
                        token_text(wm[pos - 1]),

                    "next_token_id":
                        int(wm[pos + 1]),

                    "next_token":
                        token_text(wm[pos + 1]),

                    "old_left": "GREEN",
                    "old_right": "GREEN",

                    "red_replacements":
                        red_replacements
                })

        result = {
            "sample_index": sample_index,
            "prompt_id": sample["prompt_id"],
            "original_z":
                float(sample["watermarked_z"]),
            "watermarked_token_count":
                len(wm),
            "donor_unique_token_count":
                len(donor_tokens),
            "double_green_positions":
                len(candidates),
            "candidates":
                candidates
        }

        output_path = (
            f"{OUT_DIR}/"
            f"sample_{sample_index + 1}.json"
        )

        with open(output_path, "w") as f:
            json.dump(
                result,
                f,
                indent=2
            )

        print(
            f"Found {len(candidates)} "
            f"GREEN+GREEN positions."
        )

        print(
            f"[✓ SAVED] Sample "
            f"{sample_index + 1}/{len(data)}"
        )

        print(
            f"        {output_path}"
        )


if __name__ == "__main__":
    main()

