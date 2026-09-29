import json
import math
import os

import torch
from transformers import AutoTokenizer


MODEL = "hf_models/Llama-2-7b-hf"
BASELINE = "results/raw/baseline_safe.json"

SAMPLE = 4
POSITION = 275
OLD_WORD = "plant"

REPLACEMENTS = ["station", "facility"]

OUTPUT_DIR = "results/raw/manual_attacks"

HASH_KEY1 = 15485863
HASH_KEY2 = 17624813
GAMMA = 0.5


tokenizer = AutoTokenizer.from_pretrained(MODEL)

with open(BASELINE) as f:
    baseline = json.load(f)

original_text = baseline[SAMPLE - 1]["watermarked_text"]


# ---------------------------------------------------------
# Locate position 275 directly in SAMPLE 4
# ---------------------------------------------------------

encoded = tokenizer(
    original_text,
    add_special_tokens=True,
    truncation=True,
    max_length=2048,
    return_offsets_mapping=True,
)

original_ids = encoded["input_ids"]
offsets = encoded["offset_mapping"]

start, end = offsets[POSITION]

tokenizer_span = original_text[start:end]

leading = len(tokenizer_span) - len(
    tokenizer_span.lstrip()
)

word_start = start + leading
word_end = end

actual_word = original_text[word_start:word_end]

print("SAMPLE:", SAMPLE)
print("POSITION:", POSITION)
print("TOKEN ID:", original_ids[POSITION])
print("TOKENIZER SPAN:", repr(tokenizer_span))
print("ACTUAL WORD:", repr(actual_word))

assert actual_word == OLD_WORD


# ---------------------------------------------------------
# Show the original context
# ---------------------------------------------------------

left = max(0, word_start - 250)
right = min(len(original_text), word_end + 250)

print()
print("ORIGINAL CONTEXT:")
print(original_text[left:right])


# ---------------------------------------------------------
# STA calculation
# ---------------------------------------------------------

rng = torch.Generator(device="cuda")


def is_green(now_token_id, next_token_id):
    seed = (
        HASH_KEY1 * now_token_id
        + HASH_KEY2 * next_token_id
    )

    rng.manual_seed(seed)

    value = torch.rand(
        1,
        device="cuda",
        generator=rng,
    ).item()

    return value < GAMMA


def calculate_z(ids):
    green_count = sum(
        is_green(ids[i], ids[i + 1])
        for i in range(len(ids) - 1)
    )

    pair_count = len(ids) - 1

    z = (
        (green_count - GAMMA * pair_count)
        / math.sqrt(
            GAMMA * (1 - GAMMA) * pair_count
        )
    )

    return green_count, pair_count, z


original_green, original_pairs, original_z = (
    calculate_z(original_ids)
)

print()
print("=" * 70)
print("ORIGINAL")
print("=" * 70)
print("Z:", original_z)
print("GREEN:", original_green)
print("PAIRS:", original_pairs)
print("TOKENS:", len(original_ids))


# ---------------------------------------------------------
# Test each replacement separately
# ---------------------------------------------------------

for new_word in REPLACEMENTS:

    print()
    print("=" * 70)
    print("TEST:", OLD_WORD, "->", new_word)
    print("=" * 70)


    # Check replacement tokenization.
    replacement = tokenizer(
        " " + new_word,
        add_special_tokens=False,
    )

    replacement_ids = replacement["input_ids"]

    print(
        "REPLACEMENT TOKEN IDS:",
        replacement_ids,
    )

    print(
        "REPLACEMENT DECODE:",
        repr(tokenizer.decode(replacement_ids)),
    )


    # -----------------------------------------------------
    # Replace ONLY the word.
    # Preserve the existing space.
    # -----------------------------------------------------

    attacked_text = (
        original_text[:word_start]
        + new_word
        + original_text[word_end:]
    )


    # -----------------------------------------------------
    # Verify construction.
    # -----------------------------------------------------

    assert attacked_text[word_start - 1].isspace()

    assert (
        attacked_text[
            word_start:
            word_start + len(new_word)
        ]
        == new_word
    )


    # -----------------------------------------------------
    # Display attacked context.
    # -----------------------------------------------------

    print()
    print("ATTACKED CONTEXT:")

    left = max(0, word_start - 250)
    right = min(
        len(attacked_text),
        word_start + len(new_word) + 250,
    )

    print(attacked_text[left:right])


    # -----------------------------------------------------
    # Retokenize
    # -----------------------------------------------------

    attacked_ids = tokenizer(
        attacked_text,
        add_special_tokens=True,
        truncation=True,
        max_length=2048,
    )["input_ids"]


    # -----------------------------------------------------
    # Calculate attacked STA score
    # -----------------------------------------------------

    attacked_green, attacked_pairs, attacked_z = (
        calculate_z(attacked_ids)
    )

    delta_z = original_z - attacked_z


    # -----------------------------------------------------
    # Save THIS attack immediately
    # -----------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        f"sample4_pos275_plant_to_{new_word}.json",
    )

    result = {
        "experiment": "manual_semantic_substitution",

        "sample": SAMPLE,
        "position": POSITION,

        "old_word": OLD_WORD,
        "new_word": new_word,

        "tokenizer_span": tokenizer_span,
        "actual_word": actual_word,

        "original_token_id": original_ids[POSITION],
        "replacement_token_ids": replacement_ids,

        "original_z": original_z,
        "attacked_z": attacked_z,
        "delta_z": delta_z,

        "original_green_count": original_green,
        "attacked_green_count": attacked_green,

        "original_pair_count": original_pairs,
        "attacked_pair_count": attacked_pairs,

        "original_token_count": len(original_ids),
        "attacked_token_count": len(attacked_ids),

        "detected_before": original_z > 2,
        "detected_after": attacked_z > 2,

        "original_text": original_text,
        "attacked_text": attacked_text,
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)


    # -----------------------------------------------------
    # Report
    # -----------------------------------------------------

    print()
    print("RESULT:", OLD_WORD, "->", new_word)
    print("ORIGINAL Z:", original_z)
    print("ATTACKED Z:", attacked_z)
    print("DELTA Z:", delta_z)

    print(
        "GREEN COUNT:",
        original_green,
        "->",
        attacked_green,
    )

    print(
        "PAIR COUNT:",
        original_pairs,
        "->",
        attacked_pairs,
    )

    print(
        "TOKEN COUNT:",
        len(original_ids),
        "->",
        len(attacked_ids),
    )

    print(
        "DETECTED:",
        original_z > 2,
        "->",
        attacked_z > 2,
    )

    print()
    print("[✓ SAVED]", output_path)


print()
print("=" * 70)
print("SAMPLE 4 PLANT EXPERIMENT COMPLETE")
print("=" * 70)
