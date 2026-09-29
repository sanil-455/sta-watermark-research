import json
import math
import os

import torch
from transformers import AutoTokenizer


MODEL = "hf_models/Llama-2-7b-hf"
BASELINE = "results/raw/baseline_safe.json"
WORKSHEET = "results/raw/manual_sta_worksheets/sample_1.json"

SAMPLE = 1
POSITION = 345
OLD_WORD = "might"

NEW_WORDS = [
    "may",
    "can",
]

HASH_KEY1 = 15485863
HASH_KEY2 = 17624813
GAMMA = 0.5

OUTPUT_DIR = "results/raw/manual_attacks"


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL)

with open(BASELINE) as f:
    baseline = json.load(f)[SAMPLE - 1]

with open(WORKSHEET) as f:
    worksheet = json.load(f)

original_text = baseline["watermarked_text"]


# ---------------------------------------------------------
# Verify vulnerable position
# ---------------------------------------------------------

candidate = next(
    c for c in worksheet["candidates"]
    if c["position"] == POSITION
)

assert candidate["old_token"] == OLD_WORD


# ---------------------------------------------------------
# Get tokenizer offsets
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

leading_whitespace_length = (
    len(tokenizer_span)
    - len(tokenizer_span.lstrip())
)

word_start = start + leading_whitespace_length
word_end = end

actual_word = original_text[word_start:word_end]

print("POSITION:", POSITION)
print("ORIGINAL WORD:", repr(actual_word))
print("TOKENIZER SPAN:", repr(tokenizer_span))
print("ORIGINAL TOKEN ID:", original_ids[POSITION])

assert actual_word == OLD_WORD


# ---------------------------------------------------------
# STA detector
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
print("GREEN COUNT:", original_green)
print("PAIR COUNT:", original_pairs)
print("TOKEN COUNT:", len(original_ids))


# ---------------------------------------------------------
# Test MAY and CAN independently
# ---------------------------------------------------------

for new_word in NEW_WORDS:

    print()
    print("=" * 70)
    print("TESTING:", OLD_WORD, "->", new_word)
    print("=" * 70)

    # Verify replacement is a single tokenizer token
    replacement_encoding = tokenizer(
        " " + new_word,
        add_special_tokens=False,
    )

    replacement_ids = replacement_encoding["input_ids"]

    print(
        "Replacement token IDs:",
        replacement_ids,
    )

    print(
        "Replacement decode:",
        repr(tokenizer.decode(replacement_ids)),
    )

    assert len(replacement_ids) == 1

    # -----------------------------------------------------
    # Perform text-level substitution.
    # Preserve the original whitespace.
    # -----------------------------------------------------

    attacked_text = (
        original_text[:word_start]
        + new_word
        + original_text[word_end:]
    )

    # -----------------------------------------------------
    # Verify local text.
    # -----------------------------------------------------

    attacked_context = attacked_text[
        max(0, word_start - 180):
        min(
            len(attacked_text),
            word_start + len(new_word) + 180,
        )
    ]

    print()
    print("ATTACKED CONTEXT:")
    print(attacked_context)

    # Must contain a space before the new word
    assert attacked_text[word_start - 1] == " "

    # Must contain the replacement word
    assert (
        attacked_text[
            word_start:
            word_start + len(new_word)
        ]
        == new_word
    )

    # -----------------------------------------------------
    # Re-tokenize attacked text.
    # -----------------------------------------------------

    attacked_ids = tokenizer(
        attacked_text,
        add_special_tokens=True,
        truncation=True,
        max_length=2048,
    )["input_ids"]

    # -----------------------------------------------------
    # Calculate attacked STA score.
    # -----------------------------------------------------

    attacked_green, attacked_pairs, attacked_z = (
        calculate_z(attacked_ids)
    )

    delta_z = original_z - attacked_z

    # -----------------------------------------------------
    # Save THIS attack immediately.
    # -----------------------------------------------------

    output = os.path.join(
        OUTPUT_DIR,
        f"sample1_pos345_might_to_{new_word}.json",
    )

    result = {
        "experiment": "manual_semantic_substitution",
        "sample": SAMPLE,
        "position": POSITION,

        "old_word": OLD_WORD,
        "new_word": new_word,

        "original_token_id": original_ids[POSITION],
        "replacement_token_ids": replacement_ids,

        "original_tokenizer_span": tokenizer_span,

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

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    with open(output, "w") as f:
        json.dump(
            result,
            f,
            indent=2,
        )

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
    print("[✓ SAVED]", output)


print()
print("=" * 70)
print("ALL ALTERNATIVES COMPLETE")
print("=" * 70)
