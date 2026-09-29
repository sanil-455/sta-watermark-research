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
NEW_WORD = "could"

OUTPUT = (
    "results/raw/manual_attacks/"
    "sample1_pos345_might_to_could.json"
)

HASH_KEY1 = 15485863
HASH_KEY2 = 17624813
GAMMA = 0.5


tokenizer = AutoTokenizer.from_pretrained(MODEL)

with open(BASELINE) as f:
    baseline = json.load(f)[SAMPLE - 1]

with open(WORKSHEET) as f:
    worksheet = json.load(f)


# ---------------------------------------------------------
# 1. Verify that the selected vulnerability is exactly the
#    one we intend to attack.
# ---------------------------------------------------------

candidate = next(
    c for c in worksheet["candidates"]
    if c["position"] == POSITION
)

assert candidate["old_token"] == OLD_WORD

original_text = baseline["watermarked_text"]


# ---------------------------------------------------------
# 2. Locate the exact character span corresponding to the
#    selected tokenizer position.
# ---------------------------------------------------------

encoded = tokenizer(
    original_text,
    add_special_tokens=True,
    truncation=True,
    max_length=2048,
    return_offsets_mapping=True,
)

input_ids = encoded["input_ids"]
offsets = encoded["offset_mapping"]

start, end = offsets[POSITION]

original_span = original_text[start:end]

print("POSITION:", POSITION)
print("EXPECTED WORD:", repr(OLD_WORD))
print("TOKENIZER SPAN:", repr(original_span))

assert OLD_WORD in original_span
assert original_span.strip() == OLD_WORD


# ---------------------------------------------------------
# 3. Show the local context BEFORE editing.
# ---------------------------------------------------------

context_left = max(0, start - 180)
context_right = min(len(original_text), end + 180)

print()
print("BEFORE:")
print(original_text[context_left:context_right])


# ---------------------------------------------------------
# 4. Perform the actual human-style textual substitution.
# ---------------------------------------------------------

attacked_text = (
    original_text[:start]
    + NEW_WORD
    + original_text[end:]
)


# ---------------------------------------------------------
# 5. Show the resulting context and verify the word change.
# ---------------------------------------------------------

new_start = start
new_end = start + len(NEW_WORD)

print()
print("AFTER:")
print(
    attacked_text[
        max(0, new_start - 180):
        min(len(attacked_text), new_end + 180)
    ]
)

assert attacked_text != original_text
assert NEW_WORD in attacked_text


# ---------------------------------------------------------
# 6. Re-tokenize BOTH texts.
# ---------------------------------------------------------

original_ids = tokenizer(
    original_text,
    add_special_tokens=True,
    truncation=True,
    max_length=2048,
)["input_ids"]

attacked_ids = tokenizer(
    attacked_text,
    add_special_tokens=True,
    truncation=True,
    max_length=2048,
)["input_ids"]


# ---------------------------------------------------------
# 7. Reproduce STA detector calculation.
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

attacked_green, attacked_pairs, attacked_z = (
    calculate_z(attacked_ids)
)


# ---------------------------------------------------------
# 8. Save EVERYTHING needed to reproduce the result.
# ---------------------------------------------------------

os.makedirs(
    os.path.dirname(OUTPUT),
    exist_ok=True,
)

result = {
    "experiment": "manual_semantic_substitution",
    "sample": SAMPLE,
    "position": POSITION,

    "old_word": OLD_WORD,
    "new_word": NEW_WORD,

    "original_tokenizer_span": original_span,

    "original_z": original_z,
    "attacked_z": attacked_z,
    "delta_z": original_z - attacked_z,

    "original_green_count": original_green,
    "attacked_green_count": attacked_green,

    "original_pair_count": original_pairs,
    "attacked_pair_count": attacked_pairs,

    "detected_before": original_z > 2,
    "detected_after": attacked_z > 2,

    "original_text": original_text,
    "attacked_text": attacked_text,
}

with open(OUTPUT, "w") as f:
    json.dump(result, f, indent=2)


# ---------------------------------------------------------
# 9. Report the result.
# ---------------------------------------------------------

print()
print("=" * 70)
print("RESULT")
print("=" * 70)

print("EDIT:", OLD_WORD, "->", NEW_WORD)

print("ORIGINAL Z:", original_z)
print("ATTACKED Z:", attacked_z)
print("DELTA Z:", original_z - attacked_z)

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
    "DETECTED:",
    original_z > 2,
    "->",
    attacked_z > 2,
)

print()
print("[✓ SAVED]", OUTPUT)
