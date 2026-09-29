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
    "sample1_pos345_might_to_could_corrected.json"
)

DIAGNOSTIC_OUTPUT = (
    "results/raw/manual_attacks/"
    "sample1_pos345_might_to_could_diagnostic.json"
)

HASH_KEY1 = 15485863
HASH_KEY2 = 17624813
GAMMA = 0.5


# ---------------------------------------------------------
# 1. Load tokenizer and experiment data.
# ---------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL)

with open(BASELINE) as f:
    baseline = json.load(f)[SAMPLE - 1]

with open(WORKSHEET) as f:
    worksheet = json.load(f)


# ---------------------------------------------------------
# 2. Verify selected vulnerable position.
# ---------------------------------------------------------

candidate = next(
    c for c in worksheet["candidates"]
    if c["position"] == POSITION
)

assert candidate["old_token"] == OLD_WORD

original_text = baseline["watermarked_text"]


# ---------------------------------------------------------
# 3. Recover tokenizer offsets.
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
print("TARGET WORD:", repr(OLD_WORD))
print("TOKEN ID:", input_ids[POSITION])
print("TOKENIZER SPAN:", repr(original_span))


# ---------------------------------------------------------
# 4. Record the previous failed diagnostic.
#
# The tokenizer span includes the leading whitespace.
# This is NOT an attack result.
# ---------------------------------------------------------

diagnostic = {
    "experiment": "manual_semantic_substitution",
    "status": "implementation_diagnostic",
    "sample": SAMPLE,
    "position": POSITION,
    "old_word": OLD_WORD,
    "new_word": NEW_WORD,
    "token_id": input_ids[POSITION],
    "tokenizer_span": original_span,
    "previous_error": (
        "AssertionError caused by comparing tokenizer span "
        "' might' directly with word 'might'."
    ),
    "correction": (
        "Preserve leading whitespace and replace only the "
        "word portion of the tokenizer span."
    ),
}

os.makedirs(
    os.path.dirname(DIAGNOSTIC_OUTPUT),
    exist_ok=True,
)

with open(DIAGNOSTIC_OUTPUT, "w") as f:
    json.dump(diagnostic, f, indent=2)

print(
    "[✓ SAVED] diagnostic:",
    DIAGNOSTIC_OUTPUT,
)


# ---------------------------------------------------------
# 5. Separate whitespace from the actual word.
# ---------------------------------------------------------

leading_whitespace_length = (
    len(original_span)
    - len(original_span.lstrip())
)

word_start = start + leading_whitespace_length
word_end = end

actual_word_span = original_text[
    word_start:word_end
]

print(
    "LEADING WHITESPACE:",
    repr(original_text[start:word_start]),
)

print(
    "ACTUAL WORD SPAN:",
    repr(actual_word_span),
)

assert actual_word_span == OLD_WORD


# ---------------------------------------------------------
# 6. Display original context.
# ---------------------------------------------------------

context_left = max(
    0,
    word_start - 180,
)

context_right = min(
    len(original_text),
    word_end + 180,
)

print()
print("BEFORE:")
print(
    original_text[
        context_left:context_right
    ]
)


# ---------------------------------------------------------
# 7. Perform the real human-style word substitution.
#
# IMPORTANT:
# We replace only "might".
# The existing space before it is preserved.
# ---------------------------------------------------------

attacked_text = (
    original_text[:word_start]
    + NEW_WORD
    + original_text[word_end:]
)


# ---------------------------------------------------------
# 8. Verify the resulting text.
# ---------------------------------------------------------

after_left = max(
    0,
    word_start - 180,
)

after_right = min(
    len(attacked_text),
    word_start + len(NEW_WORD) + 180,
)

print()
print("AFTER:")
print(
    attacked_text[
        after_left:after_right
    ]
)

assert attacked_text != original_text

# Verify the exact local substitution.
expected_local = (
    original_text[start:word_start]
    + NEW_WORD
    + original_text[word_end:word_end + 1]
)

actual_local = attacked_text[
    start:
    word_start + len(NEW_WORD) + 1
]

print()
print(
    "EXPECTED LOCAL:",
    repr(expected_local),
)

print(
    "ACTUAL LOCAL:",
    repr(actual_local),
)

assert expected_local == actual_local


# ---------------------------------------------------------
# 9. Re-tokenize original and attacked texts.
# ---------------------------------------------------------

original_encoded = tokenizer(
    original_text,
    add_special_tokens=True,
    truncation=True,
    max_length=2048,
)

attacked_encoded = tokenizer(
    attacked_text,
    add_special_tokens=True,
    truncation=True,
    max_length=2048,
)

original_ids = original_encoded["input_ids"]
attacked_ids = attacked_encoded["input_ids"]


# ---------------------------------------------------------
# 10. Verify that "could" is one tokenizer token when
#     preceded by a space.
# ---------------------------------------------------------

could_encoded = tokenizer(
    " could",
    add_special_tokens=False,
)

could_ids = could_encoded["input_ids"]

print()
print("COULD TOKENIZATION:", could_ids)

assert len(could_ids) == 1

print(
    "COULD DECODE:",
    repr(tokenizer.decode(could_ids))
)


# ---------------------------------------------------------
# 11. STA detector calculation.
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
        is_green(
            ids[i],
            ids[i + 1],
        )
        for i in range(len(ids) - 1)
    )

    pair_count = len(ids) - 1

    z = (
        (green_count - GAMMA * pair_count)
        / math.sqrt(
            GAMMA
            * (1 - GAMMA)
            * pair_count
        )
    )

    return (
        green_count,
        pair_count,
        z,
    )


original_green, original_pairs, original_z = (
    calculate_z(original_ids)
)

attacked_green, attacked_pairs, attacked_z = (
    calculate_z(attacked_ids)
)


# ---------------------------------------------------------
# 12. Save complete attack result.
# ---------------------------------------------------------

result = {
    "experiment": "manual_semantic_substitution_corrected",

    "sample": SAMPLE,
    "position": POSITION,

    "old_word": OLD_WORD,
    "new_word": NEW_WORD,

    "original_token_id": input_ids[POSITION],
    "original_tokenizer_span": original_span,
    "actual_word_span": actual_word_span,

    "could_token_ids": could_ids,

    "original_z": original_z,
    "attacked_z": attacked_z,
    "delta_z": original_z - attacked_z,

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

with open(OUTPUT, "w") as f:
    json.dump(
        result,
        f,
        indent=2,
    )


# ---------------------------------------------------------
# 13. Final report.
# ---------------------------------------------------------

print()
print("=" * 70)
print("RESULT")
print("=" * 70)

print(
    "EDIT:",
    OLD_WORD,
    "->",
    NEW_WORD,
)

print(
    "ORIGINAL Z:",
    original_z,
)

print(
    "ATTACKED Z:",
    attacked_z,
)

print(
    "DELTA Z:",
    original_z - attacked_z,
)

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
print(
    "[✓ SAVED]",
    OUTPUT,
)
