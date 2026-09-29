import json
import os
import re

from transformers import AutoTokenizer


MODEL = "hf_models/Llama-2-7b-hf"
BASELINE = "results/raw/baseline_safe.json"

WORKSHEET_DIR = "results/raw/manual_sta_worksheets"
OUTPUT_DIR = "results/raw/manual_attacks"


# ---------------------------------------------------------
# Load tokenizer and baseline
# ---------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL)

with open(BASELINE) as f:
    baseline = json.load(f)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------
# Words we don't want for the FIRST natural-substitution
# attack.
#
# These are valid words in English, but they are generally
# poor candidates for our current experiment because a
# natural replacement is difficult or too disruptive.
#
# We will NOT discard them permanently. They can be useful
# for later attack classes.
# ---------------------------------------------------------

SKIP_FUNCTION_WORDS = {
    "a",
    "i",
    "an",
    "the",
    "to",
    "of",
    "and",
    "or",
    "in",
    "on",
    "at",
    "by",
    "for",
    "with",
    "from",
    "as",
    "is",
    "it",
    "be",
    "was",
    "were",
    "are",
    "am",
    "has",
    "have",
    "had",
}


def get_word_from_token(text, offsets, position):
    """
    Convert the tokenizer span into the actual word.

    Llama tokenization may return:
        ' might'

    even though the actual word is:
        'might'

    We remove only tokenizer-included leading
    whitespace for classification.
    """

    start, end = offsets[position]

    span = text[start:end]

    leading = len(span) - len(span.lstrip())

    word_start = start + leading
    word_end = end

    word = text[word_start:word_end]

    return word, word_start, word_end


def is_meaningful_whole_word(
    text,
    offsets,
    position,
):
    """
    Keep only ordinary alphabetic whole words.

    Reject:
      - punctuation
      - subword fragments
      - hashtags/fragments
      - words embedded inside larger words
      - single-character tokens
    """

    if position >= len(offsets):
        return False, "", None, None

    start, end = offsets[position]

    if end <= start:
        return False, "", None, None

    word, word_start, word_end = get_word_from_token(
        text,
        offsets,
        position,
    )

    if not re.fullmatch(
        r"[A-Za-z]{2,}",
        word,
    ):
        return False, word, word_start, word_end

    # Must begin at text start or after whitespace.
    if word_start > 0:
        if not text[word_start - 1].isspace():
            return False, word, word_start, word_end

    # Must end before whitespace/punctuation/end.
    if word_end < len(text):
        if text[word_end].isalnum():
            return False, word, word_start, word_end

    return True, word, word_start, word_end


all_samples = []


# ---------------------------------------------------------
# Process all five samples
# ---------------------------------------------------------

for sample_number in range(1, 6):

    print()
    print("=" * 100)
    print("SAMPLE", sample_number)
    print("=" * 100)

    worksheet_path = os.path.join(
        WORKSHEET_DIR,
        f"sample_{sample_number}.json",
    )

    if not os.path.exists(worksheet_path):
        print(
            "[ERROR] Missing worksheet:",
            worksheet_path,
        )
        continue

    with open(worksheet_path) as f:
        worksheet = json.load(f)

    text = baseline[
        sample_number - 1
    ]["watermarked_text"]

    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=2048,
        return_offsets_mapping=True,
    )

    offsets = encoded["offset_mapping"]
    input_ids = encoded["input_ids"]

    candidates = []

    for candidate in worksheet["candidates"]:

        position = candidate["position"]

        valid, word, word_start, word_end = (
            is_meaningful_whole_word(
                text,
                offsets,
                position,
            )
        )

        if not valid:
            continue

        # For THIS first attack class, skip common
        # function words. They remain available for
        # later experiments.
        if word.lower() in SKIP_FUNCTION_WORDS:
            continue

        # Full local context around the actual word.
        context_left = max(
            0,
            word_start - 180,
        )

        context_right = min(
            len(text),
            word_end + 180,
        )

        context = text[
            context_left:context_right
        ]

        candidates.append(
            {
                "position": position,
                "target": word,
                "token_id": input_ids[position],
                "red_red_opportunities": len(
                    candidate["red_replacements"]
                ),
                "context": context,
            }
        )

    # Strongest STA-local vulnerability first.
    candidates.sort(
        key=lambda x: (
            x["red_red_opportunities"],
            -x["position"],
        ),
        reverse=True,
    )

    result = {
        "sample": sample_number,
        "num_double_green_positions": len(
            worksheet["candidates"]
        ),
        "num_meaningful_replaceable_candidates": len(
            candidates
        ),
        "candidates": candidates,
    }

    output_path = os.path.join(
        OUTPUT_DIR,
        f"sample{sample_number}_replaceable_candidates.json",
    )

    with open(output_path, "w") as f:
        json.dump(
            result,
            f,
            indent=2,
        )

    print(
        "DOUBLE-GREEN POSITIONS:",
        result["num_double_green_positions"],
    )

    print(
        "MEANINGFUL REPLACEABLE CANDIDATES:",
        result[
            "num_meaningful_replaceable_candidates"
        ],
    )

    # -----------------------------------------------------
    # SAVE CONFIRMATION FOR EACH SAMPLE
    # -----------------------------------------------------

    print(
        "[✓ SAVED]",
        output_path,
    )

    all_samples.append(result)


# ---------------------------------------------------------
# Save combined master file
# ---------------------------------------------------------

master_path = os.path.join(
    OUTPUT_DIR,
    "all5_replaceable_candidates.json",
)

master = {
    "experiment": (
        "Screen all five samples for meaningful "
        "whole-word candidates suitable for "
        "human-selected substitutions."
    ),
    "samples": all_samples,
}

with open(master_path, "w") as f:
    json.dump(
        master,
        f,
        indent=2,
    )

print()
print("=" * 100)
print("[✓ SAVED]", master_path)
print("=" * 100)


# ---------------------------------------------------------
# PRINT HUMAN-REVIEW LIST
#
# We show up to 30 candidates per sample.
# If there are more, they are still saved in JSON.
# ---------------------------------------------------------

print()
print()
print("=" * 100)
print("HUMAN-REVIEW CANDIDATES")
print("=" * 100)

for sample in all_samples:

    print()
    print()
    print("#" * 100)
    print("SAMPLE", sample["sample"])
    print("#" * 100)

    for rank, candidate in enumerate(
        sample["candidates"][:30],
        1,
    ):

        print()
        print(
            f"CANDIDATE {rank}"
        )

        print(
            "Position:",
            candidate["position"],
        )

        print(
            "Target:",
            repr(candidate["target"]),
        )

        print(
            "Token ID:",
            candidate["token_id"],
        )

        print(
            "RED-RED opportunities:",
            candidate[
                "red_red_opportunities"
            ],
        )

        print(
            "CONTEXT:"
        )

        print(
            candidate["context"]
        )

        print("-" * 100)


print()
print()
print("=" * 100)
print("SCREENING COMPLETE")
print("=" * 100)

for sample in all_samples:
    print(
        "Sample",
        sample["sample"],
        ":",
        sample[
            "num_meaningful_replaceable_candidates"
        ],
        "meaningful candidates",
    )

print()
print("[✓ SAVED]", master_path)
