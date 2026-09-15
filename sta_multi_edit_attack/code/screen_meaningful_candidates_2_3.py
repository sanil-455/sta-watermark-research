import json
import os
import re

from transformers import AutoTokenizer


# CONFIG

MODEL_PATH = "hf_models/Llama-2-7b-hf"

SAMPLES = [2, 3]

INPUT_TEMPLATE = "results/raw/manual_sta_worksheets/sample_{}.json"

OUTPUT_DIR = "results/raw/manual_attacks"

OUTPUT_TEMPLATE = OUTPUT_DIR + "/sample{}_meaningful_vulnerability_candidates.json"

TOP_N = 25


# LOAD TOKENIZER

print()
print("=" * 90)
print("LOADING LLAMA TOKENIZER")
print("=" * 90)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    use_fast=True,
)

print("[✓] Tokenizer loaded")


# HELPERS

def is_complete_alphabetic_word(text, start, end):
    """
    Check whether the tokenizer span corresponds to a complete
    alphabetic word in the actual generated text.

    This rejects fragments such as:
        ues  -> Tuesday
        adm  -> admires
        Botan -> Botanical
        Ch   -> Christie/characters etc.
    """

    if start is None or end is None:
        return False

    if start == end:
        return False

    span = text[start:end]

    # Remove whitespace surrounding the token span.
    stripped = span.strip()

    # Must contain something.
    if not stripped:
        return False

    # Must consist entirely of alphabetic characters.
    if not stripped.isalpha():
        return False

    # Locate the actual alphabetic span inside the original text.
    left = start

    while left < end and text[left].isspace():
        left += 1

    right = end

    while right > left and text[right - 1].isspace():
        right -= 1

    # If there is an alphabetic character immediately before
    # the token span, this is only a subword.
    if left > 0 and text[left - 1].isalpha():
        return False

    # If there is an alphabetic character immediately after
    # the token span, this is only a subword.
    if right < len(text) and text[right].isalpha():
        return False

    return True


def get_context(text, start, end, radius=85):
    """
    Return human-readable context around the target word.
    """

    left = max(0, start - radius)
    right = min(len(text), end + radius)

    context = text[left:right]

    # Make whitespace easier to read in terminal.
    context = context.replace("\n", " ")

    return context


def token_span_information(text):
    """
    Tokenize the exact generated text and return token offsets.
    """

    encoded = tokenizer(
        text,
        add_special_tokens=True,
        return_offsets_mapping=True,
        truncation=False,
    )

    return encoded["input_ids"], encoded["offset_mapping"]


# PROCESS EACH SAMPLE

os.makedirs(OUTPUT_DIR, exist_ok=True)

all_results = {}

for sample_num in SAMPLES:

    input_path = INPUT_TEMPLATE.format(sample_num)
    output_path = OUTPUT_TEMPLATE.format(sample_num)

    print()
    print("=" * 90)
    print(f"SAMPLE {sample_num}")
    print("=" * 90)

    # --------------------------------------------------------
    # Load worksheet
    # --------------------------------------------------------

    with open(input_path, "r") as f:
        worksheet = json.load(f)

    print("Original z:", worksheet["original_z"])
    print("Watermarked token count:", worksheet["watermarked_token_count"])
    print("Double-green positions:", worksheet["double_green_positions"])
    print("Total vulnerable positions:", len(worksheet["candidates"]))

    # --------------------------------------------------------
    # Recover the exact generated text from baseline
    # --------------------------------------------------------

    baseline_path = "results/raw/baseline_safe.json"

    with open(baseline_path, "r") as f:
        baseline = json.load(f)

    sample = baseline[sample_num - 1]

    text = sample["watermarked_text"]

    print("Text length:", len(text))

    # --------------------------------------------------------
    # Get exact tokenizer offsets
    # --------------------------------------------------------

    input_ids, offsets = token_span_information(text)

    print("Tokenizer sequence length:", len(input_ids))
    print("Offset count:", len(offsets))

    # --------------------------------------------------------
    # Screen candidates
    # --------------------------------------------------------

    meaningful = []

    rejected = {
        "invalid_position": 0,
        "empty_span": 0,
        "non_alphabetic": 0,
        "subword_fragment": 0,
        "too_short": 0,
    }

    for candidate in worksheet["candidates"]:

        position = candidate["position"]

        # Make absolutely sure the position exists.
        if position < 0 or position >= len(offsets):
            rejected["invalid_position"] += 1
            continue

        start, end = offsets[position]

        if start == end:
            rejected["empty_span"] += 1
            continue

        span = text[start:end]
        word = span.strip()

        # Must be alphabetic.
        if not word.isalpha():
            rejected["non_alphabetic"] += 1
            continue

        # Avoid extremely short words.
        # We keep 2-letter words because legitimate words like
        # "US", "UK", "to", "of", etc. can occur, but single
        # character tokens are never accepted.
        if len(word) < 2:
            rejected["too_short"] += 1
            continue

        # Most important check:
        # Is this the COMPLETE word, or merely a tokenizer piece?
        if not is_complete_alphabetic_word(text, start, end):
            rejected["subword_fragment"] += 1
            continue

        red_red = len(candidate.get("red_replacements", []))

        item = {
            "position": position,
            "target": word,
            "red_red_opportunities": red_red,
            "context": get_context(text, start, end),
            "span_start": start,
            "span_end": end,
            "old_token_id": candidate["old_token_id"],
            "previous_token": candidate["previous_token"],
            "next_token": candidate["next_token"],
            "red_replacements": candidate["red_replacements"],
        }

        meaningful.append(item)

    # --------------------------------------------------------
    # Rank
    # --------------------------------------------------------

    meaningful.sort(
        key=lambda x: (
            -x["red_red_opportunities"],
            x["position"],
        )
    )

    top = meaningful[:TOP_N]

    result = {
        "sample": sample_num,
        "original_z": worksheet["original_z"],
        "watermarked_token_count": worksheet["watermarked_token_count"],
        "double_green_positions": worksheet["double_green_positions"],
        "total_vulnerable_positions": len(worksheet["candidates"]),
        "meaningful_complete_word_candidates": len(meaningful),
        "top_n": len(top),
        "rejection_counts": rejected,
        "candidates": top,
    }

    # --------------------------------------------------------
    # SAVE IMMEDIATELY
    # --------------------------------------------------------

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print()
    print(f"[✓ SAVED] {output_path}")

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    print()
    print("-" * 90)
    print(f"TOP MEANINGFUL CANDIDATES — SAMPLE {sample_num}")
    print("-" * 90)

    for rank, item in enumerate(top, 1):

        print()
        print(
            f"#{rank} | "
            f"position={item['position']} | "
            f"target='{item['target']}' | "
            f"RED-RED opportunities={item['red_red_opportunities']}"
        )

        print("Context:")
        print(item["context"])

    # --------------------------------------------------------
    # Screening statistics
    # --------------------------------------------------------

    print()
    print("-" * 90)
    print("SCREENING SUMMARY")
    print("-" * 90)

    print(
        "Original vulnerable positions:",
        len(worksheet["candidates"])
    )

    print(
        "Meaningful complete-word candidates:",
        len(meaningful)
    )

    print("Rejected:")
    for key, value in rejected.items():
        print(f"  {key}: {value}")

    print("[✓ SAMPLE COMPLETE]")


    all_results[sample_num] = result


# MASTER FILE

master_path = OUTPUT_DIR + "/samples2_3_meaningful_vulnerability_candidates.json"

with open(master_path, "w") as f:
    json.dump(all_results, f, indent=2)

print()
print("=" * 90)
print("[✓ SAVED] " + master_path)
print("=" * 90)
print()
print("SCREENING COMPLETE")
print()
