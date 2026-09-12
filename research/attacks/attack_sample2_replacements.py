import json
import os
import math
import traceback

import torch
from transformers import AutoTokenizer

# CONFIGURATION

MODEL_PATH = "hf_models/Llama-2-7b-hf"

BASELINE_PATH = "results/raw/baseline_safe.json"
WORKSHEET_PATH = "results/raw/manual_sta_worksheets/sample_2.json"

OUTPUT_DIR = "results/raw/manual_attacks/sample2_replacements"
SUMMARY_PATH = os.path.join(
    OUTPUT_DIR,
    "sample2_replacement_summary.json"
)
GAMMA = 0.5
Z_THRESHOLD = 2.0

HASH_KEY1 = 15485863
HASH_KEY2 = 17624813

PROMPT_MAX_LENGTH = 2048

# EXACT TARGETS,not to be changed

TARGETS = [
    {
        "position": 515,
        "original": "separation",
        "replacements": [
            "severance",
            "termination",
        ],
    },

    {
        "position": 610,
        "original": "growth",
        "replacements": [
            "demand",
            "expansion",
        ],
    },

    {
        "position": 292,
        "original": "capacity",
        "replacements": [
            "output",
            "volume",
        ],
    },

    {
        "position": 377,
        "original": "releases",
        "replacements": [
            "disengages",
            "eases off",
        ],
    },

    {
        "position": 713,
        "original": "making",
        "replacements": [
            "achieving",
            "delivering",
            "driving",
        ],
    },

    {
        "position": 714,
        "original": "some",
        "replacements": [
            "steady",
            "significant",
        ],
    },
]

# SETUP

os.makedirs(OUTPUT_DIR, exist_ok=True)

print()
print("=" * 90)
print("SAMPLE 2 — TARGETED WORD SUBSTITUTION ATTACK")
print("=" * 90)

print()
print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    use_fast=True,
)

# LOAD BASELINE

with open(BASELINE_PATH, "r") as f:
    baseline = json.load(f)

with open(WORKSHEET_PATH, "r") as f:
    worksheet = json.load(f)

sample = baseline[1]

original_text = sample["watermarked_text"]
stored_z = float(sample["watermarked_z"])

print()
print("Stored baseline z:", stored_z)
print("Worksheet z:", worksheet["original_z"])

# STA DETECTOR

def calculate_sta_z(text):

    enc = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=PROMPT_MAX_LENGTH,
    )

    ids = enc["input_ids"]

    if len(ids) < 2:
        raise RuntimeError(
            "Text contains fewer than 2 tokens."
        )

    rng = torch.Generator(device="cuda")

    green_count = 0
    total_pairs = len(ids) - 1

    for i in range(total_pairs):

        now_token = int(ids[i])
        next_token = int(ids[i + 1])

        seed = (
            HASH_KEY1 * now_token
            + HASH_KEY2 * next_token
        )

        rng.manual_seed(seed)

        random_value = torch.rand(
            1,
            device="cuda",
            generator=rng,
        ).item()

        if random_value < GAMMA:
            green_count += 1

    z = (
        green_count - GAMMA * total_pairs
    ) / math.sqrt(
        GAMMA * (1.0 - GAMMA) * total_pairs
    )

    return {
        "z": float(z),
        "green_count": int(green_count),
        "total_pairs": int(total_pairs),
        "token_count": int(len(ids)),
    }

# BASELINE VALIDATION

print()
print("=" * 90)
print("BASELINE VALIDATION")
print("=" * 90)

baseline_detection = calculate_sta_z(original_text)

baseline_z_error = abs(
    stored_z - baseline_detection["z"]
)

print("Stored z:       ", stored_z)
print("Recomputed z:   ", baseline_detection["z"])
print("Green count:    ", baseline_detection["green_count"])
print("Total pairs:    ", baseline_detection["total_pairs"])
print("Token count:    ", baseline_detection["token_count"])
print("z error:        ", baseline_z_error)

if baseline_z_error > 1e-6:
    raise RuntimeError(
        "BASELINE VALIDATION FAILED. "
        "No attacks will be run."
    )

# ORIGINAL OFFSETS

original_encoded = tokenizer(
    original_text,
    add_special_tokens=True,
    return_offsets_mapping=True,
    truncation=True,
    max_length=PROMPT_MAX_LENGTH,
)

original_ids = original_encoded["input_ids"]
original_offsets = original_encoded["offset_mapping"]

print()
print("Original tokenizer length:", len(original_ids))

# EXACT WORD SPAN EXTRACTION

def get_actual_word_span(text, raw_start, raw_end):

    """
    Tokenizer may return:

        raw span = " separation"

    We separate that into:

        leading whitespace = " "
        actual word        = "separation"

    This avoids the previous false subword error.
    """

    if raw_start < 0 or raw_end > len(text):
        raise RuntimeError("Invalid offset.")

    if raw_start > raw_end:
        raise RuntimeError("Invalid offset ordering.")

    leading_end = raw_start

    while leading_end < raw_end and text[leading_end].isspace():
        leading_end += 1

    trailing_start = raw_end

    while trailing_start > leading_end and text[trailing_start - 1].isspace():
        trailing_start -= 1

    word_start = leading_end
    word_end = trailing_start

    word = text[word_start:word_end]

    return word_start, word_end, word

# VALIDATE TARGET POSITIONS

print()
print("=" * 90)
print("TARGET POSITION VALIDATION")
print("=" * 90)

validated_targets = []

for target in TARGETS:

    position = target["position"]
    expected_word = target["original"]

    print()
    print(
        f"Position {position}: "
        f"expected='{expected_word}'"
    )

    try:

        if position < 0 or position >= len(original_offsets):
            raise RuntimeError(
                "Position outside tokenizer range."
            )

        raw_start, raw_end = original_offsets[position]

        if raw_start == raw_end:
            raise RuntimeError(
                "Tokenizer returned empty span."
            )

        word_start, word_end, actual_word = (
            get_actual_word_span(
                original_text,
                raw_start,
                raw_end,
            )
        )

        print(
            "  Raw tokenizer span:",
            repr(original_text[raw_start:raw_end])
        )

        print(
            "  Actual word span:",
            repr(original_text[word_start:word_end])
        )

        print(
            "  Actual word:",
            repr(actual_word)
        )

        if actual_word != expected_word:
            raise RuntimeError(
                f"WORD MISMATCH: expected "
                f"'{expected_word}' but found "
                f"'{actual_word}'."
            )

        if not actual_word.isalpha():
            raise RuntimeError(
                "Target is not a complete alphabetic word."
            )

        # Actual word boundary check.
        if (
            word_start > 0
            and original_text[word_start - 1].isalpha()
        ):
            raise RuntimeError(
                "Actual word has an alphabetic character "
                "immediately before it."
            )

        if (
            word_end < len(original_text)
            and original_text[word_end].isalpha()
        ):
            raise RuntimeError(
                "Actual word has an alphabetic character "
                "immediately after it."
            )

        # Verify the word's surrounding context.
        before = (
            original_text[max(0, word_start - 35):word_start]
        )

        after = (
            original_text[word_end:min(
                len(original_text),
                word_end + 35
            )]
        )

        print(
            "  Context:",
            repr(before + actual_word + after)
        )

        validated_targets.append(
            {
                **target,
                "raw_start": raw_start,
                "raw_end": raw_end,
                "word_start": word_start,
                "word_end": word_end,
            }
        )

        print("  [✓ POSITION VALIDATED]")

    except Exception as e:

        print(
            "  [✗ TARGET VALIDATION ERROR]",
            repr(str(e))
        )

        raise RuntimeError(
            f"Target validation failed at "
            f"position {position}. "
            f"Stop rather than risk attacking "
            f"the wrong location."
        )


print()
print("[✓] ALL TARGET POSITIONS VALIDATED")


# ============================================================
# SUMMARY STORAGE
# ============================================================

results = []


def save_summary():

    """
    Save the aggregate after EVERY completed attempt.
    """

    with open(SUMMARY_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(
        f"[✓ SAVED SUMMARY] {SUMMARY_PATH}"
    )

# RUN EACH ATTACK INDEPENDENTLY

for target in validated_targets:

    position = target["position"]
    original_word = target["original"]

    word_start = target["word_start"]
    word_end = target["word_end"]

    context_start = max(
        0,
        word_start - 120
    )

    context_end = min(
        len(original_text),
        word_end + 120
    )

    for replacement in target["replacements"]:

        print()
        print("=" * 90)
        print(
            f"SAMPLE 2 | "
            f"position={position} | "
            f"{original_word} -> {replacement}"
        )
        print("=" * 90)

        output_name = (
            f"sample2_pos{position}_"
            f"{original_word}_to_"
            f"{replacement.replace(' ', '_')}.json"
        )

        output_path = os.path.join(
            OUTPUT_DIR,
            output_name
        )

        try:
            # Replacement tokenization
            replacement_ids = tokenizer.encode(
                " " + replacement,
                add_special_tokens=False,
            )

            replacement_token_count = len(
                replacement_ids
            )

            if replacement_token_count == 1:
                attack_class = (
                    "single_token_substitution"
                )
            else:
                attack_class = (
                    "multi_token_substitution"
                )

            print(
                "Replacement token IDs:",
                replacement_ids
            )

            print(
                "Replacement token count:",
                replacement_token_count
            )

            print(
                "Attack class:",
                attack_class
            )
            # Construct replacement while preserving whitespace

            raw_target_span = original_text[
                target["raw_start"]:
                target["raw_end"]
            ]

            leading_whitespace = (
                raw_target_span[
                    :len(raw_target_span)
                    - len(raw_target_span.lstrip())
                ]
            )

            trailing_whitespace = (
                raw_target_span[
                    len(raw_target_span.rstrip()):
                ]
            )

            replacement_span = (
                leading_whitespace
                + replacement
                + trailing_whitespace
            )

            attacked_text = (
                original_text[:target["raw_start"]]
                + replacement_span
                + original_text[target["raw_end"]:]
            )

            # Local construction validation

            before_local = original_text[
                context_start:context_end
            ]

            attacked_context = (
                attacked_text[
                    context_start:
                    min(
                        len(attacked_text),
                        context_start + len(before_local)
                        + len(replacement)
                        + 10
                    )
                ]
            )

            print()
            print("BEFORE LOCAL CONTEXT:")
            print(before_local)

            print()
            print("AFTER LOCAL CONTEXT:")
            print(attacked_context)

            # The actual character span must now contain
            # the replacement, including its preserved whitespace.
            actual_replacement_span = attacked_text[
                target["raw_start"]:
                target["raw_start"]
                + len(replacement_span)
            ]

            if actual_replacement_span != replacement_span:
                raise RuntimeError(
                    "Replacement construction validation failed."
                )

            # Detect attacked text
            attacked_detection = calculate_sta_z(
                attacked_text
            )

            attacked_z = attacked_detection["z"]

            delta_z = (
                attacked_z - stored_z
            )

            original_detected = (
                stored_z > Z_THRESHOLD
            )

            attacked_detected = (
                attacked_z > Z_THRESHOLD
            )

            watermark_broken = (
                original_detected
                and not attacked_detected
            )

            result = {
                "status": "completed",

                "sample": 2,

                "position": position,

                "original_word": original_word,

                "replacement": replacement,

                "attack_class": attack_class,

                "replacement_token_ids":
                    replacement_ids,

                "replacement_token_count":
                    replacement_token_count,

                "raw_tokenizer_span":
                    original_text[
                        target["raw_start"]:
                        target["raw_end"]
                    ],

                "word_start":
                    word_start,

                "word_end":
                    word_end,

                "original_z":
                    stored_z,

                "attacked_z":
                    attacked_z,

                "delta_z":
                    delta_z,

                "original_green_count":
                    baseline_detection[
                        "green_count"
                    ],

                "attacked_green_count":
                    attacked_detection[
                        "green_count"
                    ],

                "original_total_pairs":
                    baseline_detection[
                        "total_pairs"
                    ],

                "attacked_total_pairs":
                    attacked_detection[
                        "total_pairs"
                    ],

                "original_token_count":
                    baseline_detection[
                        "token_count"
                    ],

                "attacked_token_count":
                    attacked_detection[
                        "token_count"
                    ],

                "original_detected":
                    original_detected,

                "attacked_detected":
                    attacked_detected,

                "watermark_broken":
                    watermark_broken,

                "context":
                    original_text[
                        context_start:
                        context_end
                    ],

                "attacked_text":
                    attacked_text,
            }

            # SAVE THIS EXPERIMENT IMMEDIATELY

            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)

            print(
                f"[✓ SAVED] {output_path}"
            )

            # Add to summary and save immediately

            results.append(result)

            save_summary()

            # Print numerical result

            print()
            print("RESULT")
            print("-" * 50)

            print(
                "Original z:",
                f"{stored_z:.9f}"
            )

            print(
                "Attacked z:",
                f"{attacked_z:.9f}"
            )

            print(
                "Delta z:",
                f"{delta_z:.9f}"
            )

            print(
                "Green:",
                baseline_detection["green_count"],
                "->",
                attacked_detection["green_count"]
            )

            print(
                "Pairs:",
                baseline_detection["total_pairs"],
                "->",
                attacked_detection["total_pairs"]
            )

            print(
                "Detected:",
                original_detected,
                "->",
                attacked_detected
            )

            print(
                "WATERMARK BROKEN:",
                watermark_broken
            )

        except Exception as e:

            # ------------------------------------------------
            # IMPORTANT:
            # A single bad replacement MUST NOT stop the run.
            # ------------------------------------------------

            error_result = {
                "status": "error",

                "sample": 2,

                "position": position,

                "original_word": original_word,

                "replacement": replacement,

                "error_type": type(e).__name__,

                "error_message": str(e),

                "traceback":
                    traceback.format_exc(),

                "original_z":
                    stored_z,

                "context":
                    original_text[
                        context_start:
                        context_end
                    ],
            }

            with open(output_path, "w") as f:
                json.dump(
                    error_result,
                    f,
                    indent=2
                )

            print(
                f"[✓ SAVED ERROR] {output_path}"
            )

            results.append(error_result)

            save_summary()

            print()
            print(
                "[!] THIS REPLACEMENT FAILED."
            )
            print(
                "[!] CONTINUING TO NEXT REPLACEMENT."
            )

            continue

# FINAL REPORT
print()
print("=" * 90)
print("SAMPLE 2 — FINAL ATTACK REPORT")
print("=" * 90)

completed = [
    r for r in results
    if r["status"] == "completed"
]

errors = [
    r for r in results
    if r["status"] == "error"
]

breaks = [
    r for r in completed
    if r["watermark_broken"]
]

print()
print("Requested replacements:", len(
    [x for t in TARGETS for x in t["replacements"]]
))

print("Completed:", len(completed))
print("Errors:", len(errors))
print("Watermark breaks:", len(breaks))

print()
print("ALL COMPLETED RESULTS:")

for r in sorted(
    completed,
    key=lambda x: x["delta_z"]
):

    print(
        f"  {r['original_word']} -> "
        f"{r['replacement']} | "
        f"z={r['attacked_z']:.6f} | "
        f"Δz={r['delta_z']:.6f} | "
        f"class={r['attack_class']} | "
        f"broken={r['watermark_broken']}"
    )

print()
