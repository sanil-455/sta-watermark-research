import json
import os
import math
import itertools
import traceback

import torch
from transformers import AutoTokenizer


# CONFIG

MODEL_PATH = "hf_models/Llama-2-7b-hf"

BASELINE_PATH = "results/raw/baseline_safe.json"

GAMMA = 0.5
Z_THRESHOLD = 2.0

HASH_KEY1 = 15485863
HASH_KEY2 = 17624813

PROMPT_MAX_LENGTH = 2048


# COMBINED ATTACK DEFINITIONS
#
# IMPORTANT:
# These are the SAME replacements that were individually tested.
#
# We test every combination, but only one replacement per
# original word position.

SAMPLE2_ATTACKS = {
    515: {
        "original": "separation",
        "replacements": [
            "severance",
            "termination",
        ],
    },

    610: {
        "original": "growth",
        "replacements": [
            "demand",
            "expansion",
        ],
    },

    292: {
        "original": "capacity",
        "replacements": [
            "output",
            "volume",
        ],
    },

    377: {
        "original": "releases",
        "replacements": [
            "disengages",
            "eases off",
        ],
    },

    713: {
        "original": "making",
        "replacements": [
            "achieving",
            "delivering",
            "driving",
        ],
    },

    714: {
        "original": "some",
        "replacements": [
            "steady",
            "significant",
        ],
    },
}


SAMPLE3_ATTACKS = {
    102: {
        "original": "interest",
        "replacements": [
            "passion",
            "fascination",
            "appreciation",
        ],
    },

    213: {
        "original": "away",
        "replacements": [
            "far",
        ],
    },

    374: {
        "original": "times",
        "replacements": [
            "moments",
        ],
    },

    381: {
        "original": "stretch",
        "replacements": [
            "strain",
            "leap",
            "chore",
        ],
    },

    502: {
        "original": "reaches",
        "replacements": [
            "attains",
            "hits",
        ],
    },

    66: {
        "original": "However",
        "replacements": [
            "Nevertheless",
            "Nonetheless",
        ],
    },
}


# OUTPUT DIRECTORIES

OUT_ROOT = "results/raw/manual_attacks"

S2_OUT = os.path.join(
    OUT_ROOT,
    "sample2_combined"
)

S3_OUT = os.path.join(
    OUT_ROOT,
    "sample3_combined"
)

os.makedirs(S2_OUT, exist_ok=True)
os.makedirs(S3_OUT, exist_ok=True)


# LOAD TOKENIZER

print()
print("=" * 90)
print("LOADING TOKENIZER")
print("=" * 90)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    use_fast=True,
)

print("[✓] TOKENIZER LOADED")


# LOAD BASELINE

with open(BASELINE_PATH) as f:
    baseline = json.load(f)


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
            "Text has fewer than two tokens."
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


# LOAD SAVED INDIVIDUAL RESULTS

def load_individual_results(sample_number):

    if sample_number == 2:
        directory = (
            "results/raw/manual_attacks/"
            "sample2_replacements"
        )
    else:
        directory = (
            "results/raw/manual_attacks/"
            "sample3_replacements"
        )

    if not os.path.isdir(directory):
        raise RuntimeError(
            f"Individual results directory does not exist:\n"
            f"{directory}\n\n"
            f"Run the corresponding individual attack script first."
        )

    results = []

    for filename in os.listdir(directory):

        if not filename.endswith(".json"):
            continue

        if filename.endswith("_summary.json"):
            continue

        path = os.path.join(
            directory,
            filename
        )

        try:

            with open(path) as f:
                result = json.load(f)

            if result.get("status") == "completed":
                results.append(result)

        except Exception:
            pass

    return results


# FIND STRONGEST INDIVIDUAL ATTACK

def report_best_individual(sample_number):

    results = load_individual_results(sample_number)

    print()
    print("=" * 90)
    print(
        f"SAMPLE {sample_number} — "
        "INDIVIDUAL ATTACK ANALYSIS"
    )
    print("=" * 90)

    print(
        "Completed individual attacks:",
        len(results)
    )

    if not results:
        raise RuntimeError(
            f"No completed individual results found for Sample {sample_number}."
        )

    ordered = sorted(
        results,
        key=lambda x: x["delta_z"]
    )

    print()
    print("STRONGEST INDIVIDUAL ATTACKS")
    print("-" * 70)

    for r in ordered[:10]:

        print(
            f"{r['original_word']} -> "
            f"{r['replacement']} | "
            f"z={r['attacked_z']:.9f} | "
            f"Δz={r['delta_z']:.9f} | "
            f"broken={r['watermark_broken']}"
        )

    best = ordered[0]

    print()
    print("BEST SINGLE ATTACK")
    print("-" * 70)

    print(
        f"{best['original_word']} -> "
        f"{best['replacement']}"
    )

    print(
        "Original z:",
        best["original_z"]
    )

    print(
        "Attacked z:",
        best["attacked_z"]
    )

    print(
        "Delta z:",
        best["delta_z"]
    )

    print(
        "Watermark broken:",
        best["watermark_broken"]
    )

    # Save best attack.
    best_path = os.path.join(
        OUT_ROOT,
        f"sample{sample_number}_best_single_attack.json"
    )

    with open(best_path, "w") as f:
        json.dump(best, f, indent=2)

    print(
        f"[✓ SAVED] {best_path}"
    )

    return results


# BUILD ORIGINAL TOKEN SPANS

def get_target_spans(text, attack_definition):

    encoded = tokenizer(
        text,
        add_special_tokens=True,
        return_offsets_mapping=True,
        truncation=True,
        max_length=PROMPT_MAX_LENGTH,
    )

    offsets = encoded["offset_mapping"]

    validated = {}

    for position, target in attack_definition.items():

        if position < 0 or position >= len(offsets):
            raise RuntimeError(
                f"Position {position} outside tokenizer range."
            )

        raw_start, raw_end = offsets[position]

        if raw_start == raw_end:
            raise RuntimeError(
                f"Position {position} has empty span."
            )

        raw_span = text[
            raw_start:raw_end
        ]

        # Remove leading/trailing whitespace from
        # tokenizer's raw span.
        word_start = raw_start

        while (
            word_start < raw_end
            and text[word_start].isspace()
        ):
            word_start += 1

        word_end = raw_end

        while (
            word_end > word_start
            and text[word_end - 1].isspace()
        ):
            word_end -= 1

        actual_word = text[
            word_start:word_end
        ]

        expected_word = target["original"]

        if actual_word != expected_word:
            raise RuntimeError(
                f"POSITION MISMATCH:\n"
                f"position={position}\n"
                f"expected={expected_word!r}\n"
                f"actual={actual_word!r}"
            )

        if not actual_word.isalpha():
            raise RuntimeError(
                f"Target at position {position} "
                f"is not an alphabetic complete word."
            )

        # Check actual character boundaries,
        # NOT the tokenizer's whitespace-inclusive offset.
        if (
            word_start > 0
            and text[word_start - 1].isalpha()
        ):
            raise RuntimeError(
                f"Position {position} is actually "
                f"inside a larger alphabetic string."
            )

        if (
            word_end < len(text)
            and text[word_end].isalpha()
        ):
            raise RuntimeError(
                f"Position {position} is actually "
                f"inside a larger alphabetic string."
            )

        leading_ws = text[
            raw_start:word_start
        ]

        trailing_ws = text[
            word_end:raw_end
        ]

        validated[position] = {
            "original": expected_word,
            "raw_start": raw_start,
            "raw_end": raw_end,
            "word_start": word_start,
            "word_end": word_end,
            "leading_ws": leading_ws,
            "trailing_ws": trailing_ws,
        }

    return validated


# BUILD COMBINED TEXT

def construct_combined_text(
    original_text,
    spans,
    chosen_replacements
):

    edits = []

    for position, replacement in chosen_replacements.items():

        span = spans[position]

        replacement_span = (
            span["leading_ws"]
            + replacement
            + span["trailing_ws"]
        )

        edits.append(
            (
                span["raw_start"],
                span["raw_end"],
                replacement_span,
            )
        )

    # CRITICAL:
    #
    # Apply from RIGHT to LEFT so original character offsets
    # remain valid even when replacement lengths differ.
    #
    edits.sort(
        key=lambda x: x[0],
        reverse=True
    )

    text = original_text

    for start, end, replacement_span in edits:

        text = (
            text[:start]
            + replacement_span
            + text[end:]
        )

    return text


# GENERATE COMBINATIONS

def generate_combinations(attack_definition):

    positions = list(
        attack_definition.keys()
    )

    replacement_lists = [
        attack_definition[p]["replacements"]
        for p in positions
    ]

    for choices in itertools.product(
        *replacement_lists
    ):

        yield {
            position: replacement
            for position, replacement
            in zip(positions, choices)
        }


# COMBINED ATTACK ENGINE

def run_combined_attacks(
    sample_number,
    attack_definition,
    individual_results,
    output_directory
):

    baseline_sample = baseline[
        sample_number - 1
    ]

    original_text = (
        baseline_sample["watermarked_text"]
    )

    original_z = float(
        baseline_sample["watermarked_z"]
    )

    # --------------------------------------------------------
    # First validate original baseline.
    # --------------------------------------------------------

    baseline_detection = calculate_sta_z(
        original_text
    )

    if abs(
        baseline_detection["z"]
        - original_z
    ) > 1e-6:

        raise RuntimeError(
            f"Sample {sample_number} baseline z mismatch."
        )

    # --------------------------------------------------------
    # Validate target positions.
    # --------------------------------------------------------

    spans = get_target_spans(
        original_text,
        attack_definition
    )

    # --------------------------------------------------------
    # Determine which replacements individually produced
    # actual completed results.
    #
    # We DO NOT require individual watermark breaking.
    # A valid replacement remains eligible.
    # --------------------------------------------------------

    completed_pairs = set()

    for r in individual_results:

        completed_pairs.add(
            (
                r["position"],
                r["replacement"]
            )
        )

    # --------------------------------------------------------
    # Only use replacements that actually completed.
    # --------------------------------------------------------

    usable_definition = {}

    for position, target in attack_definition.items():

        usable = []

        for replacement in target["replacements"]:

            if (
                position,
                replacement
            ) in completed_pairs:

                usable.append(replacement)

        usable_definition[position] = {
            "original": target["original"],
            "replacements": usable,
        }

    # --------------------------------------------------------
    # If one position has no usable replacement, we cannot
    # form a full all-position combination.
    # --------------------------------------------------------

    missing = [
        position
        for position, target
        in usable_definition.items()
        if not target["replacements"]
    ]

    if missing:

        print()
        print(
            "[!] Cannot perform full combined attack."
        )

        print(
            "No completed replacement for positions:",
            missing
        )

        return []

    total_possible = 1

    for target in usable_definition.values():

        total_possible *= len(
            target["replacements"]
        )

    print()
    print("=" * 90)
    print(
        f"SAMPLE {sample_number} — "
        "COMBINED ATTACK"
    )
    print("=" * 90)

    print(
        "Usable positions:",
        len(usable_definition)
    )

    print(
        "Total combinations:",
        total_possible
    )

    print(
        "Original z:",
        original_z
    )

    print()

    combinations = generate_combinations(
        usable_definition
    )

    results = []

    for index, chosen in enumerate(
        combinations,
        1
    ):

        print()
        print(
            f"[COMBINATION {index}/{total_possible}]"
        )

        print(
            "Choices:",
            " | ".join(
                f"{p}:{r}"
                for p, r in chosen.items()
            )
        )

        filename = (
            f"sample{sample_number}_"
            f"combo_{index:04d}.json"
        )

        output_path = os.path.join(
            output_directory,
            filename
        )

        # ----------------------------------------------------
        # If already saved, load it instead of rerunning.
        # This makes the experiment crash-resumable.
        # ----------------------------------------------------

        if os.path.exists(output_path):

            try:

                with open(output_path) as f:
                    saved = json.load(f)

                results.append(saved)

                print(
                    f"[✓ ALREADY SAVED] "
                    f"{output_path}"
                )

                continue

            except Exception:
                pass

        try:

            attacked_text = construct_combined_text(
                original_text,
                spans,
                chosen
            )

            attacked_detection = calculate_sta_z(
                attacked_text
            )

            attacked_z = (
                attacked_detection["z"]
            )

            delta_z = (
                attacked_z - original_z
            )

            original_detected = (
                original_z > Z_THRESHOLD
            )

            attacked_detected = (
                attacked_z > Z_THRESHOLD
            )

            broken = (
                original_detected
                and not attacked_detected
            )

            result = {
                "status": "completed",

                "sample":
                    sample_number,

                "combination_index":
                    index,

                "chosen_replacements":
                    chosen,

                "original_z":
                    original_z,

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
                    broken,

                "attacked_text":
                    attacked_text,
            }

            with open(output_path, "w") as f:
                json.dump(
                    result,
                    f,
                    indent=2
                )

            print(
                f"[✓ SAVED] {output_path}"
            )

            results.append(result)

            print(
                f"z: {original_z:.6f} "
                f"-> {attacked_z:.6f} | "
                f"Δz={delta_z:.6f} | "
                f"broken={broken}"
            )

        except Exception as e:

            error_result = {
                "status": "error",

                "sample":
                    sample_number,

                "combination_index":
                    index,

                "chosen_replacements":
                    chosen,

                "error_type":
                    type(e).__name__,

                "error_message":
                    str(e),

                "traceback":
                    traceback.format_exc(),

                "original_z":
                    original_z,
            }

            with open(output_path, "w") as f:
                json.dump(
                    error_result,
                    f,
                    indent=2
                )

            print(
                f"[✓ SAVED ERROR] "
                f"{output_path}"
            )

            results.append(
                error_result
            )

            print(
                "[!] Combination failed."
            )

            print(
                "[!] Continuing."
            )

            continue

    # SAVE MASTER COMBINED SUMMARY

    summary_path = os.path.join(
        output_directory,
        f"sample{sample_number}_"
        "combined_summary.json"
    )

    with open(summary_path, "w") as f:
        json.dump(
            results,
            f,
            indent=2
        )

    print()
    print(
        f"[✓ SAVED] {summary_path}"
    )

    return results


# FINAL ANALYSIS

def final_analysis(
    sample_number,
    combined_results
):

    completed = [
        r
        for r in combined_results
        if r.get("status") == "completed"
    ]

    errors = [
        r
        for r in combined_results
        if r.get("status") == "error"
    ]

    if not completed:
        print(
            f"No completed combined attacks "
            f"for Sample {sample_number}."
        )
        return

    ordered = sorted(
        completed,
        key=lambda x: x["delta_z"]
    )

    best = ordered[0]

    breaks = [
        r
        for r in completed
        if r["watermark_broken"]
    ]

    report = {
        "sample": sample_number,

        "total_completed":
            len(completed),

        "total_errors":
            len(errors),

        "watermark_breaks":
            len(breaks),

        "best_combined_attack":
            best,

        "top_10_combined_attacks":
            ordered[:10],
    }

    path = os.path.join(
        OUT_ROOT,
        f"sample{sample_number}_"
        "combined_best_and_summary.json"
    )

    with open(path, "w") as f:
        json.dump(
            report,
            f,
            indent=2
        )

    print()
    print("=" * 90)
    print(
        f"SAMPLE {sample_number} — "
        "COMBINED ATTACK RESULT"
    )
    print("=" * 90)

    print(
        "Completed combinations:",
        len(completed)
    )

    print(
        "Errors:",
        len(errors)
    )

    print(
        "Watermark breaks:",
        len(breaks)
    )

    print()
    print("BEST COMBINED ATTACK:")
    print(
        best["chosen_replacements"]
    )

    print(
        "Original z:",
        best["original_z"]
    )

    print(
        "Attacked z:",
        best["attacked_z"]
    )

    print(
        "Delta z:",
        best["delta_z"]
    )

    print(
        "Watermark broken:",
        best["watermark_broken"]
    )

    print()
    print("TOP 10:")
    print("-" * 70)

    for r in ordered[:10]:

        print(
            f"Δz={r['delta_z']:.6f} | "
            f"z={r['attacked_z']:.6f} | "
            f"broken={r['watermark_broken']} | "
            f"{r['chosen_replacements']}"
        )

    print()
    print(
        f"[✓ SAVED] {path}"
    )


# RUN SAMPLE 2

s2_individual = report_best_individual(2)

s2_combined = run_combined_attacks(
    sample_number=2,
    attack_definition=SAMPLE2_ATTACKS,
    individual_results=s2_individual,
    output_directory=S2_OUT,
)

final_analysis(
    2,
    s2_combined
)


# RUN SAMPLE 3

s3_individual = report_best_individual(3)

s3_combined = run_combined_attacks(
    sample_number=3,
    attack_definition=SAMPLE3_ATTACKS,
    individual_results=s3_individual,
    output_directory=S3_OUT,
)

final_analysis(
    3,
    s3_combined
)


# MASTER CHECKPOINT DATA

master = {
    "purpose":
        "Adaptive meaningful-word substitution attack "
        "analysis and combined attacks.",

    "sample2_individual_results":
        "results/raw/manual_attacks/"
        "sample2_replacements/"
        "sample2_replacement_summary.json",

    "sample3_individual_results":
        "results/raw/manual_attacks/"
        "sample3_replacements/"
        "sample3_replacement_summary.json",

    "sample2_best_single":
        "results/raw/manual_attacks/"
        "sample2_best_single_attack.json",

    "sample3_best_single":
        "results/raw/manual_attacks/"
        "sample3_best_single_attack.json",

    "sample2_combined":
        S2_OUT,

    "sample3_combined":
        S3_OUT,

    "sample2_combined_summary":
        os.path.join(
            S2_OUT,
            "sample2_combined_summary.json"
        ),

    "sample3_combined_summary":
        os.path.join(
            S3_OUT,
            "sample3_combined_summary.json"
        ),
}

master_path = os.path.join(
    OUT_ROOT,
    "samples2_3_adaptive_substitution_checkpoint_data.json"
)

with open(master_path, "w") as f:
    json.dump(
        master,
        f,
        indent=2
    )

print()
print("=" * 90)
print(
    "[✓ SAVED] "
    + master_path
)
print("=" * 90)

print()
print("ALL ANALYSIS COMPLETE.")
