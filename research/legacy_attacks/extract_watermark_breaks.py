import json
import os


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = "results/raw/manual_attacks"

BASELINE_PATH = "results/raw/baseline_safe.json"

SAMPLE2_DIR = os.path.join(
    ROOT,
    "sample2_replacements"
)

SAMPLE3_DIR = os.path.join(
    ROOT,
    "sample3_replacements"
)

BREAKS_ROOT = os.path.join(
    ROOT,
    "BREAKS"
)

SAMPLE2_BREAKS = os.path.join(
    BREAKS_ROOT,
    "SAMPLE_2"
)

SAMPLE3_BREAKS = os.path.join(
    BREAKS_ROOT,
    "SAMPLE_3"
)

ALL_RESULTS_PATH = os.path.join(
    BREAKS_ROOT,
    "ALL_INDIVIDUAL_ATTACK_RESULTS.json"
)

ALL_BREAKS_PATH = os.path.join(
    BREAKS_ROOT,
    "ALL_WATERMARK_BREAKS.json"
)

NON_BREAKS_PATH = os.path.join(
    BREAKS_ROOT,
    "ALL_VALID_NON_BREAKING_ATTACKS.json"
)

ERRORS_PATH = os.path.join(
    BREAKS_ROOT,
    "ALL_ATTACK_ERRORS.json"
)

CHECKPOINT_PATH = os.path.join(
    BREAKS_ROOT,
    "BREAK_ANALYSIS_CHECKPOINT.json"
)

Z_THRESHOLD = 2.0


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(
    SAMPLE2_BREAKS,
    exist_ok=True
)

os.makedirs(
    SAMPLE3_BREAKS,
    exist_ok=True
)


# ============================================================
# LOAD BASELINE
# ============================================================

print()
print("=" * 100)
print("LOADING BASELINE")
print("=" * 100)

with open(
    BASELINE_PATH,
    "r"
) as f:

    baseline = json.load(f)

print(
    "[✓] Baseline loaded:",
    len(baseline),
    "samples"
)


# ============================================================
# VERIFY BASELINE STATES
# ============================================================

sample2_baseline_z = float(
    baseline[1]["watermarked_z"]
)

sample3_baseline_z = float(
    baseline[2]["watermarked_z"]
)

sample2_baseline_detected = (
    sample2_baseline_z > Z_THRESHOLD
)

sample3_baseline_detected = (
    sample3_baseline_z > Z_THRESHOLD
)

print()
print("Sample 2 baseline:")
print(
    "  z =",
    sample2_baseline_z
)
print(
    "  detected =",
    sample2_baseline_detected
)

print()
print("Sample 3 baseline:")
print(
    "  z =",
    sample3_baseline_z
)
print(
    "  detected =",
    sample3_baseline_detected
)


# ============================================================
# LOAD INDIVIDUAL RESULTS
# ============================================================

def load_results(directory, sample_number):

    results = []

    if not os.path.isdir(directory):

        print()
        print(
            "[!] Missing directory:",
            directory
        )

        return results

    for filename in sorted(
        os.listdir(directory)
    ):

        if not filename.endswith(".json"):
            continue

        if filename.endswith("_summary.json"):
            continue

        path = os.path.join(
            directory,
            filename
        )

        try:

            with open(
                path,
                "r"
            ) as f:

                result = json.load(f)

            # Ensure this is an individual attack result.
            if "position" not in result:
                continue

            result["_source_file"] = path
            result["_sample"] = sample_number

            results.append(result)

        except Exception as e:

            print(
                "[!] Could not read:",
                path
            )

            print(
                "    Error:",
                repr(e)
            )

    return results


print()
print("=" * 100)
print("LOADING SAVED INDIVIDUAL ATTACK RESULTS")
print("=" * 100)

sample2_results = load_results(
    SAMPLE2_DIR,
    2
)

print(
    "[✓] Sample 2:",
    len(sample2_results),
    "result files"
)

sample3_results = load_results(
    SAMPLE3_DIR,
    3
)

print(
    "[✓] Sample 3:",
    len(sample3_results),
    "result files"
)

all_results = (
    sample2_results
    + sample3_results
)


# ============================================================
# CLASSIFY
# ============================================================

successful_breaks = []
valid_non_breaks = []
errors = []


for result in all_results:

    if result.get("status") == "error":

        errors.append(result)

        continue

    original_detected = result.get(
        "original_detected"
    )

    attacked_detected = result.get(
        "attacked_detected"
    )

    # --------------------------------------------------------
    # ONLY TRUE -> FALSE IS A WATERMARK BREAK.
    # --------------------------------------------------------

    if (
        original_detected is True
        and attacked_detected is False
    ):

        successful_breaks.append(
            result
        )

    else:

        valid_non_breaks.append(
            result
        )


# ============================================================
# SAVE COMPLETE ARCHIVE
# ============================================================

complete_archive = {
    "description":
        "Complete archive of individual targeted "
        "word-substitution attacks on Samples 2 and 3.",

    "break_definition":
        "original_detected=True AND "
        "attacked_detected=False",

    "sample2_results":
        sample2_results,

    "sample3_results":
        sample3_results,

    "successful_breaks":
        successful_breaks,

    "valid_non_breaks":
        valid_non_breaks,

    "errors":
        errors,
}

with open(
    ALL_RESULTS_PATH,
    "w"
) as f:

    json.dump(
        complete_archive,
        f,
        indent=2
    )

print()
print(
    "[✓ SAVED]",
    ALL_RESULTS_PATH
)


# ============================================================
# SAVE BREAKS ONLY
# ============================================================

for result in successful_breaks:

    sample = result["_sample"]

    position = result["position"]

    original = result[
        "original_word"
    ]

    replacement = result[
        "replacement"
    ]

    safe_replacement = (
        replacement
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )

    filename = (
        f"SUCCESSFUL_BREAK_"
        f"sample{sample}_"
        f"pos{position}_"
        f"{original}_to_"
        f"{safe_replacement}.json"
    )

    if sample == 2:

        output_path = os.path.join(
            SAMPLE2_BREAKS,
            filename
        )

    else:

        output_path = os.path.join(
            SAMPLE3_BREAKS,
            filename
        )

    with open(
        output_path,
        "w"
    ) as f:

        json.dump(
            result,
            f,
            indent=2
        )

    print(
        "[✓ SAVED BREAK]",
        output_path
    )


# ============================================================
# SAVE MASTER BREAK FILE
# ============================================================

break_archive = {
    "definition":
        "original_detected=True AND "
        "attacked_detected=False",

    "number_of_breaks":
        len(successful_breaks),

    "sample2_breaks": [
        r
        for r in successful_breaks
        if r["_sample"] == 2
    ],

    "sample3_breaks": [
        r
        for r in successful_breaks
        if r["_sample"] == 3
    ],
}

with open(
    ALL_BREAKS_PATH,
    "w"
) as f:

    json.dump(
        break_archive,
        f,
        indent=2
    )

print()
print(
    "[✓ SAVED]",
    ALL_BREAKS_PATH
)


# ============================================================
# SAVE VALID NON-BREAKING ATTACKS
# ============================================================

with open(
    NON_BREAKS_PATH,
    "w"
) as f:

    json.dump(
        {
            "description":
                "Valid individual attacks that did not "
                "change detection from True to False.",

            "number":
                len(valid_non_breaks),

            "results":
                valid_non_breaks,
        },
        f,
        indent=2
    )

print()
print(
    "[✓ SAVED]",
    NON_BREAKS_PATH
)


# ============================================================
# SAVE ERRORS
# ============================================================

with open(
    ERRORS_PATH,
    "w"
) as f:

    json.dump(
        {
            "number":
                len(errors),

            "results":
                errors,
        },
        f,
        indent=2
    )

print()
print(
    "[✓ SAVED]",
    ERRORS_PATH
)


# ============================================================
# CLEAR REPORT
# ============================================================

print()
print()
print("#" * 100)
print("#")
print("#                    INDIVIDUAL ATTACK RESULTS")
print("#")
print("#" * 100)

print()
print(
    "Total result files:",
    len(all_results)
)

print(
    "Sample 2:",
    len(sample2_results)
)

print(
    "Sample 3:",
    len(sample3_results)
)

print(
    "Valid non-breaking attacks:",
    len(valid_non_breaks)
)

print(
    "Errors:",
    len(errors)
)

print(
    "SUCCESSFUL WATERMARK BREAKS:",
    len(successful_breaks)
)


# ============================================================
# SUCCESSFUL BREAKS
# ============================================================

print()
print()
print("#" * 100)
print("#")
print("#              >>> SUCCESSFUL WATERMARK BREAKS <<<")
print("#")
print("#" * 100)

if not successful_breaks:

    print()
    print(
        "NONE."
    )

    print(
        "No individual attack changed "
        "True -> False."
    )

else:

    for i, result in enumerate(
        successful_breaks,
        1
    ):

        print()
        print(
            "=" * 100
        )

        print(
            f"SUCCESSFUL BREAK #{i}"
        )

        print(
            "Sample:",
            result["_sample"]
        )

        print(
            "Position:",
            result["position"]
        )

        print(
            "Original:",
            result["original_word"]
        )

        print(
            "Replacement:",
            result["replacement"]
        )

        print(
            "Original z:",
            result["original_z"]
        )

        print(
            "Attacked z:",
            result["attacked_z"]
        )

        print(
            "Delta z:",
            result["delta_z"]
        )

        print(
            "Detection:",
            result["original_detected"],
            "->",
            result["attacked_detected"]
        )


# ============================================================
# BEST INDIVIDUAL ATTACKS
# ============================================================

print()
print()
print("#" * 100)
print("#")
print("#                 STRONGEST INDIVIDUAL ATTACKS")
print("#")
print("#" * 100)

if valid_non_breaks or successful_breaks:

    ordered = sorted(
        valid_non_breaks + successful_breaks,
        key=lambda r: r.get(
            "delta_z",
            float("inf")
        )
    )

    for i, result in enumerate(
        ordered[:15],
        1
    ):

        print(
            f"{i:2d}. "
            f"Sample {result['_sample']} | "
            f"{result['original_word']} -> "
            f"{result['replacement']} | "
            f"z={result['attacked_z']:.9f} | "
            f"Δz={result['delta_z']:.9f} | "
            f"True->False="
            f"{result['watermark_broken']}"
        )

else:

    print(
        "No valid individual results."
    )


# ============================================================
# ERRORS
# ============================================================

print()
print()
print("#" * 100)
print("#")
print("#                         ERRORS")
print("#")
print("#" * 100)

if not errors:

    print()
    print(
        "NONE."
    )

else:

    for result in errors:

        print()
        print(
            f"Sample {result['_sample']} | "
            f"position={result.get('position')} | "
            f"{result.get('original_word')} -> "
            f"{result.get('replacement')}"
        )

        print(
            "Error:",
            result.get("error_type")
        )

        print(
            "Message:",
            result.get("error_message")
        )


# ============================================================
# CHECKPOINT
# ============================================================

checkpoint = {

    "experiment":
        "Adaptive meaningful-word single-substitution attack",

    "samples":
        [2, 3],

    "threshold":
        Z_THRESHOLD,

    "sample2_baseline":
        {
            "z":
                sample2_baseline_z,

            "detected":
                sample2_baseline_detected,
        },

    "sample3_baseline":
        {
            "z":
                sample3_baseline_z,

            "detected":
                sample3_baseline_detected,
        },

    "total_individual_results":
        len(all_results),

    "sample2_results":
        len(sample2_results),

    "sample3_results":
        len(sample3_results),

    "valid_non_breaking_attacks":
        len(valid_non_breaks),

    "errors":
        len(errors),

    "successful_watermark_breaks":
        len(successful_breaks),

    "sample2_breaks":
        len([
            r
            for r in successful_breaks
            if r["_sample"] == 2
        ]),

    "sample3_breaks":
        len([
            r
            for r in successful_breaks
            if r["_sample"] == 3
        ]),

    "break_definition":
        "original_detected=True AND "
        "attacked_detected=False",

    "all_results_file":
        ALL_RESULTS_PATH,

    "break_results_file":
        ALL_BREAKS_PATH,

    "non_break_results_file":
        NON_BREAKS_PATH,

    "error_results_file":
        ERRORS_PATH,
}


with open(
    CHECKPOINT_PATH,
    "w"
) as f:

    json.dump(
        checkpoint,
        f,
        indent=2
    )

print()
print()
print("=" * 100)
print(
    "[✓ SAVED CHECKPOINT]",
    CHECKPOINT_PATH
)
print("=" * 100)

print()
print(
    "FINAL NUMBER OF INDIVIDUAL WATERMARK BREAKS:",
    len(successful_breaks)
)

print()
print(
    "[✓ EXTRACTION COMPLETE]"
)
