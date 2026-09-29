import os
import os
import json
import math
import os
import torch
from transformers import AutoTokenizer


# ============================================================
# 1. EXPERIMENT SETTINGS
# ============================================================

MODEL_PATH = "hf_models/Llama-2-7b-hf"
DATA_PATH = "results/raw/baseline_safe.json"

OUTPUT_PATH = "results/raw/attack_targeted_single_token.json"

HASH_KEY1 = 15485863
HASH_KEY2 = 17624813

GAMMA = 0.5
Z_THRESHOLD = 2.0
MAX_LENGTH = 2048


# ============================================================
# 2. LOAD DATA AND TOKENIZER
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required by the official STA RNG.")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

with open(DATA_PATH, "r") as f:
    data = json.load(f)

rng = torch.Generator(device="cuda")


# ============================================================
# 3. EXACT STA GREEN/RED DECISION
# ============================================================

def is_green(token_a, token_b):
    """
    Reproduce STA's pairwise green/red decision.

    STA seeds the RNG from the two adjacent token IDs and
    draws one random number. A value below gamma is green.
    """

    seed = (
        HASH_KEY1 * int(token_a)
        + HASH_KEY2 * int(token_b)
    )

    rng.manual_seed(seed)

    value = torch.rand(
        1,
        device="cuda",
        generator=rng
    ).item()

    return value < GAMMA


# ============================================================
# 4. STA DETECTOR
# ============================================================

def detect_token_ids(token_ids):
    """
    Calculate STA green-token count and z-score directly
    from a token-ID sequence.
    """

    if len(token_ids) < 2:
        return 0, 0, 0.0

    green_count = 0

    for i in range(len(token_ids) - 1):
        if is_green(token_ids[i], token_ids[i + 1]):
            green_count += 1

    pair_count = len(token_ids) - 1

    expected = GAMMA * pair_count

    variance = GAMMA * (1.0 - GAMMA) * pair_count

    z = (green_count - expected) / math.sqrt(variance)

    return green_count, pair_count, z


def detect_text(text):
    """
    Run the detector on actual decoded text, exactly using
    the Llama tokenizer.
    """

    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )

    token_ids = encoded["input_ids"]

    green_count, pair_count, z = detect_token_ids(token_ids)

    return {
        "token_ids": token_ids,
        "green_count": green_count,
        "pair_count": pair_count,
        "z": z,
    }


# ============================================================
# 5. FAST LOCAL EFFECT OF ONE TOKEN REPLACEMENT
# ============================================================

def best_single_replacement(original_ids, donor_ids):
    """
    Exhaustively search every position and every donor token.

    Only two adjacent STA pair decisions can change when
    one token is replaced.

    Therefore we can evaluate every candidate without
    reconstructing and decoding every possible text.
    """

    n = len(original_ids)

    if n < 3:
        return None

    # Calculate the original pairwise green indicators once.
    original_green = []

    for i in range(n - 1):
        original_green.append(
            1 if is_green(original_ids[i], original_ids[i + 1])
            else 0
        )

    original_green_count = sum(original_green)
    pair_count = n - 1

    # Remove duplicates and special tokens from donor vocabulary.
    donor_set = set(int(x) for x in donor_ids)

    special_ids = set(
        int(x)
        for x in tokenizer.all_special_ids
    )

    donor_set -= special_ids

    best = None

    for position in range(1, n - 1):

        old_token = int(original_ids[position])

        for new_token in donor_set:

            # Replacing a token by itself is not an attack.
            if new_token == old_token:
                continue

            # Original contributions:
            # (position-1, position)
            # (position, position+1)

            old_left = original_green[position - 1]
            old_right = original_green[position]

            # New contributions after replacing the token.
            new_left = (
                1
                if is_green(
                    original_ids[position - 1],
                    new_token
                )
                else 0
            )

            new_right = (
                1
                if is_green(
                    new_token,
                    original_ids[position + 1]
                )
                else 0
            )

            new_green_count = (
                original_green_count
                - old_left
                - old_right
                + new_left
                + new_right
            )

            expected = GAMMA * pair_count

            z = (
                new_green_count - expected
            ) / math.sqrt(
                GAMMA * (1.0 - GAMMA) * pair_count
            )

            reduction = (
                (original_green_count - expected)
                / math.sqrt(
                    GAMMA * (1.0 - GAMMA) * pair_count
                )
                - z
            )

            candidate = {
                "position": position,
                "old_token": old_token,
                "new_token": new_token,
                "old_left": old_left,
                "old_right": old_right,
                "new_left": new_left,
                "new_right": new_right,
                "new_green_count": new_green_count,
                "predicted_z": z,
                "predicted_z_reduction": reduction,
            }

            if (
                best is None
                or candidate["predicted_z"] < best["predicted_z"]
            ):
                best = candidate

    return best


# ============================================================
# 6. MAIN EXPERIMENT
# ============================================================

results = []

print()
print("=" * 70)
print("STA TARGETED SINGLE-TOKEN ATTACK")
print("=" * 70)
print(f"Samples: {len(data)}")
print(f"Gamma: {GAMMA}")
print(f"Detection threshold: z > {Z_THRESHOLD}")
print()


for sample_index, sample in enumerate(data):

    print("-" * 70)
    print(f"SAMPLE {sample_index + 1}/{len(data)}")
    print("-" * 70)

    watermarked_text = sample["watermarked_text"]
    unwatermarked_text = sample["unwatermarked_text"]

    # --------------------------------------------------------
    # Original watermarked detector result
    # --------------------------------------------------------

    original = detect_text(watermarked_text)

    stored_z = float(sample["watermarked_z"])

    verification_error = abs(
        original["z"] - stored_z
    )

    print(f"Stored STA z       : {stored_z:.6f}")
    print(f"Recomputed STA z   : {original['z']:.6f}")
    print(f"Verification error  : {verification_error:.10f}")

    if verification_error > 1e-6:
        raise RuntimeError(
            f"STA detector verification failed on sample "
            f"{sample_index}. "
            f"Stored z={stored_z}, "
            f"recomputed z={original['z']}"
        )

    # --------------------------------------------------------
    # Donor tokens come from the unwatermarked text
    # --------------------------------------------------------

    donor_encoding = tokenizer(
        unwatermarked_text,
        add_special_tokens=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )

    donor_ids = donor_encoding["input_ids"]

    print(f"Watermarked tokens : {len(original['token_ids'])}")
    print(f"Donor token pool   : {len(set(donor_ids))}")

    # --------------------------------------------------------
    # Exhaustive search
    # --------------------------------------------------------

    best = best_single_replacement(
        original["token_ids"],
        donor_ids,
    )

    if best is None:
        raise RuntimeError(
            f"No valid one-token replacement found "
            f"for sample {sample_index}."
        )

    print()
    print("BEST PREDICTED ATTACK")
    print(f"Position           : {best['position']}")
    print(f"Old token ID       : {best['old_token']}")
    print(f"New token ID       : {best['new_token']}")
    print(f"Predicted z        : {best['predicted_z']:.6f}")
    print(
        f"Predicted Δz       : "
        f"{best['predicted_z_reduction']:.6f}"
    )

    # --------------------------------------------------------
    # Actually modify the token sequence
    # --------------------------------------------------------

    attacked_ids = list(original["token_ids"])

    attacked_ids[best["position"]] = best["new_token"]

    attacked_text = tokenizer.decode(
        attacked_ids,
        skip_special_tokens=True,
    )

    # --------------------------------------------------------
    # Validate on the actual decoded text
    # --------------------------------------------------------

    attacked = detect_text(attacked_text)

    actual_z_reduction = (
        original["z"] - attacked["z"]
    )

    print()
    print("VALIDATED ATTACK")
    print(f"Original z         : {original['z']:.6f}")
    print(f"Attacked z         : {attacked['z']:.6f}")
    print(f"Actual Δz          : {actual_z_reduction:.6f}")
    print(
        f"Detected before    : "
        f"{original['z'] > Z_THRESHOLD}"
    )
    print(
        f"Detected after     : "
        f"{attacked['z'] > Z_THRESHOLD}"
    )

    result = {
        "sample_index": sample_index,
        "prompt_id": sample["prompt_id"],

        "original_z_stored": stored_z,
        "original_z_recomputed": original["z"],
        "verification_error": verification_error,

        "original_green_count": original["green_count"],
        "original_pair_count": original["pair_count"],

        "attack": {
            "type": "targeted_single_token_substitution",
            "edits": 1,
            "position": best["position"],
            "old_token": best["old_token"],
            "new_token": best["new_token"],

            "predicted_z": best["predicted_z"],
            "predicted_z_reduction":
                best["predicted_z_reduction"],

            "validated_z": attacked["z"],
            "validated_z_reduction":
                actual_z_reduction,

            "detected_before":
                original["z"] > Z_THRESHOLD,

            "detected_after":
                attacked["z"] > Z_THRESHOLD,

            "old_left": best["old_left"],
            "old_right": best["old_right"],
            "new_left": best["new_left"],
            "new_right": best["new_right"],
        },

        "original_text": watermarked_text,
        "attacked_text": attacked_text,
    }

    results.append(result)

    print()


# ============================================================
# 7. SAVE RESULTS
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True,
)

with open(OUTPUT_PATH, "w") as f:
    json.dump(
        results,
        f,
        indent=2,
    )


# ============================================================
# 8. SUMMARY
# ============================================================

successful = sum(
    1
    for r in results
    if (
        r["attack"]["detected_before"]
        and not r["attack"]["detected_after"]
    )
)

mean_reduction = sum(
    r["attack"]["validated_z_reduction"]
    for r in results
) / len(results)

print("=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

for r in results:
    a = r["attack"]

    print(
        f"Sample {r['sample_index']}: "
        f"{r['original_z_recomputed']:.3f} "
        f"-> "
        f"{a['validated_z']:.3f} "
        f"(Δz={a['validated_z_reduction']:.3f})"
    )

print()
print(
    f"Samples crossing z<=2 after one edit: "
    f"{successful}/{len(results)}"
)

print(
    f"Mean z reduction: "
    f"{mean_reduction:.6f}"
)

print()
print(f"Results saved to: {OUTPUT_PATH}")
