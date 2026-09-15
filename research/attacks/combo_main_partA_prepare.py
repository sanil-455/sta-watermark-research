import json
from pathlib import Path


SAMPLE = 1
PROMPT_ID = 0
BASELINE_Z = 3.363765319856875
SEMANTIC_THRESHOLD = 0.90
Z_THRESHOLD = 2.0

BASELINE_PATH = Path(
    "results/raw/baseline_safe.json"
)

A1_DIR = Path(
    "results/raw/manual_attacks"
)

A2_CANDIDATE_PATH = Path(
    "results/raw/attack2_candidates_by_sample/"
    "sample_1.json"
)

A2_RESULT_PATH = Path(
    "results/raw/attack2_results_by_sample/"
    "sample_1.jsonl"
)

OUT_DIR = Path(
    "results/raw/attack12_combined/"
)

MANIFEST_PATH = (
    OUT_DIR / "sample_1_combination_manifest.jsonl"
)


def load_json(path):
    with path.open() as f:
        return json.load(f)


def load_baseline():
    data = load_json(BASELINE_PATH)

    for row in data:
        if row["prompt_id"] == PROMPT_ID:
            return row

    raise RuntimeError("Sample 1 baseline not found")


def load_a1():
    files = [
        A1_DIR /
        "sample1_pos345_might_to_could_corrected.json",
        A1_DIR /
        "sample1_pos345_might_to_can.json",
    ]

    out = []

    for path in files:
        row = load_json(path)

        if row["original_text"] != BASELINE_TEXT:
            raise RuntimeError(
                f"A1 original text mismatch: {path}"
            )

        if row["old_word"] != "might":
            raise RuntimeError(
                f"Unexpected A1 old word: {path}"
            )

        out.append(row)

    if len(out) != 2:
        raise RuntimeError("Expected exactly 2 A1 attacks")

    return out


def load_a2():
    data = load_json(A2_CANDIDATE_PATH)
    candidates = data["candidates"]

    if len(candidates) != 475:
        raise RuntimeError(
            f"Expected 475 A2 candidates, "
            f"found {len(candidates)}"
        )

    results = {}

    with A2_RESULT_PATH.open() as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)
            idx = row["candidate_index"]

            if 0 <= idx < len(candidates):
                results[idx] = row

    eligible = []

    for idx, candidate in enumerate(candidates):
        result = results.get(idx)

        if result is None:
            continue

        if result.get("status") != "ok":
            continue

        if result.get("semantic_pass") is not True:
            continue

        if (
            result.get("global_similarity", 0)
            < SEMANTIC_THRESHOLD
        ):
            continue

        if (
            result.get("local_similarity", 0)
            < SEMANTIC_THRESHOLD
        ):
            continue

        eligible.append(
            {
                "candidate_index": idx,
                "candidate": candidate,
                "result": result,
            }
        )

    return eligible


def main():
    global BASELINE_TEXT

    baseline = load_baseline()

    if baseline["prompt_id"] != PROMPT_ID:
        raise RuntimeError("Wrong prompt ID")

    BASELINE_TEXT = baseline["watermarked_text"]

    if abs(
        baseline["watermarked_z"] - BASELINE_Z
    ) > 1e-6:
        raise RuntimeError(
            "Baseline z does not match "
            "validated Sample 1 baseline"
        )

    if BASELINE_Z <= Z_THRESHOLD:
        raise RuntimeError(
            "Sample 1 is not initially detected"
        )

    a1 = load_a1()
    a2 = load_a2()

    print("PRECHECK PASSED")
    print(f"Sample: {SAMPLE}")
    print(f"Prompt ID: {PROMPT_ID}")
    print(f"Baseline z: {BASELINE_Z}")
    print(f"A1 operations: {len(a1)}")
    print(f"Eligible A2 operations: {len(a2)}")

    combinations = []

    for a1_index, attack1 in enumerate(a1):
        for a2 in a2:
            combinations.append(
                {
                    "combination_index":
                        len(combinations),
                    "sample_number":
                        SAMPLE,
                    "prompt_id":
                        PROMPT_ID,
                    "a1_index":
                        a1_index,
                    "a1_old_word":
                        attack1["old_word"],
                    "a1_new_word":
                        attack1["new_word"],
                    "a1_attacked_text":
                        attack1["attacked_text"],
                    "a2_candidate_index":
                        a2["candidate_index"],
                    "a2_attack_type":
                        a2["candidate"]["attack_type"],
                    "a2_original_word":
                        a2["candidate"]["original_word"],
                    "a2_replacement":
                        a2["candidate"]["replacement"],
                    "a2_position":
                        a2["candidate"]["position"],
                    "a2_attacked_text":
                        a2["candidate"]["attacked_text"],
                    "a2_global_similarity":
                        a2["result"]["global_similarity"],
                    "a2_local_similarity":
                        a2["result"]["local_similarity"],
                }
            )

    expected = len(a1) * len(a2)

    if len(combinations) != expected:
        raise RuntimeError(
            "Combination count mismatch"
        )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with MANIFEST_PATH.open("w") as f:
        for row in combinations:
            f.write(
                json.dumps(row)
                + "\n"
            )

    print(f"Total combinations: {len(combinations)}")
    print(f"[✓ SAVED] {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
