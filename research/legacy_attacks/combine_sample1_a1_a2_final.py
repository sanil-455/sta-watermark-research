import json
import math
import re
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer


# Sample 1 only:
# Research Sample 1 -> prompt_id 0
SAMPLE_NUMBER = 1
PROMPT_ID = 0

BASELINE_PATH = Path(
    "results/raw/baseline_safe.json"
)

A1_DIR = Path(
    "results/raw/manual_attacks"
)

A1_FILES = [
    "sample1_pos345_might_to_could_corrected.json",
    "sample1_pos345_might_to_can.json",
    "sample1_pos345_might_to_may.json",
]

A2_RESULT_PATH = Path(
    "results/raw/attack2_results_by_sample/sample_1.jsonl"
)

A2_CANDIDATE_PATH = Path(
    "results/raw/attack2_candidates_by_sample/sample_1.json"
)

OUTPUT_DIR = Path(
    "results/raw/attack12_combined/sample_1"
)

SUMMARY_PATH = Path(
    "results/raw/attack12_combined/sample_1_summary.json"
)

TOKENIZER_PATH = (
    "hf_models/Llama-2-7b-hf"
)

SEMANTIC_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

GAMMA = 0.5
Z_THRESHOLD = 2.0
SEMANTIC_THRESHOLD = 0.90

MAX_LENGTH = 2048

HASH_KEY1 = 15485863
HASH_KEY2 = 17624813


def load_json(path):
    with path.open("r") as f:
        return json.load(f)


def save_json_atomic(path, data):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = path.with_suffix(
        ".tmp"
    )

    with temp_path.open("w") as f:
        json.dump(
            data,
            f,
            indent=2,
        )
        f.flush()

    temp_path.replace(path)


def get_sta_stats(text, tokenizer):
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )

    token_ids = encoded["input_ids"]

    if len(token_ids) < 2:
        return {
            "z": 0.0,
            "green_count": 0,
            "pair_count": 0,
            "token_count": len(token_ids),
        }

    rng = torch.Generator(
        device="cuda"
    )

    green_count = 0

    for i in range(len(token_ids) - 1):
        now_token = int(
            token_ids[i]
        )

        next_token = int(
            token_ids[i + 1]
        )

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

    pair_count = len(token_ids) - 1

    variance = (
        GAMMA
        * (1.0 - GAMMA)
        * pair_count
    )

    z = (
        green_count
        - GAMMA * pair_count
    ) / math.sqrt(variance)

    return {
        "z": float(z),
        "green_count": int(
            green_count
        ),
        "pair_count": int(
            pair_count
        ),
        "token_count": int(
            len(token_ids)
        ),
    }


def recover_single_edit(
    original,
    attacked,
):
    """
    Recover one localized character-level edit
    from an already-tested original/attacked pair.

    This avoids relying on saved token-span formats.
    """

    prefix = 0
    common_length = min(
        len(original),
        len(attacked),
    )

    while (
        prefix < common_length
        and original[prefix]
        == attacked[prefix]
    ):
        prefix += 1

    suffix = 0

    while (
        suffix
        < len(original) - prefix
        and suffix
        < len(attacked) - prefix
        and original[
            len(original) - 1 - suffix
        ]
        == attacked[
            len(attacked) - 1 - suffix
        ]
    ):
        suffix += 1

    original_end = (
        len(original) - suffix
    )

    attacked_end = (
        len(attacked) - suffix
    )

    old_text = original[
        prefix:original_end
    ]

    new_text = attacked[
        prefix:attacked_end
    ]

    if not old_text and not new_text:
        raise RuntimeError(
            "Original and attacked text are identical."
        )

    if not old_text:
        tag = "insert"
    elif not new_text:
        tag = "delete"
    else:
        tag = "replace"

    rebuilt = (
        original[:prefix]
        + new_text
        + original[original_end:]
    )

    if rebuilt != attacked:
        raise RuntimeError(
            "Recovered edit does not reproduce "
            "the saved attacked text."
        )

    return {
        "tag": tag,
        "start": prefix,
        "end": original_end,
        "old_text": old_text,
        "new_text": new_text,
    }


def edits_overlap(edit_a, edit_b):
    a_start = edit_a["start"]
    a_end = edit_a["end"]

    b_start = edit_b["start"]
    b_end = edit_b["end"]

    # Two insertions at the same character location
    # have ambiguous ordering.
    if (
        a_start == a_end
        and b_start == b_end
    ):
        return a_start == b_start

    # An insertion strictly inside another edit
    # is ambiguous.
    if a_start == a_end:
        return (
            b_start
            < a_start
            < b_end
        )

    if b_start == b_end:
        return (
            a_start
            < b_start
            < a_end
        )

    # Ordinary interval overlap.
    return not (
        a_end <= b_start
        or b_end <= a_start
    )


def apply_two_edits(
    original,
    edit_a,
    edit_b,
):
    if edits_overlap(
        edit_a,
        edit_b,
    ):
        raise RuntimeError(
            "A1 and A2 edits overlap or "
            "have an ambiguous insertion point."
        )

    edits = [
        edit_a,
        edit_b,
    ]

    # Both spans refer to the original text.
    # Applying from right to left preserves them.
    edits.sort(
        key=lambda edit: (
            edit["start"],
            edit["end"],
        ),
        reverse=True,
    )

    result = original

    for edit in edits:
        result = (
            result[:edit["start"]]
            + edit["new_text"]
            + result[edit["end"]:]
        )

    return result


def get_word_count(text):
    return len(
        re.findall(
            r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*",
            text,
        )
    )


def split_sentences(text):
    pieces = re.split(
        r"(?<=[.!?])\s+",
        text,
    )

    pieces = [
        piece.strip()
        for piece in pieces
        if piece.strip()
    ]

    if not pieces:
        return [text]

    return pieces


def get_semantic_scores(
    model,
    original,
    attacked,
):
    all_texts = [
        original,
        attacked,
    ]

    original_sentences = split_sentences(
        original
    )

    attacked_sentences = split_sentences(
        attacked
    )

    all_texts.extend(
        original_sentences
    )

    all_texts.extend(
        attacked_sentences
    )

    embeddings = model.encode(
        all_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    global_similarity = float(
        np.dot(
            embeddings[0],
            embeddings[1],
        )
    )

    original_start = 2
    attacked_start = (
        2 + len(original_sentences)
    )

    if (
        len(original_sentences)
        != len(attacked_sentences)
    ):
        # If an edit changes sentence boundaries,
        # use the global score for the local field
        # rather than inventing a sentence alignment.
        local_similarity = (
            global_similarity
        )

    else:
        local_scores = []

        for i in range(
            len(original_sentences)
        ):
            score = float(
                np.dot(
                    embeddings[
                        original_start + i
                    ],
                    embeddings[
                        attacked_start + i
                    ],
                )
            )

            local_scores.append(score)

        local_similarity = float(
            np.mean(local_scores)
        )

    return (
        global_similarity,
        local_similarity,
    )


def load_baseline():
    rows = load_json(
        BASELINE_PATH
    )

    matches = [
        row
        for row in rows
        if int(row["prompt_id"])
        == PROMPT_ID
    ]

    if len(matches) != 1:
        raise RuntimeError(
            "Could not uniquely identify "
            "Research Sample 1."
        )

    baseline = matches[0]

    expected_z = 3.363765319856875

    if abs(
        float(
            baseline["watermarked_z"]
        )
        - expected_z
    ) > 1e-6:
        raise RuntimeError(
            "Unexpected stored Sample 1 "
            "baseline z."
        )

    return baseline


def load_a1_operations(
    baseline_text,
):
    operations = []

    for filename in A1_FILES:
        path = A1_DIR / filename

        if not path.exists():
            raise RuntimeError(
                f"Missing A1 result file: {path}"
            )

        data = load_json(path)

        if data.get("sample") != SAMPLE_NUMBER:
            raise RuntimeError(
                f"Wrong sample in {filename}."
            )

        original_text = data.get(
            "original_text"
        )

        attacked_text = data.get(
            "attacked_text"
        )

        if original_text != baseline_text:
            raise RuntimeError(
                f"A1 original text does not "
                f"match Sample 1: {filename}"
            )

        if not attacked_text:
            raise RuntimeError(
                f"A1 attacked text missing: {filename}"
            )

        edit = recover_single_edit(
            original_text,
            attacked_text,
        )

        if edit["tag"] != "replace":
            raise RuntimeError(
                f"A1 is not a substitution: "
                f"{filename}"
            )

        if (
            edit["old_text"]
            != data["old_word"]
        ):
            raise RuntimeError(
                f"A1 old word mismatch: "
                f"{filename}"
            )

        if (
            edit["new_text"]
            != data["new_word"]
        ):
            raise RuntimeError(
                f"A1 replacement mismatch: "
                f"{filename}"
            )

        operations.append(
            {
                "source_file":
                    filename,
                "old_word":
                    data["old_word"],
                "new_word":
                    data["new_word"],
                "delta_z":
                    float(
                        data["delta_z"]
                    ),
                "attacked_z":
                    float(
                        data["attacked_z"]
                    ),
                "edit":
                    edit,
            }
        )

    # Most negative delta-z first.
    operations.sort(
        key=lambda item: item["delta_z"]
    )

    return operations


def determine_indexing(
    results,
    candidate_count,
):
    indices = [
        int(row["candidate_index"])
        for row in results
        if row.get("candidate_index")
        is not None
    ]

    if not indices:
        raise RuntimeError(
            "Attack-2 results contain no candidate indices."
        )

    minimum = min(indices)
    maximum = max(indices)

    # Clear 0-based convention:
    # 0 ... N-1
    if (
        minimum == 0
        and maximum == candidate_count - 1
    ):
        return "zero_based"

    # Clear 1-based convention:
    # 1 ... N
    if (
        minimum == 1
        and maximum == candidate_count
    ):
        return "one_based"

    # If the range is incomplete, use the strongest
    # evidence available from the boundary.
    if maximum == candidate_count:
        return "one_based"

    if minimum == 0:
        return "zero_based"

    raise RuntimeError(
        "Could not determine whether Attack-2 "
        "candidate indices are 0-based or 1-based. "
        f"Observed range: {minimum}..{maximum}; "
        f"candidate count: {candidate_count}."
    )


def resolve_candidate(
    result,
    candidates,
    indexing,
):
    index = int(
        result["candidate_index"]
    )

    if indexing == "zero_based":
        list_index = index

    else:
        list_index = index - 1

    if not (
        0 <= list_index < len(candidates)
    ):
        raise RuntimeError(
            f"Candidate index {index} cannot be "
            f"mapped using {indexing} indexing."
        )

    candidate = candidates[
        list_index
    ]

    return candidate


def load_a2_operations(
    baseline_text,
):
    candidate_data = load_json(
        A2_CANDIDATE_PATH
    )

    if int(
        candidate_data["sample_number"]
    ) != SAMPLE_NUMBER:
        raise RuntimeError(
            "A2 candidate sample number mismatch."
        )

    if int(
        candidate_data["prompt_id"]
    ) != PROMPT_ID:
        raise RuntimeError(
            "A2 candidate prompt_id mismatch."
        )

    candidates = candidate_data[
        "candidates"
    ]

    results = []

    with A2_RESULT_PATH.open("r") as f:
        for line in f:
            if line.strip():
                results.append(
                    json.loads(line)
                )

    indexing = determine_indexing(
        results,
        len(candidates),
    )

    print(
        "Attack-2 candidate indexing: "
        f"{indexing}"
    )

    latest = {}

    for result in results:
        if (
            result.get("candidate_index")
            is None
        ):
            continue

        latest[
            int(result["candidate_index"])
        ] = result

    valid = []

    for result in latest.values():
        if result.get("status") != "ok":
            continue

        if (
            result.get("semantic_pass")
            is not True
        ):
            continue

        if result.get("attacked_z") is None:
            continue

        candidate = resolve_candidate(
            result,
            candidates,
            indexing,
        )

        if not isinstance(
            candidate,
            dict,
        ):
            raise RuntimeError(
                "Resolved A2 candidate is not a dictionary."
            )

        attack_type = candidate.get(
            "attack_type"
        )

        if attack_type not in {
            "insertion",
            "deletion",
        }:
            raise RuntimeError(
                "A2 candidate has an invalid "
                f"attack type: {attack_type!r}"
            )

        attacked_text = candidate.get(
            "attacked_text"
        )

        if not attacked_text:
            raise RuntimeError(
                "A2 candidate has no attacked_text."
            )

        # This is the crucial integrity check:
        # recover the actual character edit from
        # the saved candidate text.
        edit = recover_single_edit(
            baseline_text,
            attacked_text,
        )

        expected_tag = {
            "insertion": "insert",
            "deletion": "delete",
        }[attack_type]

        if edit["tag"] != expected_tag:
            raise RuntimeError(
                "A2 candidate edit type disagrees "
                "with its metadata."
            )

        if (
            candidate.get("original_word")
            and attack_type == "deletion"
        ):
            expected_word = str(
                candidate["original_word"]
            ).strip()

            actual_word = (
                edit["old_text"]
                .strip()
            )

            if actual_word != expected_word:
                raise RuntimeError(
                    "A2 deletion word mismatch."
                )

        valid.append(
            {
                "candidate_index":
                    int(
                        result[
                            "candidate_index"
                        ]
                    ),
                "attack_type":
                    attack_type,
                "original_word":
                    candidate.get(
                        "original_word",
                        "",
                    ),
                "replacement":
                    candidate.get(
                        "replacement",
                        "",
                    ),
                "attacked_z":
                    float(
                        result["attacked_z"]
                    ),
                "delta_z":
                    float(
                        result["delta_z"]
                    ),
                "semantic_score":
                    candidate.get(
                        "semantic_score"
                    ),
                "edit":
                    edit,
            }
        )

    valid.sort(
        key=lambda item: item["delta_z"]
    )

    if len(valid) < 10:
        raise RuntimeError(
            "Fewer than 10 valid Attack-2 "
            f"candidates are available: {len(valid)}"
        )

    return valid[:10]


def load_existing_results():
    existing = {}

    if not OUTPUT_DIR.exists():
        return existing

    for path in OUTPUT_DIR.glob(
        "combo_*.json"
    ):
        try:
            row = load_json(path)

            combo_index = int(
                row["combo_index"]
            )

            existing[
                combo_index
            ] = row

        except Exception:
            # Ignore an incomplete/corrupt old file.
            # It will be regenerated.
            continue

    return existing


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    baseline = load_baseline()

    baseline_text = baseline[
        "watermarked_text"
    ]

    print(
        "Verified: Research Sample 1 "
        "(prompt_id=0)"
    )

    print(
        "Loading Llama tokenizer..."
    )

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_PATH,
        local_files_only=True,
    )

    baseline_stats = get_sta_stats(
        baseline_text,
        tokenizer,
    )

    stored_z = float(
        baseline["watermarked_z"]
    )

    if abs(
        baseline_stats["z"]
        - stored_z
    ) > 1e-6:
        raise RuntimeError(
            "Independent STA baseline "
            "verification failed."
        )

    print(
        f"Baseline z verified: "
        f"{baseline_stats['z']:.12f}"
    )

    print(
        "Loading A1 operations..."
    )

    a1_operations = load_a1_operations(
        baseline_text
    )

    print(
        "Loading A2 operations..."
    )

    a2_operations = load_a2_operations(
        baseline_text
    )

    print(
        f"A1 operations selected: "
        f"{len(a1_operations)}"
    )

    print(
        f"A2 operations selected: "
        f"{len(a2_operations)}"
    )

    total = (
        len(a1_operations)
        * len(a2_operations)
    )

    print(
        f"Total combinations: {total}"
    )

    if total != 30:
        raise RuntimeError(
            f"Expected 30 Sample-1 combinations, "
            f"but constructed {total}."
        )

    print(
        "Loading semantic model..."
    )

    semantic_model = SentenceTransformer(
        SEMANTIC_MODEL_NAME,
        device="cpu",
    )

    existing = load_existing_results()

    print(
        f"Previously saved combinations: "
        f"{len(existing)}"
    )

    combo_index = 0

    for a1_index, a1 in enumerate(
        a1_operations,
        start=1,
    ):
        for a2_index, a2 in enumerate(
            a2_operations,
            start=1,
        ):
            combo_index += 1

            if combo_index in existing:
                continue

            result = {
                "combo_index":
                    combo_index,
                "sample_number":
                    SAMPLE_NUMBER,
                "prompt_id":
                    PROMPT_ID,
                "attack_type":
                    "A1_plus_A2",
                "a1_index":
                    a1_index,
                "a2_index":
                    a2_index,
                "a1_source_file":
                    a1["source_file"],
                "a1_old_word":
                    a1["old_word"],
                "a1_new_word":
                    a1["new_word"],
                "a1_delta_z":
                    a1["delta_z"],
                "a2_candidate_index":
                    a2["candidate_index"],
                "a2_attack_type":
                    a2["attack_type"],
                "a2_original_word":
                    a2["original_word"],
                "a2_replacement":
                    a2["replacement"],
                "a2_delta_z":
                    a2["delta_z"],
                "status":
                    "error",
                "watermark_break":
                    False,
            }

            try:
                attacked_text = apply_two_edits(
                    baseline_text,
                    a1["edit"],
                    a2["edit"],
                )

                if attacked_text == baseline_text:
                    raise RuntimeError(
                        "Combined attack produced "
                        "unchanged text."
                    )

                global_similarity, local_similarity = (
                    get_semantic_scores(
                        semantic_model,
                        baseline_text,
                        attacked_text,
                    )
                )

                result[
                    "global_similarity"
                ] = global_similarity

                result[
                    "local_similarity"
                ] = local_similarity

                result[
                    "semantic_threshold"
                ] = SEMANTIC_THRESHOLD

                semantic_pass = (
                    global_similarity
                    >= SEMANTIC_THRESHOLD
                    and
                    local_similarity
                    >= SEMANTIC_THRESHOLD
                )

                result[
                    "semantic_pass"
                ] = semantic_pass

                if not semantic_pass:
                    result[
                        "status"
                    ] = "semantic_rejected"

                else:
                    attacked_stats = get_sta_stats(
                        attacked_text,
                        tokenizer,
                    )

                    original_word_count = (
                        get_word_count(
                            baseline_text
                        )
                    )

                    attacked_word_count = (
                        get_word_count(
                            attacked_text
                        )
                    )

                    attacked_z = (
                        attacked_stats["z"]
                    )

                    result.update(
                        {
                            "status":
                                "ok",

                            "original_z":
                                baseline_stats["z"],

                            "attacked_z":
                                attacked_z,

                            "delta_z":
                                (
                                    attacked_z
                                    - baseline_stats["z"]
                                ),

                            "original_green_count":
                                baseline_stats[
                                    "green_count"
                                ],

                            "attacked_green_count":
                                attacked_stats[
                                    "green_count"
                                ],

                            "green_count_change":
                                (
                                    attacked_stats[
                                        "green_count"
                                    ]
                                    - baseline_stats[
                                        "green_count"
                                    ]
                                ),

                            "original_pair_count":
                                baseline_stats[
                                    "pair_count"
                                ],

                            "attacked_pair_count":
                                attacked_stats[
                                    "pair_count"
                                ],

                            "original_token_count":
                                baseline_stats[
                                    "token_count"
                                ],

                            "attacked_token_count":
                                attacked_stats[
                                    "token_count"
                                ],

                            "token_count_change":
                                (
                                    attacked_stats[
                                        "token_count"
                                    ]
                                    - baseline_stats[
                                        "token_count"
                                    ]
                                ),

                            "original_word_count":
                                original_word_count,

                            "attacked_word_count":
                                attacked_word_count,

                            "word_count_change":
                                (
                                    attacked_word_count
                                    - original_word_count
                                ),

                            "original_character_count":
                                len(baseline_text),

                            "attacked_character_count":
                                len(attacked_text),

                            "character_count_change":
                                (
                                    len(attacked_text)
                                    - len(baseline_text)
                                ),

                            "a1_operations":
                                1,

                            "a2_operations":
                                1,

                            "total_word_operations":
                                2,

                            "substitutions":
                                1,

                            "insertions":
                                int(
                                    a2[
                                        "attack_type"
                                    ]
                                    == "insertion"
                                ),

                            "deletions":
                                int(
                                    a2[
                                        "attack_type"
                                    ]
                                    == "deletion"
                                ),

                            "edit_rate":
                                (
                                    2
                                    / original_word_count
                                ),

                            "attacked_detected":
                                (
                                    attacked_z
                                    > Z_THRESHOLD
                                ),

                            "watermark_break":
                                (
                                    baseline_stats[
                                        "z"
                                    ]
                                    > Z_THRESHOLD
                                    and
                                    attacked_z
                                    <= Z_THRESHOLD
                                ),

                            "attacked_text":
                                attacked_text,
                        }
                    )

            except Exception as exc:
                result[
                    "status"
                ] = "error"

                result[
                    "error"
                ] = repr(exc)

            save_json_atomic(
                OUTPUT_DIR
                / f"combo_{combo_index:03d}.json",
                result,
            )

            existing[
                combo_index
            ] = result

            print(
                f"[✓ SAVED] Sample 1 "
                f"combo {combo_index}/{total} "
                f"| {result['status']}"
            )

    valid = [
        row
        for row in existing.values()
        if row.get("status") == "ok"
    ]

    rejected = [
        row
        for row in existing.values()
        if row.get("status")
        == "semantic_rejected"
    ]

    errors = [
        row
        for row in existing.values()
        if row.get("status") == "error"
    ]

    breaks = [
        row
        for row in valid
        if row.get("watermark_break")
        is True
    ]

    strongest = sorted(
        valid,
        key=lambda row: row["delta_z"],
    )

    summary = {
        "sample_number":
            SAMPLE_NUMBER,
        "prompt_id":
            PROMPT_ID,
        "baseline_z":
            baseline_stats["z"],
        "a1_count":
            len(a1_operations),
        "a2_count":
            len(a2_operations),
        "total_combinations":
            total,
        "completed":
            len(existing),
        "valid":
            len(valid),
        "semantic_rejected":
            len(rejected),
        "errors":
            len(errors),
        "watermark_breaks":
            len(breaks),
        "strongest_attacks":
            strongest[:20],
        "error_records":
            errors,
    }

    save_json_atomic(
        SUMMARY_PATH,
        summary,
    )

    print()
    print(
        f"[✓ SAVED] {SUMMARY_PATH}"
    )

    print()
    print(
        "Sample 1 complete."
    )

    print(
        f"Completed: "
        f"{len(existing)}/{total}"
    )

    print(
        f"Valid: {len(valid)}"
    )

    print(
        f"Semantic rejected: "
        f"{len(rejected)}"
    )

    print(
        f"Errors: {len(errors)}"
    )

    print(
        f"Watermark breaks: "
        f"{len(breaks)}"
    )

    if strongest:
        print()
        print(
            "Strongest valid combinations:"
        )

        for rank, row in enumerate(
            strongest[:10],
            start=1,
        ):
            print(
                f"{rank:2d}. "
                f"A1 {row['a1_old_word']} -> "
                f"{row['a1_new_word']} + "
                f"A2 {row['a2_attack_type']} "
                f"{row['a2_original_word'] or row['a2_replacement']} "
                f"| z={row['attacked_z']:.6f} "
                f"| dz={row['delta_z']:.6f} "
                f"| global={row['global_similarity']:.4f} "
                f"| local={row['local_similarity']:.4f} "
                f"| break={row['watermark_break']}"
            )


if __name__ == "__main__":
    main()
