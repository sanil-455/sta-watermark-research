import json
import math
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer


BASELINE_PATH = Path("results/raw/baseline_safe.json")

CANDIDATE_DIR = Path(
    "results/raw/attack2_candidates_by_sample"
)

RESULT_DIR = Path(
    "results/raw/attack2_results_by_sample"
)

MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

SEMANTIC_THRESHOLD = 0.90
BATCH_SIZE = 32

GAMMA = 0.5
Z_THRESHOLD = 2.0

HASH_KEY1 = 15485863
HASH_KEY2 = 17624813

PROMPT_MAX_LENGTH = 2048

SAMPLE_TO_PROMPT_ID = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
}

TARGET_SAMPLE = 1

EXPECTED_BASELINE_Z = {
    1: 3.363765319856875,
    2: 2.8160458729081004,
    3: 3.049618749953805,
    4: 0.6216937087382038,
    5: 1.9549344509427915,
}


def load_json(path):
    with path.open() as f:
        return json.load(f)


def find_baseline(sample_number, baselines):
    expected_id = SAMPLE_TO_PROMPT_ID[sample_number]

    matches = [
        x
        for x in baselines
        if int(x["prompt_id"]) == expected_id
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one baseline for "
            f"Research Sample {sample_number} "
            f"(prompt_id={expected_id}), "
            f"found {len(matches)}."
        )

    sample = matches[0]

    actual_id = int(sample["prompt_id"])

    if actual_id != expected_id:
        raise RuntimeError(
            f"ID mismatch: Research Sample "
            f"{sample_number} should have "
            f"prompt_id={expected_id}, but got "
            f"{actual_id}."
        )

    expected_z = EXPECTED_BASELINE_Z[sample_number]
    actual_z = float(sample["watermarked_z"])

    if abs(actual_z - expected_z) > 1e-6:
        raise RuntimeError(
            f"Baseline z mismatch for Research "
            f"Sample {sample_number}: "
            f"expected {expected_z}, got {actual_z}."
        )

    if actual_z <= Z_THRESHOLD:
        raise RuntimeError(
            f"Research Sample {sample_number} "
            f"is not detected: z={actual_z}."
        )

    return sample


def sta_z(text, tokenizer):
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=PROMPT_MAX_LENGTH,
        return_tensors="pt",
    )

    token_ids = encoded["input_ids"][0].tolist()

    if len(token_ids) < 2:
        return {
            "z": 0.0,
            "green_count": 0,
            "pair_count": 0,
            "token_count": len(token_ids),
        }

    rng = torch.Generator(device="cuda")

    green_count = 0

    for i in range(len(token_ids) - 1):
        now_token = token_ids[i]
        next_token = token_ids[i + 1]

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

    expected = GAMMA * pair_count
    variance = (
        GAMMA
        * (1.0 - GAMMA)
        * pair_count
    )

    z = (
        (green_count - expected)
        / math.sqrt(variance)
        if variance > 0
        else 0.0
    )

    return {
        "z": float(z),
        "green_count": green_count,
        "pair_count": pair_count,
        "token_count": len(token_ids),
    }


def chunk_text(text, words_per_chunk=180):
    words = text.split()

    if not words:
        return [""]

    return [
        " ".join(
            words[i:i + words_per_chunk]
        )
        for i in range(
            0,
            len(words),
            words_per_chunk,
        )
    ]


def get_changed_regions(
    original,
    attacked,
    window=500,
):
    from difflib import SequenceMatcher

    matcher = SequenceMatcher(
        None,
        original,
        attacked,
    )

    changed = [
        op
        for op in matcher.get_opcodes()
        if op[0] != "equal"
    ]

    if not changed:
        return original, attacked

    first = changed[0]

    original_start = max(
        0,
        first[1] - window,
    )

    original_end = min(
        len(original),
        first[2] + window,
    )

    attacked_start = max(
        0,
        first[3] - window,
    )

    attacked_end = min(
        len(attacked),
        first[4] + window,
    )

    return (
        original[
            original_start:original_end
        ],
        attacked[
            attacked_start:attacked_end
        ],
    )


def semantic_scores(
    model,
    original,
    attacked,
):
    original_chunks = chunk_text(original)
    attacked_chunks = chunk_text(attacked)

    texts = (
        original_chunks
        + attacked_chunks
    )

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    original_embedding = embeddings[
        :len(original_chunks)
    ].mean(axis=0)

    attacked_embedding = embeddings[
        len(original_chunks):
    ].mean(axis=0)

    original_embedding /= np.linalg.norm(
        original_embedding
    )

    attacked_embedding /= np.linalg.norm(
        attacked_embedding
    )

    global_score = float(
        np.dot(
            original_embedding,
            attacked_embedding,
        )
    )

    original_region, attacked_region = (
        get_changed_regions(
            original,
            attacked,
        )
    )

    local_embeddings = model.encode(
        [
            original_region,
            attacked_region,
        ],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    local_score = float(
        np.dot(
            local_embeddings[0],
            local_embeddings[1],
        )
    )

    return global_score, local_score


def load_existing_results(output_path):
    existing = {}

    if not output_path.exists():
        return existing

    with output_path.open() as f:
        for line in f:
            if not line.strip():
                continue

            result = json.loads(line)

            existing[
                result["candidate_index"]
            ] = result

    return existing


def process_sample(
    sample,
    model,
    tokenizer,
):
    sample_number = int(
        sample["sample_number"]
    )

    prompt_id = int(
        sample["prompt_id"]
    )

    expected_prompt_id = (
        SAMPLE_TO_PROMPT_ID[sample_number]
    )

    if prompt_id != expected_prompt_id:
        raise RuntimeError(
            f"Output sample mapping is wrong: "
            f"Research Sample {sample_number} "
            f"must use prompt_id "
            f"{expected_prompt_id}, got "
            f"{prompt_id}."
        )

    candidate_path = (
        CANDIDATE_DIR
        / f"sample_{sample_number}.json"
    )

    candidate_data = load_json(
        candidate_path
    )

    if int(
        candidate_data["sample_number"]
    ) != sample_number:
        raise RuntimeError(
            "Candidate file sample_number "
            "does not match requested sample."
        )

    if int(
        candidate_data["prompt_id"]
    ) != prompt_id:
        raise RuntimeError(
            "Candidate file prompt_id does "
            "not match baseline prompt_id."
        )

    candidates = candidate_data[
        "candidates"
    ]

    output_path = (
        RESULT_DIR
        / f"sample_{sample_number}.jsonl"
    )

    summary_path = (
        RESULT_DIR
        / f"sample_{sample_number}_summary.json"
    )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing = load_existing_results(
        output_path
    )

    baseline_z = float(
        sample["watermarked_z"]
    )

    baseline_check = sta_z(
        sample["watermarked_text"],
        tokenizer,
    )

    if abs(
        baseline_check["z"] - baseline_z
    ) > 1e-6:
        raise RuntimeError(
            f"STA baseline mismatch for "
            f"Research Sample {sample_number}: "
            f"stored={baseline_z}, "
            f"recomputed="
            f"{baseline_check['z']}"
        )

    print(
        f"Research Sample {sample_number} "
        f"(prompt_id={prompt_id}): "
        f"{len(candidates)} candidates"
    )

    print(
        f"Already saved: {len(existing)}"
    )

    new_results = []

    for batch_start in range(
        0,
        len(candidates),
        BATCH_SIZE,
    ):
        batch = candidates[
            batch_start:
            batch_start + BATCH_SIZE
        ]

        batch_results = []

        for offset, candidate in enumerate(
            batch
        ):
            candidate_index = (
                batch_start + offset
            )

            if candidate_index in existing:
                continue

            try:
                global_score, local_score = (
                    semantic_scores(
                        model,
                        sample[
                            "watermarked_text"
                        ],
                        candidate[
                            "attacked_text"
                        ],
                    )
                )

                semantic_pass = (
                    global_score
                    >= SEMANTIC_THRESHOLD
                    and local_score
                    >= SEMANTIC_THRESHOLD
                )

                result = {
                    "candidate_index":
                        candidate_index,
                    "sample_number":
                        sample_number,
                    "prompt_id":
                        prompt_id,
                    "attack_type":
                        candidate[
                            "attack_type"
                        ],
                    "original_word":
                        candidate[
                            "original_word"
                        ],
                    "replacement":
                        candidate[
                            "replacement"
                        ],
                    "position":
                        candidate[
                            "position"
                        ],
                    "global_similarity":
                        global_score,
                    "local_similarity":
                        local_score,
                    "semantic_threshold":
                        SEMANTIC_THRESHOLD,
                    "semantic_pass":
                        semantic_pass,
                    "original_z":
                        baseline_z,
                    "original_detected":
                        baseline_z
                        > Z_THRESHOLD,
                    "status":
                        "ok",
                }

                if semantic_pass:
                    attacked_check = sta_z(
                        candidate[
                            "attacked_text"
                        ],
                        tokenizer,
                    )

                    attacked_z = (
                        attacked_check["z"]
                    )

                    result.update(
                        {
                            "attacked_z":
                                attacked_z,
                            "delta_z":
                                attacked_z
                                - baseline_z,
                            "attacked_detected":
                                attacked_z
                                > Z_THRESHOLD,
                            "watermark_break":
                                (
                                    baseline_z
                                    > Z_THRESHOLD
                                    and attacked_z
                                    <= Z_THRESHOLD
                                ),
                            "original_green_count":
                                baseline_check[
                                    "green_count"
                                ],
                            "attacked_green_count":
                                attacked_check[
                                    "green_count"
                                ],
                            "original_pair_count":
                                baseline_check[
                                    "pair_count"
                                ],
                            "attacked_pair_count":
                                attacked_check[
                                    "pair_count"
                                ],
                            "original_token_count":
                                baseline_check[
                                    "token_count"
                                ],
                            "attacked_token_count":
                                attacked_check[
                                    "token_count"
                                ],
                        }
                    )
                else:
                    result.update(
                        {
                            "attacked_z":
                                None,
                            "delta_z":
                                None,
                            "attacked_detected":
                                None,
                            "watermark_break":
                                False,
                        }
                    )

            except Exception as exc:
                result = {
                    "candidate_index":
                        candidate_index,
                    "sample_number":
                        sample_number,
                    "prompt_id":
                        prompt_id,
                    "attack_type":
                        candidate[
                            "attack_type"
                        ],
                    "original_word":
                        candidate[
                            "original_word"
                        ],
                    "replacement":
                        candidate[
                            "replacement"
                        ],
                    "position":
                        candidate[
                            "position"
                        ],
                    "status":
                        "error",
                    "error":
                        repr(exc),
                    "watermark_break":
                        False,
                }

            batch_results.append(result)

        if batch_results:
            with output_path.open(
                "a"
            ) as f:
                for result in batch_results:
                    f.write(
                        json.dumps(result)
                        + "\n"
                    )
                    f.flush()

                    new_results.append(
                        result
                    )

                    print(
                        f"[✓ SAVED] Research "
                        f"Sample "
                        f"{sample_number} "
                        f"(prompt_id="
                        f"{prompt_id}), "
                        f"candidate "
                        f"{result['candidate_index'] + 1}/"
                        f"{len(candidates)}"
                    )

    all_results = list(
        existing.values()
    ) + new_results

    all_results.sort(
        key=lambda x:
        x["candidate_index"]
    )

    semantic_passes = [
        r
        for r in all_results
        if r.get("status") == "ok"
        and r.get("semantic_pass")
        is True
    ]

    breaks = [
        r
        for r in semantic_passes
        if r.get("watermark_break")
        is True
    ]

    strongest = sorted(
        semantic_passes,
        key=lambda r:
        r["delta_z"],
    )[:20]

    summary = {
        "sample_number":
            sample_number,
        "prompt_id":
            prompt_id,
        "baseline_z":
            baseline_z,
        "baseline_detected":
            baseline_z > Z_THRESHOLD,
        "candidate_count":
            len(candidates),
        "completed_count":
            len(all_results),
        "semantic_pass_count":
            len(semantic_passes),
        "watermark_break_count":
            len(breaks),
        "strongest_attacks":
            strongest,
    }

    with summary_path.open("w") as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print(
        f"[✓ SAVED] Research Sample "
        f"{sample_number} summary -> "
        f"{summary_path}"
    )

    print(
        f"Breaks found: {len(breaks)}"
    )


def main():
    if TARGET_SAMPLE not in SAMPLE_TO_PROMPT_ID:
        raise RuntimeError(
            f"Invalid target sample: "
            f"{TARGET_SAMPLE}"
        )

    print("Loading semantic model...")

    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu",
    )

    print("Loading Llama tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        "hf_models/Llama-2-7b-hf",
        local_files_only=True,
    )

    baselines = load_json(
        BASELINE_PATH
    )

    sample = find_baseline(
        TARGET_SAMPLE,
        baselines,
    )

    print(
        f"Verified Research Sample "
        f"{TARGET_SAMPLE}: "
        f"prompt_id="
        f"{sample['prompt_id']}, "
        f"baseline_z="
        f"{sample['watermarked_z']}"
    )

    sample_with_id = dict(sample)

    sample_with_id[
        "sample_number"
    ] = TARGET_SAMPLE

    process_sample(
        sample_with_id,
        model,
        tokenizer,
    )


if __name__ == "__main__":
    main()
