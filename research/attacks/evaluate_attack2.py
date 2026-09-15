import json
import math
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


BASELINE_PATH = Path("results/raw/baseline_safe.json")
CANDIDATE_DIR = Path("results/raw/attack2_candidates")
RESULT_DIR = Path("results/raw/attack2_results")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

SEMANTIC_THRESHOLD = 0.90
BATCH_SIZE = 32

GAMMA = 0.5
Z_THRESHOLD = 2.0
HASH_KEY1 = 15485863
HASH_KEY2 = 17624813
PROMPT_MAX_LENGTH = 2048


def load_json(path):
    with path.open() as f:
        return json.load(f)


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

        seed = HASH_KEY1 * now_token + HASH_KEY2 * next_token
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
    variance = GAMMA * (1.0 - GAMMA) * pair_count

    if variance == 0:
        z = 0.0
    else:
        z = (green_count - expected) / math.sqrt(variance)

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
        " ".join(words[i:i + words_per_chunk])
        for i in range(0, len(words), words_per_chunk)
    ]


def changed_regions(original, attacked, window=500):
    from difflib import SequenceMatcher

    matcher = SequenceMatcher(None, original, attacked)

    changed = [
        op for op in matcher.get_opcodes()
        if op[0] != "equal"
    ]

    if not changed:
        return original, attacked

    first = changed[0]

    original_start = max(0, first[1] - window)
    original_end = min(len(original), first[2] + window)

    attacked_start = max(0, first[3] - window)
    attacked_end = min(len(attacked), first[4] + window)

    return (
        original[original_start:original_end],
        attacked[attacked_start:attacked_end],
    )


def semantic_scores(model, original, attacked):
    original_chunks = chunk_text(original)
    attacked_chunks = chunk_text(attacked)

    all_texts = original_chunks + attacked_chunks

    embeddings = model.encode(
        all_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    original_embedding = embeddings[:len(original_chunks)].mean(axis=0)
    attacked_embedding = embeddings[len(original_chunks):].mean(axis=0)

    original_embedding /= np.linalg.norm(original_embedding)
    attacked_embedding /= np.linalg.norm(attacked_embedding)

    global_score = float(np.dot(
        original_embedding,
        attacked_embedding,
    ))

    original_region, attacked_region = changed_regions(
        original,
        attacked,
    )

    local_embeddings = model.encode(
        [original_region, attacked_region],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    local_score = float(np.dot(
        local_embeddings[0],
        local_embeddings[1],
    ))

    return global_score, local_score


def process_sample(sample, model, tokenizer):
    sample_id = int(sample["prompt_id"])
    candidate_path = CANDIDATE_DIR / f"sample_{sample_id}.json"

    data = load_json(candidate_path)
    candidates = data["candidates"]

    output_path = RESULT_DIR / f"sample_{sample_id}.jsonl"
    summary_path = RESULT_DIR / f"sample_{sample_id}_summary.json"

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    existing = {}

    if output_path.exists():
        with output_path.open() as f:
            for line in f:
                if not line.strip():
                    continue

                result = json.loads(line)
                existing[result["candidate_index"]] = result

    baseline_z = float(sample["watermarked_z"])

    baseline_check = sta_z(
        sample["watermarked_text"],
        tokenizer,
    )

    if abs(baseline_check["z"] - baseline_z) > 1e-6:
        raise RuntimeError(
            f"Sample {sample_id}: baseline z mismatch. "
            f"Stored={baseline_z}, recomputed={baseline_check['z']}"
        )

    print(
        f"Sample {sample_id}: {len(candidates)} candidates "
        f"(already saved: {len(existing)})"
    )

    new_results = []

    for batch_start in range(0, len(candidates), BATCH_SIZE):
        batch = candidates[
            batch_start:batch_start + BATCH_SIZE
        ]

        batch_results = []

        for offset, candidate in enumerate(batch):
            index = batch_start + offset

            if index in existing:
                continue

            try:
                global_score, local_score = semantic_scores(
                    model,
                    sample["watermarked_text"],
                    candidate["attacked_text"],
                )

                semantic_pass = (
                    global_score >= SEMANTIC_THRESHOLD
                    and local_score >= SEMANTIC_THRESHOLD
                )

                result = {
                    "candidate_index": index,
                    "sample_id": sample_id,
                    "attack_type": candidate["attack_type"],
                    "original_word": candidate["original_word"],
                    "replacement": candidate["replacement"],
                    "position": candidate["position"],
                    "global_similarity": global_score,
                    "local_similarity": local_score,
                    "semantic_threshold": SEMANTIC_THRESHOLD,
                    "semantic_pass": semantic_pass,
                }

                if semantic_pass:
                    attacked_check = sta_z(
                        candidate["attacked_text"],
                        tokenizer,
                    )

                    attacked_z = attacked_check["z"]

                    result.update({
                        "original_z": baseline_z,
                        "attacked_z": attacked_z,
                        "delta_z": attacked_z - baseline_z,
                        "original_detected": baseline_z > Z_THRESHOLD,
                        "attacked_detected": attacked_z > Z_THRESHOLD,
                        "watermark_break": (
                            baseline_z > Z_THRESHOLD
                            and attacked_z <= Z_THRESHOLD
                        ),
                        "original_green_count": baseline_check["green_count"],
                        "attacked_green_count": attacked_check["green_count"],
                        "original_pair_count": baseline_check["pair_count"],
                        "attacked_pair_count": attacked_check["pair_count"],
                        "original_token_count": baseline_check["token_count"],
                        "attacked_token_count": attacked_check["token_count"],
                    })
                else:
                    result.update({
                        "original_z": baseline_z,
                        "attacked_z": None,
                        "delta_z": None,
                        "original_detected": baseline_z > Z_THRESHOLD,
                        "attacked_detected": None,
                        "watermark_break": False,
                    })

                result["status"] = "ok"

            except Exception as exc:
                result = {
                    "candidate_index": index,
                    "sample_id": sample_id,
                    "attack_type": candidate["attack_type"],
                    "original_word": candidate["original_word"],
                    "replacement": candidate["replacement"],
                    "position": candidate["position"],
                    "status": "error",
                    "error": repr(exc),
                    "watermark_break": False,
                }

            batch_results.append(result)

        if batch_results:
            with output_path.open("a") as f:
                for result in batch_results:
                    f.write(json.dumps(result) + "\n")
                    f.flush()

                    print(
                        f"[✓ SAVED] Sample {sample_id}, "
                        f"candidate {result['candidate_index'] + 1}/"
                        f"{len(candidates)}"
                    )

                    new_results.append(result)

    all_results = list(existing.values()) + new_results

    valid_semantic = [
        r for r in all_results
        if r.get("status") == "ok"
        and r.get("semantic_pass") is True
    ]

    breaks = [
        r for r in valid_semantic
        if r.get("watermark_break") is True
    ]

    strongest = sorted(
        valid_semantic,
        key=lambda r: r["delta_z"],
    )[:20]

    summary = {
        "sample_id": sample_id,
        "baseline_z": baseline_z,
        "baseline_detected": baseline_z > Z_THRESHOLD,
        "candidate_count": len(candidates),
        "completed_count": len(all_results),
        "semantic_pass_count": len(valid_semantic),
        "watermark_break_count": len(breaks),
        "strongest_attacks": strongest,
    }

    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"[✓ SAVED] Sample {sample_id} summary -> {summary_path}"
    )

    return summary


def main():
    from transformers import AutoTokenizer

    print("Loading semantic model...")
    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu",
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "hf_models/Llama-2-7b-hf",
        local_files_only=True,
    )

    baselines = load_json(BASELINE_PATH)

    summaries = []

    for sample in baselines:
        sample_id = int(sample["prompt_id"])

        if sample_id not in {1, 2, 3}:
            continue

        summaries.append(
            process_sample(
                sample,
                model,
                tokenizer,
            )
        )

    combined_path = RESULT_DIR / "attack2_summary.json"

    with combined_path.open("w") as f:
        json.dump(summaries, f, indent=2)

    print(f"[✓ SAVED] Combined summary -> {combined_path}")


if __name__ == "__main__":
    main()
