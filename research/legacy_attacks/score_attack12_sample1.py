import json
import math
import re
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer


BASELINE_PATH = Path(
    "results/raw/baseline_safe.json"
)

COMBO_PATH = Path(
    "results/raw/attack12_combined/"
    "sample_1_all_combinations.jsonl"
)

OUT_PATH = Path(
    "results/raw/attack12_scored/"
    "sample_1_scored.jsonl"
)

SUMMARY_PATH = Path(
    "results/raw/attack12_scored/"
    "sample_1_scored_summary.json"
)

TOKENIZER_PATH = "hf_models/Llama-2-7b-hf"

SEMANTIC_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

PROMPT_ID = 0
SAMPLE_NUMBER = 1

GAMMA = 0.5
Z_THRESHOLD = 2.0
MAX_LENGTH = 2048

HASH_KEY1 = 15485863
HASH_KEY2 = 17624813

WORD_RE = re.compile(
    r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*"
)


def load_json(path):
    with path.open() as f:
        return json.load(f)


def sta_stats(text, tokenizer, rng):
    """
    STA-1 detection statistics.

    Green/red is decided per adjacent token-ID pair using a
    seed derived from both IDs. z follows Eq. (3) of the
    paper: (|S|_G - gamma*T) / sqrt(gamma*(1-gamma)*T),
    where T is the number of scored pairs.
    """

    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )

    token_ids = encoded["input_ids"]

    if len(token_ids) < 2:
        raise RuntimeError(
            "Text has fewer than 2 tokens."
        )

    green_count = 0
    pair_count = len(token_ids) - 1

    for i in range(pair_count):

        now_token = int(token_ids[i])
        next_token = int(token_ids[i + 1])

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
        green_count - GAMMA * pair_count
    ) / math.sqrt(
        GAMMA * (1.0 - GAMMA) * pair_count
    )

    return {
        "z": float(z),
        "green_count": int(green_count),
        "pair_count": int(pair_count),
        "token_count": int(len(token_ids)),
    }


def sentence_parts(text):
    parts = [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+", text)
        if part.strip()
    ]

    return parts if parts else [text]


def semantic_scores(model, original, attacked):
    """
    Global similarity compares full texts.

    Local similarity compares sentence-aligned pairs, which
    catches damage confined to one sentence that a global
    embedding would average away. When sentence counts
    differ, alignment is undefined and local falls back to
    global; the fallback is flagged in the output.
    """

    embeddings = model.encode(
        [original, attacked],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    global_similarity = float(
        np.dot(embeddings[0], embeddings[1])
    )

    original_sentences = sentence_parts(original)
    attacked_sentences = sentence_parts(attacked)

    if len(original_sentences) != len(attacked_sentences):
        return (
            global_similarity,
            global_similarity,
            True,
        )

    scores = []

    for original_sentence, attacked_sentence in zip(
        original_sentences,
        attacked_sentences,
    ):
        pair_embeddings = model.encode(
            [original_sentence, attacked_sentence],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        scores.append(
            float(
                np.dot(
                    pair_embeddings[0],
                    pair_embeddings[1],
                )
            )
        )

    return (
        global_similarity,
        float(np.mean(scores)),
        False,
    )


def load_completed(path):
    """
    Resume support. Returns the set of already-scored keys
    and the rows themselves.

    The attacked text is part of the key so that an old result
    for the same A1/A2 identifiers cannot be reused if the
    combination text itself differs.
    """

    if not path.exists():
        return set(), []

    keys = set()
    rows = []

    with path.open() as f:
        for line in f:
            if not line.strip():
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # Truncated final line from an interrupted
                # run. Drop it; it will be rescored.
                continue

            key = (
                row.get("a1_replacement"),
                row.get("a2_candidate_index"),
                row.get("attacked_text"),
            )

            if key in keys:
                continue

            keys.add(key)
            rows.append(row)

    return keys, rows


def main():

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required by the STA RNG."
        )

    # --------------------------------------------------
    # Baseline
    # --------------------------------------------------

    baseline_rows = load_json(BASELINE_PATH)

    matches = [
        r for r in baseline_rows
        if int(r["prompt_id"]) == PROMPT_ID
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Could not uniquely identify "
            f"prompt_id={PROMPT_ID}"
        )

    baseline_row = matches[0]

    baseline_text = baseline_row["watermarked_text"]

    stored_z = float(
        baseline_row["watermarked_z"]
    )

    print("========== SETUP ==========")
    print("Loading Llama-2 tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_PATH,
        local_files_only=True,
    )

    rng = torch.Generator(device="cuda")

    baseline_stats = sta_stats(
        baseline_text,
        tokenizer,
        rng,
    )

    baseline_z = baseline_stats["z"]

    if abs(baseline_z - stored_z) > 1e-6:
        raise RuntimeError(
            f"Detector verification failed. "
            f"Stored z={stored_z}, "
            f"recomputed z={baseline_z}. "
            f"Refusing to score."
        )

    print(
        f"[OK] Detector verified: "
        f"z={baseline_z:.12f}"
    )

    print(
        f"     green={baseline_stats['green_count']} "
        f"pairs={baseline_stats['pair_count']} "
        f"tokens={baseline_stats['token_count']}"
    )

    baseline_word_count = len(
        WORD_RE.findall(baseline_text)
    )

    baseline_char_count = len(baseline_text)

    baseline_detected = baseline_z > Z_THRESHOLD

    if not baseline_detected:
        raise RuntimeError(
            "Baseline is not detected. A watermark "
            "break is undefined for this sample."
        )

    # --------------------------------------------------
    # Combined texts
    # --------------------------------------------------

    if not COMBO_PATH.exists():
        raise RuntimeError(
            f"Missing combination file: {COMBO_PATH}"
        )

    combos = []

    with COMBO_PATH.open() as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                combos.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Malformed JSON at "
                    f"{COMBO_PATH}:{line_number}: {exc}"
                )

    print(f"Combinations to score: {len(combos)}")

    print("Loading semantic model...")

    semantic_model = SentenceTransformer(
        SEMANTIC_MODEL,
        device="cpu",
    )

    OUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    completed_keys, completed_rows = load_completed(
        OUT_PATH
    )

    print(
        f"Previously scored: {len(completed_keys)}"
    )

    scored = list(completed_rows)

    # --------------------------------------------------
    # Score
    # --------------------------------------------------

    print()
    print("========== SCORING ==========")

    with OUT_PATH.open("a") as out:

        for index, combo in enumerate(combos, start=1):

            attacked_text_for_key = combo.get(
                "combined_text"
            )

            key = (
                combo.get("a1_replacement"),
                combo.get("a2_candidate_index"),
                attacked_text_for_key,
            )

            if key in completed_keys:
                continue

            row = {
                "sample_number": SAMPLE_NUMBER,
                "prompt_id": PROMPT_ID,
                "a1_original_word":
                    combo.get("a1_original_word"),
                "a1_replacement":
                    combo.get("a1_replacement"),
                "a2_candidate_index":
                    combo.get("a2_candidate_index"),
                "a2_attack_type":
                    combo.get("a2_attack_type"),
                "a2_original_word":
                    combo.get("a2_original_word"),
                "a2_position":
                    combo.get("a2_position"),
                "a2_replacement":
                    combo.get("a2_replacement"),
                "status": "error",
                "watermark_break": False,
            }

            try:

                attacked_text = combo["combined_text"]

                if attacked_text == baseline_text:
                    raise RuntimeError(
                        "Combined text equals baseline."
                    )

                attacked_stats = sta_stats(
                    attacked_text,
                    tokenizer,
                    rng,
                )

                attacked_z = attacked_stats["z"]

                (
                    global_similarity,
                    local_similarity,
                    local_fallback,
                ) = semantic_scores(
                    semantic_model,
                    baseline_text,
                    attacked_text,
                )

                attacked_word_count = len(
                    WORD_RE.findall(attacked_text)
                )

                attacked_char_count = len(attacked_text)

                attack_type = combo.get(
                    "a2_attack_type"
                )

                insertions = int(
                    attack_type == "insertion"
                )

                deletions = int(
                    attack_type == "deletion"
                )

                # Diagnostic: z recomputed holding the pair
                # count at its baseline value. Isolates the
                # green-flip effect from the denominator
                # shift caused by insertion/deletion.
                reference_pairs = baseline_stats[
                    "pair_count"
                ]

                z_at_original_pairs = (
                    attacked_stats["green_count"]
                    - GAMMA * reference_pairs
                ) / math.sqrt(
                    GAMMA
                    * (1.0 - GAMMA)
                    * reference_pairs
                )

                row.update(
                    {
                        "status": "ok",

                        "baseline_z": baseline_z,
                        "attacked_z": attacked_z,
                        "delta_z":
                            attacked_z - baseline_z,

                        "baseline_green_count":
                            baseline_stats[
                                "green_count"
                            ],
                        "attacked_green_count":
                            attacked_stats[
                                "green_count"
                            ],
                        "green_count_change":
                            attacked_stats[
                                "green_count"
                            ]
                            - baseline_stats[
                                "green_count"
                            ],

                        "baseline_pair_count":
                            baseline_stats[
                                "pair_count"
                            ],
                        "attacked_pair_count":
                            attacked_stats[
                                "pair_count"
                            ],
                        "pair_count_change":
                            attacked_stats[
                                "pair_count"
                            ]
                            - baseline_stats[
                                "pair_count"
                            ],

                        "baseline_token_count":
                            baseline_stats[
                                "token_count"
                            ],
                        "attacked_token_count":
                            attacked_stats[
                                "token_count"
                            ],
                        "token_count_change":
                            attacked_stats[
                                "token_count"
                            ]
                            - baseline_stats[
                                "token_count"
                            ],

                        "z_at_original_pair_count":
                            float(z_at_original_pairs),
                        "delta_z_at_original_pair_count":
                            float(
                                z_at_original_pairs
                                - baseline_z
                            ),

                        "baseline_word_count":
                            baseline_word_count,
                        "attacked_word_count":
                            attacked_word_count,
                        "word_count_change":
                            attacked_word_count
                            - baseline_word_count,

                        "baseline_char_count":
                            baseline_char_count,
                        "attacked_char_count":
                            attacked_char_count,
                        "char_count_change":
                            attacked_char_count
                            - baseline_char_count,

                        "global_similarity":
                            global_similarity,
                        "local_similarity":
                            local_similarity,
                        "local_similarity_is_fallback":
                            local_fallback,

                        "substitutions": 1,
                        "insertions": insertions,
                        "deletions": deletions,
                        "total_operations": 2,
                        "edit_rate":
                            2.0 / baseline_word_count,

                        "baseline_detected":
                            baseline_detected,
                        "attacked_detected":
                            attacked_z > Z_THRESHOLD,
                        "watermark_break":
                            bool(
                                baseline_detected
                                and attacked_z
                                <= Z_THRESHOLD
                            ),

                        "attacked_text":
                            attacked_text,
                    }
                )

            except Exception as exc:

                row["error_type"] = type(exc).__name__
                row["error"] = str(exc)

                print(
                    f"[! FAILED] {index}/{len(combos)} "
                    f"| {type(exc).__name__}: {exc}"
                )

            out.write(json.dumps(row) + "\n")
            out.flush()

            completed_keys.add(key)
            scored.append(row)

            if row["status"] == "ok":
                marker = (
                    " *** BREAK ***"
                    if row["watermark_break"]
                    else ""
                )

                print(
                    f"[{index}/{len(combos)}] "
                    f"{row['a1_original_word']}->"
                    f"{row['a1_replacement']} + "
                    f"{row['a2_attack_type']}"
                    f"@{row['a2_position']} | "
                    f"z={row['attacked_z']:.4f} "
                    f"dz={row['delta_z']:+.4f} "
                    f"g={row['green_count_change']:+d} "
                    f"p={row['pair_count_change']:+d}"
                    f"{marker}"
                )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    ok_rows = [
        r for r in scored
        if r.get("status") == "ok"
    ]

    error_rows = [
        r for r in scored
        if r.get("status") != "ok"
    ]

    breaks = [
        r for r in ok_rows
        if r.get("watermark_break")
    ]

    ranked = sorted(
        ok_rows,
        key=lambda r: r["delta_z"],
    )

    if ok_rows:
        deltas = [r["delta_z"] for r in ok_rows]
        min_z = min(r["attacked_z"] for r in ok_rows)
        mean_delta = float(np.mean(deltas))
        median_delta = float(np.median(deltas))
        best_delta = min(deltas)
    else:
        min_z = None
        mean_delta = None
        median_delta = None
        best_delta = None

    summary = {
        "sample_number": SAMPLE_NUMBER,
        "prompt_id": PROMPT_ID,
        "z_threshold": Z_THRESHOLD,
        "gamma": GAMMA,
        "tokenizer": TOKENIZER_PATH,
        "tokenizer_max_length": MAX_LENGTH,
        "hash_key1": HASH_KEY1,
        "hash_key2": HASH_KEY2,

        "baseline_z": baseline_z,
        "baseline_stored_z": stored_z,
        "baseline_green_count":
            baseline_stats["green_count"],
        "baseline_pair_count":
            baseline_stats["pair_count"],
        "baseline_token_count":
            baseline_stats["token_count"],
        "baseline_word_count": baseline_word_count,
        "baseline_detected": baseline_detected,

        "total_combinations": len(combos),
        "scored": len(scored),
        "ok": len(ok_rows),
        "errors": len(error_rows),
        "watermark_breaks": len(breaks),

        "min_attacked_z": min_z,
        "best_delta_z": best_delta,
        "mean_delta_z": mean_delta,
        "median_delta_z": median_delta,

        "top_50_by_delta_z": [
            {
                k: v
                for k, v in r.items()
                if k != "attacked_text"
            }
            for r in ranked[:50]
        ],

        "break_records": [
            {
                k: v
                for k, v in r.items()
                if k != "attacked_text"
            }
            for r in breaks
        ],

        "error_records": error_rows,
    }

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with SUMMARY_PATH.open("w") as f:
        json.dump(summary, f, indent=2)

    # --------------------------------------------------
    # Report
    # --------------------------------------------------

    print()
    print("========== RESULT ==========")
    print(f"Baseline z:        {baseline_z:.6f}")
    print(f"Threshold:         {Z_THRESHOLD}")
    print(f"Scored:            {len(scored)}")
    print(f"OK:                {len(ok_rows)}")
    print(f"Errors:            {len(error_rows)}")
    print(f"Watermark breaks:  {len(breaks)}")

    if ok_rows:
        print(f"Lowest z reached:  {min_z:.6f}")
        print(f"Best delta z:      {best_delta:+.6f}")
        print(f"Mean delta z:      {mean_delta:+.6f}")
        print(f"Median delta z:    {median_delta:+.6f}")

        print()
        print("Top 20 by delta z:")

        for rank, r in enumerate(ranked[:20], start=1):
            print(
                f"{rank:3d}. "
                f"{r['a1_original_word']}->"
                f"{r['a1_replacement']} + "
                f"{r['a2_attack_type']}"
                f"@{r['a2_position']} "
                f"{r['a2_original_word'] or r['a2_replacement']!r} "
                f"| z={r['attacked_z']:.4f} "
                f"dz={r['delta_z']:+.4f} "
                f"g={r['green_count_change']:+d} "
                f"p={r['pair_count_change']:+d} "
                f"gsim={r['global_similarity']:.4f} "
                f"lsim={r['local_similarity']:.4f} "
                f"break={r['watermark_break']}"
            )

    if breaks:
        print()
        print(f"*** {len(breaks)} WATERMARK BREAK(S) ***")
    else:
        print()
        print(
            "No watermark break. STA-1 held above "
            f"z={Z_THRESHOLD} for all scored "
            "combinations."
        )

    print()
    print(f"[SAVED] {OUT_PATH}")
    print(f"[SAVED] {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
