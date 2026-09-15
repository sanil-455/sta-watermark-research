import json
import os
import math
import itertools
import torch
from transformers import AutoTokenizer

from simultaneous_edit_engine import (
    apply_simultaneous_edits,
    compute_sta_stats
)

from defensive_sensitivity_benchmark import SemanticChecker


MODEL_PATH = 'hf_models/Llama-2-7b-hf'
BASELINE_PATH = 'results/raw/baseline_safe.json'
MANIFEST_PATH = (
    'results/raw/manual_attacks/'
    'simultaneous_candidates_sample1.json'
)
OUTPUT_PATH = (
    'results/raw/manual_attacks/'
    'optimal_simultaneous_attack_sample1.json'
)

TOP_K_DELETIONS = 5
TOP_K_INSERTIONS = 5

TOP_SUBSTITUTIONS = [
    {
        'pos': 301,
        'type': 'replace',
        'new_token_id': 322,
        'label': 'and->or'
    },
    {
        'pos': 345,
        'type': 'replace',
        'new_token_id': 1033,
        'label': 'might->could'
    }
]


def load_candidate_pool(manifest_file):
    with open(manifest_file, 'r') as f:
        data = json.load(f)

    deletions = [
        {
            'pos': item['pos'],
            'type': 'delete',
            'label': f"del({item['token_str']})"
        }
        for item in data['candidate_deletions'][:TOP_K_DELETIONS]
    ]

    insertions = [
        {
            'pos': item['pos'],
            'type': 'insert',
            'new_token_id': item['insert_token_id'],
            'label': f"ins({item['token_str']})"
        }
        for item in data['candidate_insertions'][:TOP_K_INSERTIONS]
    ]

    return TOP_SUBSTITUTIONS, deletions, insertions


def evaluate_combinatorial_attacks():
    print("=" * 70)
    print("MODULE 3: COMBINATORIAL MULTI-EDIT ATTACK OPTIMIZER")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    checker = SemanticChecker()

    with open(BASELINE_PATH, 'r') as f:
        baseline_data = json.load(f)[0]

    original_text = baseline_data['watermarked_text']

    orig_encoded = tokenizer(
        original_text,
        add_special_tokens=True,
        truncation=True,
        max_length=2048
    )

    orig_ids = orig_encoded['input_ids']

    orig_stats = compute_sta_stats(orig_ids)

    print(
        f"Original Baseline Z-Score: "
        f"{orig_stats['z']:.4f} | "
        f"Green Pairs: "
        f"{orig_stats['green_count']}/"
        f"{orig_stats['pair_count']}"
    )

    substitutions, deletions, insertions = load_candidate_pool(
        MANIFEST_PATH
    )

    print(
        f"Loaded Candidate Pools: "
        f"{len(substitutions)} Substitutions, "
        f"{len(deletions)} Deletions, "
        f"{len(insertions)} Insertions."
    )

    all_combinations = list(
        itertools.product(
            substitutions,
            deletions,
            insertions
        )
    )

    print(
        f"Total Combinations to Evaluate: "
        f"{len(all_combinations)}"
    )

    print("-" * 70)

    best_attack = None
    min_z_score = orig_stats['z']
    results = []

    for idx, (sub, dele, ins) in enumerate(all_combinations):

        positions = [
            sub['pos'],
            dele['pos'],
            ins['pos']
        ]

        if len(positions) != len(set(positions)):
            continue

        combo_edits = [sub, dele, ins]

        modified_ids = apply_simultaneous_edits(
            orig_ids,
            combo_edits
        )

        attacked_text = tokenizer.decode(
            modified_ids,
            skip_special_tokens=True
        )

        retokenized_ids = tokenizer(
            attacked_text,
            add_special_tokens=True,
            truncation=True,
            max_length=2048
        )['input_ids']

        stats = compute_sta_stats(retokenized_ids)

        sim_score = checker.compute_similarity(
            original_text,
            attacked_text
        )

        edit_dist, edit_ratio = (
            checker.compute_token_edit_distance(
                orig_ids,
                retokenized_ids
            )
        )

        delta_z = orig_stats['z'] - stats['z']

        record = {
            'combo_id': idx + 1,
            'edits': [
                e['label']
                for e in combo_edits
            ],
            'z_score': stats['z'],
            'delta_z': delta_z,
            'green_count': stats['green_count'],
            'similarity': sim_score,
            'levenshtein_distance': edit_dist,
            'edit_ratio': edit_ratio,
            'detected': stats['detected']
        }

        results.append(record)

        if (
            sim_score >= 0.90
            and stats['z'] < min_z_score
        ):
            min_z_score = stats['z']

            best_attack = {
                'edits': combo_edits,
                'stats': stats,
                'similarity': sim_score,
                'levenshtein_distance': edit_dist,
                'edit_ratio': edit_ratio,
                'delta_z': delta_z,
                'attacked_text': attacked_text
            }

    print("\n" + "=" * 70)
    print("OPTIMAL SIMULTANEOUS ATTACK RESULTS")
    print("=" * 70)

    if best_attack:

        print(
            f"Applied Edits: "
            f"{[e['label'] for e in best_attack['edits']]}"
        )

        print(
            f"New Z-Score: "
            f"{best_attack['stats']['z']:.4f} "
            f"(Original: {orig_stats['z']:.4f})"
        )

        print(
            f"Delta Z Drop: "
            f"{best_attack['delta_z']:.4f}"
        )

        print(
            f"Green Count: "
            f"{best_attack['stats']['green_count']} / "
            f"{best_attack['stats']['pair_count']}"
        )

        print(
            f"Semantic Cosine Similarity: "
            f"{best_attack['similarity']:.4f}"
        )

        print(
            f"Levenshtein Edit Distance: "
            f"{best_attack['levenshtein_distance']} "
            f"tokens "
            f"({best_attack['edit_ratio']:.4f} ratio)"
        )

        print(
            "Watermark Detection Status: "
            f"{'DETECTED' if best_attack['stats']['detected'] else 'BROKEN (PASSED)'}"
        )

        print("=" * 70)

        os.makedirs(
            os.path.dirname(OUTPUT_PATH),
            exist_ok=True
        )

        with open(OUTPUT_PATH, 'w') as f:
            json.dump(
                {
                    'original_z': orig_stats['z'],
                    'best_attack': best_attack,
                    'top_evaluated_combos': sorted(
                        results,
                        key=lambda x: x['z_score']
                    )[:10]
                },
                f,
                indent=2
            )

        print(
            f"[✓ SAVED] Optimal attack report saved to: "
            f"{OUTPUT_PATH}\n"
        )

    else:
        print(
            "[✗ WARNING] No combination passed "
            "semantic similarity threshold (>= 0.90)."
        )


if __name__ == '__main__':
    evaluate_combinatorial_attacks()
