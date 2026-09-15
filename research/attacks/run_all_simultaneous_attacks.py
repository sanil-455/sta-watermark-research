import json
import os
import math
import itertools
import torch
from transformers import AutoTokenizer

from simultaneous_edit_engine import apply_simultaneous_edits, compute_sta_stats
from semantic_checker import SemanticChecker
from generate_simultaneous_candidates import extract_simultaneous_candidates

MODEL_PATH = 'hf_models/Llama-2-7b-hf'
BASELINE_PATH = 'results/raw/baseline_safe.json'
OUTPUT_DIR = 'results/raw/manual_attacks'

TOP_K_DELETIONS = 5
TOP_K_INSERTIONS = 5

# Sample-specific manual full-word substitutions
SAMPLE_SUBSTITUTIONS = {
    0: [
        {'pos': 301, 'type': 'replace', 'new_token_id': 322, 'label': 'and->or'},
        {'pos': 345, 'type': 'replace', 'new_token_id': 1033, 'label': 'might->could'},
    ],
    1: [
        {'pos': 120, 'type': 'replace', 'new_token_id': 1033, 'label': 'might->could'},
        {'pos': 210, 'type': 'replace', 'new_token_id': 322, 'label': 'and->or'},
    ],
    2: [
        {'pos': 150, 'type': 'replace', 'new_token_id': 1033, 'label': 'might->could'},
        {'pos': 280, 'type': 'replace', 'new_token_id': 322, 'label': 'and->or'},
    ]
}


def process_sample(sample_idx, sample_data, tokenizer, checker):
    text = sample_data['watermarked_text']
    encoded = tokenizer(text, add_special_tokens=True, truncation=True, max_length=2048)
    orig_ids = encoded['input_ids']

    orig_stats = compute_sta_stats(orig_ids)
    print("=" * 70)
    print(f"PROCESSING SAMPLE {sample_idx + 1}")
    print(f"Original Z-Score: {orig_stats['z']:.4f} | Green Pairs: {orig_stats['green_count']}/{orig_stats['pair_count']}")

    candidates = extract_simultaneous_candidates(orig_ids, tokenizer)
    
    deletions = [{'pos': item['pos'], 'type': 'delete', 'label': f"del({item['token_str']})"} 
                 for item in candidates['candidate_deletions'][:TOP_K_DELETIONS]]

    insertions = [{'pos': item['pos'], 'type': 'insert', 'new_token_id': item['insert_token_id'], 'label': f"ins({item['token_str']})"} 
                  for item in candidates['candidate_insertions'][:TOP_K_INSERTIONS]]

    substitutions = SAMPLE_SUBSTITUTIONS.get(sample_idx, SAMPLE_SUBSTITUTIONS[0])

    all_combinations = list(itertools.product(substitutions, deletions, insertions))

    best_attack = None
    min_z_score = orig_stats['z']
    results = []

    for idx, (sub, dele, ins) in enumerate(all_combinations):
        positions = [sub['pos'], dele['pos'], ins['pos']]
        if len(positions) != len(set(positions)):
            continue

        combo_edits = [sub, dele, ins]
        modified_ids = apply_simultaneous_edits(orig_ids, combo_edits)
        attacked_text = tokenizer.decode(modified_ids, skip_special_tokens=True)

        retokenized_ids = tokenizer(attacked_text, add_special_tokens=True, truncation=True, max_length=2048)['input_ids']
        stats = compute_sta_stats(retokenized_ids)

        sim_score = checker.compute_similarity(text, attacked_text)
        edit_dist, edit_ratio = checker.compute_token_edit_distance(orig_ids, retokenized_ids)
        delta_z = orig_stats['z'] - stats['z']

        record = {
            'combo_id': idx + 1,
            'edits': [e['label'] for e in combo_edits],
            'z_score': stats['z'],
            'delta_z': delta_z,
            'green_count': stats['green_count'],
            'similarity': sim_score,
            'levenshtein_distance': edit_dist,
            'edit_ratio': edit_ratio,
            'detected': stats['detected']
        }
        results.append(record)

        if sim_score >= 0.90 and stats['z'] < min_z_score:
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

    if best_attack:
        print(f"Optimal Edits: {[e['label'] for e in best_attack['edits']]}")
        print(f"Attacked Z-Score: {best_attack['stats']['z']:.4f} (Delta Z: -{best_attack['delta_z']:.4f})")
        print(f"Cosine Similarity: {best_attack['similarity']:.4f} | Levenshtein Ratio: {best_attack['edit_ratio']:.4f}")
    else:
        print("[✗ WARNING] No edit combination passed semantic threshold.")

    summary = {
        'sample_index': sample_idx + 1,
        'original_z': orig_stats['z'],
        'best_attack': best_attack,
        'top_evaluated_combos': sorted(results, key=lambda x: x['z_score'])[:5]
    }

    out_file = os.path.join(OUTPUT_DIR, f"optimal_simultaneous_attack_sample{sample_idx + 1}.json")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"[✓ SAVED] Sample {sample_idx + 1} report: {out_file}\n")
    return summary


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    checker = SemanticChecker()

    with open(BASELINE_PATH, 'r') as f:
        baseline_samples = json.load(f)

    all_summaries = []
    # Execute across Sample 1 (index 0), Sample 2 (index 1), Sample 3 (index 2)
    for i in range(min(3, len(baseline_samples))):
        summary = process_sample(i, baseline_samples[i], tokenizer, checker)
        all_summaries.append(summary)

    consolidated_path = os.path.join(OUTPUT_DIR, "consolidated_samples_1_2_3_report.json")
    with open(consolidated_path, 'w') as f:
        json.dump(all_summaries, f, indent=2)

    print("=" * 70)
    print(f"[✓ COMPLETE] Consolidated report saved to: {consolidated_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()