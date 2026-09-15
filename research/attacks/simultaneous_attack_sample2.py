import json
import os
import itertools
from transformers import AutoTokenizer

from simultaneous_edit_engine import apply_simultaneous_edits, compute_sta_stats
from semantic_checker import SemanticChecker
from generate_simultaneous_candidates import extract_simultaneous_candidates

MODEL_PATH = 'hf_models/Llama-2-7b-hf'
BASELINE_PATH = 'results/raw/baseline_safe.json'
OUTPUT_PATH = 'results/raw/manual_attacks/optimal_simultaneous_attack_sample2.json'

TOP_K_DELETIONS = 5
TOP_K_INSERTIONS = 5

SAMPLE_2_SUBSTITUTIONS = [
    {'pos': 120, 'type': 'replace', 'new_token_id': 1033, 'label': 'might->could'},
    {'pos': 210, 'type': 'replace', 'new_token_id': 322, 'label': 'and->or'},
]


def run_sample2():
    print("=" * 70)
    print("RUNNING STANDALONE ATTACK OPTIMIZER ON SAMPLE 2")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    checker = SemanticChecker()

    with open(BASELINE_PATH, 'r') as f:
        baseline_data = json.load(f)[1] # Index 1 corresponding to Sample 2

    original_text = baseline_data['watermarked_text']
    orig_encoded = tokenizer(original_text, add_special_tokens=True, truncation=True, max_length=2048)
    orig_ids = orig_encoded['input_ids']

    orig_stats = compute_sta_stats(orig_ids)
    print(f"Sample 2 Original Z-Score: {orig_stats['z']:.4f} | Green Pairs: {orig_stats['green_count']}/{orig_stats['pair_count']}")

    candidates = extract_simultaneous_candidates(orig_ids, tokenizer)
    deletions = [{'pos': item['pos'], 'type': 'delete', 'label': f"del({item['token_str']})"} 
                 for item in candidates['candidate_deletions'][:TOP_K_DELETIONS]]
    insertions = [{'pos': item['pos'], 'type': 'insert', 'new_token_id': item['insert_token_id'], 'label': f"ins({item['token_str']})"} 
                  for item in candidates['candidate_insertions'][:TOP_K_INSERTIONS]]

    all_combinations = list(itertools.product(SAMPLE_2_SUBSTITUTIONS, deletions, insertions))
    
    best_attack = None
    min_z_score = orig_stats['z']

    for idx, (sub, dele, ins) in enumerate(all_combinations):
        positions = [sub['pos'], dele['pos'], ins['pos']]
        if len(positions) != len(set(positions)):
            continue

        combo_edits = [sub, dele, ins]
        modified_ids = apply_simultaneous_edits(orig_ids, combo_edits)
        attacked_text = tokenizer.decode(modified_ids, skip_special_tokens=True)

        retokenized_ids = tokenizer(attacked_text, add_special_tokens=True, truncation=True, max_length=2048)['input_ids']
        stats = compute_sta_stats(retokenized_ids)

        sim_score = checker.compute_similarity(original_text, attacked_text)
        edit_dist, edit_ratio = checker.compute_token_edit_distance(orig_ids, retokenized_ids)
        delta_z = orig_stats['z'] - stats['z']

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

    print("\nSAMPLE 2 OPTIMAL ATTACK RESULT")
    print(f"Applied Edits: {[e['label'] for e in best_attack['edits']]}")
    print(f"New Z-Score: {best_attack['stats']['z']:.4f} (Original: {orig_stats['z']:.4f})")
    print(f"Delta Z Drop: {best_attack['delta_z']:.4f}")
    print(f"Semantic Cosine Similarity: {best_attack['similarity']:.4f}")
    print(f"Levenshtein Edit Ratio: {best_attack['edit_ratio']:.4f}")
    print("=" * 70)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump({'original_z': orig_stats['z'], 'best_attack': best_attack}, f, indent=2)


if __name__ == '__main__':
    run_sample2()