import json
import math
import os
import itertools
import torch
from transformers import AutoTokenizer

from simultaneous_edit_engine import apply_simultaneous_edits, compute_sta_stats
from semantic_checker import SemanticChecker

# -----------------------------------------------------------------------------
# STA Constants & Model Paths
# -----------------------------------------------------------------------------
H1 = 15485863
H2 = 17624813
GAMMA = 0.5

MODEL_PATH = 'hf_models/Llama-2-7b-hf'
BASELINE_PATH = 'results/raw/baseline_safe.json'
OUTPUT_DIR = 'results/raw/automated_attacks'

# Dictionary mapping common English functional words to context-preserving synonyms
SYNONYM_DICTIONARY = {
    'might': ['could', 'may'],
    'could': ['might', 'would'],
    'and': ['or', 'as well as'],
    'or': ['and'],
    'first': ['initial', 'primary'],
    'small': ['little', 'compact'],
    'large': ['big', 'huge'],
    'company': ['firm', 'business'],
    'car': ['vehicle', 'auto'],
    'help': ['assist', 'aid'],
    'fast': ['quick', 'swift']
}


def is_green_pair(token_a, token_b, rng_device='cuda' if torch.cuda.is_available() else 'cpu'):
    rng = torch.Generator(device=rng_device)
    seed = H1 * token_a + H2 * token_b
    rng.manual_seed(seed)
    return torch.rand(1, device=rng_device, generator=rng).item() < GAMMA


def extract_automated_candidates(input_ids, tokenizer, top_k=5):
    """
    Scans the entire token sequence to automatically discover high-impact
    Substitutions, Deletions, and Insertions.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n = len(input_ids)
    
    candidate_substitutions = []
    candidate_deletions = []
    candidate_insertions = []

    punctuation_ids = [29892, 29889, 259]  # Comma, Period, Space

    for i in range(1, n - 1):
        prev_id = input_ids[i - 1]
        curr_id = input_ids[i]
        next_id = input_ids[i + 1]

        token_str = tokenizer.decode([curr_id]).strip().lower()

        # 1. Automated Dynamic Substitutions
        if token_str in SYNONYM_DICTIONARY:
            for syn in SYNONYM_DICTIONARY[token_str]:
                syn_tokens = tokenizer.encode(syn, add_special_tokens=False)
                if len(syn_tokens) == 1:
                    syn_id = syn_tokens[0]
                    # Check if substitution destroys green pair
                    g1_orig = is_green_pair(prev_id, curr_id, device)
                    g2_orig = is_green_pair(curr_id, next_id, device)
                    g1_new = is_green_pair(prev_id, syn_id, device)
                    g2_new = is_green_pair(syn_id, next_id, device)

                    drop = (int(g1_orig) + int(g2_orig)) - (int(g1_new) + int(g2_new))
                    if drop > 0:
                        candidate_substitutions.append({
                            'pos': i,
                            'type': 'replace',
                            'new_token_id': syn_id,
                            'label': f"{token_str}->{syn}",
                            'green_drop': drop
                        })

        # 2. Automated Deletions (only on whole words)
        if len(token_str) > 1 and token_str.isalnum():
            g1 = is_green_pair(prev_id, curr_id, device)
            g2 = is_green_pair(curr_id, next_id, device)
            g_merged = is_green_pair(prev_id, next_id, device)
            if g1 and g2 and not g_merged:
                candidate_deletions.append({
                    'pos': i,
                    'type': 'delete',
                    'label': f"del({token_str})",
                    'green_drop': 2
                })

        # 3. Automated Insertions
        for p_id in punctuation_ids:
            g_orig = is_green_pair(curr_id, next_id, device)
            if g_orig:
                g1_ins = is_green_pair(curr_id, p_id, device)
                g2_ins = is_green_pair(p_id, next_id, device)
                if not g1_ins and not g2_ins:
                    p_str = tokenizer.decode([p_id])
                    candidate_insertions.append({
                        'pos': i,
                        'type': 'insert',
                        'new_token_id': p_id,
                        'label': f"ins({p_str})",
                        'green_drop': 1
                    })
                    break

    # Rank candidates by green pair reduction impact
    sorted_subs = sorted(candidate_substitutions, key=lambda x: x['green_drop'], reverse=True)[:top_k]
    sorted_dels = sorted(candidate_deletions, key=lambda x: x['green_drop'], reverse=True)[:top_k]
    sorted_inss = sorted(candidate_insertions, key=lambda x: x['green_drop'], reverse=True)[:top_k]

    return sorted_subs, sorted_dels, sorted_inss


def run_automated_pipeline():
    print("=" * 70)
    print("RUNNING AUTOMATED MULTI-EDIT ATTACK PIPELINE")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    checker = SemanticChecker()

    if not os.path.exists(BASELINE_PATH):
        raise FileNotFoundError(f"Missing baseline file at {BASELINE_PATH}")

    with open(BASELINE_PATH, 'r') as f:
        samples = json.load(f)

    all_results = []

    for idx, sample in enumerate(samples[:3]):
        text = sample['watermarked_text']
        encoded = tokenizer(text, add_special_tokens=True, truncation=True, max_length=2048)
        orig_ids = encoded['input_ids']

        orig_stats = compute_sta_stats(orig_ids)
        print(f"\n[SAMPLE {idx + 1}] Tokens: {len(orig_ids)} | Baseline Z: {orig_stats['z']:.4f} | Green: {orig_stats['green_count']}/{orig_stats['pair_count']}")

        subs, dels, inss = extract_automated_candidates(orig_ids, tokenizer, top_k=5)
        print(f"Discovered Candidates: {len(subs)} Substitutions, {len(dels)} Deletions, {len(inss)} Insertions")

        if not subs or not dels or not inss:
            print(f"[✗ SKIPPED] Insufficient candidate combinations found for Sample {idx + 1}")
            continue

        all_combinations = list(itertools.product(subs, dels, inss))
        best_attack = None
        min_z_score = orig_stats['z']

        for sub, dele, ins in all_combinations:
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

            if sim_score >= 0.90 and stats['z'] < min_z_score:
                min_z_score = stats['z']
                best_attack = {
                    'edits': combo_edits,
                    'stats': stats,
                    'similarity': sim_score,
                    'levenshtein_distance': edit_dist,
                    'edit_ratio': edit_ratio,
                    'delta_z': orig_stats['z'] - stats['z'],
                    'attacked_text': attacked_text
                }

        if best_attack:
            print(f"Optimal Edits: {[e['label'] for e in best_attack['edits']]}")
            print(f"New Z-Score: {best_attack['stats']['z']:.4f} (Delta Z: -{best_attack['delta_z']:.4f})")
            print(f"Cosine Similarity: {best_attack['similarity']:.4f} | Levenshtein Ratio: {best_attack['edit_ratio']:.4f}")

            all_results.append({
                'sample': idx + 1,
                'original_z': orig_stats['z'],
                'best_attack': best_attack
            })

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    consolidated_file = os.path.join(OUTPUT_DIR, "automated_attack_summary.json")
    with open(consolidated_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 70)
    print(f"[✓ COMPLETE] Automated attack summary saved to: {consolidated_file}")
    print("=" * 70)


if __name__ == '__main__':
    run_automated_pipeline()