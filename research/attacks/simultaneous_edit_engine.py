import json
import math
import os
import torch
from transformers import AutoTokenizer

# STA PRG Constants (from STA Paper)
H1 = 15485863
H2 = 17624813
GAMMA = 0.5

MODEL_PATH = 'hf_models/Llama-2-7b-hf'
BASELINE_PATH = 'results/raw/baseline_safe.json'
OUTPUT_PATH = 'results/raw/manual_attacks/simultaneous_attack_sample1.json'


def compute_sta_stats(input_ids, rng_device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Calculates green pair count, pair count, Z-score, and detection state.
    """
    if len(input_ids) <= 1:
        return {'z': 0.0, 'green_count': 0, 'pair_count': 0, 'detected': False}

    rng = torch.Generator(device=rng_device)
    green_count = 0
    pair_count = len(input_ids) - 1

    for i in range(pair_count):
        prev_token = input_ids[i]
        curr_token = input_ids[i + 1]
        
        # PRG seed hashing formula
        seed = H1 * prev_token + H2 * curr_token
        rng.manual_seed(seed)
        
        # Evaluate green list condition
        random_val = torch.rand(1, device=rng_device, generator=rng).item()
        if random_val < GAMMA:
            green_count += 1

    # Theoretical mean and std dev under H0 (unwatermarked)
    expected_mean = GAMMA * pair_count
    std_dev = math.sqrt(GAMMA * (1.0 - GAMMA) * pair_count)
    
    z_score = (green_count - expected_mean) / std_dev if std_dev > 0 else 0.0

    return {
        'z': float(z_score),
        'green_count': green_count,
        'pair_count': pair_count,
        'detected': z_score > 2.0
    }


def apply_simultaneous_edits(token_list, edits):
    """
    Applies replacements, deletions, and insertions in right-to-left order.
    """
    # Sort edits descending by position index to maintain index stability
    sorted_edits = sorted(edits, key=lambda item: item['pos'], reverse=True)
    modified = list(token_list)

    for edit in sorted_edits:
        pos = edit['pos']
        op_type = edit['type']

        if op_type == 'replace':
            modified[pos] = edit['new_token_id']
        elif op_type == 'delete':
            del modified[pos]
        elif op_type == 'insert':
            modified.insert(pos + 1, edit['new_token_id'])

    return modified


def main():
    if not os.path.exists(BASELINE_PATH):
        raise FileNotFoundError(f"Baseline file missing at: {BASELINE_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    with open(BASELINE_PATH, 'r') as f:
        baseline_samples = json.load(f)

    sample = baseline_samples[0]
    original_text = sample['watermarked_text']
    
    encoded = tokenizer(original_text, add_special_tokens=True, truncation=True, max_length=2048)
    orig_ids = encoded['input_ids']

    # 1. Baseline Stats
    orig_stats = compute_sta_stats(orig_ids)
    print("=" * 60)
    print("MODULE 1: STA BASELINE EVALUATION")
    print(f"Original Z-Score: {orig_stats['z']:.4f}")
    print(f"Green Pair Count: {orig_stats['green_count']} / {orig_stats['pair_count']}")
    print(f"Watermark Detected: {orig_stats['detected']}")
    print("=" * 60)

    # 2. Test Edits (1 Replace, 1 Delete, 1 Insert)
    sample_edits = [
        {'type': 'replace', 'pos': 301, 'new_token_id': 322},    # Replace token at 301
        {'type': 'delete',  'pos': 305},                         # Delete token at 305
        {'type': 'insert',  'pos': 310, 'new_token_id': 29889}   # Insert period after 310
    ]

    # 3. Apply Edits and Retokenize
    modified_ids = apply_simultaneous_edits(orig_ids, sample_edits)
    attacked_text = tokenizer.decode(modified_ids, skip_special_tokens=True)
    
    retokenized_ids = tokenizer(attacked_text, add_special_tokens=True, truncation=True, max_length=2048)['input_ids']
    attacked_stats = compute_sta_stats(retokenized_ids)

    print("\nSIMULTANEOUS ATTACK RESULTS")
    print(f"Attacked Z-Score: {attacked_stats['z']:.4f}")
    print(f"Green Pair Count: {attacked_stats['green_count']} / {attacked_stats['pair_count']}")
    print(f"Delta Z: {orig_stats['z'] - attacked_stats['z']:.4f}")
    print(f"Watermark Detected: {attacked_stats['detected']}")
    print("=" * 60)

    # 4. Save Artifact
    output_payload = {
        'sample': 1,
        'attack_type': 'simultaneous_replace_insert_delete',
        'edits_applied': sample_edits,
        'original_stats': orig_stats,
        'attacked_stats': attacked_stats,
        'original_text': original_text,
        'attacked_text': attacked_text
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output_payload, f, indent=2)

    print(f"[✓ SAVED] Module 1 result written to: {OUTPUT_PATH}\n")


if __name__ == '__main__':
    main()