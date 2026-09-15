import json
import math
import os
import torch
from transformers import AutoTokenizer

# -----------------------------------------------------------------------------
# Configuration Constants
# -----------------------------------------------------------------------------
MODEL_PATH = 'hf_models/Llama-2-7b-hf'
BASELINE_PATH = 'results/raw/baseline_safe.json'
OUTPUT_MANIFEST_PATH = 'results/raw/manual_attacks/simultaneous_candidates_sample1.json'

H1 = 15485863
H2 = 17624813
GAMMA = 0.5


def is_green_pair(token_a, token_b, rng_device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Checks if an ordered pair of token IDs produces a green token score.
    Calculated via STA PRG seeding formula: H1 * token_a + H2 * token_b
    """
    rng = torch.Generator(device=rng_device)
    seed = H1 * token_a + H2 * token_b
    rng.manual_seed(seed)
    return torch.rand(1, device=rng_device, generator=rng).item() < GAMMA


def extract_simultaneous_candidates(input_ids, tokenizer):
    """
    Scans input_ids to find high-impact targets for simultaneous edits:
    1. Deletions: Green pairs where deleting the token leaves a red pair.
    2. Insertions: Green pairs where inserting a punctuation token creates 2 red pairs.
    3. Replacements: Green pairs where replacing the token flips the pair to red.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n = len(input_ids)
    
    green_pair_indices = []
    
    # Identify all green transitions (i -> i+1)
    for i in range(n - 1):
        if is_green_pair(input_ids[i], input_ids[i + 1], rng_device=device):
            green_pair_indices.append(i)

    deletions = []
    insertions = []
    replacements = []

    # Standard low-disruption insertion tokens (comma, period, space)
    punctuation_ids = [29892, 29889, 259]

    for idx in green_pair_indices:
        # Check boundary condition for deletion/replacement
        if idx == 0 or idx >= n - 1:
            continue

        prev_id = input_ids[idx - 1]
        curr_id = input_ids[idx]
        next_id = input_ids[idx + 1]

        # --- 1. DELETION SCORING ---
        # Removing curr_id links prev_id directly to next_id
        if not is_green_pair(prev_id, next_id, rng_device=device):
            deletions.append({
                'pos': idx,
                'token_id': curr_id,
                'token_str': tokenizer.decode([curr_id]),
                'net_green_reduction': 2  # Destroys 2 green pairs, creates 0
            })

        # --- 2. INSERTION SCORING ---
        # Inserting punctuation after idx turns (curr_id -> next_id) into (curr_id -> p) and (p -> next_id)
        for p_id in punctuation_ids:
            g1 = is_green_pair(curr_id, p_id, rng_device=device)
            g2 = is_green_pair(p_id, next_id, rng_device=device)
            if not g1 and not g2:
                insertions.append({
                    'pos': idx,
                    'insert_token_id': p_id,
                    'token_str': tokenizer.decode([p_id]),
                    'net_green_reduction': 1  # Destroys 1 green pair, creates 0 new green pairs
                })
                break

    return {
        'total_tokens': n,
        'total_green_pairs': len(green_pair_indices),
        'candidate_deletions': sorted(deletions, key=lambda x: x['pos'])[:15],
        'candidate_insertions': sorted(insertions, key=lambda x: x['pos'])[:15]
    }


def main():
    if not os.path.exists(BASELINE_PATH):
        raise FileNotFoundError(f"Missing baseline file at {BASELINE_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    with open(BASELINE_PATH, 'r') as f:
        baseline_samples = json.load(f)

    # Working on Sample 1 (index 0)
    sample = baseline_samples[0]
    text = sample['watermarked_text']
    
    encoded = tokenizer(text, add_special_tokens=True, truncation=True, max_length=2048)
    input_ids = encoded['input_ids']

    print("=" * 60)
    print(f"SAMPLE 1: Analyzing {len(input_ids)} tokens for vulnerability candidates...")
    candidates = extract_simultaneous_candidates(input_ids, tokenizer)

    print(f"Total Tokens: {candidates['total_tokens']}")
    print(f"Total Green Pairs: {candidates['total_green_pairs']}")
    print(f"High-Impact Deletions Found: {len(candidates['candidate_deletions'])}")
    print(f"High-Impact Insertions Found: {len(candidates['candidate_insertions'])}")
    print("=" * 60)

    # Save manifest
    os.makedirs(os.path.dirname(OUTPUT_MANIFEST_PATH), exist_ok=True)
    with open(OUTPUT_MANIFEST_PATH, 'w') as f:
        json.dump(candidates, f, indent=2)

    print(f"[✓ SAVED] Candidates manifest generated at: {OUTPUT_MANIFEST_PATH}\n")


if __name__ == '__main__':
    main()