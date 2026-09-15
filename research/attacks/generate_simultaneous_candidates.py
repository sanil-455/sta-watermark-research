import json
import math
import os
import torch
from transformers import AutoTokenizer

# -----------------------------------------------------------------------------
# Selective Text Analysis (STA) PRG Constants
# -----------------------------------------------------------------------------
H1 = 15485863
H2 = 17624813
GAMMA = 0.5

MODEL_PATH = 'hf_models/Llama-2-7b-hf'
BASELINE_PATH = 'results/raw/baseline_safe.json'
OUTPUT_MANIFEST_PATH = 'results/raw/manual_attacks/simultaneous_candidates_sample1.json'


def is_green_pair(token_a, token_b, rng_device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Evaluates whether two adjacent token IDs produce a green token state under the STA PRG hash.
    Formula: seed = H1 * token_a + H2 * token_b
    """
    rng = torch.Generator(device=rng_device)
    seed = H1 * token_a + H2 * token_b
    rng.manual_seed(seed)
    return torch.rand(1, device=rng_device, generator=rng).item() < GAMMA


def extract_simultaneous_candidates(input_ids, tokenizer):
    """
    Scans input_ids to identify high-impact candidates for deletion and insertion.
    Ensures that deleted and target tokens correspond to complete, meaningful words.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n = len(input_ids)
    
    green_pair_indices = []
    
    # 1. Identify all adjacent green token transitions (idx -> idx + 1)
    for i in range(n - 1):
        if is_green_pair(input_ids[i], input_ids[i + 1], rng_device=device):
            green_pair_indices.append(i)

    deletions = []
    insertions = []

    # Standard low-disruption insertion tokens in LLaMA vocabulary (Comma, Period, Space)
    punctuation_ids = [29892, 29889, 259]

    # 2. Evaluate impact for each green position
    for idx in green_pair_indices:
        if idx == 0 or idx >= n - 1:
            continue

        prev_id = input_ids[idx - 1]
        curr_id = input_ids[idx]
        next_id = input_ids[idx + 1]

        token_str = tokenizer.decode([curr_id]).strip()

        # Skip non-meaningful tokens or standalone punctuation for deletion
        if len(token_str) <= 1 or not token_str.isalnum():
            continue

        # --- DELETION SCORING ---
        # Removing curr_id connects prev_id directly to next_id
        if not is_green_pair(prev_id, next_id, rng_device=device):
            deletions.append({
                'pos': idx,
                'token_id': curr_id,
                'token_str': token_str,
                'net_green_reduction': 2  # Destroys 2 green pairs, creates 0
            })

        # --- INSERTION SCORING ---
        # Inserting punctuation after idx turns (curr_id -> next_id) into (curr_id -> P) and (P -> next_id)
        for p_id in punctuation_ids:
            g1 = is_green_pair(curr_id, p_id, rng_device=device)
            g2 = is_green_pair(p_id, next_id, rng_device=device)
            if not g1 and not g2:
                insertions.append({
                    'pos': idx,
                    'insert_token_id': p_id,
                    'token_str': tokenizer.decode([p_id]),
                    'net_green_reduction': 1  # Destroys 1 green pair, creates 0
                })
                break

    return {
        'total_tokens': n,
        'total_green_pairs': len(green_pair_indices),
        'candidate_deletions': sorted(deletions, key=lambda x: x['pos']),
        'candidate_insertions': sorted(insertions, key=lambda x: x['pos'])
    }


def main():
    if not os.path.exists(BASELINE_PATH):
        raise FileNotFoundError(f"Missing baseline file at {BASELINE_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    with open(BASELINE_PATH, 'r') as f:
        baseline_samples = json.load(f)

    # Analyze Sample 1 (index 0)
    sample = baseline_samples[0]
    text = sample['watermarked_text']
    
    encoded = tokenizer(text, add_special_tokens=True, truncation=True, max_length=2048)
    input_ids = encoded['input_ids']

    print("=" * 60)
    print(f"MODULE 2: Analyzing {len(input_ids)} tokens for vulnerability candidates...")
    candidates = extract_simultaneous_candidates(input_ids, tokenizer)

    print(f"Total Tokens: {candidates['total_tokens']}")
    print(f"Total Green Pairs: {candidates['total_green_pairs']}")
    print(f"High-Impact Deletions Found: {len(candidates['candidate_deletions'])}")
    print(f"High-Impact Insertions Found: {len(candidates['candidate_insertions'])}")
    print("=" * 60)

    # Export Manifest
    os.makedirs(os.path.dirname(OUTPUT_MANIFEST_PATH), exist_ok=True)
    with open(OUTPUT_MANIFEST_PATH, 'w') as f:
        json.dump(candidates, f, indent=2)

    print(f"[✓ SAVED] Candidate manifest saved to: {OUTPUT_MANIFEST_PATH}\n")


if __name__ == '__main__':
    main()