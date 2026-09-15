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


def compute_sta_stats(input_ids, rng_device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Computes green pair counts, total adjacent token pairs, Z-score statistic,
    and watermark detection status for a sequence of token IDs.
    """
    if len(input_ids) <= 1:
        return {'z': 0.0, 'green_count': 0, 'pair_count': 0, 'detected': False}

    rng = torch.Generator(device=rng_device)
    green_count = 0
    pair_count = len(input_ids) - 1

    for i in range(pair_count):
        prev_token, curr_token = input_ids[i], input_ids[i + 1]

        # Deterministic STA seed formula based on token pair hash
        seed = H1 * prev_token + H2 * curr_token
        rng.manual_seed(seed)

        # Check if PRG random value falls within green list fraction (gamma)
        random_val = torch.rand(
            1,
            device=rng_device,
            generator=rng
        ).item()

        if random_val < GAMMA:
            green_count += 1

    # Expected mean and standard deviation under Null Hypothesis
    expected_mean = GAMMA * pair_count
    std_dev = math.sqrt(GAMMA * (1.0 - GAMMA) * pair_count)

    z_score = (
        (green_count - expected_mean) / std_dev
        if std_dev > 0
        else 0.0
    )

    return {
        'z': float(z_score),
        'green_count': green_count,
        'pair_count': pair_count,
        'detected': z_score > 2.0
    }


def apply_simultaneous_edits(token_list, edits):
    """
    Executes replacements, deletions, and insertions on a token array.

    Edits are sorted from right to left so that modifications to later
    positions do not invalidate the indices of earlier positions.
    """
    sorted_edits = sorted(
        edits,
        key=lambda item: item['pos'],
        reverse=True
    )

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
