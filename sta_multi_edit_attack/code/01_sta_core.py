import math
import torch
from transformers import AutoTokenizer

H1 = 15485863
H2 = 17624813
GAMMA = 0.5
Z_THRESHOLD = 2.0
MAX_LENGTH = 2048
MODEL_PATH = "hf_models/Llama-2-7b-hf"


def load_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_PATH)


def text_to_ids(tokenizer, text):
    return tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )["input_ids"]


def is_green_pair(prev_id, curr_id, device=None):
    device = device or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    rng = torch.Generator(device=device)
    rng.manual_seed(H1 * prev_id + H2 * curr_id)
    return (
        torch.rand(1, device=device, generator=rng).item()
        < GAMMA
    )


def sta_stats(input_ids, device=None):
    pairs = len(input_ids) - 1
    if pairs <= 0:
        return {
            "z": 0.0,
            "green_count": 0,
            "pair_count": 0,
            "detected": False,
        }

    green = sum(
        is_green_pair(
            input_ids[i],
            input_ids[i + 1],
            device,
        )
        for i in range(pairs)
    )
