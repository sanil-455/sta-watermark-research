import math

import torch
from transformers import AutoTokenizer

H1 = 15485863
H2 = 17624813
GAMMA = 0.5
Z_THRESHOLD = 2.0
MAX_LENGTH = 2048
MODEL_PATH = "hf_models/Llama-2-7b-hf"

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_RNG = torch.Generator(device=_DEVICE)
_GREEN_CACHE = {} # doing memoization ie. will look up the same prev,curr id every time so that it 
                # does not compute the same pair again and again

def load_tokenizer():
    # wrapper which causes every file to load tokenizer in same way from exact same path
    return AutoTokenizer.from_pretrained(MODEL_PATH) 

def text_to_ids(tokenizer, text):
    # Converts a text string into a list of token id numbers.
    return tokenizer(
        text,
        add_special_tokens=True, # adds models' begining of seq marker as first token has no token before it
        truncation=True, # in case if more tokens are generated than 2048 it is trimmed off
        max_length=MAX_LENGTH,
    )["input_ids"]


def is_green_pair(prev_id, curr_id):
    """
    Green/red verdict for one adjacent token-ID pair.

    Matches SampleWatermarkDetector._seed_rng exactly:
        seed = H1 * now_token_id + H2 * next_token_id
        green iff torch.rand(1, generator=rng) < gamma

    Deterministic in (prev_id, curr_id), so it is cached.
    The generator is module-level because manual_seed fully
    determines each draw; a new one per call is overhead.
    """
    key = (prev_id, curr_id)

    cached = _GREEN_CACHE.get(key)
    if cached is not None: # we return none when nothing is stored, else false could mean red/ nothing is returned
        return cached

    _RNG.manual_seed(H1 * prev_id + H2 * curr_id)

    verdict = (
        torch.rand(1, device=_DEVICE, generator=_RNG).item() # reuisng the same seed again and again to reproduce the same result
        < GAMMA
    )

    _GREEN_CACHE[key] = verdict
    return verdict


def sta_stats(input_ids):
    """
    STA-1 detection statistics for a token-ID sequence.

    T is the PAIR count, not the token count. This follows
    the official detector, where T = len(result) and result
    has one entry per adjacent pair.
    # there has been a descrepancy with algo in paper which uses no. of tokens instead of number of pairs as in code
    """
    pairs = len(input_ids) - 1
    # if text is short that it tokenizes to 0 or 1 token then there are 0 pairs to check and z scord
    # formula divides by sqrt(0) and thus crashes... thsu handling the edge case
    if pairs <= 0:
        return {
            "z": 0.0,
            "green_count": 0,
            "pair_count": 0,
            "detected": False,
        }
    # checks every pairs and calling is_green_pair fn on each and sums up true and false results
    # thus counting number of green tokens as each true =1
    green = sum(
        is_green_pair(input_ids[i], input_ids[i + 1])
        for i in range(pairs)
    )
    # z=observed-what expect by chance divided by std of the expectation
    z = (green - GAMMA * pairs) / math.sqrt(GAMMA * (1.0 - GAMMA) * pairs)
    
    # returns a dictionary of req data
    return {
        "z": float(z),
        "green_count": int(green),
        "pair_count": int(pairs),
        "detected": z > Z_THRESHOLD,
    }


def score_text(tokenizer, text):
    """
    Tokenize then score. A real detector receives text, not
    token IDs, so every edit must survive a decode/re-encode
    round trip. # though only encode happens here
    """
    # actually aincluding the edit_ops we do:
    # encode → edit-as-numbers → decode-to-text → re-encode-from-text
    return sta_stats(text_to_ids(tokenizer, text))
