import json
import os
import random
import numpy as np
import torch

from types import SimpleNamespace
from transformers import AutoTokenizer, AutoModelForCausalLM

from sample_watermark.sample_watermark_processor import (
    SampleWatermark,
    SampleWatermarkDetector,
)

MODEL_PATH = "hf_models/Llama-2-7b-hf"
DATA_PATH = "data/C4/sampled_dataset_train.json"

MAX_NEW_TOKENS = 200
NUM_PROMPTS = 5
MAX_PROMPT_TOKENS = 1500

SEED = 42
GAMMA = 0.5
N_SAMPLE_PER_TOKEN = 1
SAMPLING_TEMP = 1.0

os.makedirs("results/raw", exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

with open(DATA_PATH, "r") as f:
    data = json.load(f)

print("=" * 60)
print("STA SAFE BASELINE")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    use_fast=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
)

model.eval()

model_device = next(model.parameters()).device

args = SimpleNamespace(
    max_new_tokens=MAX_NEW_TOKENS,
    sampling_temp=SAMPLING_TEMP,
    use_sampling=True,
    gamma=GAMMA,
)

vocab = list(range(tokenizer.vocab_size))

watermarker = SampleWatermark(
    vocab=vocab,
    n_sample_per_token=N_SAMPLE_PER_TOKEN,
    seeding_scheme="simple_1",
)

detector = SampleWatermarkDetector(
    vocab=vocab,
    n_sample_per_token=N_SAMPLE_PER_TOKEN,
    z_threshold=2,
    gamma=GAMMA,
    seeding_scheme="simple_1",
)

results = []
valid_count = 0

prompt_max_length = 2048 - MAX_NEW_TOKENS

for prompt_id, prompt in enumerate(data):

    if valid_count >= NUM_PROMPTS:
        break

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
    )

    input_ids = inputs["input_ids"].to(model_device)
    prompt_tokens = input_ids.shape[1]

    print("\nPrompt ID:", prompt_id)
    print("Prompt tokens:", prompt_tokens)

    if prompt_tokens > MAX_PROMPT_TOKENS:
        print("SKIPPED: prompt too long")
        continue

    print("Generating WATERMARKED...")

    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    with torch.no_grad():
        watermarked_ids, _ = watermarker.generate_with_watermark(
            input_ids,
            args,
            model,
            model_device,
            tokenizer,
        )

    watermarked_text = tokenizer.decode(
        watermarked_ids[0],
        skip_special_tokens=True,
    )

    wm_z, wm_p = detector.detect(
        watermarked_text,
        prompt_max_length,
        model_device,
        tokenizer,
    )

    print("WM z:", wm_z)
    print("WM p:", wm_p)

    print("Generating UNWATERMARKED...")

    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    with torch.no_grad():
        unwatermarked_ids, _ = watermarker.generate_without_watermark(
            input_ids,
            args,
            model,
            model_device,
            tokenizer,
        )

    unwatermarked_text = tokenizer.decode(
        unwatermarked_ids[0],
        skip_special_tokens=True,
    )

    uwm_z, uwm_p = detector.detect(
        unwatermarked_text,
        prompt_max_length,
        model_device,
        tokenizer,
    )

    print("UWM z:", uwm_z)
    print("UWM p:", uwm_p)

    results.append({
        "prompt_id": prompt_id,
        "prompt": prompt,
        "prompt_tokens": prompt_tokens,
        "max_new_tokens": MAX_NEW_TOKENS,
        "seed": SEED,
        "gamma": GAMMA,
        "n_sample_per_token": N_SAMPLE_PER_TOKEN,
        "sampling_temp": SAMPLING_TEMP,
        "watermarked_z": float(wm_z),
        "watermarked_p": float(wm_p),
        "unwatermarked_z": float(uwm_z),
        "unwatermarked_p": float(uwm_p),
        "watermarked_text": watermarked_text,
        "unwatermarked_text": unwatermarked_text,
    })

    valid_count += 1

    with open("results/raw/baseline_safe.json", "w") as f:
        json.dump(results, f, indent=2)

    print("SAVED completed prompt:", valid_count)
    
print("\nBASELINE FINISHED")
print("Valid prompts:", valid_count)
print("Saved: results/raw/baseline_safe.json")
