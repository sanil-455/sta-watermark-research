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

prompt = data[0]

print("\nPROMPT:")
print(prompt)

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    use_fast=True,
)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
)

model.eval()

model_device = next(model.parameters()).device

print("Model loaded.")
print("Input device:", model_device)
print("GPU available:", torch.cuda.is_available())

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

inputs = tokenizer(
    prompt,
    return_tensors="pt",
    add_special_tokens=True,
)

input_ids = inputs["input_ids"].to(model_device)

prompt_tokens = input_ids.shape[1]

print("\nPrompt tokens:", prompt_tokens)
print("Generating:", MAX_NEW_TOKENS, "tokens")

prompt_max_length = 2048 - MAX_NEW_TOKENS

print("\nGenerating WATERMARKED text...")

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

with torch.no_grad():
    watermarked_ids, wm_stats = watermarker.generate_with_watermark(
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

print("\nWATERMARKED TEXT:")
print(watermarked_text)

wm_z, wm_p = detector.detect(
    watermarked_text,
    prompt_max_length,
    model_device,
    tokenizer,
)

print("\nWatermarked z-score:", wm_z)
print("Watermarked p-value:", wm_p)

print("\nGenerating UNWATERMARKED text...")

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

with torch.no_grad():
    unwatermarked_ids, uwm_stats = watermarker.generate_without_watermark(
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

print("\nUNWATERMARKED TEXT:")
print(unwatermarked_text)

uwm_z, uwm_p = detector.detect(
    unwatermarked_text,
    prompt_max_length,
    model_device,
    tokenizer,
)

print("\nUnwatermarked z-score:", uwm_z)
print("Unwatermarked p-value:", uwm_p)

result = {
    "experiment": "baseline_smoke_test",
    "model": "meta-llama/Llama-2-7b-hf",
    "seed": SEED,
    "max_new_tokens": MAX_NEW_TOKENS,
    "gamma": GAMMA,
    "n_sample_per_token": N_SAMPLE_PER_TOKEN,
    "sampling_temp": SAMPLING_TEMP,
    "use_sampling": True,
    "dtype": "float16",
    "device_map": "auto",
    "prompt": prompt,
    "prompt_tokens": prompt_tokens,
    "watermarked_z": float(wm_z),
    "watermarked_p": float(wm_p),
    "unwatermarked_z": float(uwm_z),
    "unwatermarked_p": float(uwm_p),
    "watermarked_text": watermarked_text,
    "unwatermarked_text": unwatermarked_text,
}

out_path = "results/raw/baseline_smoke_test.json"

with open(out_path, "w") as f:
    json.dump(result, f, indent=2)

print("\nSaved:", out_path)
print("\nSMOKE TEST COMPLETE.")
