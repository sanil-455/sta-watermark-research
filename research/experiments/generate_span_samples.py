"""
Generate watermarked samples for the attack experiments.

The earlier baseline scripts saved prompt + generation as one string
and scored that whole string. The prompt is unwatermarked, so it
diluted the score: one document read z = 6.5 on its generated span
and 2.8 on the whole string. This script keeps the generated span
separate and scores only that span, which is what the paper evaluates.

The exported files are prompt-free: "watermarked_text" is the generated
span alone and "watermarked_z" is the detector's score on that text, which
is what an attacker holding only the generated text would query. The
prompt is kept in "source_prompt" for provenance and nothing else uses it.
The attack scripts treat the prompt as empty, so their offsets are zero.

One long generation (default 500 tokens) is made per prompt. The
generator is autoregressive and seeded the same way every time, so its
first 200 tokens are what a 200-token run would have produced (up to
GPU numerical noise). The 200-token protocol therefore comes out of
the same run. Two files are exported in the layout the attack sweeps
already read.

Each sample is saved as soon as it finishes, and a rerun resumes where
the last one stopped.

Run:  python research/experiments/generate_span_samples.py
"""

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

GAMMA = 0.5  # same as sta_core.GAMMA; checked in main()
SEED = 42
MODEL_PATH = "hf_models/Llama-2-7b-hf"
DATA_PATH = "data/C4/sampled_dataset_train.json"
MASTER = "results/raw/gen_master.json"


def span_z(flags, k):
    """z over the first k generated pairs.

    flags[j] is 1 when the pair (token before generated token j,
    generated token j) is green, so flags[0] is the pair that
    straddles the prompt boundary.
    """
    k = min(k, len(flags))
    if k < 2:
        return 0.0
    greens = sum(flags[:k])
    return (greens - GAMMA * k) / math.sqrt(GAMMA * (1 - GAMMA) * k)


def save_json(obj, path, indent=None):
    """Write to a temp file, then rename, so a crash cannot corrupt it."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=indent)
    os.replace(tmp, path)


def new_master(cfg):
    return {
        "config": {"model": MODEL_PATH, "seed": SEED, "gamma": GAMMA,
                   "long": cfg.long, "sampling_temp": 1.0,
                   "n_sample_per_token": 1},
        "done": [], "too_short": [], "skipped_long": [], "failed": [],
        "oom_events": [],
    }


def build_rows(master, tokenizer, length, score_fn):
    """Prompt-free rows for one span length, in the layout the sweeps read.

    "prompt" is empty and "prompt_tokens" is 0, so the sweeps' offsets are
    zero and every sentence and token belongs to the span. "watermarked_z"
    is score_fn(span text): the z a detector gets when it is handed the
    generated text alone. "z_from_ids" is the z over the generated token
    ids exactly as the watermark produced them, kept for comparison.
    """
    rows = []
    for s in master["done"]:
        k = min(length, s["n_generated"])
        if k < length - 5:          # the paper allows 200 +/- 5
            continue
        p, ids = s["prompt_tokens"], s["full_ids"]
        span_text = tokenizer.decode(ids[p:p + k], skip_special_tokens=True)
        flags = s["green_flags"]
        rows.append({
            "prompt_id": s["prompt_id"],
            "prompt": "",
            "prompt_tokens": 0,
            "n_generated": k,
            "watermarked_z": score_fn(span_text),
            "z_from_ids": span_z(flags, k),
            "green_fraction": sum(flags[:k]) / k,
            "watermarked_text": span_text,
            "source_prompt": s["prompt"],
            "source_prompt_tokens": p,
        })
    return rows


def write_exports(master, tokenizer, cfg, score_fn):
    for length in (cfg.short, cfg.long):
        rows = build_rows(master, tokenizer, length, score_fn)
        save_json(rows, "results/raw/samples_len%d.json" % length, indent=1)


def generate_all(data, tokenizer, model, device, watermarker,
                 is_green_pair, score_fn, rt, master, cfg, log=print):
    args = SimpleNamespace(max_new_tokens=cfg.long, sampling_temp=1.0,
                           use_sampling=True, gamma=GAMMA)
    master.setdefault("oom_events", [])
    seen = ({s["prompt_id"] for s in master["done"]}
            | {x["prompt_id"] for x in master["too_short"]}
            | set(master["skipped_long"]))
    last = min(len(data), cfg.start + cfg.max_tries)
    t_start, processed = time.time(), 0

    for pid in range(cfg.start, last):
        if len(master["done"]) >= cfg.target:
            break
        if pid in seen:
            continue

        prompt = data[pid]
        input_ids = tokenizer(prompt, return_tensors="pt",
                              add_special_tokens=True)["input_ids"]
        p = input_ids.shape[1]

        if p > cfg.max_prompt_tokens:
            master["skipped_long"].append(pid)
            save_json(master, MASTER)
            log("prompt %d skipped: %d tokens is over the %d cap"
                % (pid, p, cfg.max_prompt_tokens))
            continue

        t1 = time.time()
        rt.seed(SEED)
        oom, oom_msg = False, ""
        try:
            out_ids, _ = watermarker.generate_with_watermark(
                input_ids.to(device), args, model, device, tokenizer)
        except rt.oom_exc as e:
            oom, oom_msg = True, (str(e).splitlines() or [""])[0][:300]
        if oom:
            # free the memory only after the except block has ended:
            # until then the traceback still holds the tensors
            rt.free()
            if pid not in master["failed"]:
                master["failed"].append(pid)
            master["oom_events"].append(
                {"prompt_id": pid, "prompt_tokens": p, "message": oom_msg})
            save_json(master, MASTER)
            log("prompt %d (%d prompt tokens): OUT OF MEMORY, skipped. %s"
                % (pid, p, oom_msg))
            continue

        full = out_ids[0].tolist()
        n_gen = len(full) - p
        flags = [int(is_green_pair(full[i - 1], full[i]))
                 for i in range(p, len(full))]
        secs = time.time() - t1
        processed += 1

        if n_gen < cfg.short - 5:
            master["too_short"].append({"prompt_id": pid, "n_generated": n_gen})
            log("prompt %d: model stopped after %d tokens, not counted"
                % (pid, n_gen))
        else:
            if pid in master["failed"]:
                master["failed"].remove(pid)
            master["done"].append({
                "prompt_id": pid, "prompt": prompt, "prompt_tokens": p,
                "n_generated": n_gen, "full_ids": full,
                "green_flags": flags, "seconds": round(secs, 1)})
            n_done = len(master["done"])
            per = (time.time() - t_start) / processed
            eta = per * (cfg.target - n_done) / 60
            log("[%d/%d] prompt %d | %d prompt tok | %d generated | "
                "z%d=%.2f z%d=%.2f | %.0fs | about %.0f min left"
                % (n_done, cfg.target, pid, p, n_gen, cfg.short,
                   span_z(flags, cfg.short), cfg.long,
                   span_z(flags, cfg.long), secs, eta))

        save_json(master, MASTER)
        write_exports(master, tokenizer, cfg, score_fn)

    log("finished: %d usable, %d too short, %d skipped for length, %d out of memory events"
        % (len(master["done"]), len(master["too_short"]),
           len(master["skipped_long"]), len(master["oom_events"])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=50,
                    help="usable samples to collect")
    ap.add_argument("--long", type=int, default=500,
                    help="new tokens generated per prompt")
    ap.add_argument("--short", type=int, default=200,
                    help="paper protocol length, cut from the same run")
    ap.add_argument("--start", type=int, default=28,
                    help="first C4 entry; 0-4 and 20-27 are already used")
    ap.add_argument("--max-prompt-tokens", type=int, default=2048,
                    help="skip longer prompts; Llama-2's context is 4096, so "
                         "prompt + --long must stay under that")
    ap.add_argument("--max-tries", type=int, default=150)
    cfg = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    os.chdir(root)
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "research" / "attacks_core"))

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from sample_watermark.sample_watermark_processor import SampleWatermark
    import sta_core

    assert sta_core.GAMMA == GAMMA, "gamma differs from sta_core"

    master = new_master(cfg)
    if Path(MASTER).exists():
        master = json.load(open(MASTER))
        old = master["config"]
        if old["long"] != cfg.long or old["seed"] != SEED:
            sys.exit("%s was made with long=%s, you asked for %s. Use the same "
                     "--long or move the old file away." % (MASTER, old["long"], cfg.long))
        print("resuming: %d usable samples already saved" % len(master["done"]))

    data = json.load(open(DATA_PATH))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.float16, device_map="auto",
        low_cpu_mem_usage=True)
    model.eval()
    device = next(model.parameters()).device

    watermarker = SampleWatermark(
        vocab=list(range(tokenizer.vocab_size)),
        n_sample_per_token=1, seeding_scheme="simple_1")

    def free():
        gc.collect()
        torch.cuda.empty_cache()

    rt = SimpleNamespace(seed=torch.manual_seed, free=free,
                         oom_exc=torch.cuda.OutOfMemoryError)

    def score_fn(text):
        return sta_core.score_text(tokenizer, text)["z"]

    generate_all(data, tokenizer, model, device, watermarker,
                 sta_core.is_green_pair, score_fn, rt, master, cfg)


if __name__ == "__main__":
    main()
