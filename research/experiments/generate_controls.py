"""
Unwatermarked control spans, for false-positive rates, ROC and AUC.

Same model, same sampling (temperature 1.0), same seed rule and same span-making rule as
generate_span_samples.py, but generated WITHOUT the watermark (the authors' own
generate_without_watermark). Prompts start at --start (28), so the first controls use the same
prompts as the watermarked spans. One 500-token generation per prompt gives both lengths: its
first 200 tokens are the 200-token control. A control is kept only if the model really produced
the length (195 for 200, 495 for 500), the same rule as the watermarked spans.

Each sample is saved as soon as it finishes and a rerun resumes. Exports (span text only, no prompt):
results/raw/controls_len200.json and controls_len500.json with "unwatermarked_z" = the z of the span text alone.

Run:  python research/experiments/generate_controls.py --target 200
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

SEED = 42
MODEL_PATH = "hf_models/Llama-2-7b-hf"
DATA_PATH = "data/C4/sampled_dataset_train.json"
MASTER = "results/raw/controls_master.json"


def save_json(obj, path, indent=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=indent)
    os.replace(tmp, path)


def export_rows(master, tokenizer, length, score_fn):
    rows = []
    for s in master["done"]:
        k = min(length, s["n_generated"])
        if k < length - 5:
            continue
        text = tokenizer.decode(s["gen_ids"][:k], skip_special_tokens=True)
        rows.append({"prompt_id": s["prompt_id"], "n_generated": k, "unwatermarked_z": score_fn(text), "text": text})
    return rows


def write_exports(master, tokenizer, cfg, score_fn):
    for length in (cfg.short, cfg.long):
        save_json(export_rows(master, tokenizer, length, score_fn), "results/raw/controls_len%d.json" % length, indent=1)


def generate_controls(data, tokenizer, model, device, watermarker, score_fn, rt, master, cfg, log=print, save=save_json, export=write_exports):
    args = SimpleNamespace(max_new_tokens=cfg.long, sampling_temp=1.0, use_sampling=True, gamma=0.5)
    seen = {s["prompt_id"] for s in master["done"]} | {x["prompt_id"] for x in master["too_short"]} | set(master["skipped_long"])
    t_start, processed = time.time(), 0
    for pid in range(cfg.start, min(len(data), cfg.start + cfg.max_tries)):
        if len(master["done"]) >= cfg.target:
            break
        if pid in seen:
            continue
        input_ids = tokenizer(data[pid], return_tensors="pt", add_special_tokens=True)["input_ids"]
        p = input_ids.shape[1]
        if p > cfg.max_prompt_tokens:
            master["skipped_long"].append(pid)
            save(master, MASTER)
            log("prompt %d skipped: %d tokens is over the %d cap" % (pid, p, cfg.max_prompt_tokens))
            continue
        t1 = time.time()
        rt.seed(SEED)
        oom, msg = False, ""
        try:
            out_ids, _ = watermarker.generate_without_watermark(input_ids.to(device), args, model, device, tokenizer)
        except rt.oom_exc as e:
            oom, msg = True, (str(e).splitlines() or [""])[0][:300]
        if oom:
            rt.free()
            master["oom_events"].append({"prompt_id": pid, "prompt_tokens": p, "message": msg})
            save(master, MASTER)
            log("prompt %d: OUT OF MEMORY, skipped. %s" % (pid, msg))
            continue
        full = out_ids[0].tolist()
        n_gen = len(full) - p
        processed += 1
        if n_gen < cfg.short - 5:
            master["too_short"].append({"prompt_id": pid, "n_generated": n_gen})
            log("prompt %d: model stopped after %d tokens, not counted" % (pid, n_gen))
        else:
            master["done"].append({"prompt_id": pid, "prompt_tokens": p, "n_generated": n_gen, "gen_ids": full[p:], "seconds": round(time.time() - t1, 1)})
            per = (time.time() - t_start) / processed
            log("[%d/%d] prompt %d | %d generated | %.0fs | about %.0f min left" % (
                len(master["done"]), cfg.target, pid, n_gen, time.time() - t1, per * (cfg.target - len(master["done"])) / 60))
        save(master, MASTER)
        export(master, tokenizer, cfg, score_fn)
    log("finished: %d usable, %d too short, %d skipped for length, %d out of memory events" % (
        len(master["done"]), len(master["too_short"]), len(master["skipped_long"]), len(master["oom_events"])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=200, help="number of usable controls")
    ap.add_argument("--long", type=int, default=500)
    ap.add_argument("--short", type=int, default=200)
    ap.add_argument("--start", type=int, default=28)
    ap.add_argument("--max-prompt-tokens", type=int, default=2048)
    ap.add_argument("--max-tries", type=int, default=400)
    cfg = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    os.chdir(root)
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "research" / "attacks_core"))
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from sample_watermark.sample_watermark_processor import SampleWatermark
    import sta_core

    master = {"config": {"model": MODEL_PATH, "seed": SEED, "long": cfg.long, "sampling_temp": 1.0, "watermark": False},
              "done": [], "too_short": [], "skipped_long": [], "oom_events": []}
    if Path(MASTER).exists():
        master = json.load(open(MASTER))
        if master["config"]["long"] != cfg.long or master["config"]["seed"] != SEED:
            sys.exit("%s was made with different settings; use the same --long or move it away." % MASTER)
        print("resuming: %d usable controls already saved" % len(master["done"]))
    data = json.load(open(DATA_PATH))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True)
    model.eval()
    device = next(model.parameters()).device
    watermarker = SampleWatermark(vocab=list(range(tokenizer.vocab_size)), n_sample_per_token=1, seeding_scheme="simple_1")

    def free():
        gc.collect()
        torch.cuda.empty_cache()

    rt = SimpleNamespace(seed=torch.manual_seed, free=free, oom_exc=torch.cuda.OutOfMemoryError)
    generate_controls(data, tokenizer, model, device, watermarker, lambda t: sta_core.score_text(tokenizer, t)["z"], rt, master, cfg)


if __name__ == "__main__":
    main()
