"""
Generate STA-16 (tau=1.35) watermarked spans: the paper's own most-robust reported
variant (Mao et al., ACL 2025, "Watermarking Large Language Models: An Unbiased and
Low-risk Method"). Same green/red hash and z-test as STA-1 (confirmed by diffing
sample_watermark/ against sample_watermark_threshold/: only the generation step
differs, via use_entropy_threshold / entropy_threshold / n_sample_per_token_above_
entropy_threshold). Reuses generate_span_samples.generate_all, build_rows, write_exports,
span_z and new_master unchanged, so every downstream script (samples.py, run_attack.py,
report.py, final_all.py, bound_check.py, repair.py) reads these files exactly as they
read the STA-1 ones.

Exports results/raw/sta16_len200.json and sta16_len500.json, same schema as
samples_len*.json. Separate master file (results/raw/gen_master_sta16.json), so
it never touches or resumes the STA-1 master.

Run:  python research/experiments/generate_sta16_samples.py --target 100
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

TAU = 1.35   # the paper's entropy threshold for its "STA-16" row (Table 2 / Appendix E)
M = 16       # n_sample_per_token_above_entropy_threshold


def wrap_generate(watermarker, tau):
    """Returns a generate_with_watermark that always turns on the entropy threshold at tau."""
    original = watermarker.generate_with_watermark

    def wrapped(input_ids, args, model, device, tokenizer):
        args.use_entropy_threshold, args.entropy_threshold = True, tau
        return original(input_ids, args, model, device, tokenizer)
    return wrapped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=100, help="usable samples to collect")
    ap.add_argument("--long", type=int, default=500)
    ap.add_argument("--short", type=int, default=200)
    ap.add_argument("--start", type=int, default=28, help="same prompt stream as the STA-1 generator, so spans line up")
    ap.add_argument("--max-prompt-tokens", type=int, default=2048)
    ap.add_argument("--max-tries", type=int, default=300)
    cfg = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    os.chdir(root)
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "research" / "attacks_core"))
    sys.path.insert(0, str(root / "research" / "experiments"))

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from sample_watermark_threshold.sample_watermark_processor import SampleWatermark
    import sta_core
    import generate_span_samples as base

    assert sta_core.GAMMA == base.GAMMA, "gamma differs from sta_core"

    master_path = "results/raw/gen_master_sta16.json"
    base.MASTER = master_path   # build_rows/write_exports/generate_all save through base.save_json(obj, base.MASTER)
    master = base.new_master(cfg)
    if Path(master_path).exists():
        master = json.load(open(master_path))
        old = master["config"]
        if old["long"] != cfg.long or old["seed"] != base.SEED:
            sys.exit("%s was made with long=%s, you asked for %s. Use the same --long or move the old file away."
                      % (master_path, old["long"], cfg.long))
        print("resuming: %d usable STA-16 samples already saved" % len(master["done"]))

    data = json.load(open(base.DATA_PATH))
    tokenizer = AutoTokenizer.from_pretrained(base.MODEL_PATH, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(base.MODEL_PATH, dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True)
    model.eval()
    device = next(model.parameters()).device
    watermarker = SampleWatermark(vocab=list(range(tokenizer.vocab_size)), n_sample_per_token=1,
                                  n_sample_per_token_above_entropy_threshold=M, seeding_scheme="simple_1")
    watermarker.generate_with_watermark = wrap_generate(watermarker, TAU)

    def free():
        gc.collect()
        torch.cuda.empty_cache()
    rt = SimpleNamespace(seed=torch.manual_seed, free=free, oom_exc=torch.cuda.OutOfMemoryError)

    def score_fn(text):
        return sta_core.score_text(tokenizer, text)["z"]

    def write_exports_sta16(master, tokenizer, cfg, score_fn):
        # base.write_exports is hard-coded to "results/raw/samples_len%d.json" -- the STA-1 files.
        # generate_all calls write_exports on EVERY iteration, so this patch must be in place for
        # the whole call, not applied only after it returns, or the STA-1 files get overwritten mid-run.
        for length in (cfg.short, cfg.long):
            rows = base.build_rows(master, tokenizer, length, score_fn)
            base.save_json(rows, "results/raw/sta16_len%d.json" % length, indent=1)
    real_write_exports = base.write_exports
    base.write_exports = write_exports_sta16
    try:
        assert base.write_exports is write_exports_sta16  # the patch must be live before generate_all is entered
        base.generate_all(data, tokenizer, model, device, watermarker, sta_core.is_green_pair, score_fn, rt, master, cfg)
    finally:
        base.write_exports = real_write_exports  # restore even on error, so a later STA-1 run in the same process is unaffected
    print("exported: results/raw/sta16_len%d.json, results/raw/sta16_len%d.json" % (cfg.short, cfg.long))


if __name__ == "__main__":
    main()
