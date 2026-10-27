"""
Can the whole experiment run on this GPU without a memory crash?

Each essential process runs in its own subprocess, one after another,
exactly as the experiment runs them, on the worst-case input the
experiment allows. For each one this reports how much GPU memory it
needed and how much was left. A stage that crashes is reported as
crashed, which is itself the answer.

  generate      Llama-2-7B fp16, 1000-token prompt + 500 new tokens
                (the prompt cap and length used by generate_span_samples.py),
                plus a 1500-token forward pass as a conservative bound
  token_attack  Llama-2-7B fp16 (fit + perplexity) + MiniLM on a
                1500-token document (token-level attack scoring)
  paraphrase    Qwen2.5-7B 4-bit + MiniLM, 60 rewrites of the longest
                sentence in the data (the paraphrase attack)

needed   = GPU memory in use before the stage (other processes and the
           CUDA context) + the stage's peak PyTorch memory
headroom = GPU total - needed. PASS when headroom >= --margin.

LanguageTool and spaCy run on the CPU and are not part of this check.

Run:  python research/experiments/memory_check.py
"""

import argparse
import json
import os
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

GIB = 2 ** 30
LLAMA = "hf_models/Llama-2-7b-hf"
DATA_PATH = "data/C4/sampled_dataset_train.json"
STAGES = {
    "generate": "sample generation (Llama fp16, 1000 prompt + 500 new tokens)",
    "token_attack": "token-level attack scoring (Llama fp16 + MiniLM, 1500 tokens)",
    "paraphrase": "paraphrase attack (Qwen 4-bit + MiniLM, 60 rewrites)",
}


# ---------------------------------------------------------------- child side

def child_setup():
    root = Path(__file__).resolve().parents[2]
    os.chdir(root)
    for sub in ("", "research/attacks_core", "research/blackbox"):
        sys.path.insert(0, str(root / sub) if sub else str(root))
    import torch
    if not torch.cuda.is_available():
        sys.exit("no CUDA device available")
    torch.zeros(1, device="cuda")              # create the CUDA context first
    free, total = torch.cuda.mem_get_info()
    return torch, (total - free) / GIB, total / GIB


def child_report(stage, torch, other, total, extra):
    import resource
    res = {
        "stage": stage,
        "total_gib": total,
        "other_used_gib": other,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / GIB,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / GIB,
        "host_peak_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024,
    }
    res.update(extra)
    print("RESULT " + json.dumps(res), flush=True)


def stage_generate():
    torch, other, total = child_setup()
    from types import SimpleNamespace
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from sample_watermark.sample_watermark_processor import SampleWatermark

    data = json.load(open(DATA_PATH))
    tok = AutoTokenizer.from_pretrained(LLAMA, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        LLAMA, dtype=torch.float16, device_map="auto",
        low_cpu_mem_usage=True).eval()
    dev = next(model.parameters()).device

    ids = tok(max(data, key=len), return_tensors="pt",
              add_special_tokens=True)["input_ids"]
    assert ids.shape[1] >= 1500, "longest prompt is shorter than 1500 tokens"

    wm = SampleWatermark(vocab=list(range(tok.vocab_size)),
                         n_sample_per_token=1, seeding_scheme="simple_1")
    args = SimpleNamespace(max_new_tokens=500, sampling_temp=1.0,
                           use_sampling=True, gamma=0.5)
    torch.manual_seed(42)
    out, _ = wm.generate_with_watermark(ids[:, :1000].to(dev), args, model, dev, tok)
    n_gen = out.shape[1] - 1000
    peak_generation = torch.cuda.max_memory_reserved() / GIB

    # the model may stop early, so also bound the worst case directly:
    # one forward pass over 1500 tokens holds the same final cache
    del out
    torch.cuda.empty_cache()
    with torch.no_grad():
        model(input_ids=ids[:, :1500].to(dev), use_cache=True)
    child_report("generate", torch, other, total,
                 {"tokens_generated": n_gen,
                  "peak_reserved_generation_only_gib": peak_generation})


def stage_token_attack():
    torch, other, total = child_setup()
    from transformers import AutoTokenizer
    from fluency import Fluency
    from context_fit import ContextFit
    from beam import Semantic

    data = json.load(open(DATA_PATH))
    tok = AutoTokenizer.from_pretrained(LLAMA)
    ids = tok(max(data, key=len), add_special_tokens=True, truncation=True,
              max_length=1500)["input_ids"]
    text = tok.decode(ids, skip_special_tokens=True)

    flu = Fluency(tok)
    fit = ContextFit(tok, flu.model, flu.device)
    sem = Semantic()
    flu.set_reference(text)
    ratio = flu.ratio(text)
    pos = len(ids) - 3
    fit_score = fit.score(ids, pos, ids[pos])
    sem.similarity(text, [text])
    child_report("token_attack", torch, other, total,
                 {"document_tokens": len(ids), "ppl_ratio": float(ratio),
                  "fit_score": float(fit_score)})


def stage_paraphrase():
    torch, other, total = child_setup()
    import re
    from paraphrase import Paraphraser
    from beam import Semantic

    data = json.load(open(DATA_PATH))
    sentences = [s for d in data for s in re.split(r"(?<=[.!?])\s+", d)]
    sent = max((s for s in sentences if len(s) <= 900), key=len)

    para = Paraphraser()
    sem = Semantic()
    cands = para.rewrite(sent, n=60)
    for c in cands:
        para.perplexity(c)
    if cands:
        sem.similarity(sent, cands)
    child_report("paraphrase", torch, other, total,
                 {"sentence_chars": len(sent), "distinct_rewrites": len(cands)})


# --------------------------------------------------------------- parent side

def gpu_used_mib():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"], text=True, timeout=10)
        used, total = out.strip().splitlines()[0].split(",")
        return float(used), float(total)
    except Exception:
        return None


def judge(res, code, log_text, margin):
    """Verdict for one stage, and the result dict with needed/headroom."""
    if code != 0 or res is None:
        oom = "out of memory" in log_text.lower()
        return ("CRASHED: out of memory" if oom
                else "CRASHED: other error, see log"), res
    res["needed_gib"] = res["other_used_gib"] + res["peak_reserved_gib"]
    res["headroom_gib"] = res["total_gib"] - res["needed_gib"]
    return ("PASS" if res["headroom_gib"] >= margin else "TOO TIGHT"), res


def run_stage(name, margin, cmd=None):
    log = Path("memcheck_%s.log" % name)
    cmd = cmd or [sys.executable, "-u", os.path.abspath(__file__), "--stage", name]
    smi_peak = 0.0
    with open(log, "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            used = gpu_used_mib()
            if used:
                smi_peak = max(smi_peak, used[0])
            time.sleep(0.5)
    text = log.read_text()
    res = None
    for line in text.splitlines():
        if line.startswith("RESULT "):
            res = json.loads(line[len("RESULT "):])
    verdict, res = judge(res, proc.returncode, text, margin)
    if res is not None:
        res["nvidia_smi_peak_gib"] = smi_peak / 1024 if smi_peak else None
    return verdict, res, log


def version(pkg):
    try:
        return metadata.version(pkg)
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=list(STAGES), help=argparse.SUPPRESS)
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--margin", type=float, default=1.0,
                    help="GiB that must remain free at the peak")
    cfg = ap.parse_args()

    if cfg.stage:
        {"generate": stage_generate, "token_attack": stage_token_attack,
         "paraphrase": stage_paraphrase}[cfg.stage]()
        return

    lines = []
    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    smi = gpu_used_mib()
    say("torch %s | transformers %s | bitsandbytes %s"
        % (version("torch"), version("transformers"), version("bitsandbytes")))
    if smi:
        say("GPU in use before the check: %.2f GiB of %.2f GiB (should be near 0)"
            % (smi[0] / 1024, smi[1] / 1024))
    say("margin required at the peak: %.1f GiB\n" % cfg.margin)

    rows = []
    for name in cfg.stages.split(","):
        say("running %s: %s ..." % (name, STAGES[name]))
        verdict, res, log = run_stage(name, cfg.margin)
        rows.append((name, verdict, res))
        say("  -> %s   (log: %s)" % (verdict, log))

    say()
    say("%-13s %-27s %8s %8s %8s %9s %8s" % ("stage", "verdict", "other", "peak", "needed", "headroom", "host RAM"))
    for name, verdict, res in rows:
        if res and "needed_gib" in res:
            say("%-13s %-27s %7.2fG %7.2fG %7.2fG %8.2fG %7.1fG"
                % (name, verdict, res["other_used_gib"], res["peak_reserved_gib"],
                   res["needed_gib"], res["headroom_gib"], res["host_peak_gib"]))
        else:
            say("%-13s %-27s" % (name, verdict))
    total = next((r["total_gib"] for _, _, r in rows if r), None)
    if total:
        say("\nGPU total: %.2f GiB. 'other' is memory in use before the stage "
            "(CUDA context and any other process)." % total)
    ok = all(v == "PASS" for _, v, _ in rows)
    say("\nOVERALL: " + ("every essential process fits with at least %.1f GiB to spare."
                         % cfg.margin if ok else "NOT every process passed; see above."))
    Path("memcheck_report.txt").write_text("\n".join(lines) + "\n\n" +
        json.dumps([r for _, _, r in rows], indent=1))
    say("report saved: memcheck_report.txt")


if __name__ == "__main__":
    main()
