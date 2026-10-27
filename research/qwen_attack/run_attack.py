"""
Module 7: the runner. Attacks every flagged span, writes one JSON file per
span as soon as it is finished (so a rerun with --resume continues where it
stopped), then builds the report.

  python research/qwen_attack/run_attack.py --only baseline_safe:1 --n-cands 20   (quick check)
  python research/qwen_attack/run_attack.py                                        (everything)
  python research/qwen_attack/run_attack.py --resume results/qwen_attack/run_...   (continue)

Each run writes to its own folder and never overwrites an earlier one.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FILES = ["results/raw/baseline_safe.json", "results/raw/baseline_extra.json",
         "results/raw/samples_len200.json", "results/raw/samples_len500.json"]


def sample_file(rundir, sample_id):
    return Path(rundir) / "samples" / (sample_id.replace(":", "__") + ".json")


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, path)


def gate_config(args, gates):
    cfg = asdict(gates.PRESETS[args.preset])
    if args.semantic is not None:
        cfg["semantic"] = args.semantic
    if args.fluency is not None:
        cfg["fluency"] = args.fluency
    if args.names:
        cfg["check_names"] = True
    if args.no_spelling:
        cfg["spelling"] = False
    return gates.GateConfig(**cfg)


def llama2_parameters():
    """Llama-2-7B from its published configuration (hidden 4096, 11008, 32 layers, vocab 32000)."""
    h, inter, layers, vocab = 4096, 11008, 32, 32000
    return 2 * vocab * h + layers * (4 * h * h + 3 * h * inter + 2 * h) + h


def prompt_snippets(files):
    """sample id -> the start of its prompt, used to prove the prompt never reaches the detector."""
    out = {}
    for f in files:
        for row in json.load(open(f)):
            p = (row.get("source_prompt") or row.get("prompt_raw") or row.get("prompt") or "").strip()
            out["%s:%s" % (Path(f).stem, row["prompt_id"])] = p[:80] if len(p) >= 40 else ""
    return out


def git_commit():
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*", default=FILES)
    ap.add_argument("--preset", choices=["lenient", "strict"], default="lenient")
    ap.add_argument("--semantic", type=float, default=None)
    ap.add_argument("--fluency", type=float, default=None)
    ap.add_argument("--names", action="store_true")
    ap.add_argument("--no-spelling", action="store_true")
    ap.add_argument("--n-cands", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--max-minutes", type=float, default=0, help="optional time budget per span; 0 = none")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    os.chdir(ROOT)
    sys.path[:0] = [str(HERE), str(ROOT / "research" / "blackbox"), str(ROOT / "research" / "attacks_core")]
    import torch
    import attack, gates, paraphraser, report, samples, sentences, sta_core
    from oracle import DetectorOracle

    files = [f for f in args.files if os.path.exists(f)]
    rundir = Path(args.resume) if args.resume else ROOT / "results" / "qwen_attack" / ("run_%s_%s" % (time.strftime("%Y%m%d_%H%M%S"), args.preset))
    rundir.mkdir(parents=True, exist_ok=True)

    tok = sta_core.load_tokenizer()
    loaded = samples.load_samples(files)
    flagged, not_flagged = samples.screen(loaded, DetectorOracle(tok, "score"))
    write_json(rundir / "screening.json", [{"sample_id": s.sample_id, "z0": s.z0, "flagged": s.z0 > samples.Z_THRESHOLD, "source": s.source} for s in loaded])
    selected = [s for s in flagged if not args.only or s.sample_id in args.only]
    snippets = prompt_snippets(files)

    para = paraphraser.Paraphraser()
    minilm = gates.MiniLM()
    wordnet = gates.load_wordnet()
    gcfg = gate_config(args, gates)
    gate = gates.Gates(gcfg, gates.KnownWords(gates.vocabulary_from_c4(), wordnet), minilm.similarity, para.perplexity)
    cfg = SimpleNamespace(n_cands=args.n_cands, seed=args.seed, max_minutes=args.max_minutes)
    ids_fn = lambda text: sta_core.text_to_ids(tok, text)

    gen_run = None
    if os.path.exists("results/raw/gen_master.json"):
        m = json.load(open("results/raw/gen_master.json"))
        gen_run = {"kept": len(m["done"]), "too_short": len(m["too_short"]), "skipped_long": len(m["skipped_long"]), "oom_events": len(m.get("oom_events", m.get("failed", [])))}
    run = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "ended": None,
           "args": dict(vars(args), command=" ".join(sys.argv), files=files),
           "gates": {"config": asdict(gcfg), "wordnet": wordnet is not None},
           "attack_config": vars(cfg), "segmentation": sentences.settings(), "model": para.describe(),
           "minilm": {"name": "sentence-transformers/all-MiniLM-L6-v2", "parameters": sum(p.numel() for p in minilm.model.parameters())},
           "generator_parameters": "{:,}".format(llama2_parameters()), "generation_run": gen_run,
           "environment": {"git_commit": git_commit(), "gpu": torch.cuda.get_device_name(0), "python": sys.version.split()[0]}}
    write_json(rundir / "run.json", run)

    started, done_now = time.time(), 0
    for n, s in enumerate(selected, 1):
        path = sample_file(rundir, s.sample_id)
        if path.exists():
            continue
        seg = sentences.split_span(s.span)
        summary = seg.summary()
        tokens = s.n_generated or (len(ids_fn(s.span)) - 1)
        needed, delta = attack.feasibility(s.z0, summary["eligible_char_share"], tokens)
        record = {"sample_id": s.sample_id, "source": s.source, "layout": s.layout, "tokens": tokens,
                  "eligible_share": summary["eligible_char_share"], "needed_share": needed, "delta_needed": delta,
                  "segmentation": {k: v for k, v in summary.items() if k not in ("leading_fragment", "trailing_fragment")}}
        try:
            result = attack.attack_span(seg, para, gate, attack.SpanOracle(DetectorOracle(tok, "score"), snippets.get(s.sample_id, "")), cfg)
            repl = {int(k): v["rewrite"] for k, v in result["rewrites"].items()}
            record["attack"] = result
            record["measures"] = attack.change_measures(seg, repl, result["final_span"], ids_fn, sta_core.is_green_pair)
        except Exception:
            record["error"] = traceback.format_exc()
        write_json(path, record)
        done_now += 1
        left = len(selected) - n
        per = (time.time() - started) / done_now
        a = record.get("attack")
        print("[%d/%d] %-22s z0 %.2f -> %s | %s | %s | %.1f min | about %.0f min left" % (
            n, len(selected), s.sample_id, s.z0, ("%.2f" % a["z_final"]) if a else "ERROR", a["status"] if a else "see error in the sample file",
            "%d rewrites, %d queries" % (len(a["rewrites"]), a["queries"]) if a else "", (a["seconds"] / 60) if a else 0, per * left / 60), flush=True)

    run["ended"] = time.strftime("%Y-%m-%d %H:%M:%S")
    run["peak_gpu"] = "allocated %.2f GiB, reserved %.2f GiB" % (torch.cuda.max_memory_allocated() / 2 ** 30, torch.cuda.max_memory_reserved() / 2 ** 30)
    write_json(rundir / "run.json", run)
    print("\nreport:", report.build(rundir))
    print("to save the result to your repo, after reading it:  cat %s/STATUS_entry.md >> research/STATUS.md" % rundir)


if __name__ == "__main__":
    main()
