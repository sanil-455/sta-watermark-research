"""
Live progress monitor for the Qwen attack. Read-only: it never touches the run.

  python research/qwen_attack/monitor.py          refreshes every 5 seconds until Ctrl+C
  python research/qwen_attack/monitor.py --once   one snapshot

Progress is the finished share of the work. Each flagged span is weighted by
its number of eligible sentences (the work the attack does per span), and a
span counts when its result file is written, so the bar moves once per span.
Runs started with --only (quick checks) are ignored.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
GROUPS = {"200": ("200-token spans", ["results/raw/baseline_safe.json", "results/raw/baseline_extra.json", "results/raw/samples_len200.json"]),
          "500": ("500-token spans", ["results/raw/samples_len500.json"])}


def bar(frac, width=36):
    n = int(round(max(0.0, min(1.0, frac)) * width))
    return "[" + "#" * n + "-" * (width - n) + "]"


def hms(seconds):
    seconds = int(max(0, seconds))
    return "%dh%02dm" % (seconds // 3600, seconds % 3600 // 60) if seconds >= 3600 else "%dm%02ds" % (seconds // 60, seconds % 60)


def group_of(run):
    return "500" if any("len500" in f for f in run["args"].get("files", [])) else "200"


def full_runs(runs_dir):
    """Run folders that cover a whole group, oldest first (quick checks made with --only are skipped)."""
    out = []
    for d in sorted(Path(runs_dir).glob("run_*")):
        try:
            run = json.load(open(d / "run.json"))
        except Exception:
            continue
        if not run["args"].get("only"):
            out.append((d, run))
    return out


def weights_for(files):
    """sample id -> eligible sentence count (at least 1), from the span files."""
    sys.path[:0] = [str(HERE)]
    import samples
    import sentences
    return {s.sample_id: max(1, len(sentences.split_span(s.span).eligible())) for s in samples.load_samples(files)}


def snapshot(runs_dir, weights, now=None):
    now = now or time.time()
    runs = full_runs(runs_dir)
    groups, last_file, current, started_at = {}, None, None, None
    for key, (label, _) in GROUPS.items():
        mine = [(d, r) for d, r in runs if group_of(r) == key]
        d, run = mine[-1] if mine else (None, None)
        if d is not None and (d / "screening.json").exists():
            ids = [x["sample_id"] for x in json.load(open(d / "screening.json")) if x["flagged"]]
            estimated = False
        else:                                           # the group has not started: assume every span in its files is flagged
            ids = [i for i in weights if ("len500" in i) == (key == "500")]
            estimated = True
        done = {}
        if d is not None:
            for p in (d / "samples").glob("*.json"):
                sid = p.stem.replace("__", ":")
                try:
                    done[sid] = json.load(open(p))
                except Exception:
                    continue
                if last_file is None or p.stat().st_mtime > last_file[0]:
                    last_file = (p.stat().st_mtime, sid, done[sid])
        total = sum(weights.get(i, 1) for i in ids)
        finished = sum(weights.get(i, 1) for i in ids if i in done)
        seconds = sum(done[i]["attack"]["seconds"] for i in ids if i in done and "attack" in done[i])
        running = d is not None and run.get("ended") is None and any(i not in done for i in ids)
        if running:
            current = next(i for i in ids if i not in done)
        if run is not None and started_at is None:
            started_at = time.mktime(time.strptime(run["started"], "%Y-%m-%d %H:%M:%S"))
        groups[key] = {"label": label, "spans": len(ids), "spans_done": sum(1 for i in ids if i in done), "total": total,
                       "finished": finished, "seconds": seconds, "state": "running" if running else ("finished" if d is not None and finished >= total else ("waiting" if d is None else "stopped")),
                       "estimated": estimated, "run": d.name if d else None,
                       "broken": sum(1 for i in ids if i in done and done[i].get("attack", {}).get("status") == "broken")}
    total = sum(g["total"] for g in groups.values())
    finished = sum(g["finished"] for g in groups.values())
    secs, units = sum(g["seconds"] for g in groups.values()), sum(g["finished"] for g in groups.values())
    remaining = 0.0                         # each group is timed at its own pace once it has done enough work
    for g in groups.values():
        left = g["total"] - g["finished"]
        if left <= 0:
            continue
        rate = g["seconds"] / g["finished"] if g["finished"] >= 20 else (secs / units if units else None)
        if rate is None:
            remaining = None
            break
        remaining += left * rate
    since = None
    if current is not None:
        ref = last_file[0] if last_file else None
        if ref is None:
            ref = max((d / "run.json").stat().st_mtime for d, r in runs if r.get("ended") is None)
        since = now - ref
    return {"groups": groups, "total": total, "finished": finished, "fraction": finished / total if total else 0.0,
            "elapsed": (now - started_at) if started_at else None, "remaining": remaining, "current": current,
            "current_for": since, "last": last_file, "now": now, "no_runs": not runs}


def machine():
    alive = subprocess.run(["pgrep", "-f", "run_attack.py"], capture_output=True, text=True).stdout.split()
    try:
        gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"], text=True).strip().split("\n")[0].split(",")
        gpu = "GPU %s%% busy, %.1f of %.1f GiB" % (gpu[0].strip(), float(gpu[1]) / 1024, float(gpu[2]) / 1024)
    except Exception:
        gpu = "GPU unknown"
    return ("process running" if alive else "NO run_attack.py PROCESS FOUND"), gpu


def render(s, proc="", gpu=""):
    if s["no_runs"]:
        return "Qwen attack progress\n\nwaiting: no run folder yet under results/qwen_attack (the run has not started, or only quick checks exist)"
    lines = ["Qwen attack progress   %s" % time.strftime("%a %H:%M:%S", time.localtime(s["now"])), "",
             "OVERALL  %s %5.1f%%   %d of %d spans" % (bar(s["fraction"]), 100 * s["fraction"], sum(g["spans_done"] for g in s["groups"].values()), sum(g["spans"] for g in s["groups"].values()))]
    for g in s["groups"].values():
        frac = g["finished"] / g["total"] if g["total"] else 0.0
        lines.append("  %-16s %s %5.1f%%   %d of %d spans, %d broken   (%s%s)" % (g["label"], bar(frac, 28), 100 * frac, g["spans_done"], g["spans"], g["broken"], g["state"], ", span count assumed" if g["estimated"] else ""))
    lines.append("")
    if s["remaining"] is not None:
        lines.append("elapsed %s | estimated remaining %s | finish about %s" % (hms(s["elapsed"] or 0), hms(s["remaining"]), time.strftime("%a %H:%M", time.localtime(s["now"] + s["remaining"]))))
    else:
        lines.append("elapsed %s | estimated remaining: after the first span finishes" % hms(s["elapsed"] or 0))
    if s["current"]:
        lines.append("now attacking %s, running for %s (a span takes minutes; the bar moves when it finishes)" % (s["current"], hms(s["current_for"] or 0)))
    if s["last"]:
        a = s["last"][2].get("attack")
        lines.append("last finished: %s -> %s (z %.2f -> %.2f), %s ago" % (s["last"][1], a["status"] if a else "ERROR", a["z0"] if a else 0, a["z_final"] if a else 0, hms(s["now"] - s["last"][0])))
    if proc:
        lines.append("%s | %s" % (proc, gpu))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--every", type=float, default=5.0)
    args = ap.parse_args()
    os.chdir(ROOT)
    files = [f for _, fl in GROUPS.values() for f in fl if os.path.exists(f)]
    weights = weights_for(files)
    runs_dir = ROOT / "results" / "qwen_attack"
    try:
        while True:
            text = render(snapshot(runs_dir, weights), *machine())
            print(text if args.once else "\033[H\033[J" + text, flush=True)
            if args.once:
                break
            time.sleep(args.every)
    except KeyboardInterrupt:
        print("\nmonitor stopped (the run is not affected)")


if __name__ == "__main__":
    main()
