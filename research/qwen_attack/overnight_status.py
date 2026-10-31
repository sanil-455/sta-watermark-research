"""
Status of the whole overnight queue: which stage is active, its own real ETA, and which
stages haven't started yet. Read-only, CPU-only, safe to run any time.

  python research/qwen_attack/overnight_status.py
"""

import os
import re
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

# in run order; each entry is (label, log path, kind). "generator" logs need the usable-rate
# correction (some attempts are too-short and don't count); "attacker" logs give an already-
# unbiased ETA (every span gets exactly one progress line, no skipped attempts).
STAGES = [
    ("batch2: 200-token attack (213 new spans)", "scale200.log", "attacker"),
    ("STA-16: generation (target 10)", "overnight.log", "generator"),
    ("STA-16: attack", "overnight.log", "attacker"),
    ("ablation: n_cands=20", "results/qwen_attack_ablation/ncands20.log", "attacker"),
    ("ablation: seed=777", "results/qwen_attack_ablation/seed777.log", "attacker"),
    ("ablation: n_cands=90", "results/qwen_attack_ablation/ncands90.log", "attacker"),
    ("ablation: preset=strict", "results/qwen_attack_ablation/strict.log", "attacker"),
    ("ablation: target z<=1.5", "results/qwen_attack_ablation/z15.log", "attacker"),
]

ATTACKER_RE = re.compile(r"\[(\d+)/(\d+)\]\s+(\S+)\s+z0\s+[\d.]+\s*->\s*[\d.]+\s*\|\s*(\w[\w\s]*?)\s*\|.*?about\s+(\d+)\s+min left")
GENERATOR_RE = re.compile(r"\[(\d+)/(\d+)\]\s+prompt\s+\d+.*?about\s+(\d+)\s+min left")
TOO_SHORT_RE = re.compile(r"^prompt \d+: model stopped", re.M)
FINISHED_RE = re.compile(r"^(finished:|done$|report:)", re.M)   # generator scripts print "finished:"; run_attack.py prints "report:" (verified in its source)


def read(path):
    p = ROOT / path
    return p.read_text() if p.exists() else None


def parse_attacker(text):
    matches = ATTACKER_RE.findall(text)
    if not matches:
        return None
    k, n, last_id, last_status, left = matches[-1]
    return {"k": int(k), "n": int(n), "last_id": last_id, "last_status": last_status.strip(), "left_min": int(left), "finished": bool(FINISHED_RE.search(text))}


def parse_generator(text):
    matches = GENERATOR_RE.findall(text)
    if not matches:
        return None
    k, n, left = matches[-1]
    short = len(TOO_SHORT_RE.findall(text))
    rate = int(k) / max(1, int(k) + short)
    return {"k": int(k), "n": int(n), "left_min_reported": int(left), "left_min_real": int(left) / rate if rate else None,
            "usable_rate_pct": 100 * rate, "finished": bool(FINISHED_RE.search(text))}


def main():
    os.chdir(ROOT)
    print("Overnight queue status  (%s)\n" % time.strftime("%a %H:%M:%S"))
    active_found, known_remaining = False, 0.0
    unknown_stages = []
    for label, path, kind in STAGES:
        text = read(path)
        if text is None or not text.strip() or text.strip() == "nohup: ignoring input":
            print("  [not started]   %s" % label)
            unknown_stages.append(label)
            continue
        if kind == "attacker" and path == "overnight.log":
            # the STA-16 generation and attack phases share this one log file; a generator-phase
            # "finished:" line must never count as the ATTACK phase being done. Only look at text
            # after generation's own "exported:" line (its real end marker), if that line exists yet.
            cut = text.rfind("exported:")
            text = text[cut:] if cut >= 0 else ""
        info = (parse_generator(text) if kind == "generator" else parse_attacker(text))
        if info is None:
            print("  [starting up]   %s  (log exists but no progress line yet)" % label)
            continue
        if info["finished"]:
            print("  [DONE]          %s" % label)
            continue
        active_found = True
        if kind == "attacker":
            print("  [RUNNING]       %s -- %d of %d done, last: %s (%s), about %d min left (this stage's own estimate)" %
                  (label, info["k"], info["n"], info["last_id"], info["last_status"], info["left_min"]))
            known_remaining += info["left_min"]
        else:
            print("  [RUNNING]       %s -- %d of %d done, %.0f%% of attempts usable so far; reported ETA %d min, REAL (bias-corrected) ETA ~%.0f min" %
                  (label, info["k"], info["n"], info["usable_rate_pct"], info["left_min_reported"], info["left_min_real"]))
            known_remaining += info["left_min_real"]
    print()
    if active_found:
        print("Time left in the CURRENTLY RUNNING stage only: ~%.0f min (%.1f h)" % (known_remaining, known_remaining / 60))
    if unknown_stages:
        print("Stages with NO timing data yet (haven't started): %s" % ", ".join(unknown_stages))
        print("These cannot be given a real ETA until they start -- any number for them would be a guess, not a measurement.")
    print("\nThis script never estimates a stage before it has run at least one span/prompt of its own: a queued stage's cost is genuinely unknown until then.")


if __name__ == "__main__":
    main()
