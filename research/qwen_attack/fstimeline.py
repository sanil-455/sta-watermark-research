import glob, os, time

OUT = "results/qwen_attack_fullscale"
# (label, result-dir or None for gen, kind)
STAGES = [
    ("nobox500",     OUT+"/nobox500",   "nobox"),
    ("sta16-gen",    None,              "gen"),
    ("sta16-attack", OUT+"/sta16_attack","adaptive"),
    ("sta16-nobox",  OUT+"/sta16_nobox", "nobox"),
    ("adaptive500",  OUT+"/adaptive500", "adaptive"),
]

def count(d):
    return len(glob.glob(d+"/samples/*.json")) if d and os.path.isdir(d+"/samples") else 0

def eta_from_files(d):
    # pace from the two newest sample-file mtimes
    fs = sorted(glob.glob(d+"/samples/*.json"), key=os.path.getmtime)
    if len(fs) < 2: return None
    span = os.path.getmtime(fs[-1]) - os.path.getmtime(fs[0])
    return span / (len(fs)-1) / 60.0  # minutes per span, averaged

print("Full-scale timeline  (%s)\n" % time.strftime("%H:%M:%S"))
if os.path.exists(OUT+"/queue.log"):
    for l in open(OUT+"/queue.log").read().splitlines()[-6:]:
        print("  "+l)
    print()
for label, d, kind in STAGES:
    fail = (d+".PILOT_FAIL.txt") if d else None
    if fail and os.path.exists(fail):
        print("  [PILOT FAILED] %-13s -> paste %s" % (label, fail)); continue
    if kind == "gen":
        n = len(__import__("json").load(open("results/raw/sta16_len200.json"))) if os.path.exists("results/raw/sta16_len200.json") else 0
        print("  [gen] %-13s %d STA-16 samples on disk" % (label, n)); continue
    c = count(d)
    mpr = eta_from_files(d)
    note = ""
    if mpr:
        note = " | ~%.1f min/span" % mpr
    print("  [%s] %-13s %d spans done%s" % (kind, label, c, note))
print("\nrun: tail results/qwen_attack_fullscale/queue.log for stage transitions")
