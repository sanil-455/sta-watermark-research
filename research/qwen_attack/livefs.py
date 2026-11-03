import glob, json, os, time

OUT = "results/qwen_attack_fullscale"
STAGES = [
    ("nobox500",     OUT+"/nobox500"),
    ("sta16-attack", OUT+"/sta16_attack"),
    ("sta16-nobox",  OUT+"/sta16_nobox"),
    ("adaptive500",  OUT+"/adaptive500"),
]

def flagged(d):
    s = d+"/screening.json"
    if os.path.exists(s):
        r = json.load(open(s)); return sum(1 for x in r if x.get("flagged"))
    return None

def cnt(d):
    return len(glob.glob(d+"/samples/*.json"))

try:
    while True:
        os.system("clear")
        print("LIVE  %s   (Ctrl+C to stop watching; runs keep going)\n" % time.strftime("%H:%M:%S"))
        if os.path.exists(OUT+"/queue.log"):
            for l in open(OUT+"/queue.log").read().splitlines()[-3:]:
                print("  " + l)
        print()
        for label, d in STAGES:
            if os.path.exists(d+".PILOT_FAIL.txt"):
                print("  %-13s  PILOT FAILED -> paste %s.PILOT_FAIL.txt" % (label, d)); continue
            tot = flagged(d); c = cnt(d)
            bar = ("%d / %d" % (c, tot)) if tot else ("%d done" % c if c else "waiting")
            print("  %-13s  %s" % (label, bar))
        n = len(json.load(open("results/raw/sta16_len200.json"))) if os.path.exists("results/raw/sta16_len200.json") else 0
        print("\n  sta16 samples generated: %d" % n)
        time.sleep(5)
except KeyboardInterrupt:
    print("\nstopped watching (runs unaffected)")
