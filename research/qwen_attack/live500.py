import glob, json, os, time

OUT = "results/qwen_attack_sta16_500"
AD, NB = OUT+"/attack", OUT+"/nobox"

def flagged():
    s = AD+"/screening.json"
    return sum(1 for x in json.load(open(s)) if x.get("flagged")) if os.path.exists(s) else None

def cnt(d):
    return len(glob.glob(d+"/samples/*.json"))

def pace_eta(d, tot):
    fs = sorted(glob.glob(d+"/samples/*.json"), key=os.path.getmtime)
    if len(fs) < 2 or not tot:
        return ""
    recent = fs[-6:]
    span = os.path.getmtime(recent[-1]) - os.path.getmtime(recent[0])
    if span <= 0:
        return ""
    mpr = span/(len(recent)-1)/60
    left = (tot-len(fs))*mpr
    eta = time.strftime("%a %H:%M", time.localtime(time.time()+left*60))
    return "  | %.1f min/span | ~%.1f h left | eta %s" % (mpr, left/60, eta)

try:
    while True:
        os.system("clear")
        print("STA-16 500-tok LIVE  %s  (Ctrl+C stops watcher; run unaffected)\n" % time.strftime("%H:%M:%S"))
        if os.path.exists(OUT+"/log.txt"):
            for l in open(OUT+"/log.txt").read().splitlines()[-5:]:
                print("  "+l)
        print()
        tot = flagged()
        for label, d in [("attack", AD), ("nobox", NB)]:
            if os.path.exists(d+".PILOT_FAIL.txt"):
                print("  %-7s PILOT FAILED -> paste %s.PILOT_FAIL.txt" % (label, d)); continue
            print("  %-7s %d / %s%s" % (label, cnt(d), tot if tot else "?", pace_eta(d, tot)))
        time.sleep(10)
except KeyboardInterrupt:
    print("\nstopped watching (run unaffected)")
