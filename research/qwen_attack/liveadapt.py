import glob, json, os, time

D = "results/qwen_attack_fullscale/adaptive500"
tot = None
if os.path.exists(D+"/screening.json"):
    tot = sum(1 for x in json.load(open(D+"/screening.json")) if x.get("flagged"))

def newest_mtime(fs):
    return max(os.path.getmtime(f) for f in fs) if fs else 0

try:
    while True:
        os.system("clear")
        fs = glob.glob(D+"/samples/*.json")
        c = len(fs)
        print("adaptive500 LIVE  %s   (Ctrl+C to stop; run unaffected)\n" % time.strftime("%H:%M:%S"))
        print("  done: %d / %s" % (c, tot if tot else "?"))
        # measured pace from the last up-to-8 files
        if c >= 3:
            recent = sorted(fs, key=os.path.getmtime)[-8:]
            span = os.path.getmtime(recent[-1]) - os.path.getmtime(recent[0])
            if span > 0 and len(recent) > 1:
                mpr = span/(len(recent)-1)/60
                print("  pace: %.1f min/span (recent)" % mpr)
                if tot:
                    left = (tot-c)*mpr
                    print("  ~%d min (%.1f h) left | eta %s" % (left, left/60, time.strftime("%H:%M", time.localtime(time.time()+left*60))))
        age = (time.time()-newest_mtime(fs))/60 if fs else 0
        if age > 15:
            print("  WARNING: newest file %.0f min old - check if alive" % age)
        if tot and c >= tot:
            print("\n  DONE"); break
        time.sleep(10)
except KeyboardInterrupt:
    print("\nstopped watching (run unaffected)")
