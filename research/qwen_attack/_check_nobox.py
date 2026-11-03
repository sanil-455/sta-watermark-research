import json, glob, sys
d = sys.argv[1]; files = glob.glob(d + "/samples/*.json")
if not files: print("PILOT FAIL: no sample files in " + d); sys.exit(1)
for f in files:
    s = json.load(open(f))
    if "nobox" in s and "z_final" in s["nobox"]: print("pilot ok: real nobox result"); sys.exit(0)
print("PILOT FAIL: files exist but no real nobox result"); sys.exit(1)
