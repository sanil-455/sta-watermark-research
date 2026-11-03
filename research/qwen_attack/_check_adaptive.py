import json, glob, sys
d = sys.argv[1]; files = glob.glob(d + "/samples/*.json")
if not files: print("PILOT FAIL: no sample files in " + d); sys.exit(1)
ok = 0
for f in files:
    s = json.load(open(f))
    if "error" in s: print("PILOT FAIL: %s errored:\n%s" % (s.get("sample_id"), s["error"][:2000])); sys.exit(1)
    if "attack" in s and "z_final" in s["attack"]: ok += 1
if ok == 0: print("PILOT FAIL: files but no real attack result (empty --only match?)"); sys.exit(1)
print("pilot ok: %d attacked span(s)" % ok)
