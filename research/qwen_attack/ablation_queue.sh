#!/bin/bash
# Queues items 5, 6, 8 behind whatever GPU work is already running, smallest-first.
# FIX (v2): --files is read from the MAIN verified run's own run.json, not hardcoded --
# the subset ids can come from ANY of the files that run originally loaded (baseline_extra,
# baseline_safe, samples_len200, ...), and hardcoding one file meant --only silently matched
# nothing for ids from the others, so every stage "completed" instantly having attacked zero
# spans. A smoke check (one span, --only) now runs FIRST and the whole script aborts if that
# smoke check doesn't actually do real work, instead of silently producing five empty reports.
cd "$(dirname "$0")/../.." || exit 1
MAIN=results/qwen_attack/run_20261010_043023_lenient
OUT=results/qwen_attack_ablation
N_SPANS=${N_SPANS:-12}

wait_for_gpu() {
    while pgrep -f "[g]enerate_span_samples\.py|[g]enerate_sta16_samples\.py|[g]enerate_controls\.py|[r]un_attack\.py|[r]un_attack_target\.py" >/dev/null; do
        sleep "${WAIT_POLL:-60}"
    done
}

mkdir -p "$OUT"
python - <<PY
import json
run = json.load(open("$MAIN/run.json"))
files = [f for f in run["args"]["files"] if f]
open("$OUT/files.txt", "w").write(" ".join(files))
ids = sorted(x["sample_id"] for x in json.load(open("$MAIN/screening.json")) if x["flagged"])[:$N_SPANS]
open("$OUT/subset_ids.txt", "w").write(" ".join(ids))
print("files to load (from the main run's own run.json): %s" % files)
print("subset: %d spans -> $OUT/subset_ids.txt" % len(ids))
PY
FILES=$(cat "$OUT/files.txt")
IDS=$(cat "$OUT/subset_ids.txt")
echo "ablation subset ($N_SPANS spans): $IDS"
echo "files: $FILES"

FIRST_ID=$(echo $IDS | cut -d' ' -f1)
wait_for_gpu
echo "[0/5] SMOKE CHECK: one span ($FIRST_ID), must show real elapsed time and a z0 before trusting the rest"
python -u research/qwen_attack/run_attack.py --files $FILES --only "$FIRST_ID" --resume "$OUT/smoke_check" > "$OUT/smoke_check.log" 2>&1
python - <<PY
import json, sys
d = "$OUT/smoke_check"
run = json.load(open(d + "/run.json"))
if run["started"] == run["ended"]:
    print("ABORT: the smoke check completed instantly (started == ended), meaning the span was never actually attacked. Check files.txt and subset_ids.txt match before retrying.")
    sys.exit(1)
import glob
samples = glob.glob(d + "/samples/*.json")
if not samples:
    print("ABORT: no sample result file was written at all.")
    sys.exit(1)
s = json.load(open(samples[0]))
if "attack" not in s:
    print("ABORT: the sample file has no 'attack' key, so no attack actually ran on it.")
    sys.exit(1)
print("smoke check passed: span %s really ran, z0=%.2f, status=%s" % (s["sample_id"], s["attack"]["z0"], s["attack"]["status"]))
PY
[ $? -ne 0 ] && { echo "STOPPING: smoke check failed, the rest of the queue was not run"; exit 1; }

wait_for_gpu
echo "[1/5] n_cands=20"
python -u research/qwen_attack/run_attack.py --files $FILES --only $IDS --n-cands 20 --resume "$OUT/ncands20" > "$OUT/ncands20.log" 2>&1

wait_for_gpu
echo "[2/5] seed=777"
python -u research/qwen_attack/run_attack.py --files $FILES --only $IDS --seed 777 --resume "$OUT/seed777" > "$OUT/seed777.log" 2>&1

wait_for_gpu
echo "[3/5] n_cands=90"
python -u research/qwen_attack/run_attack.py --files $FILES --only $IDS --n-cands 90 --resume "$OUT/ncands90" > "$OUT/ncands90.log" 2>&1

wait_for_gpu
echo "[4/5] preset=strict"
python -u research/qwen_attack/run_attack.py --files $FILES --only $IDS --preset strict --resume "$OUT/strict" > "$OUT/strict.log" 2>&1

wait_for_gpu
echo "[5/5] target z<=1.5"
python -u research/qwen_attack/run_attack_target.py --target 1.5 --files $FILES --only $IDS --resume "$OUT/z15" > "$OUT/z15.log" 2>&1

echo "ablation queue finished"
