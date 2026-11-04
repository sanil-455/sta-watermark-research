#!/bin/bash
# STA-16 500-token: attack then no-box, on the already-generated sta16_len500.json (n=53).
# Pilot-gated: pilots one span, stops with diagnostics if it fails, else runs full.
set -u
cd "$(dirname "$0")/../.." || exit 1
OUT=results/qwen_attack_sta16_500
mkdir -p "$OUT"
L="$OUT/log.txt"
FILE=results/raw/sta16_len500.json
AD="$OUT/attack"
NB="$OUT/nobox"
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$L"; }
wait_pid() { while kill -0 "$1" 2>/dev/null; do sleep "${WAIT_POLL:-60}"; done; }
first_flagged() { python3 -c "import json,sys;r=json.load(open(sys.argv[1]));f=[x['sample_id'] for x in r if x.get('flagged')];print(f[0] if f else '')" "$1"; }
check_key() { python3 -c "import json,glob,sys;d,k=sys.argv[1],sys.argv[2];fs=glob.glob(d+'/samples/*.json');print('ok' if any(k in json.load(open(f)) and 'z_final' in json.load(open(f)).get(k,{}) for f in fs) else 'fail')" "$1" "$2"; }

# ---- STAGE 1: adaptive attack ----
say "===== STA-16 500-tok started ====="
[ -f "$FILE" ] || { say "ABORT: $FILE missing"; exit 1; }

mkdir -p "$AD"
if [ ! -f "$AD/screening.json" ]; then
    say "screening STA-16 500 (no attack) to list flagged ids"
    python3 research/qwen_attack/run_attack.py --files "$FILE" --only __none__ --resume "$AD" > "$AD.screen.log" 2>&1 &
    wait_pid $!
fi
[ -f "$AD/screening.json" ] || { { echo "no screening.json"; tail -40 "$AD.screen.log"; } > "$AD.PILOT_FAIL.txt"; say "screening failed -> $AD.PILOT_FAIL.txt"; exit 1; }
PID=$(first_flagged "$AD/screening.json")
[ -z "$PID" ] && { say "no flagged STA-16 500 spans; nothing to attack"; exit 0; }
say "attack PILOT on $PID"
python3 research/qwen_attack/run_attack.py --files "$FILE" --only "$PID" --resume "$AD" > "$AD.pilot.log" 2>&1 &
wait_pid $!
if [ "$(check_key "$AD" attack)" != "ok" ]; then
    { echo "=== STA-16 500 attack PILOT FAIL ==="; tail -40 "$AD.pilot.log"; } > "$AD.PILOT_FAIL.txt"
    say "attack pilot FAILED -> $AD.PILOT_FAIL.txt"; exit 1
fi
say "attack pilot passed -- full attack"
python3 research/qwen_attack/run_attack.py --files "$FILE" --resume "$AD" > "$AD.full.log" 2>&1 &
wait_pid $!
say "full attack done"

# ---- STAGE 2: no-box (needs the attack's screening for flagged set + z0) ----
mkdir -p "$NB"
NPID=$(first_flagged "$AD/screening.json")
say "nobox PILOT on $NPID"
python3 research/qwen_attack/nobox_attack.py --files "$FILE" --screening "$AD/screening.json" --out "$NB" --only "$NPID" > "$NB.pilot.log" 2>&1 &
wait_pid $!
if [ "$(check_key "$NB" nobox)" != "ok" ]; then
    { echo "=== STA-16 500 nobox PILOT FAIL ==="; tail -40 "$NB.pilot.log"; } > "$NB.PILOT_FAIL.txt"
    say "nobox pilot FAILED -> $NB.PILOT_FAIL.txt ; attack results still saved"; exit 1
fi
say "nobox pilot passed -- full nobox"
python3 research/qwen_attack/nobox_attack.py --files "$FILE" --screening "$AD/screening.json" --out "$NB" > "$NB.full.log" 2>&1 &
wait_pid $!
say "===== STA-16 500-tok finished ====="
