#!/bin/bash
set -u
cd "$(dirname "$0")/../.." || { echo "cannot cd to project root"; exit 1; }
OUT=results/qwen_attack_fullscale
mkdir -p "$OUT"
LOG="$OUT/queue.log"
FILES500="results/raw/samples_len500.json"
MAIN500=results/qwen_attack/run_20261010_061701_lenient
STA16_TARGET=${STA16_TARGET:-100}
PY=python
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
wait_pid() { while kill -0 "$1" 2>/dev/null; do sleep "${WAIT_POLL:-60}"; done; }
first_flagged() { $PY -c 'import json,sys; rows=json.load(open(sys.argv[1])); ids=[r["sample_id"] for r in rows if r.get("flagged")]; print(ids[0] if ids else "")' "$1"; }
check_adaptive() { $PY research/qwen_attack/_check_adaptive.py "$1"; }
check_nobox() { $PY research/qwen_attack/_check_nobox.py "$1"; }
stage_nobox() {
    local label="$1" scr="$2" files="$3" dir="$4"
    if [ ! -f "$scr" ]; then say "STAGE $label: screening $scr missing, skipped"; return; fi
    local pid_id; pid_id=$(first_flagged "$scr")
    if [ -z "$pid_id" ]; then say "STAGE $label: no flagged spans, skipped"; return; fi
    say "STAGE $label: pilot on $pid_id"
    $PY research/qwen_attack/nobox_attack.py --files $files --screening "$scr" --out "$dir" --only "$pid_id" > "$dir.pilot.log" 2>&1 &
    wait_pid $!
    local chk; chk=$(check_nobox "$dir")
    if ! echo "$chk" | grep -q "pilot ok"; then
        { echo "=== PILOT FAIL: $label ==="; echo "$chk"; echo "--- log tail ---"; tail -40 "$dir.pilot.log"; } > "$dir.PILOT_FAIL.txt"
        say "STAGE $label: PILOT FAILED -> $dir.PILOT_FAIL.txt ; next stage"; return
    fi
    say "STAGE $label: pilot passed -- full run"
    $PY research/qwen_attack/nobox_attack.py --files $files --screening "$scr" --out "$dir" > "$dir.full.log" 2>&1 &
    wait_pid $!
    say "STAGE $label: done"
}
stage_adaptive() {
    local label="$1" files="$2" dir="$3"; shift 3
    local extra=("$@"); mkdir -p "$dir"
    if [ ! -f "$dir/screening.json" ]; then
        say "STAGE $label: screening (no attack) to list flagged ids"
        $PY research/qwen_attack/run_attack.py --files $files --only __none__ --resume "$dir" "${extra[@]}" > "$dir.screen.log" 2>&1 &
        wait_pid $!
    fi
    if [ ! -f "$dir/screening.json" ]; then
        { echo "=== PILOT FAIL: $label -- no screening.json ==="; tail -40 "$dir.screen.log"; } > "$dir.PILOT_FAIL.txt"
        say "STAGE $label: screening failed -> $dir.PILOT_FAIL.txt ; next stage"; return
    fi
    local pid_id; pid_id=$(first_flagged "$dir/screening.json")
    if [ -z "$pid_id" ]; then say "STAGE $label: no flagged spans, skipped"; return; fi
    say "STAGE $label: pilot on $pid_id"
    $PY research/qwen_attack/run_attack.py --files $files --only "$pid_id" --resume "$dir" "${extra[@]}" > "$dir.pilot.log" 2>&1 &
    wait_pid $!
    local chk; chk=$(check_adaptive "$dir")
    if ! echo "$chk" | grep -q "pilot ok"; then
        { echo "=== PILOT FAIL: $label ==="; echo "$chk"; echo "--- log tail ---"; tail -40 "$dir.pilot.log"; } > "$dir.PILOT_FAIL.txt"
        say "STAGE $label: PILOT FAILED -> $dir.PILOT_FAIL.txt ; next stage"; return
    fi
    say "STAGE $label: pilot passed -- full run"
    $PY research/qwen_attack/run_attack.py --files $files --resume "$dir" "${extra[@]}" > "$dir.full.log" 2>&1 &
    wait_pid $!
    say "STAGE $label: done"
}
say "========== full_scale_queue started =========="
stage_nobox "nobox500" "$MAIN500/screening.json" "$FILES500" "$OUT/nobox500"
if [ -f research/experiments/generate_sta16_samples.py ]; then
    say "STAGE sta16-gen: target $STA16_TARGET"
    $PY research/experiments/generate_sta16_samples.py --target "$STA16_TARGET" > "$OUT/sta16_gen.log" 2>&1 &
    wait_pid $!
    md5sum -c /tmp/before_sta16.md5 2>/dev/null || say "note: sta16 md5 guard not present"
    if [ -f results/raw/sta16_len200.json ]; then
        stage_adaptive "sta16-attack" "results/raw/sta16_len200.json" "$OUT/sta16_attack"
        if [ -f "$OUT/sta16_attack/screening.json" ]; then
            stage_nobox "sta16-nobox" "$OUT/sta16_attack/screening.json" "results/raw/sta16_len200.json" "$OUT/sta16_nobox"
        fi
    else
        say "STAGE sta16: sta16_len200.json not produced, skipping"
    fi
else
    say "STAGE sta16: generator missing, skipped"
fi
stage_adaptive "adaptive500" "$FILES500" "$OUT/adaptive500"
say "========== full_scale_queue finished =========="
