#!/usr/bin/env bash
# Set the acceptance ceilings from a measurement. Run this INSIDE a sandbox, where the
# model env is present:
#
#   kubectl cp <capstone dir> agenticai/agenticaiu1-0:/home/jovyan/work/cap
#   kubectl exec -n agenticai agenticaiu1-0 -- bash /home/jovyan/work/cap/_generators/calibrate.sh
#
# It scores the reference service twice -- thinking off, then thinking on -- because the
# whole point of the ceilings is that neither the cheap careless configuration nor the
# slow careful one passes on its own. Copy the numbers it prints into acceptance.py and
# into index.html's "Where the numbers came from", and say when they were measured.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"

start() {                       # start(thinking) -> serves on :8000
  pkill -f "uvicorn app:app" 2>/dev/null || true
  sleep 1
  ( cd "$ROOT/reference" && LAB_LLM_THINKING="$1" PYTHONPATH="$ROOT/starter" \
      nohup python3 -m uvicorn app:app --host 127.0.0.1 --port 8000 \
      > /tmp/calibrate-$1.log 2>&1 & )
  for _ in $(seq 1 60); do
    curl -sf http://127.0.0.1:8000/healthz >/dev/null && return 0
    sleep 1
  done
  echo "uvicorn did not come up; see /tmp/calibrate-$1.log"; return 1
}

echo "=================== thinking OFF (all 45 cases) ==================="
start off && python3 acceptance.py --url http://127.0.0.1:8000 --workers 4 \
  --json /tmp/calibrate-off.json

echo
echo "=================== thinking ON (15 cases -- it is ~12x slower) ==================="
start on && python3 acceptance.py --url http://127.0.0.1:8000 --workers 2 --limit 15 \
  --json /tmp/calibrate-on.json

pkill -f "uvicorn app:app" 2>/dev/null || true
echo
echo "Both scorecards are in /tmp/calibrate-*.json."
echo "The floor must sit clearly above 84.4% -- the score of an agent that never screens"
echo "the counterparty -- and the reference must clear it with margin, or one of the two"
echo "is wrong and it is worth finding out which."
