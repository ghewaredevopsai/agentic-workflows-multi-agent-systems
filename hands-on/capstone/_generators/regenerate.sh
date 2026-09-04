#!/usr/bin/env bash
# Rebuild everything generated for the capstone, then verify the gate itself.
# Offline: no cluster, no model, no network beyond a loopback socket.
set -euo pipefail
cd "$(dirname "$0")"
python3 gen_capstone.py
echo; echo "=== the gate must accept a perfect service and reject five broken ones ==="
python3 verify_capstone.py
