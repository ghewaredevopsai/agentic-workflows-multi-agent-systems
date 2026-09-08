#!/usr/bin/env bash
# Rebuild every Module 9 lab + solution from the single source, then verify both directions.
set -euo pipefail
cd "$(dirname "$0")"
python3 gen_labs.py
echo; echo "=== solutions must score full marks ==="; python3 verify.py
echo; echo "=== untouched labs must survive Run All ==="; python3 verify_labs.py
echo; echo "=== solutions against the LIVE model (cluster only) ==="
echo "    python3 verify_live.py        # needs LAB_LLM_*; scrubs APP_* and LANGFUSE_*"
