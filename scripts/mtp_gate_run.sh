#!/usr/bin/env bash
set -uo pipefail
cd /home/patrickd/glm-5.3-reap
LOG=logs/mtp_gate.log
say() { echo "[$(date -Is)] mtp_gate: $*" >> "$LOG"; }
say "pausing extraction chain"
./scripts/overnight_ctl.sh stop >> "$LOG" 2>&1
sleep 5
say "=== gate2 ==="
./.venv/bin/python scripts/nvfp4_mtp_gate.py >> "$LOG" 2>&1
say "=== rc=$? ==="
say "resuming extraction chain"
./scripts/overnight_ctl.sh resume >> "$LOG" 2>&1
say "done"
